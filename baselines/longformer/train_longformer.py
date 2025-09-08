import os
import numpy as np
from tqdm import tqdm
import torch
from loader import OmniDataset, EOS_TOKEN, PAD_TOKEN
from torch.utils.data import DataLoader
from datetime import timedelta
import argparse
import json
import time
import socket

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from dataloaders.text_loader import StreamingJsonlTokenDataset
from baselines.longformer.model import SparseAttentionConfig, SparseAttentionTransformer  # uses your sparse model

import sentencepiece as spm

MASK_TOKEN = 2
dtype = torch.bfloat16

torch.backends.cudnn.benchmark = True  # enable cuDNN benchmarking for faster training


# -------------- helpers --------------
def _parse_global_indices(s: str):
    """Parse comma-separated indices -> sorted tuple[int] or None."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    toks = [t for t in s.replace(" ", "").split(",") if t != ""]
    if not toks:
        return None
    return tuple(sorted({int(t) for t in toks}))


def _build_scheduler(optimizer, total_steps, pct_start, inverse_dmodel_lr, fixed_lr):
    """Make OneCycleLR with the right number of max_lr values for the optimizer's param groups."""
    if len(optimizer.param_groups) == 1:
        max_lr = fixed_lr
    elif len(optimizer.param_groups) == 3:
        max_lr = [inverse_dmodel_lr, fixed_lr, fixed_lr]
    else:
        # Fallback: same LR for each group
        max_lr = [fixed_lr] * len(optimizer.param_groups)
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps + 2,
        pct_start=pct_start,
        div_factor=1e5,
        final_div_factor=1e5,
    )


def run(args):
    # -------------- distributed init --------------
    dist.init_process_group("nccl")
    gloo_group = dist.new_group(backend="gloo", timeout=timedelta(hours=1))
    nccl_group = dist.new_group(backend="nccl", timeout=timedelta(hours=1))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = f"cuda:{rank % torch.cuda.device_count()}"
    node_name = socket.gethostname()
    print(f"batch: {args.batch_size}, world: {world_size}, minibatch {args.mini_batch_size}")

    assert args.batch_size % (world_size * args.mini_batch_size) == 0, \
        "Batch size must be divisible by the number of processes * mini batch size"

    # -------------- data --------------
    tokenizer_offset = spm.SentencePieceProcessor(model_file=args.tokenizer).get_piece_size()
    base_dir = args.base_dir
    print(f'base dir: {base_dir}')

    # ### UPDATED: dataset ctx_len now respects --ctx_len
    train_dataset = StreamingJsonlTokenDataset(
        directory=f"{base_dir}/train",
        sp_model_path=args.tokenizer,
        ctx_len=args.ctx_len,       # <---
        buf_lines=8192,
        char_cap=None,
        concat_docs=True,
        append_bos=False,
        insert_eos_between_docs=True,
        drop_short_docs_under_chars=0,
        min_tokens_after_encode=0,
        shuffle_files=True,
        repeat_forever=True,
    )

    test_dataset = StreamingJsonlTokenDataset(
        directory=f"{base_dir}/val",
        sp_model_path=args.tokenizer,
        ctx_len=args.ctx_len,       # <---
        buf_lines=8192,
        char_cap=None,
        concat_docs=True,
        append_bos=False,
        insert_eos_between_docs=True,
        drop_short_docs_under_chars=0,
        min_tokens_after_encode=0,
        shuffle_files=True,
        repeat_forever=True,
    )

    logging = True
    batch_size = args.batch_size // world_size
    ctx_len = args.ctx_len

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=2).__iter__()
    test_loader = DataLoader(test_dataset, batch_size=args.mini_batch_size, shuffle=False, num_workers=2, prefetch_factor=2).__iter__()

    # -------------- model config --------------
    # ### UPDATED: map legacy --longformer_window to --block_size when provided
    block_size = args.block_size if args.block_size is not None else int(args.longformer_window)

    config = SparseAttentionConfig(
        context_length=args.ctx_len,
        vocab_size=tokenizer_offset,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=False,  # keep your default
        autoregressive=args.autoregressive,  # ### NEW
        position_encoding=args.position_encoding,
        memory_efficient=False,  # inherited field; keep default unless you need checkpointing switch
        padding_idx=PAD_TOKEN,

        # ### NEW: sparsity knobs
        block_size=block_size,
        local_span_blocks=args.local_span_blocks,
        num_random=args.num_random,
        random_per_head=args.random_per_head,
        global_blocks=_parse_global_indices(args.global_blocks),
        random_seed=args.random_seed,
        allow_dense_sdpa_dropout_fallback=args.dense_dropout_fallback,
    )

    # -------------- instantiate model --------------
    m = SparseAttentionTransformer(config)
    m = m.to(dtype).to(device)
    raw_model = m  # keep a handle to the *unwrapped* model for saving/resume in FSDP

    # Save config (as plain dict; dataclass or not)
    try:
        from dataclasses import asdict, is_dataclass
        cfg_dump = asdict(config) if is_dataclass(config) else config.__dict__
    except Exception:
        cfg_dump = config.__dict__
    with open(f"{args.save_name}_config.json", "w") as f:
        json.dump(cfg_dump, f, indent=2)

    num_model_params = raw_model.get_num_params()

    # -------------- optional resume --------------
    starting_step = 0
    trained_tokens = 0
    last_test = 0
    last_save = 0
    wandb_id = None

    if os.path.exists(f"{args.save_name}_last_step.txt"):
        with open(f"{args.save_name}_last_step.txt", "r") as f:
            lines = f.readlines()
            starting_step_file = int(lines[0])
            resume_from = int(lines[1])
            wandb_id = lines[2].strip()

        # load raw model (unwrapped), then re-wrap later
        raw_model = torch.load(f"{args.save_name}_{resume_from}.pt", map_location=device)
        raw_model.to(dtype).to(device)
        m = raw_model  # keep names consistent downstream
        torch.cuda.empty_cache()
        print(f"Loaded model from {resume_from} token checkpoint")

    if args.compile:
        m = torch.compile(m)  # compile the raw model

    # -------------- DDP/FSDP wrap --------------
    if args.FSDP:
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        model = FSDP(m, use_orig_params=True)
    else:
        model = DDP(m, device_ids=[rank % torch.cuda.device_count()])

    # -------------- optimizer & scheduler --------------
    token_budget = args.token_budget
    total_iters = int(token_budget / (world_size * batch_size * ctx_len))
    base_lr = args.lr * np.sqrt(args.batch_size) / 32
    inverse_dmodel_lr = base_lr * 32 / args.n_embd  # keep muP relation

    if args.force_lr:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
            fused=True
        )
    else:
        def fixed_lr(param, name):
            keys = ["lm_head"]
            if any([k in name for k in keys]): return True
            if len(param.shape) == 1: return True
            return False

        def no_wd(param, name):
            keys = ["ln", "wte", "pos_emb"]
            if any([k in name for k in keys]): return True
            return False

        fixed_lr_params = [p for n, p in model.named_parameters() if fixed_lr(p, n) and not no_wd(p, n)]
        inverse_dmodel_params = [p for n, p in model.named_parameters() if not fixed_lr(p, n) and not no_wd(p, n)]
        no_wd_params = [p for n, p in model.named_parameters() if no_wd(p, n)]

        fixed_lr_scalar = base_lr
        inverse_dmodel_lr_scalar = inverse_dmodel_lr

        param_groups = [
            {
                "params": inverse_dmodel_params,
                "lr": inverse_dmodel_lr_scalar,
                "weight_decay": args.weight_decay * args.n_embd / 32
            },
            {
                "params": fixed_lr_params,
                "lr": fixed_lr_scalar,
                "weight_decay": args.weight_decay
            },
            {
                "params": no_wd_params,
                "lr": fixed_lr_scalar,
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
            fused=True
        )

    # Build scheduler consistent with the optimizer's number of param groups
    scheduler = _build_scheduler(
        optimizer=optimizer,
        total_steps=total_iters,
        pct_start=args.warmup_period,
        inverse_dmodel_lr=inverse_dmodel_lr,
        fixed_lr=base_lr,
    )

    # -------------- resume optimizer/scheduler if present --------------
    if os.path.exists(f"{args.save_name}_last_step.txt"):
        with open(f"{args.save_name}_last_step.txt", "r") as f:
            lines = f.readlines()
            resume_from = int(lines[1])
            starting_step = int(resume_from / (world_size * batch_size * ctx_len))
            wandb_id = lines[2].strip()

        last_test = resume_from
        last_save = resume_from
        trained_tokens = resume_from

        print(f"Found checkpoint at {resume_from} tokens, resuming training...")

        try:
            opt_loaded = torch.load(f"{args.save_name}_optimizer_{resume_from}.pt", map_location=device)
            optimizer.load_state_dict(opt_loaded.state_dict())
            del opt_loaded
            print(f"Loaded optimizer from {resume_from} token checkpoint")
        except Exception as e:
            print(f"Warning: failed to load optimizer state: {e}")

        # Rebuild scheduler with last_epoch; try loading state
        scheduler = _build_scheduler(
            optimizer=optimizer,
            total_steps=total_iters,
            pct_start=args.warmup_period,
            inverse_dmodel_lr=inverse_dmodel_lr,
            fixed_lr=base_lr,
        )
        try:
            sched_state = torch.load(f"{args.save_name}_sched_{resume_from}.pt", map_location=device)
            scheduler.load_state_dict(sched_state)
        except Exception as e:
            print(f"Warning: failed to load scheduler state (continuing with fresh scheduler): {e}")

    # -------------- wandb --------------
    if logging and rank == 0:
        import wandb
        needs_init = False
        if wandb_id is None:
            needs_init = True
            wandb_id = wandb.util.generate_id()
        wandb.init(project=args.wandb_project_name, id=wandb_id, resume="allow")
        if needs_init:
            wandb.run.name = f"N{args.n_layer} {args.position_encoding}"

        # Log config once
        wandb.config.update(cfg_dump, allow_val_change=True)

    # -------------- training loop --------------
    if rank == 0:
        pbar = tqdm(range(starting_step, total_iters))
    else:
        pbar = range(starting_step, total_iters)

    print(f"Rank {rank} ready to start training")
    dist.barrier(group=gloo_group)

    for i in pbar:
        start_time = time.time()

        batch_start = time.time()
        input_ids = next(train_loader).to(device)
        batch_end = time.time()

        if rank == 0 and logging:
            wandb.log({"timing/batch_fetch_time": batch_end - batch_start}, step=trained_tokens)

        cum_loss = 0
        optimizer.zero_grad(set_to_none=True)

        mask_prob = 0.15  # MLM

        # mask tokens
        mask = np.random.binomial(1, mask_prob, input_ids.shape)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        mask = mask & (input_ids != PAD_TOKEN) & (input_ids != EOS_TOKEN)
        masked_ids = input_ids.masked_fill(mask, MASK_TOKEN)

        forward_times, backward_times = [], []
        for j in range(input_ids.shape[0] // args.mini_batch_size):
            mini_batch_x = masked_ids[j*args.mini_batch_size:(j+1)*args.mini_batch_size]
            mini_batch_y = input_ids[j*args.mini_batch_size:(j+1)*args.mini_batch_size]

            forward_start = time.time()
            # NOTE: attn_mask=None is fine—SparseAttentionTransformer will auto-build from padding_idx
            logits = model(mini_batch_x, attn_mask=None)
            forward_times.append(time.time() - forward_start)

            # per-token CE, normalize by grad-accum steps
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                mini_batch_y.view(-1),
                reduction="none"
            ) / (batch_size // args.mini_batch_size)

            # only on masked tokens
            cur_mask = mask[j*args.mini_batch_size:(j+1)*args.mini_batch_size].view(-1)
            loss = (loss * cur_mask.float()).sum() / (cur_mask.sum().clamp_min(1.0))

            backward_start = time.time()
            loss.backward()
            backward_times.append(time.time() - backward_start)

            cum_loss += loss.item()

        optimizer_start = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        try:
            scheduler.step()
        except Exception as e:
            # If scheduler config mismatched param groups (shouldn't happen with _build_scheduler)
            print(f"Error in scheduler: {e} (continuing)")

        optimizer_end = time.time()

        if rank == 0 and logging:
            wandb.log({
                "timing/forward_time": float(np.mean(forward_times)),
                "timing/backward_time": float(np.mean(backward_times)),
                "timing/optimizer_time": optimizer_end - optimizer_start
            }, step=trained_tokens)

        # get current learning rate(s)
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        if len(lrs) == 1:
            lrs.append(lrs[0])

        # gather and log loss across ranks
        all_cum_loss = [None for _ in range(world_size)]
        dist.all_gather_object(all_cum_loss, cum_loss, group=gloo_group)
        cum_loss = float(np.mean(all_cum_loss))

        if rank == 0:
            if trained_tokens < 1e6:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e3:.3f}K")
            elif trained_tokens < 1e9:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e6:.3f}M")
            else:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e9:.3f}B")

            if logging:
                wandb.log({"loss": cum_loss, "lr": lrs[0], "batch_size": batch_size * world_size}, step=trained_tokens)

        # count non-pad tokens
        num_tokens = (input_ids != PAD_TOKEN).sum().item()
        all_num_tokens = [None for _ in range(world_size)]
        dist.all_gather_object(all_num_tokens, num_tokens, group=gloo_group)

        end_time = time.time()

        tokens_per_sec = sum(all_num_tokens) / (end_time - start_time)
        actual_flops = 6 * num_model_params + 12 * args.n_layer * args.n_embd * args.ctx_len
        actual_flops *= sum(all_num_tokens)
        if 'H100' in torch.cuda.get_device_name(0):
            gpu_flops_per_sec = 1513e12
        else:
            gpu_flops_per_sec = 312e12
        efficiency = actual_flops / (gpu_flops_per_sec * world_size) * 100

        if rank == 0 and logging:
            wandb.log({
                "timing/tokens_per_sec": tokens_per_sec,
                "timing/total_train_step_time": end_time - start_time,
                "GPU efficiency": efficiency
            }, step=trained_tokens)

        trained_tokens += sum(all_num_tokens)

        # -------------- validation --------------
        if trained_tokens - last_test > args.test_freq:
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                num_samples = 1024 // world_size + 1
                for _ in range(num_samples):
                    test_batch = next(test_loader).to(device)
                    mask = np.random.binomial(1, mask_prob, test_batch.shape)
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
                    mask = mask & (test_batch != PAD_TOKEN) & (test_batch != EOS_TOKEN)
                    masked_ids = test_batch.masked_fill(mask, MASK_TOKEN)
                    logits = model(masked_ids, attn_mask=None)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        test_batch.view(-1),
                        reduction="none"
                    )
                    loss *= mask.view(-1).float()
                    # average per-rank
                    loss = loss.sum() / mask.view(-1).sum().clamp_min(1.0) / world_size
                    total_loss += loss.item()

                loss = total_loss / num_samples

                all_losses = [None for _ in range(world_size)]
                dist.all_gather_object(all_losses, loss, group=gloo_group)

                if logging and rank == 0:
                    wandb.log({f"test_loss": float(np.sum(all_losses))}, step=trained_tokens)

            model.train()
            last_test = trained_tokens

        # -------------- checkpointing --------------
        if rank == 0 and trained_tokens - last_save > args.save_freq:
            # Save the *raw* model for both DDP and FSDP cases
            torch.save(raw_model, f"{args.save_name}_{trained_tokens}.pt")
            torch.save(optimizer, f"{args.save_name}_optimizer_{trained_tokens}.pt")
            torch.save(scheduler.state_dict(), f"{args.save_name}_sched_{trained_tokens}.pt")

            with open(f"{args.save_name}_last_step.txt", "w") as f:
                f.write(str(i) + "\n")
                f.write(str(trained_tokens) + "\n")
                f.write(wandb_id if wandb_id is not None else "")

            if last_save > 0:
                try:
                    os.remove(f"{args.save_name}_{last_save}.pt")
                    os.remove(f"{args.save_name}_optimizer_{last_save}.pt")
                except Exception as e:
                    print(f"Warning: failed to remove old checkpoints: {e}")

            last_save = trained_tokens

        total_loop_time = time.time() - start_time
        if rank == 0 and logging:
            wandb.log({"timing/total_loop_time": total_loop_time}, step=trained_tokens)

    if rank == 0:
        torch.save(raw_model, f"{args.save_name}.pt")
        torch.save(optimizer, f"{args.save_name}_optimizer.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024, help="The total batch size across all processes")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="The batch size for gradient accumulation, and the batch size for each process")
    parser.add_argument("--n_head", type=int, default=2, help="The number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256, help="The embedding dimension")
    parser.add_argument("--n_layer", type=int, default=2, help="The number of transformer layers")
    parser.add_argument("--ctx_len", type=int, default=1024, help="The context length")
    parser.add_argument("--dropout", type=float, default=0.05, help="The dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate (scaled by muP, unless --force_lr is set)")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for AdamW")
    parser.add_argument("--beta2", type=float, default=0.999, help="The beta2 parameter for AdamW")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="The epsilon parameter for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="The weight decay parameter for AdamW")
    parser.add_argument("--token_budget", type=float, default=20e9, help="The number of tokens to train on")
    parser.add_argument("--test_freq", type=int, default=1e7, help="The number of tokens between tests")
    parser.add_argument("--save_freq", type=int, default=1e9, help="The number of tokens between saves")
    parser.add_argument("--save_name", type=str, default="omnibiota", help="The prefix name to save the model as")
    parser.add_argument("--disable_flash", action="store_true", default=False, help="Whether to disable flash attention")
    parser.add_argument("--wandb_project_name", type=str, default="omnibiota", help="The name of the wandb project to log to")
    parser.add_argument("--base_dir", type=str, default="", help="The base directory for the training and validation data")
    parser.add_argument("--force_lr", action="store_true", default=False, help="Whether to override muP's learning rate scaling")
    parser.add_argument("--checkpoint_freq", type=int, default=0, help="The frequency at which activations are checkpointed")
    parser.add_argument("--warmup_period", type=float, default=0.01, help="The proportion of the total iterations to warm up")
    parser.add_argument("--FSDP", action="store_true", default=False, help="Whether to use FullyShardedDataParallel")
    parser.add_argument("--use_padding", action="store_true", default=False, help="Whether to pad sequences with PAD_TOKEN when lines don't fit")
    parser.add_argument("--position_encoding", type=str, default="rope", help="The type of position encoding to use")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer model")
    parser.add_argument("--compile", action="store_true", default=False, help="Whether to compile the model")

    # ### UPDATED / NEW sparse-related flags
    parser.add_argument("--block_size", type=int, default=None, help="Block size for block-sparse attention (defaults to --longformer_window if not set)")
    parser.add_argument("--local_span_blocks", type=int, default=1, help="Number of neighbor blocks on each side for local attention")
    parser.add_argument("--num_random", type=int, default=0, help="Number of random key blocks per query block")
    # togglable boolean: --random_per_head / --no_random_per_head
    parser.add_argument("--random_per_head", dest="random_per_head", action="store_true")
    parser.add_argument("--no_random_per_head", dest="random_per_head", action="store_false")
    parser.set_defaults(random_per_head=True)
    parser.add_argument("--global_blocks", type=str, default="", help="Comma-separated token indices to mark as global (0-based); empty for none")
    parser.add_argument("--random_seed", type=int, default=0, help="Seed for random block selection")
    parser.add_argument("--dense_dropout_fallback", action="store_true", default=False,
                        help="Use dense SDPA (slow) to enable true attention-dropout; for debugging/short sequences only")
    parser.add_argument("--autoregressive", action="store_true", default=False,
                        help="Enable causal attention (k<=q). For masked LM keep this False.")

    # legacy arg kept for backward compat; maps to block_size if --block_size not set
    parser.add_argument("--longformer_window", type=int, default=512, help="(Deprecated) maps to --block_size")

    args = parser.parse_args()
    run(args)
