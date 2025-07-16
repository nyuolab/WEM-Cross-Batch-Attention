import os
import numpy as np
from tqdm import tqdm
import torch
from loader import WEMDataset, EOS_TOKEN, PAD_TOKEN
from torch.utils.data import DataLoader
from train_WEM.model import OLT2D, OLT2DConfig, CrossBatchDistributed
from torch.distributed.device_mesh import init_device_mesh
from datetime import timedelta
import argparse
import json
import time
import socket
from cut_cross_entropy import linear_cross_entropy

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import math
import sentencepiece as spm

MASK_TOKEN = 2
dtype = torch.bfloat16

torch.backends.cudnn.benchmark = True 

def run(args):
    world_size = int(os.environ["WORLD_SIZE"])
    tp_size = args.tp_size
    dp_size = world_size//tp_size
    device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("replicate", "tp"))

    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["replicate"]
    rank = dist.get_rank()

    tp_group = device_mesh.get_group(mesh_dim="tp")
    dp_group = device_mesh.get_group(mesh_dim="replicate")

    dp_rank = dp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()

    device = f"cuda:{rank % torch.cuda.device_count()}"
    node_name = socket.gethostname()

    distributed_comms = CrossBatchDistributed(tp_rank=tp_rank, dp_rank=dp_rank, group=tp_group, tp_size=tp_size)
    # print(f"Running on rank {rank}, node {os.environ['SLURMD_NODENAME']}, device {device}, node name {node_name}")
    print(f"batch: {args.batch_size },world: {world_size},minibatch {args.mini_batch_size}")

    assert args.batch_size % (dp_size * args.mini_batch_size) == 0, "Batch size must be divisible by the number of processes * mini batch size"

    ########### INITIALIZE DATALOADERS #################
    DNA_tokenizer = spm.SentencePieceProcessor()
    DNA_tokenizer.Load(args.tokenizer)

    tokenizer = DNA_tokenizer
    vocab_size = DNA_tokenizer.get_piece_size()

    if rank == 0:
        print(f"Using {args.tokenizer_suffix} tokenizer")

    base_dir = args.base_dir

    
    train_dataset = WEMDataset(directory=f"{base_dir}/x_chrom/feather/train", ctx_len=args.sequence_length, tokenizer=tokenizer)
    test_dataset = WEMDataset(directory=f"{base_dir}/x_chrom/feather/val", ctx_len=args.sequence_length, tokenizer=tokenizer)

    ################################################################

    logging = True # whether to log the training loss on wandb
    batch_size = args.batch_size // dp_size # batch size per process
    ctx_len = args.sequence_length * args.num_genes # context length of the model
    print(f"Real Batch Size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=2).__iter__()
    test_loader = DataLoader(test_dataset, batch_size=args.mini_batch_size, shuffle=False, num_workers=2, prefetch_factor=2).__iter__()

    # set up the model
    config = OLT2DConfig()
    config.length= args.num_genes 
    config.width = args.sequence_length
    config.vocab_size = vocab_size
    config.n_embd = args.n_embd
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.dropout = args.dropout
    config.position_encoding = args.position_encoding

    with open(f"{args.save_name}_config.json", "w") as f:
        json.dump(config.__dict__, f)

    m = OLT2D(config, distributed_comms)
    m = m.to(dtype).to(device)

    num_model_params = m.get_num_params()

    if os.path.exists(f"{args.save_name}_last_step.txt"):
        with open(f"{args.save_name}_last_step.txt", "r") as f:
            lines = f.readlines()
            starting_step = int(lines[0])
            resume_from = int(lines[1])
            wandb_id = lines[2].strip()
        
        m = torch.load(f"{args.save_name}_{resume_from}.pt", map_location=device)
        m.to(dtype).to(device)
        torch.cuda.empty_cache()
        print(f"Loaded model from {resume_from} token checkpoint")
    
    m.distributed_comms = distributed_comms

    if args.compile:
        m = torch.compile(m) # compile the model

    if args.FSDP:
        torch.cuda.set_device(rank % 8)
        model = FSDP(m, use_orig_params=True)
    else:
        model = DDP(m, device_ids=[rank % torch.cuda.device_count()], gradient_as_bucket_view=True)

    token_budget = args.token_budget # the number of tokens to train on
    total_iters = int(token_budget / (dp_size * batch_size * ctx_len)) # the total number of iterations to train for
    lr = args.lr*np.sqrt(args.batch_size) / 32 # the learning rate to use, scaled by batch size (default batch size is 1024)
    if args.force_lr:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.epsilon, fused=True) # initialize optimizer without muP
    else:
        def fixed_lr(param, name):
            '''
            Determines how LR should scale with a param given muP scaling laws
            '''
            keys = ["lm_head"]
            if any([k in name for k in keys]):
                return True
            if len(param.shape) == 1:
                return True
            
            return False
        
        def no_wd(param, name):
            '''
            Determines which parameters should not have weight decay
            '''
            keys = ["ln", "wte", "pos_emb"]
            if any([k in name for k in keys]):
                return True
            
            return False

        fixed_lr_params = [p for n, p in model.named_parameters() if fixed_lr(p, n) and not no_wd(p, n)]
        inverse_dmodel_params = [p for n, p in model.named_parameters() if not fixed_lr(p, n) and not no_wd(p, n)]
        no_wd_params = [p for n, p in model.named_parameters() if no_wd(p, n)]

        fixed_lr = lr
        inverse_dmodel_lr = lr * 32 / config.n_embd
        
        param_groups = [
            {"params": inverse_dmodel_params, "lr": inverse_dmodel_lr,
             "weight_decay": args.weight_decay * config.n_embd / 32}, # scales weight decay by inverse LR to keep weight decay constant (since AdamW weight decay is wd * lr)
            {"params": fixed_lr_params, "lr": fixed_lr, "weight_decay": args.weight_decay},
            {"params": no_wd_params, "lr": fixed_lr, "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(param_groups,
                                      betas=(args.beta1, args.beta2),
                                      eps=args.epsilon,
                                      fused=True)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[inverse_dmodel_lr, fixed_lr, fixed_lr], total_steps=total_iters + 2, pct_start=args.warmup_period, div_factor=1e5, final_div_factor=1e5)
    mini_batch_size = args.mini_batch_size # the mini batch size for gradient accumulation

    trained_tokens = 0
    starting_step = 0

    last_test = 0
    last_save = 0

    wandb_id = None

    if os.path.exists(f"{args.save_name}_last_step.txt"):
        with open(f"{args.save_name}_last_step.txt", "r") as f:
            lines = f.readlines()
            #starting_step = int(lines[0])
            resume_from = int(lines[1])
            starting_step = int(resume_from / (dp_size * batch_size * ctx_len)) # compute the starting step from the resume_from token in case the batch size has changed
            wandb_id = lines[2].strip()
        
        last_test = resume_from
        last_save = resume_from
        trained_tokens = resume_from

        print(f"Found checkpoint at {resume_from} tokens, resuming training...")

        optimizer_state_dict = torch.load(f"{args.save_name}_optimizer_{resume_from}.pt", map_location=device).state_dict()
        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        
        print(f"Loaded optimizer from {resume_from} token checkpoint")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[inverse_dmodel_lr, fixed_lr, fixed_lr],
                                                        total_steps=total_iters, pct_start=args.warmup_period,
                                                        div_factor=1e5, final_div_factor=1e5,
                                                        last_epoch=starting_step)
    if logging and rank == 0:
        import wandb

        needs_init = False
        if wandb_id is None:
            needs_init = True
            wandb_id = wandb.util.generate_id()
        
        wandb.init(project=args.wandb_project_name,
                   id=wandb_id,
                   resume="allow")
        
        if needs_init:
            wandb.run.name = f"N{args.n_layer} {args.position_encoding}-{args.tokenizer_suffix}"

    if rank == 0:
        pbar = tqdm(range(starting_step, total_iters))
    else:
        pbar = range(starting_step, total_iters)
    
    print(f"Rank {rank} ready to start training")
    dist.barrier()
    for i in pbar:
        start_time = time.time()
        batch_start = time.time()
        input_ids = next(train_loader).to(device)
        batch_end = time.time()
        input_ids = input_ids[:, :config.length, :]
        if rank == 0 and logging:
            wandb.log({"timing/batch_fetch_time": batch_end - batch_start}, step=trained_tokens)

        cum_loss = 0 # cumulative loss for this iteration
        optimizer.zero_grad(set_to_none=True)

        mask_prob = 0.15 # probability of masking a token

        # mask tokens
        mask = np.random.binomial(1, mask_prob, input_ids.shape)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        mask = mask & (input_ids != PAD_TOKEN) & (input_ids != EOS_TOKEN)
        
        masked_ids = input_ids.masked_fill(mask, MASK_TOKEN)
        masked_ids = distributed_comms.scatter_tensor(masked_ids, dim = 1)
        input_ids = distributed_comms.scatter_tensor(input_ids, dim = 1)

        mask = distributed_comms.scatter_tensor(mask, dim = 1)

        forward_times = []
        backward_times = []
        for j in range(input_ids.shape[0] // mini_batch_size): # gradient accumulation if we don't have enough GPUs
            mini_batch_x = masked_ids[j*mini_batch_size:(j+1)*mini_batch_size]
            mini_batch_y = input_ids[j*mini_batch_size:(j+1)*mini_batch_size]
            
            

            forward_start = time.time()
            embeddings = model.forward(mini_batch_x, return_embeddings=True).flatten(start_dim=0, end_dim=1) # (B, L, W, C) -> (B*L, W, C)
            lm_head = model.module.lm_head.weight
            forward_times.append(time.time() - forward_start)
            loss = linear_cross_entropy(embeddings, lm_head, mini_batch_y.flatten(start_dim=0, end_dim=1), reduction="none", impl="torch_compile") / (batch_size // mini_batch_size)
            loss *= mask[j*mini_batch_size:(j+1)*mini_batch_size].flatten(start_dim=0, end_dim=1).float()
            loss = loss.sum() / mask[j*mini_batch_size:(j+1)*mini_batch_size].view(-1).sum()

            backward_start = time.time()
            loss.backward()
            
            backward_times.append(time.time() - backward_start)

            cum_loss += loss.item()

        optimizer_start = time.time()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        try:
            scheduler.step()
        except Exception as e:
            print(f"Error in scheduler: {e}. This is expected if the scheduler is not configured correctly.")

        optimizer_end = time.time()

        if rank == 0 and logging:
            wandb.log({"timing/forward_time": np.mean(forward_times), "timing/backward_time": np.mean(backward_times), "timing/optimizer_time": optimizer_end - optimizer_start}, step=trained_tokens)

        # get current learning rate
        lrs = []
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            lrs.append(lr)

        if len(lrs) == 1:
            lrs.append(lrs[0])

        all_cum_loss = [None for _ in range(world_size)]
        dist.all_gather_object(all_cum_loss, cum_loss)
        cum_loss = np.mean(all_cum_loss) # this technically isn't exactly the same as the loss (due to different padding lengths), but it's close enough

        if rank == 0:
            if trained_tokens < 1e6:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e3:.3f}K")
            elif trained_tokens < 1e9:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e6:.3f}M")
            else:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e9:.3f}B")

            if logging:
                wandb.log({"loss": cum_loss, "lr": lrs[0], "batch_size": batch_size * dp_size}, step=trained_tokens)

        # count total number of tokens in the batch
        
        num_tokens = (input_ids != PAD_TOKEN).sum().item()

        all_num_tokens = [None for _ in range(world_size)]
        # the first argument is the collected lists, the second argument is the data unique in each process
        dist.all_gather_object(all_num_tokens, num_tokens)

        end_time = time.time()

        # compute efficiency relative to A100
        tokens_per_sec = sum(all_num_tokens) / (end_time - start_time)
        actual_flops = 6  * num_model_params + 12 * args.n_layer * args.n_embd * ((args.sequence_length + args.num_genes)/2) # 6N + 12*LHQT estimate of flops per token
        actual_flops *= sum(all_num_tokens)
        flops_per_sec = actual_flops / (end_time - start_time)
        if 'H100' in torch.cuda.get_device_name(0):
            gpu_flops_per_sec = 1513e12
        else:
            gpu_flops_per_sec = 312e12

        efficiency = flops_per_sec / (gpu_flops_per_sec * world_size) * 100

        if rank == 0 and logging:
            wandb.log({"timing/tokens_per_sec": tokens_per_sec, "timing/total_train_step_time": end_time - start_time, "GPU efficiency": efficiency}, step=trained_tokens)

        trained_tokens += sum(all_num_tokens)

        if trained_tokens - last_test > args.test_freq:
            # test the model
            model.eval()

            with torch.no_grad():
                total_loss = 0
                num_samples = 1024 // world_size + 1 # sample ~1M tokens
                for _ in range(num_samples): # do 100 mini batches to get an estimate of the validation loss
                    test_batch = next(test_loader).to(device)
                    test_batch = distributed_comms.scatter_tensor(test_batch, dim=1)

                    mask = np.random.binomial(1, mask_prob, test_batch.shape)
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
                    mask = mask & (test_batch != PAD_TOKEN) & (test_batch != EOS_TOKEN)
                    masked_ids = test_batch.masked_fill(mask, MASK_TOKEN)
                    

                    logits = model.forward(masked_ids)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), test_batch.view(-1), reduction="none")
                    loss *= mask.view(-1).float()
                    loss = loss.sum() / mask.view(-1).sum() / world_size

                    total_loss += loss.item()

                loss = total_loss / num_samples

                all_losses = [None for _ in range(world_size)]
                dist.all_gather_object(all_losses, loss)

                all_num_tokens = [None for _ in range(world_size)]
                dist.all_gather_object(all_num_tokens, mask.view(-1).sum().item())

                # log the test loss
                if logging and rank == 0:
                    wandb.log({f"test_loss": float(np.sum(all_losses))}, step=trained_tokens)

            model.train()
            del test_batch, mask, masked_ids, logits, loss
            torch.cuda.empty_cache()


            last_test = trained_tokens

        if rank == 0 and trained_tokens - last_save > args.save_freq:
            torch.save(model.module, f"{args.save_name}_{trained_tokens}.pt")
            torch.save(optimizer, f"{args.save_name}_optimizer_{trained_tokens}.pt")

            with open(f"{args.save_name}_last_step.txt", "w") as f:
                f.write(str(i) + "\n")
                f.write(str(trained_tokens) + "\n")
                f.write(wandb_id)

            if last_save > 0:
                os.remove(f"{args.save_name}_{last_save}.pt")
                os.remove(f"{args.save_name}_optimizer_{last_save}.pt")

            last_save = trained_tokens

        total_loop_time = time.time() - start_time
        if rank == 0 and logging:
            wandb.log({"timing/total_loop_time": total_loop_time}, step=trained_tokens)

    if rank == 0:
        torch.save(model.module, f"{args.save_name}.pt")
        torch.save(optimizer, f"{args.save_name}_optimizer.pt")


    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024, help="The total batch size across all processes")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="The batch size for gradient accumulation, and the batch size for each process")
    parser.add_argument("--n_head", type=int, default=2, help="The number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256, help="The embedding dimension")
    parser.add_argument("--n_layer", type=int, default=2, help="The number of transformer layers")
    parser.add_argument("--sequence_length", type=int, default=4096, help="The context length")
    parser.add_argument("--num_genes", type=int, default=893, help="The context length")
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
    parser.add_argument("--use_padding", action="store_true", default=False, help="Whether to pad the sequence with PAD_TOKEN instead of truncating a line if it doesn't fit in")
    parser.add_argument("--position_encoding", type=str, default="rope", help="The type of position encoding to use")
    parser.add_argument("--tokenizer", type=str, help="The path to the DNA tokenizer")
    parser.add_argument("--tokenizer_suffix", type=str, required=True, help="The suffix of the tokenizer to use")
    parser.add_argument("--compile", action="store_true", default=False, help="Whether to compile the model")
    parser.add_argument("--gpu_type", type=str, default='h100', help="GPU type, only H100 and A100 are supported")
    parser.add_argument("--tp_size", type=int, default=4, help="Number of GPUs in the sequence parallel")
    args = parser.parse_args()

    run(args)