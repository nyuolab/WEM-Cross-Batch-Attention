import os, gzip, json, io, random
from typing import Iterator, List, Optional
from collections import deque

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import sentencepiece as spm
from torch.utils.data import DataLoader
import time

def _safe_sp_ids(tok: spm.SentencePieceProcessor):
    # Return (pad_id, bos_id, eos_id) with safe fallbacks
    def safe(x):
        return x if x is not None and x >= 0 else None
    pad_id = safe(getattr(tok, "pad_id", lambda: -1)())
    bos_id = safe(getattr(tok, "bos_id", lambda: -1)())
    eos_id = safe(getattr(tok, "eos_id", lambda: -1)())
    return pad_id, bos_id, eos_id

def iter_json_text_gz(path: str) -> Iterator[str]:
    """Stream 'text' from a .json.gz file, tolerant to glitches."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                # Parse per line; skip bad lines fast
                try:
                    obj = json.loads(raw)
                    s = obj.get("text", "")
                    if s:
                        yield s
                except Exception:
                    continue
    except Exception:
        # Skip unreadable/corrupt files without killing the worker
        return

def shuffled_buffer(source: Iterator[str], buf_lines: int, rng: np.random.RandomState) -> Iterator[str]:
    """Small-buffer shuffle for random-ish order with bounded memory."""
    buf: List[str] = []
    for s in source:
        buf.append(s)
        if len(buf) >= buf_lines:
            rng.shuffle(buf)
            for x in buf:
                yield x
            buf.clear()
    if buf:
        rng.shuffle(buf)
        for x in buf:
            yield x

def random_char_window(s: str, max_chars: Optional[int], rng: np.random.RandomState) -> str:
    if not max_chars or len(s) <= max_chars:
        return s
    start = rng.randint(0, len(s) - max_chars + 1)
    return s[start:start + max_chars]

class StreamingJsonlTokenDataset(IterableDataset):
    """
    Streams .json.gz files (one JSON per line with a 'text' field), lightly shuffles lines,
    tokenizes with SentencePiece, and yields fixed-length [ctx_len] tensors.

    Key features:
    - Concatenate multiple documents (with EOS id) to pack tokens to ctx_len
    - Deterministic per-worker RNG
    - Char-cap before tokenization to bound CPU spikes from very long docs
    - Robust to corrupt files/lines
    """

    def __init__(
        self,
        directory: str,
        sp_model_path: str,
        ctx_len: int,
        *,
        buf_lines: int = 8192,
        char_cap: Optional[int] = 4096,
        concat_docs: bool = True,
        append_bos: bool = False,
        insert_eos_between_docs: bool = True,
        drop_short_docs_under_chars: int = 0,
        min_tokens_after_encode: int = 0,
        shuffle_files: bool = True,
        repeat_forever: bool = True,
        file_ext: str = ".json.gz",
        pad_sequence: bool = True,
    ):
        super().__init__()
        # Recursively collect files ending in file_ext
        files: List[str] = []
        for root, _dirs, names in os.walk(directory):
            for nm in names:
                if nm.endswith(file_ext):
                    files.append(os.path.join(root, nm))
        if not files:
            raise FileNotFoundError(f"No {file_ext} files found under: {directory}")

        self.files = files
        self.sp_model_path = sp_model_path
        self.ctx_len = int(ctx_len)
        self.buf_lines = int(buf_lines)
        self.char_cap = char_cap
        self.concat_docs = concat_docs
        self.append_bos = append_bos
        self.insert_eos_between_docs = insert_eos_between_docs
        self.drop_short_docs_under_chars = int(drop_short_docs_under_chars)
        self.min_tokens_after_encode = int(min_tokens_after_encode)
        self.shuffle_files = shuffle_files
        self.repeat_forever = repeat_forever
        self.pad_sequences = pad_sequence

    def _worker_files(self) -> List[str]:
        info = get_worker_info()
        if info is None:
            return list(self.files)
        # Round-robin shard across workers
        return [f for i, f in enumerate(self.files) if i % info.num_workers == info.id]

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Initialize tokenizer & RNG inside the worker
        tok = spm.SentencePieceProcessor(model_file=self.sp_model_path)
        pad_id, bos_id, eos_id = _safe_sp_ids(tok)

        # If no PAD in the vocab and we ever need to pad, pick a conventional fallback
        # (Ideally, train the tokenizer with a PAD; otherwise choose a reserved/unused ID.)
        if pad_id is None:
            pad_id = 1  # fallback (adjust if your vocab uses a different convention)

        # Per-worker RNG seeded from torch's worker seed
        # torch.initial_seed() is unique per worker & epoch when using DataLoader
        worker_seed = torch.initial_seed() % (2**32)
        rng = np.random.RandomState(worker_seed)

        files = self._worker_files()
        if not files:
            return

        carry: List[int] = []  # rolling token buffer to pack into fixed ctx_len
        while True:
            files_local = list(files)
            if self.shuffle_files:
                rng.shuffle(files_local)

            for path in files_local:
                src = iter_json_text_gz(path)
                # Small-buffer shuffle to keep order random-ish with bounded memory
                for raw_text in shuffled_buffer(src, self.buf_lines, rng):
                    if self.drop_short_docs_under_chars and len(raw_text) < self.drop_short_docs_under_chars:
                        continue

                    # Char-cap to cap CPU work on long docs
                    s = random_char_window(raw_text, self.char_cap, rng)

                    # Encode; optionally add BOS/EOS around each doc
                    ids: List[int] = []
                    if self.append_bos and bos_id is not None:
                        ids.append(bos_id)
                    ids.extend(tok.encode(s))
                    if self.insert_eos_between_docs and eos_id is not None:
                        ids.append(eos_id)

                    if self.min_tokens_after_encode and len(ids) < self.min_tokens_after_encode:
                        continue

                    if not ids:
                        continue

                    if self.concat_docs:
                        # Add to rolling buffer and yield fixed-size chunks
                        carry.extend(ids)
                        while len(carry) >= self.ctx_len:
                            out = carry[:self.ctx_len]
                            del carry[:self.ctx_len]   # maintain small rolling buffer
                            yield torch.tensor(out, dtype=torch.long)
                    else:
                        # Each doc -> 1 sample (truncate or pad)
                        if len(ids) >= self.ctx_len:
                            out = ids[:self.ctx_len]
                        else:
                            out = ids + [pad_id] * (self.ctx_len - len(ids))
                        yield torch.tensor(out, dtype=torch.long)

            if not self.repeat_forever:
                # Optionally flush any remaining carry (padded) once and return
                if carry:
                    # pad the last partial if you want a final sample
                    last = carry[:self.ctx_len]
                    if len(last) < self.ctx_len:
                        last = last + [pad_id] * (self.ctx_len - len(last))
                    yield torch.tensor(last, dtype=torch.long)
                return


if __name__ == "__main__":
    # --- Your dataset (unchanged) ---
    ds = StreamingJsonlTokenDataset(
        directory="/gpfs/data/oermannlab/public_data/dolma/train",
        sp_model_path="/gpfs/data/oermannlab/users/steelr04/WEM-Cross-Batch-Attention/tokenizers/dolma_4k.model",
        ctx_len=2048,
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

    global_batch_size = 8  # adjust as needed

    loader = DataLoader(
        ds,
        batch_size=global_batch_size,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
    )

    # --- Benchmark loop ---
    def benchmark(loader, num_batches=100):
        start = time.time()
        n_samples, n_tokens = 0, 0

        for i, batch in enumerate(loader):
            if i >= num_batches:
                break

            # If your dataset returns tokenized tensors already:
            if torch.is_tensor(batch):
                tokens = batch
            elif isinstance(batch, dict) and "input_ids" in batch:
                tokens = batch["input_ids"]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            n_samples += tokens.size(0)
            n_tokens += tokens.numel()

            if i == 0:
                print(f"Batch shape: {tokens.shape}")

        elapsed = time.time() - start
        print(f"\nProcessed {n_samples} samples in {elapsed:.2f} sec")
        print(f"Throughput: {n_samples/elapsed:.2f} samples/sec")
        print(f"Tokens/sec: {n_tokens/elapsed:.2f}")

    benchmark(loader, num_batches=200)
