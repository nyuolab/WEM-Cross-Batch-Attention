import os, gzip, json, random
from typing import Iterator, List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import sentencepiece as spm
import time
from tqdm import tqdm


def _safe_sp_ids(tok: spm.SentencePieceProcessor) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    def safe(v): return v if (v is not None and v >= 0) else None
    return safe(tok.pad_id()), safe(tok.bos_id()), safe(tok.eos_id())


def iter_json_text_gz(path: str) -> Iterator[str]:
    """Yield 'text' from JSONL .gz, tolerant to glitches."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                try:
                    obj = json.loads(raw)
                    s = obj.get("text", "")
                    if s:
                        yield s
                except Exception:
                    continue
    except Exception:
        return  # unreadable file → skip


def shuffled_buffer(source: Iterator[str], buf_lines: int, rng: np.random.RandomState) -> Iterator[str]:
    """Small-buffer shuffle (random-ish order, bounded RAM)."""
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


class DocPackedJsonlDataset(IterableDataset):
    """
    Per-document packing with optional multi-document packing.
      - If pack_documents=False (default behavior from before):
          * Yield one sample per document (possibly multiple rows P for long docs)
          * Respect max_chunks if provided (truncate beyond max_chunks * ctx_len)
      - If pack_documents=True:
          * Keep appending documents into a single sample until you reach max_chunks rows
          * Between documents, insert exactly one full spacer row:
              - Prefer EOS tokens; if EOS unavailable, use PADs
              - EOS spacer has attention=True; PAD spacer has attention=False
    """

    def __init__(
        self,
        directory: str,
        sp_model_path: str,
        ctx_len: int,
        *,
        buf_lines: int = 8192,
        append_bos: bool = False,
        append_eos: bool = True,
        pad_to_multiple_rows: bool = True,   # kept for API compatibility; last row is padded
        file_ext: str = ".json.gz",
        shuffle_files: bool = True,
        repeat_forever: bool = True,
        max_chunks: Optional[int] = None,    # cap rows per yielded sample
        pack_documents: bool = False,        # pack multiple docs into one sample
    ):
        super().__init__()
        self.ctx_len = int(ctx_len)
        self.buf_lines = int(buf_lines)
        self.append_bos = append_bos
        self.append_eos = append_eos
        self.pad_to_multiple_rows = pad_to_multiple_rows
        self.shuffle_files = shuffle_files
        self.repeat_forever = repeat_forever
        self.max_chunks = int(max_chunks) if (max_chunks is not None) else None
        self.pack_documents = bool(pack_documents)

        files: List[str] = []
        for root, _dirs, names in os.walk(directory):
            for nm in names:
                if nm.endswith(file_ext):
                    files.append(os.path.join(root, nm))
        if not files:
            raise FileNotFoundError(f"No {file_ext} files found under: {directory}")
        self.files = files
        self.sp_model_path = sp_model_path

    def _worker_files(self) -> List[str]:
        info = get_worker_info()
        if info is None:
            return list(self.files)
        return [f for i, f in enumerate(self.files) if i % info.num_workers == info.id]

    def _encode_document_rows(
        self,
        doc: str,
        tok: spm.SentencePieceProcessor,
        pad_id: int,
        bos_id: Optional[int],
        eos_id: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a single document and pack into rows of length ctx_len.
        Returns (out_ids [P,S], attn [P,S]).
        """
        ids: List[int] = []
        if self.append_bos and bos_id is not None:
            ids.append(bos_id)
        ids.extend(tok.encode(doc))
        if self.append_eos and eos_id is not None:
            ids.append(eos_id)

        if not ids:
            return torch.empty((0, self.ctx_len), dtype=torch.long), torch.empty((0, self.ctx_len), dtype=torch.bool)

        L = len(ids)
        P = (L + self.ctx_len - 1) // self.ctx_len

        out = torch.full((P, self.ctx_len), fill_value=pad_id, dtype=torch.long)
        attn = torch.zeros((P, self.ctx_len), dtype=torch.bool)

        full_rows = L // self.ctx_len
        if full_rows:
            pre = np.asarray(ids[: full_rows * self.ctx_len], dtype=np.int64)
            out[:full_rows].copy_(torch.from_numpy(pre.reshape(full_rows, self.ctx_len)))
            attn[:full_rows] = True

        rem = L - full_rows * self.ctx_len
        if rem > 0:
            last = np.asarray(ids[-rem:], dtype=np.int64)
            out[full_rows, :rem] = torch.from_numpy(last)
            attn[full_rows, :rem] = True

        return out, attn

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        tok = spm.SentencePieceProcessor(model_file=self.sp_model_path)
        pad_id, bos_id, eos_id = _safe_sp_ids(tok)
        if pad_id is None:
            pad_id = 1  # fallback; ideally the tokenizer should define PAD

        # spacer row preference: EOS if available, else PAD
        spacer_uses_eos = eos_id is not None
        spacer_token_id = eos_id if spacer_uses_eos else pad_id

        worker_seed = torch.initial_seed() % (2**32)
        rng = np.random.RandomState(worker_seed)

        files = self._worker_files()
        if not files:
            return

        def maybe_truncate_rows(
            rows_ids: torch.Tensor, rows_attn: torch.Tensor, max_rows: Optional[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if max_rows is None:
                return rows_ids, rows_attn
            if rows_ids.size(0) <= max_rows:
                return rows_ids, rows_attn
            return rows_ids[:max_rows], rows_attn[:max_rows]

        # Pre-build a spacer row
        spacer_ids = torch.full((1, self.ctx_len), int(spacer_token_id), dtype=torch.long)
        spacer_attn = torch.ones((1, self.ctx_len), dtype=torch.bool) if spacer_uses_eos else torch.zeros((1, self.ctx_len), dtype=torch.bool)

        while True:
            files_local = list(files)
            if self.shuffle_files:
                rng.shuffle(files_local)

            # If not packing docs, we yield as soon as we finish each doc
            # If packing docs, we accumulate rows until hitting max_chunks (required), then yield
            rows_bucket_ids: List[torch.Tensor] = []
            rows_bucket_attn: List[torch.Tensor] = []

            for path in files_local:
                for doc in shuffled_buffer(iter_json_text_gz(path), self.buf_lines, rng):
                    doc_ids, doc_attn = self._encode_document_rows(doc, tok, pad_id, bos_id, eos_id)

                    if self.max_chunks is not None:
                        # If not packing, just truncate doc rows to max_chunks and yield immediately
                        if not self.pack_documents:
                            doc_ids, doc_attn = maybe_truncate_rows(doc_ids, doc_attn, self.max_chunks)
                            if doc_ids.size(0) == 0:
                                continue
                            yield {
                                "input_ids": doc_ids,
                                "attention_mask": doc_attn,
                            }
                            continue

                    if not self.pack_documents:
                        # No max_chunks provided: just yield doc as-is
                        if doc_ids.size(0) == 0:
                            continue
                        yield {
                            "input_ids": doc_ids,
                            "attention_mask": doc_attn,
                        }
                        continue

                    # pack_documents=True below:
                    # Start new bucket if empty; otherwise add spacer before appending next doc
                    need_spacer = (len(rows_bucket_ids) > 0)

                    # compute how many rows can we still add
                    if self.max_chunks is None:
                        remaining = None
                    else:
                        current_rows = sum(t.size(0) for t in rows_bucket_ids)
                        remaining = max(self.max_chunks - current_rows, 0)

                    # If we need a spacer but have no room even for the spacer → flush bucket first
                    if need_spacer and remaining is not None and remaining == 0:
                        # yield current bucket
                        out_ids = torch.cat(rows_bucket_ids, dim=0)
                        out_attn = torch.cat(rows_bucket_attn, dim=0)
                        yield {"input_ids": out_ids, "attention_mask": out_attn}
                        rows_bucket_ids, rows_bucket_attn = []
                        # recompute remaining for new bucket
                        remaining = self.max_chunks if self.max_chunks is not None else None
                        need_spacer = False

                    # Add spacer if needed
                    if need_spacer:
                        if remaining is None or remaining > 0:
                            rows_bucket_ids.append(spacer_ids)
                            rows_bucket_attn.append(spacer_attn)
                            if remaining is not None:
                                remaining -= 1
                        else:
                            # should not happen due to flush above
                            pass

                    # Now add as many rows from the current doc as will fit
                    if remaining is None:
                        take = doc_ids.size(0)
                    else:
                        take = min(remaining, doc_ids.size(0))

                    if take > 0:
                        rows_bucket_ids.append(doc_ids[:take])
                        rows_bucket_attn.append(doc_attn[:take])

                    # If the doc had more rows than we could take, we truncate (by design)
                    # Check if we’ve filled the bucket to max_chunks; if so, flush
                    if self.max_chunks is not None:
                        current_rows = sum(t.size(0) for t in rows_bucket_ids)
                        if current_rows >= self.max_chunks:
                            out_ids = torch.cat(rows_bucket_ids, dim=0)
                            out_attn = torch.cat(rows_bucket_attn, dim=0)
                            yield {"input_ids": out_ids, "attention_mask": out_attn}
                            rows_bucket_ids, rows_bucket_attn = []

            # End of files: if we have a partial bucket (pack_documents=True), flush it
            if self.pack_documents and rows_bucket_ids:
                out_ids = torch.cat(rows_bucket_ids, dim=0)
                out_attn = torch.cat(rows_bucket_attn, dim=0)
                yield {"input_ids": out_ids, "attention_mask": out_attn}

            if not self.repeat_forever:
                return



def docpack_collate(batch: List[Dict[str, torch.Tensor]], maxP) -> Dict[str, torch.Tensor]:
    """
    Collate a list of doc-packed samples with different P into a batch.
    Pads the P dimension to max_P across the batch.
    Returns:
    {
        'input_ids': LongTensor [B, max_P, S],
        'attention_mask': BoolTensor [B, max_P, S],
        'row_mask': BoolTensor [B, max_P],  # True where a row exists for that doc
    }
    """
    B = len(batch)
    S = batch[0]["input_ids"].shape[-1]
    Ps = [x["input_ids"].shape[0] for x in batch]
    if maxP is None:
        maxP = max(Ps)

    # dtype/ids from first element
    pad_id = int(batch[0]["input_ids"][0, 0].item())  # not strictly the pad_id, but we’ll pad with first element's pad
    # Safer: detect pad_id by finding a masked location, else default to 0
    detected_pad = None
    for x in batch:
        m = (~x["attention_mask"]).nonzero(as_tuple=False)
        if m.numel() > 0:
            detected_pad = int(x["input_ids"][m[0,0], m[0,1]].item())
            break
    if detected_pad is not None:
        pad_id = detected_pad

    input_ids = torch.full((B, maxP, S), pad_id, dtype=torch.long)

    for i, sample in enumerate(batch):
        P = sample["input_ids"].shape[0]
        input_ids[i, :P] = sample["input_ids"]

    return input_ids



if __name__ == "__main__":

    from torch.utils.data import DataLoader

    ds = DocPackedJsonlDataset(
        directory="/gpfs/data/oermannlab/public_data/dolma/train",
        sp_model_path="/gpfs/data/oermannlab/users/steelr04/WEM-Cross-Batch-Attention/tokenizers/dolma_4k.model",
        ctx_len=2048,
        buf_lines=8192,
        append_bos=False,
        append_eos=True,
        shuffle_files=True,
        repeat_forever=True,
    )

    global_batch_size = 1  # adjust as needed

    def make_docpack_collate(maxP=None):
        def collate(batch):
            return docpack_collate(batch, maxP)
        return collate

    loader = DataLoader(
        ds,
        batch_size=global_batch_size,   # number of documents per batch
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=make_docpack_collate(maxP=8),     # << important
    )

    # --- Benchmark loop ---
    def benchmark(loader, num_batches=100):
        start = time.time()
        n_samples, n_tokens = 0, 0

        n_p = 0
        max_p = 0

        for i, batch in tqdm(enumerate(loader)):
            if i >= num_batches:
                break

            # If your dataset returns tokenized tensors already:
            print(batch.shape)
            if torch.is_tensor(batch):
                tokens = batch
            elif isinstance(batch, dict) and "input_ids" in batch:
                tokens = batch["input_ids"]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            n_p += tokens.size(1)
            max_p = max(max_p, tokens.size(1))
            n_samples += tokens.size(0)
            n_tokens += tokens.numel()


        elapsed = time.time() - start
        print(f"Max P {max_p}")
        print(f"\nProcessed {n_samples} samples in {elapsed:.2f} sec")
        print(f"Average P: {n_p/num_batches}")
        print(f"Throughput: {n_samples/elapsed:.2f} samples/sec")
        print(f"Tokens/sec: {n_tokens/elapsed:.2f}")

    benchmark(loader, num_batches=100)
