from pathlib import Path
from threading import Thread
from queue import Queue
from typing import List, Dict

import random
import numpy as np
import pyarrow as pa, pyarrow.feather as feather
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

UNKNOWN_TOKEN = 0
EOS_TOKEN     = 3
MASK_TOKEN    = 2
PAD_TOKEN     = 1


def arrow_matrix_reader(feather_files: List[Path],
                        tokenizer,
                        context_length: int,
                        queue_size: int = 4):
    """∞ generator that yields a 2-D token matrix per Arrow row, padded to context_length per gene."""
    file_q: "Queue[Path]"   = Queue(maxsize=queue_size)
    tbl_q:  "Queue[pa.Table]" = Queue(maxsize=queue_size)

    def cycle_files():
        while True:
            for p in feather_files:
                file_q.put(p)

    def load_tables():
        while True:
            p = file_q.get()
            try:
                tbl = feather.read_table(p, memory_map=True)
            except Exception as e:
                print(f"[reader] failed {p}: {e}")
                continue
            tbl_q.put(tbl)

    Thread(target=cycle_files, daemon=True).start()
    Thread(target=load_tables, daemon=True).start()

    # Establish a **stable gene order** once from the very first shard
    first_tbl = tbl_q.get()
    gene_cols = [c for c in first_tbl.column_names if c != "__sample__"]
    gene_cols_sorted = sorted(gene_cols)
    col_index = {name: first_tbl.column_names.index(name)
                 for name in gene_cols_sorted}
    tbl_q.put(first_tbl)

    rng = np.random.default_rng()

    while True:
        tbl: pa.Table = tbl_q.get()
        row_indices = rng.permutation(tbl.num_rows)
        for r in row_indices:
            matrix: List[List[int]] = []
            for g in gene_cols_sorted:
                seq = tbl[col_index[g]][r].as_py()
                if not seq:
                    matrix.append([PAD_TOKEN] * context_length)
                    continue
                tokens = tokenizer.Encode(seq)
                tokens = tokens[:context_length]

                padded = tokens + [PAD_TOKEN] * (context_length - len(tokens))
                matrix.append(padded)
            yield matrix
            


class WEMDataset(Dataset):
    def __init__(
        self,
        directory: str,
        ctx_len: int,
        tokenizer
    ):
        """
        Load text data from a list of directories

        Parameters:
        - directories: list of dictionaries containign three keys: "path", "fraction"
        - batch_size: batch size for DataLoader
        """
        self.ctx_len = ctx_len

        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        random.shuffle(files)
        self.tokenizer = tokenizer

        self.sequence_generator = arrow_matrix_reader(files, tokenizer, self.ctx_len)

    def __len__(self):
        return int(1e12) # return a large number to allow for infinite iterations

    def __getitem__(self, _):
        sequence = next(self.sequence_generator)
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence
    
if __name__ == "__main__":
    tokenizer = "/gpfs/data/oermannlab/users/steelr04/Whole_Exome_Modeling/tokenizers/x_chrom_bpe_4k.model"

    sp = spm.SentencePieceProcessor(tokenizer)
    directory = "/gpfs/data/oermannlab/users/steelr04/Whole_Exome_Modeling/data/x_chrom/feather/train"
    dataset = WEMDataset(directory, 3500, sp)
    
    print(dataset.__getitem__(None).shape)




