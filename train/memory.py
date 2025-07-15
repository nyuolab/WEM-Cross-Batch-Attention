# --- utils/memory.py --------------------------------------------------------
import sys
import torch
import json
import time
import torch.distributed as dist

class MemTracker:
    """
    Resets a fresh peak counter before the layer runs forward and prints the
    layer-local backward peak right after grads are computed.
    """
    def __init__(self):
        self.peaks = {}     # name → MB

    def fw(self, name):
        def _h(*_):
            rank = dist.get_rank()
            torch.cuda.reset_peak_memory_stats()
            mb = torch.cuda.memory_allocated() / 2**20
            self.peaks[name] = mb
            print(f"[mem_fw{rank}] {name:>20s}  {mb:7.1f} MB"); sys.stdout.flush()
        return _h

    def bw(self, name):
        def _h(*_):
            rank = dist.get_rank()
            mb = torch.cuda.memory_allocated() / 2**20
            self.peaks[name] = mb
            print(f"[mem_bw{rank}] {name:>20s}  {mb:7.1f} MB"); sys.stdout.flush()
        return _h
    
def start_memory_trace(max_entries: int = 200_000) -> None:
    """
    Begin recording every allocation/free with full Python stack traces.
    Call this *once* near program start (before tensors are created).

    After calling, the pickle produced by `dump_cuda_snapshot()` will contain
    a complete history that you can inspect with
        python -m torch.cuda._memory_viz trace snapshot.pickle -o trace.json
    and open in chrome://tracing.
    """
    torch.cuda.memory._record_memory_history(enabled="all",
                                             context="all",
                                             stacks="all",
                                             max_entries=max_entries)
    print(f"[mem] memory history recording started "
          f"(max {max_entries:_} events)"); sys.stdout.flush()
    
def cuda_dump_snapshot():
    fname = f"/workspace/Whole_Exome_Modeling/train_WEM/cuda_snapshot_{int(time.time())}"
    torch.cuda.memory._dump_snapshot(f"{fname}.pkl")


def dump_cuda_snapshot(tag=""):
    torch.cuda.synchronize()                 # make sizes stable
    snap = torch.cuda.memory_snapshot()
    fname = f"/workspace/Whole_Exome_Modeling/train_WEM/cuda_snapshot_{tag}_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump(snap, f)
    print(f"[mem] wrote {fname}")
