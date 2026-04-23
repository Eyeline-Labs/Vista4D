import gc
import torch


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
