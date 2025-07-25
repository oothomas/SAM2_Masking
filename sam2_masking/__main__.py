import sys
import torch
from .cli import main

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
