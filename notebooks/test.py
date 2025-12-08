import torch

print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("Reserved: ", torch.cuda.memory_reserved() / 1024**2, "MB")
print("Max Alloc:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

