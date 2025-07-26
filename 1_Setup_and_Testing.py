import sys
import pip
import torch
print("Python version:", sys.version)
# Print pip version
print("Pip version:", pip.__version__)
# Print pytorch version
print("Pytorch version:", torch.__version__)
# Print CUDA version
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available.")

# Print GPU information
if torch.cuda.is_available():
    print("GPU is available.")
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")

# Check if pytorch can use CUDA
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    print("Pytorch can use CUDA ✅Tensor on GPU:", x)
else:
    print("Pytorch is not using CUDA.")