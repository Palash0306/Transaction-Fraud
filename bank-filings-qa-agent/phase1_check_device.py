"""
Phase 1, Step 3: Tensors vs numpy
 
Goal: build hands-on intuition for what a torch.Tensor actually is,
so that later when sentence-transformers hands you a tensor back,
you're not staring at a black box.
 
Run this file section by section (or just top to bottom) and read
the print output against the comments.
"""

import torch
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------
# 1. Creating tensors — three common ways
# ---------------------------------------------------------------
# A tensor from a plain Python list. This is the same as np.array([...]).

t_from_list = torch.tensor([1.0, 2.0, 3.0])
print(f"From list: {t_from_list}")

# A tensor of random values. Shape (3, 3) means 3 rows, 3 columns —
# same shape convention as numpy.
t_random = torch.rand(3,3)
print(f"Random: {t_random}")

# A tensor built FROM a numpy array. This matters because a lot of
# real-world data (e.g. loaded from CSV/pandas) starts life as numpy,
# and you'll often need to hand it to a PyTorch model.
np_array = np.array([[1,2],[3,4]], dtype = np.float32)
t_from_numpy = torch.from_numpy(np_array)
print(f"From numpy: {t_from_numpy}")

# ---------------------------------------------------------------
# 2. Inspecting a tensor: shape, dtype, device
# ---------------------------------------------------------------
# .shape tells you the dimensions — critical for debugging. Almost
# every bug you'll hit with embeddings later is a shape mismatch
# (e.g. expecting (384,) but getting (1, 384)).
print("\nShape:", t_random.shape)
 
# .dtype tells you the data type. float32 is the standard for most
# ML work — enough precision, half the memory of float64.
print("Dtype:", t_random.dtype)
 
# .device tells you WHERE this tensor's data physically lives.
# Right now it's on CPU by default, even though MPS is available —
# PyTorch never silently moves things for you.
print("Device:", t_random.device)
 

# ---------------------------------------------------------------
# 4. Basic operations — matmul and elementwise multiply
# ---------------------------------------------------------------
# Elementwise multiply: each position multiplied independently.
# Shapes must match exactly.
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
elementwise = a * b
print("\nElementwise multiply:", elementwise)
 
# Matrix multiplication: this is the core operation inside every
# neural network layer, including the transformer layers inside
# sentence-transformers. (3,3) @ (3,3) -> (3,3).
m1 = torch.rand(3, 3)
m2 = torch.rand(3, 3)
matmul_result = m1 @ m2   # same as torch.matmul(m1, m2)
print("\nMatmul result:\n", matmul_result)
 
# Run the same matmul on MPS, to see the device-aware pattern
# you'll use for real work later.
m1_mps = m1.to(device)
m2_mps = m2.to(device)
matmul_on_device = m1_mps @ m2_mps
print(f"\nMatmul on {device}:\n", matmul_on_device)

