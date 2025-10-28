"""
A small test to check saving and loading NumPy arrays using the iodata module.
"""
import numpy as np
from iodata import save_checkpoint, load_checkpoint

# Create test arrays
a = np.arange(6).reshape(2, 3)
b = np.linspace(0, 1, 5)

# Save them to file
save_checkpoint("../tests/test_cp.npz", arr=a, vec=b)

# Load them back
data = load_checkpoint("../tests/test_cp.npz")

print("Keys:", data.keys())
print("arr:\n", data["arr"])
print("vec:\n", data["vec"])