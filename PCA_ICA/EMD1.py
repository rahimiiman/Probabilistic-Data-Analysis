"""
Empirical Mode Decomposition (EMD) using the 'emd' package.
This code demonstrates how to use the 'emd' package to perform EMD on a test signal.
EMD frequently used in signal processing and time series analysis to decompose a signal into intrinsic mode functions (IMFs) and a residue.
The Idea is that original signal is summation of several oscilatory functions (IMFs) and a residue (trend)."""

import numpy as np
import matplotlib.pyplot as plt
import emd

# -----------------------------
# Generate test signal
# -----------------------------
t = np.linspace(0, 1, 2000)
x = (
    np.sin(2*np.pi*5*t) +
    0.5*np.sin(2*np.pi*20*t) +
    0.2*t
)

plt.figure()
plt.plot(t, x)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Original Signal")
plt.show()
# -----------------------------
# Empirical Mode Decomposition
# -----------------------------
# Returns array of shape (N, n_imfs)
imfs = emd.sift.sift(x)

# Convert to (n_imfs, N)
imfs = imfs.T

# Residue
residue = x - imfs.sum(axis=0)

# Reconstruction
x_rec = imfs.sum(axis=0) + residue


# -----------------------------
# Plot 2: IMFs + Residue
# -----------------------------
plt.figure()
for i, imf in enumerate(imfs):
    plt.plot(t, imf, label=f"IMF {i+1}")

plt.plot(t, residue, "--", label="Residue")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("IMFs and Residue (emd package)")
plt.legend()
plt.show()

# -----------------------------
# Plot 3: Original vs Reconstruction
# -----------------------------
plt.figure()
plt.plot(t, x, label="Original signal")
plt.plot(t, x_rec, "--", label="Sum(IMFs) + Residue")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Original Signal vs Detected Components")
plt.legend()
plt.show()
