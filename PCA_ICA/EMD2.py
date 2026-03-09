"""
Empirical Mode Decomposition (EMD) using the 'emd' package.
This code demonstrates how to use the 'emd' package to perform EMD on a test signal.
EMD frequently used in signal processing and time series analysis to decompose a signal into intrinsic mode functions (IMFs) and a residue.
The Idea is that original signal is summation of several oscilatory functions (IMFs) and a residue (trend)."""


import numpy as np
import matplotlib.pyplot as plt
import emd
from scipy.signal import chirp
import math

# -----------------------------
# Time axis
# -----------------------------
fs = 2000
t = np.linspace(0, 1, fs)

# -----------------------------
# Two chirps + noise
# -----------------------------
chirp1 = chirp(t, f0=5,  f1=50,  t1=1, method='linear')
chirp2 = chirp(t, f0=60, f1=200, t1=1, method='quadratic')
noise = 0.2 * np.random.randn(len(t))

x = chirp1 + chirp2 + noise
plt.figure()
plt.plot(t, x)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Original Signal")
plt.show()
# -----------------------------
# EMD (emd package)
# -----------------------------
imfs = emd.sift.sift(x).T     # (n_imfs, N)

residue = x - imfs.sum(axis=0)
x_rec = imfs.sum(axis=0) + residue



# -----------------------------
# Figure 2: IMFs + Residue (Adaptive Subplots)
# -----------------------------
signals = list(imfs) + [residue]
labels  = [f"IMF {i+1}" for i in range(imfs.shape[0])] + ["Residue"]

n_plots = len(signals)

# Choose layout
if n_plots <= 4:
    ncols = 1
else:
    ncols = 2

nrows = math.ceil(n_plots / ncols)

fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(10, 2*nrows))
axes = np.atleast_1d(axes).flatten()

for ax, sig, lab in zip(axes, signals, labels):
    ax.plot(t, sig)
    ax.set_ylabel(lab)

# Remove unused axes
for ax in axes[len(signals):]:
    ax.remove()

axes[-1].set_xlabel("Time [s]")
fig.suptitle("EMD: IMFs and Residue (Two Chirps + Noise)")
plt.tight_layout()
plt.show()

# -----------------------------
# Figure 3: Original vs Reconstruction
# -----------------------------
plt.figure()
plt.plot(t, x, label="Original (chirps + noise)")
plt.plot(t, x_rec, "--", label="Sum(IMFs) + Residue")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Original Signal vs Reconstruction (emd)")
plt.legend()
plt.show()