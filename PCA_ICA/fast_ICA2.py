"""FastICA implementation.
This code implements the FastICA algorithm for Independent Component Analysis (ICA) using a deflation approach.
The FastICA algorithm is an efficient method for separating mixed signals into their independent components.

Number of components is known in this approach , if it is not provided it will be set to number of features (n_features) in the data.
The Method used an iterative fixed-point algorithm to find the unmixing matrix W that separates the mixed signals into independent components.
it is important to note that FastICA assumes that the independent components are non-Gaussian and statistically independent.
Also input data should be preprocessed (centered and whitened) before applying FastICA for better performance and convergence. """



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# -------------------------------------------------
# Generate example data
# -------------------------------------------------

np.random.seed(0)

n_samples = 2000
t = np.linspace(0, 8, n_samples)

# True independent sources
s1 = np.sin(2 * t)
s2 = np.sign(np.sin(3 * t))
s3 = np.random.normal(size=n_samples)

S_true = np.vstack([s1, s2, s3]).T   # shape: (n_samples, n_sources)

# Mixing matrix
A = np.array([[1.0, 1.0, 1.0],
              [0.5, 2.0, 1.0],
              [1.5, 1.0, 2.0]])

# Observed mixtures
X = S_true @ A.T    # shape: (n_samples, n_features)

# -------------------------------------------------
# FastICA (scikit-learn)
# -------------------------------------------------

ica = FastICA(
    n_components=3,        # optional (default = n_features)
    algorithm='deflation', # or 'parallel' : deflation is for one component at a time
    fun='cube',         # the opjective function we want to optimize cube = kurtosis based
    max_iter=300,       # maximum number of iterations per component
    tol=1e-4,           # tolerance for stopping condition
    random_state=0      
)

S_est = ica.fit_transform(X)   # Estimated sources
A_est = ica.mixing_            # Estimated mixing matrix
W_est = ica.components_        # Estimated unmixing matrix

# -------------------------------------------------
# Plot: True vs Estimated Sources
# -------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(12, 7))

for i in range(3):
    axes[i, 0].plot(S_true[:, i])
    axes[i, 0].set_title(f"True Source {i+1}")
    axes[i, 0].grid(True)

    axes[i, 1].plot(S_est[:, i])
    axes[i, 1].set_title(f"Estimated Source {i+1}")
    axes[i, 1].grid(True)

axes[-1, 0].set_xlabel("Samples")
axes[-1, 1].set_xlabel("Samples")

plt.tight_layout()
plt.show()
