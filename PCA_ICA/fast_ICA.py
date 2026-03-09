"""FastICA implementation.
This code implements the FastICA algorithm for Independent Component Analysis (ICA) using a deflation approach.
The FastICA algorithm is an efficient method for separating mixed signals into their independent components.

Number of components is known in this approach , if it is not provided it will be set to number of features (n_features) in the data.
The Method used an iterative fixed-point algorithm to find the unmixing matrix W that separates the mixed signals into independent components.
it is important to note that FastICA assumes that the independent components are non-Gaussian and statistically independent.
Also input data should be preprocessed (centered and whitened) before applying FastICA for better performance and convergence. """



import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------

def center(X):
    """Center data to zero mean."""
    return X - np.mean(X, axis=1, keepdims=True)

def whiten(X):
    """Whiten data using eigenvalue decomposition."""
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    X_white = D_inv_sqrt @ E.T @ X
    return X_white

# -------------------------------------------------
# FastICA algorithm (deflation approach)
# -------------------------------------------------

def fastica(X, n_components, max_iter=200, tol=1e-6):
    """
    FastICA using fixed-point iteration.

    Parameters:
        X : ndarray (n_features, n_samples)
        n_components : int
    """
    X = center(X)
    X = whiten(X)

    n_features, n_samples = X.shape
    W = np.zeros((n_components, n_features))

    for i in range(n_components):
        w = np.random.randn(n_features)
        w /= np.linalg.norm(w)

        for _ in range(max_iter):
            w_old = w.copy()

            wx = w @ X
            g = np.tanh(wx)
            g_prime = 1 - g**2

            # Fixed-point update
            w = (X @ g.T) / n_samples - np.mean(g_prime) * w

            # Orthogonalization
            if i > 0:
                w -= W[:i].T @ (W[:i] @ w)

            w /= np.linalg.norm(w)

            # Convergence check
            if np.abs(np.abs(w @ w_old) - 1) < tol:
                break

        W[i, :] = w

    S = W @ X
    return S, W

# -------------------------------------------------
# Plotting function
# -------------------------------------------------

def plot_sources(S_true, S_est):
    """
    Plot true and estimated sources.
    Each row corresponds to one source.
    """
    n_sources = S_true.shape[0]
    fig, axes = plt.subplots(n_sources, 2, figsize=(12, 2.5 * n_sources))

    for i in range(n_sources):
        axes[i, 0].plot(S_true[i])
        axes[i, 0].set_title(f"True Source {i+1}")
        axes[i, 0].grid(True)

        axes[i, 1].plot(S_est[i])
        axes[i, 1].set_title(f"Estimated Source {i+1}")
        axes[i, 1].grid(True)

    axes[-1, 0].set_xlabel("Samples")
    axes[-1, 1].set_xlabel("Samples")

    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Example: Source separation
# -------------------------------------------------

if __name__ == "__main__":

    np.random.seed(0)

    # Generate sources
    n_samples = 2000
    t = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t))
    s3 = np.random.normal(size=n_samples)

    S = np.vstack([s1, s2, s3])

    # Mixing matrix
    A = np.array([[1.0, 1.0, 1.0],
                  [0.5, 2.0, 1.0],
                  [1.5, 1.0, 2.0]])

    X = A @ S  # Observed mixtures

    # Apply FastICA
    S_est, W = fastica(X, n_components=3)

    # Plot results
    plot_sources(S, S_est)
