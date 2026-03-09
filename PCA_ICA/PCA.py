"""PCA implementation.
This code implements Principal Component Analysis (PCA) for dimensionality reduction and data visualization.
PCA is a linear technique that transforms the data into a new coordinate system such that the greatest
variance of the data lies on the first coordinate (called the first principal component), 
the second greatest variance on the second coordinate, and so on.

Here we apply PCA to an image, treating each pixel's RGB values as a 3D point.
at the end we visualize the original image and the first 3 principal components as color images.
and we show howmany percentage of total information (variance) is captured by each principal component. 

the results show we can represent the 3 channel image using just one channel while keeping 97.9% of information (variance) in the data,
and the first PC captures most of the color information while the other PCs capture less significant variations. 
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# ==========================
# Load image
# ==========================
img = cv2.imread("PCA_Test.jpg")      # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W, C = img.shape

# ==========================
# Reshape image to 3 x N
# ==========================
X = img.reshape(-1, 3).T.astype(np.float32)  # shape: (3, N_pixels)

# ==========================
# Compute mean and center data
# ==========================
mu_hat = np.mean(X, axis=1, keepdims=True)   # shape: (3,1)
X_mu = X - mu_hat

# ==========================
# Compute covariance
# ==========================
# covariance: C = (1/N) * X_mu * X_mu^T
cov = (X_mu @ X_mu.T) / X_mu.shape[1]  # shape: (3,3)

# ==========================
# Eigen decomposition
# ==========================
eig_vals, eig_vecs = np.linalg.eigh(cov)

idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]  # columns = eigenvectors
print("Eigenvalues:", ", ".join([f"{v:.2f}" for v in eig_vals]))
print("info percentage per Component:", ", ".join([f"{v/sum(eig_vals)*100:.1f}%" for v in eig_vals]))
print("first eigenvector (PC1):", eig_vecs[:,0])
print("second eigenvector (PC2):", eig_vecs[:,1])
print("third eigenvector (PC3):", eig_vecs[:,2])
# ==========================
# Project pixels into eigen space
# ==========================
X_pca = eig_vecs.T @ X_mu     # shape: (3,N_pixels)

# ==========================
# Reshape each PC to image
# ==========================
pc1 = X_pca[0,:].reshape(H,W)
pc2 = X_pca[1,:].reshape(H,W)
pc3 = X_pca[2,:].reshape(H,W)

# ==========================
# Reconstruct color images for each PC
# ==========================
pc_imgs = []

for i in range(3):
    # X_pca[i,:] = 1 x N_pixels
    # eig_vecs[:,i] = 3 x 1
    X_pc = eig_vecs[:,i][:,None] * X_pca[i,:][None,:] + mu_hat  # shape 3 x N
    X_pc = np.clip(X_pc, 0, 255)
    pc_img = X_pc.T.reshape(H,W,3)  # now shape H x W x 3
    pc_imgs.append(pc_img.astype(np.uint8))

# ==========================
# Plot original + first 3 PCs
# ==========================
fig, axes = plt.subplots(2,2, figsize=(10,10))

axes[0,0].imshow(img.astype(np.uint8))
axes[0,0].set_title("Original Image")
axes[0,0].axis('off')

for i, pc_img in enumerate(pc_imgs):
    ax = axes[(i+1)//2, (i+1)%2]
    ax.imshow(pc_img)
    ax.set_title(f"PC {i+1} (color)")
    ax.axis('off')

plt.tight_layout()
plt.show()