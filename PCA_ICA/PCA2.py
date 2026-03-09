"""
Principal Component Analysis (PCA) implementation
This code implements Principal Component Analysis (PCA) for dimension redunction and data visualization.
Dataset used is BrestCancer wisconsin dataset.
Dataset contains 569 samples with 30 features each, and a binary target variable indicating whether the tumor is malignant (1) or benign (0).

Here we want to find among 30 features which are the most important ones that capture the most variance in the data, 
and we want to visualize the data in 2D using the first 2 principal components.

It turns out the first two principal components capture 99.82% of the variance in the data, 
which means we can represent the 30D data in just 2 dimensions while keeping almost all the information.
PC1 : 98.2% of variance, PC2 : 1.62% of variance."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================
# Load dataset
# ==========================
df= pd.read_csv("Breast Cancer Dataset.csv")
df.drop(columns=["Unnamed: 32"], inplace=True)   # drop empty column
df=df.dropna()
df_diagnosis = df["diagnosis"].values
df_features = df.drop(columns=["diagnosis", "id"]).values

# ==========================
# Create Data Matrix
# ==========================
X = df_features.T   # shape: (n_features, n_samples)
mu_hat = np.mean(X, axis=1, keepdims=True)   # shape: (n_features, 1)
X_mu = X - mu_hat

# ==========================
# Compute covariance and eigen decomposition
# ==========================
cov = (X_mu @ X_mu.T) / X_mu.shape[1]  # shape: (n_features, n_features)

eigenvalues, eigenvectors = np.linalg.eig(cov)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("Eigenvalues:", ", ".join([f"{v:.1f}" for v in eigenvalues]))
print("info percentage per Component:", ", ".join([f"{v/sum(eigenvalues)*100:.2f}%" for v in eigenvalues]))
print("first eigenvector (PC1):", eigenvectors[:,0])
print("second eigenvector (PC2):", eigenvectors[:,1])
print("third eigenvector (PC3):", eigenvectors[:,2])
# ==========================
# Project data into eigen space
# ==========================
X_pca = eigenvectors.T @ X_mu     # shape: (n_features, n_samples)
# ==========================
# Plot: PC1 vs PC2
# ==========================
plt.figure()
for label in np.unique(df_diagnosis):
    idx = df_diagnosis == label
    plt.scatter(X_pca[0, idx], X_pca[1, idx], label=label, alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")   
plt.title("PCA: PC1 vs PC2")
plt.legend()
plt.show()

# ==========================
# Plot: PC1 & PC2 & PC3
# ==========================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(df_diagnosis):
    idx = df_diagnosis == label
    ax.scatter(X_pca[0, idx], X_pca[1, idx], X_pca[2, idx], label=label, alpha=0.5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA: PC1 vs PC2 vs PC3")
ax.legend()
plt.show()