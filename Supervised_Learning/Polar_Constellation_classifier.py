"""
The code generates a synthetic dataset of points in a 2D plane, where each class corresponds
to points that lie within specific radial and angular ranges (like a polar constellation).
The dataset is noisy, and a simple feedforward neural network is trained to classify the points
into their respective classes. Finally, the predicted classes are visualized using a scatter plot with color coding.

Supervised learning is used here, as the model is trained on labeled data (the class labels for each point).
Objective function: Cross-entropy loss, which is standard for multi-class classification problems.
Optimizer: Adam, which is an adaptive learning rate optimization algorithm that is widely used for training neural networks."""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Parameters
# ==========================

samples_per_class = 4000

sigma_rho = 1
sigma_phi = 5 * np.pi/180  # convert degrees to radians

np.random.seed(0)
torch.manual_seed(0)

# radial ranges
rho_min = np.array([1,2,3,4,5])
rho_max = np.array([5,4,3.5,8,5.5])

# angular ranges in degrees
phi_min = np.array([0,60,100,200,280]) * np.pi/180
phi_max = np.array([60,100,200,280,360]) * np.pi/180

# ==========================
# Generate dataset
# ==========================
X = []
y = []
N_classes = 5
for k in range(N_classes):

    rho = np.random.uniform(rho_min[k], rho_max[k], samples_per_class)
    phi = np.random.uniform(phi_min[k], phi_max[k], samples_per_class)

    # add noise
    rho += np.random.normal(0, sigma_rho, samples_per_class)
    phi += np.random.normal(0, sigma_phi, samples_per_class)

    x = rho * np.cos(phi)
    y_coord = rho * np.sin(phi)

    pts = np.column_stack((x,y_coord))

    X.append(pts)
    y.append(np.full(samples_per_class,k))

X = np.vstack(X)
y = np.concatenate(y)

# ==========================
# Visualize dataset
# ==========================
plt.figure(figsize=(7,7))

plt.scatter(X[:,0], X[:,1], c='blue', s=5, alpha=0.5)

plt.legend()
plt.axis("equal")
plt.title("Noisy Dataset")
plt.show()

# ==========================
# Convert to torch
# ==========================
X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.long)

# ==========================
# Train/Test split
# ==========================
N = X.shape[0]
perm = torch.randperm(N)

n_train = int(0.8*N)

train_idx = perm[:n_train]
test_idx = perm[n_train:]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

# ==========================
# Model
# ==========================
model = nn.Sequential(
    nn.Linear(2,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,N_classes)
    # no softmax needed since we use CrossEntropyLoss which applies it internally
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

# ==========================
# Training
# ==========================
epochs = 500

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs,y_train)

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch} loss {loss.item():.4f}")

# ==========================
# Test accuracy
# ==========================
with torch.no_grad():

    outputs = model(X_test)

    pred = torch.argmax(outputs,dim=1)

    acc = (pred==y_test).float().mean()

print("Test accuracy:",acc.item())



# ==========================
# Predict class for all data
# ==========================
with torch.no_grad():
    outputs = model(X)
    pred_all = torch.argmax(outputs, dim=1)

pred_np = pred_all.numpy()
X_np = X.numpy()

# ==========================
# Visualization using color index
# ==========================
plt.figure(figsize=(7,7))

scatter = plt.scatter(
    X_np[:,0],
    X_np[:,1],
    c=pred_np,
    cmap='tab10',   # good for up to 10 classes
    s=5,
    alpha=0.7
)

plt.colorbar(scatter, label="Predicted Class")
plt.axis("equal")
plt.title("Predicted Classes by Network")
plt.xlabel("x")
plt.ylabel("y")

plt.show()