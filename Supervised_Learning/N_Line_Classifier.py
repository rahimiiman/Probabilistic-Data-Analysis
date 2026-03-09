"""N-Line Classifier
This code generates a dataset of points around N noisy lines and trains a neural network to classify which line each point belongs to.
Supervised learning is used here, as the model is trained on labeled data (the class labels for each point).
Objective function: Cross-entropy loss, which is standard for multi-class classification problems.
Optimizer: Adam, which is an adaptive learning rate optimization algorithm that is widely used for training neural networks.
The dataset is visualized using a scatter plot with different colors for each line. 
The model's performance is evaluated on a test set, and the accuracy is printed at the end.

"""



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Parameters
# ==========================
N_lines = 5
samples_per_line = 4000
sigma = 0.4
torch.manual_seed(0)
np.random.seed(0)

# slopes and intercepts
a = np.linspace(-1.5, 1.5, N_lines)
b = np.linspace(-4, 4, N_lines)

# ==========================
# Generate data
# ==========================
X = []
y = []

for i in range(N_lines):

    x_vals = np.random.uniform(-15, 15, samples_per_line)
    noise = np.random.normal(0, sigma, samples_per_line)

    y_vals = a[i]*x_vals + b[i] + noise

    pts = np.column_stack((x_vals, y_vals))

    X.append(pts)
    y.append(np.full(samples_per_line, i))

X = np.vstack(X)
y = np.concatenate(y)

# Plot dataset
plt.figure(figsize=(7,7))
plt.scatter(X[:,0], X[:,1], s=10, alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset")
plt.show()

# ==========================
# Convert to torch
# ==========================
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# ==========================
# Train / Test split
# ==========================
N = X.shape[0]
perm = torch.randperm(N)

train_ratio = 0.8
n_train = int(train_ratio * N)

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
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, N_lines)
    #No softmax needed since CrossEntropyLoss applies it internally
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================
# Training
# ==========================
epochs = 500

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

# ==========================
# Test accuracy
# ==========================
with torch.no_grad():

    outputs = model(X_test)

    pred = torch.argmax(outputs, dim=1)

    acc = (pred == y_test).float().mean()

print("Test Accuracy:", acc.item())

# ==========================
# Plot labeled dataset
# ==========================
X_np = X.numpy()
y_np = y.numpy()

plt.figure(figsize=(7,7))

for i in range(N_lines):
    idx = y_np == i
    plt.scatter(X_np[idx,0], X_np[idx,1], s=10, label=f"Line {i}")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("5 noisy lines dataset")
plt.show()