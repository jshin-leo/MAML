import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# Example Embedding Network (Simple CNN)
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Embedding of size 64


# Prototypical Network
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(PrototypicalNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, support, support_labels, query):
        # Embed the support and query samples
        support_embeddings = self.embedding_net(support)
        query_embeddings = self.embedding_net(query)

        # Compute prototypes for each class
        unique_classes = torch.unique(support_labels)
        prototypes = torch.stack([support_embeddings[support_labels == cls].mean(0) for cls in unique_classes])

        # Compute distance from each query to each prototype
        dists = torch.cdist(query_embeddings, prototypes)

        return dists  # Distances to prototypes

    def loss(self, dists, query_labels):
        # Loss is cross-entropy based on the distances (logits)
        return F.cross_entropy(-dists, query_labels)


# Initialize the embedding network and prototypical network
embedding_net = EmbeddingNet()
proto_net = PrototypicalNetwork(embedding_net)

# Dummy Data (3 classes, 4 examples per class, query of 2 images)
support_images = torch.randn(12, 1, 28, 28)  # 12 support images (3 classes * 4 examples)
support_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])  # 3 classes
query_images = torch.randn(6, 1, 28, 28)  # 6 query images
query_labels = torch.tensor([0, 0, 1, 1, 2, 2])

# Forward pass
dists = proto_net(support_images, support_labels, query_images)
loss = proto_net.loss(dists, query_labels)

# Training Step
optimizer = Adam(proto_net.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()

loss_value = loss.item()
dists_value = dists.detach().numpy()
loss_value, dists_value