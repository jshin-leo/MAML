# import numpy as np
# import torch
# from prototypical_batch_sampler import PrototypicalBatchSampler
# from data_loader import encoded_labels as labels, vectors

# def main():
#     # Define parameters based on your data
#     classes_per_it = 5  # Number of classes per batch
#     num_samples = 10    # Number of samples per class (support + query)
#     iterations = 200    # Number of iterations per epoch

#     # Initialize the PrototypicalBatchSampler
#     batch_sampler = PrototypicalBatchSampler(
#         labels=labels,
#         classes_per_it=classes_per_it,
#         num_samples=num_samples,
#         iterations=iterations
#     )

#     # Iterate over the batches
#     for batch in batch_sampler:
#         # batch is a tensor containing indices for the current batch
#         print("Batch indices:", batch)
#         # Convert tensor to numpy array for indexing
#         batch_vectors = vectors[batch.numpy()]
#         print("Batch vectors shape:", batch_vectors.shape)
#         # Add your processing code here (e.g., feeding to a model)
        
#         # Example processing (just for demonstration):
#         print("Batch vectors:", batch_vectors)
#         break  # Remove this break to process all batches

# if __name__ == "__main__":
#     main()



import torch
import torch.optim as optim
from prototypical_batch_sampler import PrototypicalBatchSampler
from data_loader import encoded_labels as labels, vectors
from protonet import ProtoNet
from prototypical_loss import PrototypicalLoss

# Hyperparameters
input_dim = vectors.shape[1]  # Dimension of your feature vectors
hidden_dim = 64
z_dim = 64
classes_per_it = 5
num_samples = 10
iterations = 200
n_support = 5  # Number of support samples per class
learning_rate = 1e-3
num_epochs = 5

# Initialize the ProtoNet model
model = ProtoNet(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim)

# Define the PrototypicalLoss function and optimizer
loss_comp = PrototypicalLoss(n_support=n_support)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the PrototypicalBatchSampler
batch_sampler = PrototypicalBatchSampler(
    labels=labels,
    classes_per_it=classes_per_it,
    num_samples=num_samples,
    iterations=iterations
)

def train():
    model.train()
    for epoch in range(num_epochs):
        for batch in batch_sampler:
            # Get the vectors for the current batch
            batch_vectors = vectors[batch.numpy()]
            batch_vectors = torch.tensor(batch_vectors, dtype=torch.float32)

            # Dummy target for this example; in practice, use true labels for training
            batch_labels = torch.tensor([labels[i] for i in batch.numpy()])

            # Forward pass
            embeddings = model(batch_vectors)

            # Compute loss and accuracy
            loss, acc = loss_comp(embeddings, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {acc.item()}")


if __name__ == "__main__":
    train()
