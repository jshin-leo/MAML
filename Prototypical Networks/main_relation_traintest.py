import pickle
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Dataset
from relation_networks import RelationNetworks
from collections import defaultdict
import random
from sklearn.preprocessing import LabelEncoder
from sampler import BalancedBatchSampler
import torch.nn as nn


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item['vector']), item['label']


# Load your data from the .pkl file
with open('document_vectors_with_labels.pkl', 'rb') as f:
    data = pickle.load(f)  # Load the list of dictionaries

# Extract labels and perform label encoding
labels = [item['label'] for item in data]
vectors = [item['vector'] for item in data]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(np.unique(labels))

# Create a mapping from original labels to encoded labels
label_mapping = {original: encoded for original, encoded in zip(set(labels), encoded_labels)}

# Update your dataset to use encoded labels
for idx, item in enumerate(data):
    item['label'] = label_mapping[item['label']]

# Group data by labels for better organization
data_by_label = defaultdict(list)
for item in data:
    data_by_label[item['label']].append(item['vector'])

# Split classes into training and testing sets
all_labels = list(data_by_label.keys())
print("All labels {}".format(all_labels))
#random.shuffle(all_labels)

train_classes = all_labels[:int(0.8 * len(all_labels))]  # 70% classes for training
test_classes = all_labels[int(0.8 * len(all_labels)):]  # 30% classes for testing
print(train_classes, test_classes)




# Prepare support and query sets for training classes
def prepare_data(data_by_label, classes, num_support=10, num_query=10):#Remember that this changes depending on the number of classes for test and train
    support_vectors, support_labels = [], []
    query_vectors, query_labels = [], []

    for label in classes:
        vectors = data_by_label[label]
        random.shuffle(vectors)  # Shuffle to randomize selection
        support_samples = vectors[:num_support]
        query_samples = vectors[num_support:num_support + num_query]  # Take the next samples for query

        # Append support samples
        for vector in support_samples:
            support_vectors.append(vector)
            support_labels.append(label)

        # Append query samples
        for vector in query_samples:
            query_vectors.append(vector)
            query_labels.append(label)

    return support_vectors, support_labels, query_vectors, query_labels


# Prepare training data (support and query)
support_vectors, support_labels, query_vectors, query_labels = prepare_data(data_by_label, train_classes)
print("Support labels {}".format(set(support_labels)))
print("Support labels shape {}".format(len(support_labels)))


# Prepare test data (query only, as we will evaluate with unseen classes)
test_support_vectors, test_support_labels, test_query_vectors, test_query_labels = prepare_data(data_by_label,
                                                                                                test_classes)
print("Test support labels {}".format(test_support_labels))

# Convert to Tensors for training and testing
support_vectors = torch.stack([torch.tensor(vec) for vec in support_vectors])
support_labels = torch.tensor(support_labels)
query_vectors = torch.stack([torch.tensor(vec) for vec in query_vectors])
query_labels = torch.tensor(query_labels)

test_support_vectors = torch.stack([torch.tensor(vec) for vec in test_support_vectors])
test_support_labels = torch.tensor(test_support_labels)
test_query_vectors = torch.stack([torch.tensor(vec) for vec in test_query_vectors])
test_query_labels = torch.tensor(test_query_labels)

print("Training Classes Support and Query shapes:")
print("support_vectors", support_vectors.shape, "support_labels", support_labels.shape)
print("query_vectors", query_vectors.shape, "query_labels", query_labels.shape)

print("\nTest Classes Support and Query shapes:")
print("test_support_vectors", test_support_vectors.shape, "test_support_labels", test_support_labels.shape)
print("test_query_vectors", test_query_vectors.shape, "test_query_labels", test_query_labels.shape)

# Create dataset and dataloader for training and testing
support_dataset = CustomDataset([{'vector': v, 'label': l} for v, l in zip(support_vectors, support_labels)])
query_dataset = CustomDataset([{'vector': v, 'label': l} for v, l in zip(query_vectors, query_labels)])

test_support_dataset = CustomDataset(
    [{'vector': v, 'label': l} for v, l in zip(test_support_vectors, test_support_labels)])
test_query_dataset = CustomDataset([{'vector': v, 'label': l} for v, l in zip(test_query_vectors, test_query_labels)])
batch_size_support = 25  # Batch size for support
batch_size_query = 35    # Batch size for query

# Creating samplers
support_sampler = BalancedBatchSampler(support_labels.tolist(), batch_size_support)
query_sampler = BalancedBatchSampler(query_labels.tolist(), batch_size_query)
# Creating dataloaders using the custom sampler
support_dataloader = DataLoader(support_dataset, batch_sampler=support_sampler)
query_dataloader = DataLoader(query_dataset, batch_sampler=query_sampler)

test_support_sampler = BalancedBatchSampler(test_support_labels.tolist(), batch_size_support)
test_query_sampler = BalancedBatchSampler(test_query_labels.tolist(), batch_size_query)

test_support_dataloader = DataLoader(test_support_dataset, batch_sampler=test_support_sampler)
test_query_dataloader = DataLoader(test_query_dataset, batch_sampler=test_query_sampler)

# Define the model
feature_dimension = 50  # Your feature dimension
model = RelationNetworks(feature_dimension=feature_dimension)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


# Training function
def train(model, support_dataloader, query_dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for support_batch, query_batch in zip(support_dataloader, query_dataloader):
        support_vectors, support_labels = support_batch
        query_vectors, query_labels = query_batch

        # Process the support set
        model.process_support_set(support_vectors, support_labels)

        # Forward pass on the query set
        relation_scores = model.forward(query_vectors)

        # Compute the loss
        loss = criterion(relation_scores.float(), query_labels)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(support_dataloader)


# Testing function
def test(model, support_dataloader, query_dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for support_batch, query_batch in zip(support_dataloader, query_dataloader):
            support_vectors, support_labels = support_batch
            query_vectors, query_labels = query_batch

            model.process_support_set(support_vectors, support_labels)
            relation_scores = model.forward(query_vectors)

            # Get the predicted labels
            predicted = torch.argmax(relation_scores, dim=1)
            total += query_labels.size(0)
            correct += (predicted == query_labels).sum().item()

    accuracy = correct / total
    return accuracy


# Main training loop
num_epochs = 100
avg_acc = 0
avg_loss = 0
best_acc = 0
best_loss = 999
for epoch in range(num_epochs):
    support_sampler.reset()
    query_sampler.reset()
    train_loss = train(model, support_dataloader, query_dataloader, optimizer, criterion)
    accuracy = test(model, test_support_dataloader, test_query_dataloader)

    avg_loss += train_loss
    avg_acc += accuracy

    if train_loss < best_loss:
        best_loss = train_loss
    
    if best_acc < accuracy:
        best_acc = accuracy

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Test Accuracy: {accuracy:.4f}")

avg_acc = avg_acc / num_epochs
avg_loss = avg_loss / num_epochs

print(f"Average Accuracy {avg_acc}, Average Loss {avg_loss}")
print(f"Best Accuracy {best_acc}, Best Loss {best_loss}")


# Save the model
torch.save(model.state_dict(), 'relation_networks_model.pth')
print("Model saved to 'relation_networks_model.pth'.")
