import pickle
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Dataset
from relation_networks import RelationNetworks 
from collections import defaultdict
import random
from sklearn.preprocessing import LabelEncoder
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


# Prepare support and query sets
support_vectors = []
support_labels = []
query_vectors = []
query_labels = []

for label, vectors in data_by_label.items():
    random.shuffle(vectors)  # Shuffle to randomize selection
    num_support = 50  # Adjust this number as needed for few-shot capability
    support_samples = vectors[:num_support]
    query_samples = vectors[num_support:]  # Remaining samples for query

    # Append support samples
    for vector in support_samples:
        support_vectors.append(vector)
        support_labels.append(label)
    
    # Append query samples
    for vector in query_samples:
        query_vectors.append(vector)
        query_labels.append(label)

# Convert to Tensors for training
support_vectors = torch.stack([torch.tensor(vec) for vec in support_vectors])
support_labels = torch.tensor(support_labels)  # Labels are now encoded as integers
query_vectors = torch.stack([torch.tensor(vec) for vec in query_vectors])
query_labels = torch.tensor(query_labels)  # Same for query labels
print("support_vectors",support_vectors.shape, len(support_vectors))
print("support_labels",support_labels.shape, len(support_labels))
print("query_vectors",query_vectors.shape, len(query_vectors))
print("query_labels", query_labels.shape, len(query_labels))



# Create dataset and dataloader for support and query sets
support_dataset = CustomDataset([{'vector': v, 'label': l} for v, l in zip(support_vectors, support_labels)])
query_dataset = CustomDataset([{'vector': v, 'label': l} for v, l in zip(query_vectors, query_labels)])


support_dataloader = DataLoader(support_dataset, batch_size=50, shuffle=True)  # Adjust batch_size as needed
query_dataloader = DataLoader(query_dataset, batch_size=50, shuffle=True)  # Adjust batch_size as needed

# Define the model
feature_dimension = 50  # Your feature dimension
model = RelationNetworks(feature_dimension=feature_dimension)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed
criterion = torch.nn.CrossEntropyLoss()  # Cross Entropy Loss for the relation score

# Training loop
num_epochs = 5  # Adjust number of epochs as needed
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    i = 0
    for support_batch in support_dataloader:
        support_vectors, support_labels = support_batch
        print('support vector',support_vectors.shape, len(support_vectors), 'this is i', i)
        i+= 1
        
        # Use the same logic to prepare query vectors and labels
        # For this example, we can randomly sample from the query dataset
        query_batch = next(iter(query_dataloader))
        query_vectors, query_labels = query_batch
        # print('query vector',query_vectors.shape, len(query_vectors), 'this is i', i)
        
        
        # Process the support set
        model.process_support_set(support_vectors, support_labels)
        print(i)

        # Forward pass
        relation_scores = model.forward(query_vectors)
        
        # Compute the loss (assuming binary labels for simplicity)
        print(relation_scores.shape)
        # print(query_labels, query_labels.shape)
        # print(torch.argmax(relation_scores, dim=1), torch.argmax(relation_scores, dim=1).shape)

        loss = criterion(relation_scores.float(), query_labels)  # Ensure labels are float FIX ME: I think that FIXME

        #one got encode ground truth 



        # loss = criterion(torch.argmax(relation_scores, dim=1).float(), query_labels.float())  # Ensure labels are float FIX ME: I think that


#         total_loss += loss.item()
        
#         # Backward pass and optimization
#         optimizer.zero_grad()  # Zero the gradients
#         loss.backward()  # Backpropagation
#         optimizer.step()  # Update weights

#     # Print epoch loss
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(support_dataloader):.4f}")

# # Save the model
# torch.save(model.state_dict(), 'relation_networks_model.pth')
# print("Model saved to 'relation_networks_model.pth'.")
