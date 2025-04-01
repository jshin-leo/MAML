# data_loader.py
# import pickle
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# # Load the dataset
# with open('src/document_vectors_with_labels.pkl', 'rb') as file:
#     data = pickle.load(file)

# # Extract labels and vectors
# labels = np.array([item['label'] for item in data])
# vectors = np.array([item['vector'] for item in data])

# # Encode labels to numeric values
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(labels)

import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file, 'rb') as file:
            self.data = pickle.load(file)
        # Assume data is a list of dicts
        self.labels = [item['label'] for item in self.data]
        self.vectors = [item['vector'] for item in self.data]
        
        # Optionally, map labels to indices if needed
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.labels_idx = [self.label_to_idx[label] for label in self.labels]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vector = self.vectors[idx]
        label = self.labels_idx[idx]
        
        # Convert the vector to a PyTorch tensor and reshape it
        vector = np.array(vector, dtype=np.float32)
        vector = torch.tensor(vector).unsqueeze(0)  # Add channel dimension
        
        return vector, label

# Path to your pickle file
#data_file = 'document_vectors_with_labels.pkl'

# Create dataset and dataloader
#dataset = CustomDataset(data_file)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# print("This is the shape of labels ", type(dataset.labels), len(dataset.labels))
# print("This is the shape of vectors", type(dataset.vectors), len(dataset.vectors))
# print("These is the vector at index 0:", dataset.vectors[0])
# print("This is the label of vector at 0:", dataset.labels_idx[0])
# print("This is the type of the vector at 0::", type(dataset.vectors[0]))



