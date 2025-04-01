import random
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.label_to_indices = {label: [] for label in set(labels)}
        self.original_label_to_indices = {label: [] for label in set(labels)}

        # Populate the dictionary with indices corresponding to each label
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
            self.original_label_to_indices[label].append(idx)

        # Shuffle the indices for each label
        self.shuffle_indices()

    def shuffle_indices(self):
        """Shuffle the indices of each label."""
        for label in self.label_to_indices:
            random.shuffle(self.label_to_indices[label])

    def reset(self):
        """Reset the indices to their original state for the next epoch."""
        self.label_to_indices = {label: list(indices) for label, indices in self.original_label_to_indices.items()}
        self.shuffle_indices()

    def __iter__(self):
        # Make a copy of the label indices for iteration
        label_indices = {label: list(indices) for label, indices in self.label_to_indices.items()}
        batch = []

        while True:
            available_labels = [label for label in label_indices if label_indices[label]]

            if not available_labels:
                break  # Stop if all classes are exhausted

            # Determine the number of classes and calculate the samples needed for each
            num_classes = len(available_labels)
            samples_per_class = max(1, self.batch_size // num_classes)

            for label in available_labels:
                if not label_indices[label]:
                    continue  # Skip labels that have no available indices

                # Randomly select the number of samples for this label
                selected_indices = random.sample(label_indices[label], min(samples_per_class, len(label_indices[label])))
                batch.extend(selected_indices)

                # Remove the selected indices from the label's pool
                for idx in selected_indices:
                    label_indices[label].remove(idx)

                # If the batch reaches the specified size, yield it
                if len(batch) >= self.batch_size:
                    yield batch[:self.batch_size]  # Yield exactly `batch_size` samples
                    batch = batch[self.batch_size:]  # Retain any leftover samples for the next batch

        if batch:  # Yield any remaining samples at the end
            yield batch

    def __len__(self):
        # The total number of batches
        total_samples = sum(len(indices) for indices in self.label_to_indices.values())
        return max(total_samples // self.batch_size, 1)  # Ensure at least one batch is returned
