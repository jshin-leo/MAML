from collections import OrderedDict

import torch
from torch import Tensor

def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    n_way = len(torch.unique(support_labels))
    n_way = 10
    feature_dim = support_features.shape[1]  # Assuming support_features is [num_samples, feature_dim]

    # Prototype i is the mean of all instances of features corresponding to labels == i
    prototypes = []
    for label in range(n_way):
        # Get the indices where support_labels == label
        indices = torch.nonzero(support_labels == label, as_tuple=False).squeeze()

        if indices.numel() == 0:
            # If no examples for this label, create a zero tensor for the prototype
            prototypes.append(torch.zeros(feature_dim))
        else:
            # Compute the mean of features for this label
            prototypes.append(support_features[indices].mean(0))

    # Concatenate prototypes into a single tensor
    return torch.stack(prototypes)

def compute_prototypes_2(support_features: Tensor, support_labels: Tensor) -> Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """

    n_way = len(torch.unique(support_labels))
    n_way = 10
    # Prototype i is the mean of all instances of features corresponding to labels == i
    # print(torch.cat(
    #     [
    #         support_features[torch.nonzero(support_labels == label)].mean(0)
    #         for label in range(n_way)
    #     ]
    # ).shape)
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )


def entropy(logits: Tensor) -> Tensor:
    """
    Compute entropy of prediction.
    WARNING: takes logit as input, not probability.
    Args:
        logits: shape (n_images, n_way)
    Returns:
        Tensor: shape(), Mean entropy.
    """
    probabilities = logits.softmax(dim=1)
    return (-(probabilities * (probabilities + 1e-12).log()).sum(dim=1)).mean()


def k_nearest_neighbours(features: Tensor, k: int, p_norm: int = 2) -> Tensor:
    """
    Compute k nearest neighbours of each feature vector, not included itself.
    Args:
        features: input features of shape (n_features, feature_dimension)
        k: number of nearest neighbours to retain
        p_norm: use l_p distance. Defaults: 2.

    Returns:
        Tensor: shape (n_features, k), indices of k nearest neighbours of each feature vector.
    """
    distances = torch.cdist(features, features, p_norm)

    return distances.topk(k, largest=False).indices[:, 1:]


def power_transform(features: Tensor, power_factor: float) -> Tensor:
    """
    Apply power transform to features.
    Args:
        features: input features of shape (n_features, feature_dimension)
        power_factor: power to apply to features

    Returns:
        Tensor: shape (n_features, feature_dimension), power transformed features.
    """
    return (features.relu() + 1e-6).pow(power_factor)


def strip_prefix(state_dict: OrderedDict, prefix: str):
    """
    Strip a prefix from the keys of a state_dict. Can be used to address compatibility issues from
    a loaded state_dict to a model with slightly different parameter names.
    Example usage:
        state_dict = torch.load("model.pth")
        # state_dict contains keys like "module.encoder.0.weight" but the model expects keys like "encoder.0.weight"
        state_dict = strip_prefix(state_dict, "module.")
        model.load_state_dict(state_dict)
    Args:
        state_dict: pytorch state_dict, as returned by model.state_dict() or loaded via torch.load()
            Keys are the names of the parameters and values are the parameter tensors.
        prefix: prefix to strip from the keys of the state_dict. Usually ends with a dot.

    Returns:
        copy of the state_dict with the prefix stripped from the keys
    """
    return OrderedDict(
        [
            (k[len(prefix) :] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )
