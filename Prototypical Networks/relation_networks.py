# from typing import List, Dict, Optional
# import torch
# from torch import Tensor, nn
# from few_shot_classifier import FewShotClassifier
# from utils import compute_prototypes


# class RelationNetworks(FewShotClassifier):

#     def __init__(
#         self,
#         *args,
#         feature_dimension: int,
#         hidden_dimension: int = 64,  # You can choose an appropriate hidden layer size
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)

#         self.feature_dimension = feature_dimension

#         # Create your custom MLP relation module
#         self.relation_module = nn.Sequential(
#             nn.Linear(2 * self.feature_dimension, hidden_dimension),  # Input size is 100
#             nn.ReLU(),
#             nn.Linear(hidden_dimension, 1),  # Output a single relation score
#             nn.Sigmoid()  # Ensure the score is between 0 and 1
#         )

#     def process_support_set(self, support_data: List[Dict[str, any]]):
#         """
#         Overrides process_support_set of FewShotClassifier.
#         Extract feature maps from the support set and store class prototypes.
#         Args:
#             support_data: List of dictionaries containing 'label' and 'vector'.
#         """
#         support_vectors = [torch.tensor(item['vector']) for item in support_data]  # Extract vectors
#         support_labels = [item['label'] for item in support_data]  # Extract labels
        
#         support_features = torch.stack(support_vectors)  # Convert list of tensors to a single tensor
#         self._validate_features_shape(support_features)
#         self.prototypes = compute_prototypes(support_features, support_labels)

#     def forward(self, query_data: List[Dict[str, any]]) -> Tensor:
#         """
#         Overrides method forward in FewShotClassifier.
#         Predict the label of query samples by processing input in a list format.
#         Args:
#             query_data: List of dictionaries containing 'label' and 'vector'.
#         """
#         query_vectors = [torch.tensor(item['vector']) for item in query_data]  # Extract vectors
#         query_features = torch.stack(query_vectors)  # Convert list of tensors to a single tensor
#         self._validate_features_shape(query_features)

#         # Reshape query_features to match the expected input shape for relation computation
#         query_prototype_feature_pairs = torch.cat(
#             (
#                 self.prototypes.unsqueeze(dim=0).expand(query_features.shape[0], -1, -1),
#                 query_features.unsqueeze(dim=1).expand(-1, self.prototypes.shape[0], -1),
#             ),
#             dim=2
#         ).view(-1, 2 * self.feature_dimension)  # No extra height/width dimensions

#         # Each pair (query, prototype) is assigned a relation score
#         relation_scores = self.relation_module(query_prototype_feature_pairs).view(
#             -1, self.prototypes.shape[0]
#         )

#         return self.softmax_if_specified(relation_scores)

#     def _validate_features_shape(self, features):
#         if len(features.shape) != 2:  # Updated to 2D for (n_samples, feature_dimension)
#             raise ValueError(
#                 "Illegal backbone for Relation Networks. "
#                 "Expected output for a sample is a 1D tensor of shape (feature_dimension)."
#             )
#         if features.shape[1] != self.feature_dimension:
#             raise ValueError(
#                 f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
#             )

#     @staticmethod
#     def is_transductive() -> bool:
#         return False


from typing import List, Dict
import torch
from torch import Tensor, nn
from few_shot_classifier import FewShotClassifier
from utils import compute_prototypes

class RelationNetworks(FewShotClassifier):

    def __init__(
        self,
        *args,
        feature_dimension: int,
        hidden_dimension: int = 64,  # You can choose an appropriate hidden layer size
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.feature_dimension = feature_dimension

 
        self.relation_module = nn.Sequential(
            nn.Linear(2 * self.feature_dimension, hidden_dimension),  
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1)
        )
        #self.relation_module2 = nn.Sigmoid()

    def process_support_set(self, support_vectors: Tensor, support_labels: Tensor):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature maps from the support set and store class prototypes.
        Args:
            support_vectors: Tensor of support vectors.
            support_labels: Tensor of support labels.
        """
        self._validate_features_shape(support_vectors)  # Validate shape of support vectors
        print("Support labels {}".format(torch.unique(support_labels)))
        self.prototypes = compute_prototypes(support_vectors, support_labels)  # Compute prototypes

    def forward(self, query_vectors: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict the label of query samples by processing input as a tensor.
        Args:
            query_vectors: Tensor of shape (n_samples, feature_dimension).
        """
        self._validate_features_shape(query_vectors)  # Validate the shape of the query vectors

        # Reshape query_vectors to match the expected input shape for relation computation
        query_prototype_feature_pairs = torch.cat(
            (
                self.prototypes.unsqueeze(dim=0).expand(query_vectors.shape[0], -1, -1),
                query_vectors.unsqueeze(dim=1).expand(-1, self.prototypes.shape[0], -1),
            ),
            dim=2
        ).view(-1, 2 * self.feature_dimension)  # No extra height/width dimensions
        #print("Prototypes {}".format(self.prototypes))
        #print("Prototypes shape {}".format(self.prototypes.shape))
        #print("Query vectors {}".format(query_vectors.shape))
        # Each pair (query, prototype) is assigned a relation score
        relation_scores = self.relation_module(query_prototype_feature_pairs).view(
            -1, self.prototypes.shape[0]
        )
        relation_scores = torch.clamp(relation_scores,-50,50)
        #relation_scores = nn.Softmax(relation_scores)
        return self.softmax_if_specified(relation_scores)


    def _validate_features_shape(self, features):
        if len(features.shape) != 2:  # Updated to 2D for (n_samples, feature_dimension)
            raise ValueError(
                "Illegal backbone for Relation Networks. "
                "Expected output for a sample is a 1D tensor of shape (feature_dimension)."
            )
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
            )

    @staticmethod
    def is_transductive() -> bool:
        return False
