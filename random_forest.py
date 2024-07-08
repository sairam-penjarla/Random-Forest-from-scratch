import numpy as np
from decision_tree import CustomDecisionTree

class RandomForestClassifier:
    def __init__(self, num_trees=100, depth_limit=None):
        """Initialize the RandomForestClassifier with the number of trees and an optional depth limit."""
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.trees = []

    def fit(self, features, labels):
        """Fit the RandomForestClassifier to the provided features and labels."""
        self.trees = []

        for _ in range(self.num_trees):
            tree = CustomDecisionTree(depth_limit=self.depth_limit)
            sample_features, sample_labels = self._create_bootstrap_sample(features, labels)
            tree.fit(sample_features, sample_labels)
            self.trees.append(tree)

    def predict(self, features):
        """Predict the class labels for the given features."""
        predictions = np.zeros((features.shape[0], self.num_trees))

        for idx, tree in enumerate(self.trees):
            predictions[:, idx] = tree.predict(features)

        return np.mean(predictions, axis=1)

    def _create_bootstrap_sample(self, features, labels):
        """Generate a bootstrap sample from the dataset."""
        num_samples = features.shape[0]
        sample_indices = np.random.choice(num_samples, size=num_samples, replace=True)
        return features[sample_indices], labels[sample_indices]
