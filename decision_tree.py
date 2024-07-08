import numpy as np
from tree import Node, resample_data

class CustomDecisionTree:
    def __init__(self, depth_limit=None):
        """Initialize the CustomDecisionTree with an optional depth limit."""
        self.depth_limit = depth_limit

    def fit(self, features, labels):
        """Fit the CustomDecisionTree to the provided features and labels."""
        self.num_classes = len(np.unique(labels))
        self.num_features = features.shape[1]
        self.root = self._build_tree(features, labels)

    def _build_tree(self, features, labels, current_depth=0):
        """Recursively build the tree."""
        class_counts = [np.sum(labels == c) for c in range(self.num_classes)]
        majority_class = np.argmax(class_counts)
        node = Node(current_depth, self.depth_limit)

        # Check if we should split further
        if current_depth < self.depth_limit and len(features) > 1:
            selected_features = np.random.choice(self.num_features, size=int(np.sqrt(self.num_features)), replace=False)
            best_feature, best_split = self._find_optimal_split(features, labels, selected_features)
            if best_feature is not None:
                node.split_feature = best_feature
                node.split_value = best_split
                left_mask = features[:, best_feature] < best_split
                right_mask = ~left_mask
                features_left, labels_left = features[left_mask], labels[left_mask]
                features_right, labels_right = features[right_mask], labels[right_mask]
                if len(labels_left) > 0 and len(labels_right) > 0:
                    node.left_child = self._build_tree(features_left, labels_left, current_depth + 1)
                    node.right_child = self._build_tree(features_right, labels_right, current_depth + 1)
                else:
                    node.leaf_value = majority_class
            else:
                node.leaf_value = majority_class
        else:
            node.leaf_value = majority_class

        return node

    def _find_optimal_split(self, features, labels, feature_indices):
        """Find the best feature and threshold for splitting the data."""
        lowest_gini = float('inf')
        optimal_feature, optimal_threshold = None, None

        if len(feature_indices) == 0:
            return optimal_feature, optimal_threshold

        for feature in feature_indices:
            sorted_thresholds = np.unique(features[:, feature])
            for threshold in sorted_thresholds:
                left_mask = features[:, feature] < threshold
                right_mask = features[:, feature] >= threshold
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini_score = self._compute_gini(labels[left_mask], labels[right_mask])

                if gini_score < lowest_gini:
                    lowest_gini = gini_score
                    optimal_feature = feature
                    optimal_threshold = threshold

        return optimal_feature, optimal_threshold

    def _compute_gini(self, left_labels, right_labels):
        """Calculate the Gini impurity of a split."""
        n_left, n_right = len(left_labels), len(right_labels)
        if n_left == 0 or n_right == 0:
            return 0
        total_samples = n_left + n_right
        p_left = n_left / total_samples
        p_right = n_right / total_samples
        gini_left = 1 - sum((np.sum(left_labels == c) / n_left) ** 2 for c in np.unique(left_labels))
        gini_right = 1 - sum((np.sum(right_labels == c) / n_right) ** 2 for c in np.unique(right_labels))
        return p_left * gini_left + p_right * gini_right

    def predict(self, data):
        """Predict the class labels for the given data."""
        return np.array([self._predict_single(sample, self.root) for sample in data])

    def _predict_single(self, sample, node):
        """Recursively predict the class label for a single sample."""
        if node.leaf_value is not None:
            return node.leaf_value
        if sample[node.split_feature] < node.split_value:
            return self._predict_single(sample, node.left_child)
        else:
            return self._predict_single(sample, node.right_child)
