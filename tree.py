import numpy as np

def resample_data(features, labels):
    """Generate a bootstrap sample from the dataset."""
    num_samples = features.shape[0]
    sample_indices = np.random.choice(num_samples, size=num_samples, replace=True)
    return features[sample_indices], labels[sample_indices]

class Node:
    def __init__(self, current_depth=0, depth_limit=None):
        """Initialize a tree node."""
        self.current_depth = current_depth
        self.depth_limit = depth_limit
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None  # Value for leaf nodes
