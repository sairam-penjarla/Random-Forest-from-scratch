# Random Forest from Scratch

This repository contains an implementation of a Random Forest classifier from scratch using Python. The code is organized into separate files for clarity and modularity.

## Files

- `tree.py`: Contains the `Node` class and the `resample_data` function.
- `decision_tree.py`: Contains the `CustomDecisionTree` class, which implements a single decision tree.
- `random_forest.py`: Contains the `RandomForestClassifier` class, which implements the random forest using multiple decision trees.
- `inference.py`: Example usage of the Random Forest classifier on the Iris dataset.

## Installation

To install the necessary dependencies, you can use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### Cloning the Repository

To get a local copy of this repository, use the following commands:

```bash
# Clone the repository
git clone https://github.com/sairam-penjarla/Random-Forest-from-scratch.git

# Navigate to the project directory
cd Random-Forest-from-scratch
```

To see an example of how to use the Random Forest classifier, run the `inference.py` script. This script demonstrates loading the Iris dataset, training the Random Forest classifier, making predictions, and evaluating the model's accuracy.

```bash
python inference.py
```

### Example

Here is a brief overview of the steps performed in `inference.py`:

1. Load the Iris dataset.
2. Split the dataset into training and testing sets.
3. Train the Random Forest classifier on the training set.
4. Make predictions on the testing set.
5. Evaluate the accuracy of the predictions.

```python
# Example usage with Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest import RandomForestClassifier

# Load the Iris dataset
iris_data = load_iris()
features, labels = iris_data.data, iris_data.target

# Split the dataset into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
random_forest = RandomForestClassifier(num_trees=100, depth_limit=5)
random_forest.fit(features_train, labels_train)

# Make predictions on the test set
predicted_labels = random_forest.predict(features_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(labels_test, predicted_labels.round())
print(f'Accuracy: {accuracy * 100:.2f}%')
```

## Repository Structure

- `tree.py`: Contains the helper function and class for tree nodes.
- `decision_tree.py`: Implements the decision tree logic.
- `random_forest.py`: Implements the random forest logic.
- `inference.py`: Example script for using the random forest classifier.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.