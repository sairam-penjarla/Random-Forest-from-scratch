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
