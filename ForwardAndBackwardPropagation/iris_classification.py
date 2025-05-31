import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Import our custom neural network implementation
from NeuralNetwork import NeuralNetwork

# Set random seed for reproducibility
np.random.seed(19)

# Load the Iris dataset
print("Loading Iris dataset...")
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv('data/iris.csv', header=None, names=column_names)

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Basic dataset information
print("\nDataset information:")
print(data.info())

print("\nDataset statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Visualize the data
print("\nCreating data visualizations...")

# Create a directory for saving visualizations
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Pairplot to visualize relationships between features
plt.figure(figsize=(10, 8))
sns.pairplot(data, hue='species')
plt.savefig('visualizations/pairplot.png')
plt.close()

# Distribution of features by species
plt.figure(figsize=(15, 10))
for i, feature in enumerate(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
    plt.subplot(2, 2, i+1)
    for species in data['species'].unique():
        sns.kdeplot(data[data['species'] == species][feature], label=species)
    plt.title(f'Distribution of {feature} by Species')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
plt.tight_layout()
plt.savefig('visualizations/feature_distributions.png')
plt.close()

# Prepare data for neural network
print("\nPreparing data for neural network training...")

# Encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(data['species'].values.reshape(-1, 1))

# Extract features
X = data.iloc[:, :-1].values

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Create and train the neural network
print("\nCreating and training the neural network...")

# Define network architecture: 4 input features, 8 neurons in hidden layer, 3 output classes
nn = NeuralNetwork(
    layer_sizes=[4, 8, 3],
    activations=['sigmoid', 'softmax'],
    learning_rate=0.1,
    epochs=1000
)

# Train the model
nn.fit(X_train, y_train)

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(nn.loss_history)
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('visualizations/loss_curve.png')
plt.close()

# Evaluate the model
print("\nEvaluating the model...")
y_pred_prob = nn.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred == y_true)
print(f"Test accuracy: {accuracy:.4f}")

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = list(data['species'].unique())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualize decision boundaries (for 2 features)
def plot_decision_boundary(X, y, model, feature_indices=[0, 1], feature_names=None):
    """
    Plot the decision boundary for a model trained on 2 features.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature data.
    y : numpy.ndarray
        Target labels.
    model : object
        Trained model with predict method.
    feature_indices : list
        Indices of the two features to plot.
    feature_names : list
        Names of the features.
    """
    # Extract the two features we want to plot
    X_plot = X[:, feature_indices]
    
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create a mesh of points to predict on
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For each mesh point, we need to create a full feature vector
    # by filling in the other features with their mean values
    X_full = np.zeros((mesh_points.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        if i in feature_indices:
            idx = feature_indices.index(i)
            X_full[:, i] = mesh_points[:, idx]
        else:
            X_full[:, i] = X[:, i].mean()
    
    # Predict on the mesh
    Z = model.predict(X_full)
    Z = np.argmax(Z, axis=1)
    
    # Plot the decision boundary
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot the training points
    for i, class_name in enumerate(class_names):
        idx = np.where(np.argmax(y, axis=1) == i)
        plt.scatter(X_plot[idx, 0], X_plot[idx, 1], label=class_name, edgecolors='k')
    
    plt.xlabel(feature_names[feature_indices[0]] if feature_names else f'Feature {feature_indices[0]}')
    plt.ylabel(feature_names[feature_indices[1]] if feature_names else f'Feature {feature_indices[1]}')
    plt.title('Decision Boundary')
    plt.legend()
    return plt

# Plot decision boundaries for sepal length and sepal width
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
plot_decision_boundary(X, y, nn, [0, 1], feature_names)
plt.savefig('visualizations/decision_boundary_sepal.png')
plt.close()

# Plot decision boundaries for petal length and petal width
plot_decision_boundary(X, y, nn, [2, 3], feature_names)
plt.savefig('visualizations/decision_boundary_petal.png')
plt.close()

print("\nAll visualizations have been saved to the 'visualizations' directory.")
print("Neural network implementation and evaluation complete!")
