import numpy as np

class NeuralNetwork:
    """
    A simple neural network implementation with forward and backward propagation.
    This implementation supports multiple layers with customizable activation functions.
    """
    
    def __init__(self, layer_sizes, activations=None, learning_rate=0.01, epochs=1000):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        layer_sizes : list
            List containing the number of neurons in each layer including input and output layers.
        activations : list, optional
            List of activation functions for each layer (except input layer).
            Supported activations: 'sigmoid', 'relu', 'tanh', 'softmax' (for output layer only).
        learning_rate : float, optional
            Learning rate for gradient descent.
        epochs : int, optional
            Number of training epochs.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # He initialization for weights
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))
        
        # Set activation functions
        if activations is None:
            self.activations = ['sigmoid'] * (self.num_layers - 1)
            if self.layer_sizes[-1] > 1:  # If output layer has multiple neurons, use softmax
                self.activations[-1] = 'softmax'
        else:
            if len(activations) != self.num_layers - 1:
                raise ValueError("Number of activation functions must match number of layers - 1")
            self.activations = activations
        
        # Initialize lists to store values during forward pass
        self.z_values = []  # Pre-activation values
        self.a_values = []  # Post-activation values
        
        # For tracking training progress
        self.loss_history = []
        
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow
    
    def _sigmoid_derivative(self, a):
        """Derivative of sigmoid function."""
        return a * (1 - a)
    
    def _relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """Derivative of ReLU function."""
        return np.where(z > 0, 1, 0)
    
    def _tanh(self, z):
        """Tanh activation function."""
        return np.tanh(z)
    
    def _tanh_derivative(self, a):
        """Derivative of tanh function."""
        return 1 - np.power(a, 2)
    
    def _softmax(self, z):
        """Softmax activation function."""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _apply_activation(self, z, activation_name):
        """Apply specified activation function."""
        if activation_name == 'sigmoid':
            return self._sigmoid(z)
        elif activation_name == 'relu':
            return self._relu(z)
        elif activation_name == 'tanh':
            return self._tanh(z)
        elif activation_name == 'softmax':
            return self._softmax(z)
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
    
    def _apply_activation_derivative(self, a, z, activation_name):
        """Apply derivative of specified activation function."""
        if activation_name == 'sigmoid':
            return self._sigmoid_derivative(a)
        elif activation_name == 'relu':
            return self._relu_derivative(z)
        elif activation_name == 'tanh':
            return self._tanh_derivative(a)
        elif activation_name == 'softmax':
            # For softmax, the derivative is handled differently in backpropagation
            # when combined with cross-entropy loss
            return 1
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns:
        --------
        numpy.ndarray
            Output predictions of shape (n_samples, n_output_features).
        """
        self.z_values = []
        self.a_values = [X]  # Input layer activation
        
        a = X
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            a = self._apply_activation(z, self.activations[i])
            self.a_values.append(a)
        
        return a  # Output layer activation
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute the loss between true and predicted values.
        Uses mean squared error for regression or cross-entropy for classification.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True target values.
        y_pred : numpy.ndarray
            Predicted values.
            
        Returns:
        --------
        float
            Computed loss value.
        """
        if self.activations[-1] == 'softmax':
            # Cross-entropy loss for softmax
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            # Mean squared error for regression
            return np.mean(np.sum(np.square(y_true - y_pred), axis=1)) / 2
    
    def backward_propagation(self, X, y):
        """
        Perform backward propagation to compute gradients.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples, n_output_features).
            
        Returns:
        --------
        tuple
            Lists of weight and bias gradients for each layer.
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradients
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        if self.activations[-1] == 'softmax':
            # For softmax with cross-entropy, the derivative simplifies
            delta = self.a_values[-1] - y
        else:
            # For other activation functions
            delta = (self.a_values[-1] - y) * self._apply_activation_derivative(
                self.a_values[-1], self.z_values[-1], self.activations[-1]
            )
        
        # Compute gradients for output layer
        dw[-1] = np.dot(self.a_values[-2].T, delta) / m
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Backpropagate error through hidden layers
        for l in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-l+1].T) * self._apply_activation_derivative(
                self.a_values[-l], self.z_values[-l], self.activations[-l]
            )
            dw[-l] = np.dot(self.a_values[-l-1].T, delta) / m
            db[-l] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dw, db
    
    def update_parameters(self, dw, db):
        """
        Update weights and biases using gradient descent.
        
        Parameters:
        -----------
        dw : list
            List of weight gradients.
        db : list
            List of bias gradients.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def fit(self, X, y, verbose=True):
        """
        Train the neural network on the given data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples, n_output_features).
        verbose : bool, optional
            Whether to print training progress.
            
        Returns:
        --------
        self
            Trained neural network instance.
        """
        self.loss_history = []
        
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.forward_propagation(X)
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward pass
            dw, db = self.backward_propagation(X, y)
            
            # Update parameters
            self.update_parameters(dw, db)
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions for the given input data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns:
        --------
        numpy.ndarray
            Predicted values.
        """
        return self.forward_propagation(X)
    
    def predict_classes(self, X):
        """
        Predict class labels for classification problems.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns:
        --------
        numpy.ndarray
            Predicted class indices.
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)

