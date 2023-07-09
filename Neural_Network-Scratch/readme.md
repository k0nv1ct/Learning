# Neural Network from Scratch using Vanilla Python and NumPy

This notebook explores the creation of a neural network from scratch using only vanilla Python and the NumPy library. The network is trained on the MNIST digits dataset and consists of an input layer, a hidden layer, and an output layer, with each layer containing 10 neurons.

## Prerequisites

To run the code in this notebook, you'll need the following dependencies:

- Python 3.x
- NumPy

## Code Structure

The code is structured into several functions, each responsible for a specific part of the neural network implementation.

### 1. Initialization

The `init_params()` function initializes the weights and biases of the neural network with random values.

### 2. Activation Functions

- The `ReLU(z)` function implements the rectified linear unit (ReLU) activation function, which returns the maximum between 0 and the input value.
- The `softmax(z)` function calculates the softmax activation for the output layer, which transforms the output into a probability distribution across the 10 classes.

### 3. Forward Propagation

The `forward_prop(w1, b1, w2, b2, X)` function performs forward propagation through the neural network. It computes the weighted sum and applies the activation functions to generate the outputs of each layer.

### 4. One-Hot Encoding

The `ohe(y)` function converts the target labels into a one-hot encoded format. It creates a binary matrix where each row corresponds to a label and has a 1 at the index representing the label and 0s elsewhere.

### 5. Derivative of the Activation Function

The `d_a(z)` function calculates the derivative of the ReLU activation function. It returns 1 for positive values and 0 for negative values.

### 6. Backpropagation

The `back_prop(z1, a1, z2, a2, w2, X, Y)` function implements the backpropagation algorithm. It computes the gradients of the weights and biases by propagating the errors from the output layer to the hidden layer and then to the input layer.

### 7. Parameter Update

The `update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)` function updates the weights and biases of the neural network using the gradients calculated during backpropagation. It performs gradient descent by subtracting the product of the learning rate (`alpha`) and the gradients from the current parameter values.

### 8. Prediction and Accuracy

The `get_predictions(a2)` function returns the predicted labels based on the output probabilities. It selects the index with the highest probability as the predicted class.

The `get_accuracy(pred, Y)` function calculates the accuracy of the predictions by comparing them to the true labels.

### 9. Gradient Descent

The `gradient_descent(X, Y, iter, alpha)` function performs the training of the neural network using gradient descent. It iteratively updates the parameters of the network based on the gradients computed through backpropagation. The function also prints the accuracy of the predictions at regular intervals during training.

## Running the Code

To run the code and train the neural network, follow these steps:

1. Ensure that Python 3.x is installed on your system.
2. Install the NumPy library by running `pip install numpy`.
3. Copy and paste the provided code into a Python environment or a Jupyter Notebook.
4. Execute the code to train the neural network on the MNIST digits dataset.
5. Observe the training progress and the accuracy of the predictions at regular intervals.
6. Experiment with different hyperparameters such as the number of iterations (`iter`) and the learning rate (`alpha`) to optimize the network's performance.

Note: This implementation is intended for educational purposes to understand the basics of neural networks and backpropagation. For practical use cases, it is recommended to use established deep learning libraries such as TensorFlow or PyTorch.

Happy exploring and learning with neural networks from scratch!
