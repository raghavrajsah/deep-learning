# AIT Budapest - Deep Learning course
## Assignment 1 - backpropagation
## Created by: Bálint Gyires-Tóth

# Introduction

In this assignment, we focus on some concepts of neural networks and backpropagation. Neural networks are a fundamental component of deep learning, and understanding how they work is crucial for building and training effective models. Backpropagation is an algorithm used to train neural networks by adjusting the weights and biases of the network based on the error between the predicted and actual outputs. Through this assignment, we will delve into some details of the theory behind neural networks and backpropagation, and implement them in code to gain hands-on experience.

Please always write your anwser between the "------" lines. 

Let's get started!

## Rules

Your final score will be penalized if you engage in the following behaviors:

1. 20% penalty for not using or following correctly the structure of this file.
2. 20% penalty for lengthy answers.
3. 40% penalty for for joint work or copying, including making the same, non-tipical mistakes, to all students concerned.
4. Late submission is not allowed. 
5. The use of generative AI is not allowed. 

# Theory (50 points)
We have the following neural network architecture:

Input: 10 variables
Hidden layer: Fully-connected layer with 16 neurons and sigmoid activation, with bias
Output layer: Fully-connected layer with 1 neuron and sigmoid activation, no bias

The fully-connected layer is defined as z^(i+1) = a^(i)*W^(i), where 

z^(i): output of the i-th fully-connected layer before the activation function.
a^(i): output of the activation function of the i-th layer.
a^(1): X (the input data).
W^(i): weight matrix between the i-th layer and (i+1)-th layer. 

Use the notation as above. For partial deriavative use the @ sign. The cost function is MSE denoted by C, and the ground truth is y. 

Question 1: Define the number of parameters in the neural network. (10 points)
------
Answer 1: 
------

Question 2: Define the output of Layer 1 after the activation w.r.t. the input data. (10 points)
------
Answer 2:
------

Question 3: Define the output of Layer 2 after the activation w.r.t. the input data. (10 points)
------
Answer 3:
------

Question 4: Define the gradient of the loss w.r.t. W^(2). (10 points)
------
Answer 4:
------

Question 5: Define the gradient of the loss w.r.t. W^(1). (10 points)
------
Answer 5:
------

# Practice (50 points)
Use the sample code for basic back propagation below. You will implement a train-validation-test split, introduce early stopping, and wandb.com logging.After these modifications, you should update the code to achieve at least 95% accuracy. 
When you are done, copy the final code back to this file and upload it to Moodle. Please remove your WandB API key from the code before upload.

Tasks: 
1. Data Splitting:
   Split the dataset into train (70%), validation (20%), and test (10%) sets.
   Print the number of samples in each set as follows:
   
   print("Number of samples in train set:", len(X_train))
   print("Number of samples in validation set:", len(X_val))
   print("Number of samples in test set:", len(X_test))
   
2. Implement early stopping based on the validation loss.
   Print a message indicating when early stopping is triggered:
   
   print(f"Early stopping at epoch {epoch} with validation loss: {val_loss:.4f}")

3. Log the training and validation accuracy and loss to Weights & Biases. Use this resource: https://docs.wandb.ai/quickstart/ 

4. Modify the code to achieve at least 95% accuracy on the test data. Use the following code for evaluation: 

   test_preds = (A2_test > 0.5).astype(int)
   test_accuracy = np.mean(test_preds == y_test)
   print(f"Final test accuracy: {test_accuracy * 100:.2f}%")


--------------------code-----------------------

import numpy as np
import matplotlib.pyplot as plt

# Fix the random seed, to have the same random numbers each time we run the code
np.random.seed(42)

# We will create 200 samples in total
m = 200

# The four base points for the classic XOR problem
base_points = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
# The corresponding XOR labels (0, 1, 1, 0)
base_labels = np.array([0, 1, 1, 0])

# Randomly pick which base point each of the 200 samples comes from
idx = np.random.randint(0, 4, size=m)
X = base_points[idx]
y = base_labels[idx].reshape(m, 1)

# Add noise to the base points
noise = 0.2 * np.random.randn(m, 2)  # Gaussian noise with std dev of 0.2
X_noisy = X + noise

# 2. DEFINE THE NETWORK ARCHITECTURE
n_input = 2    # number of input features (x1, x2)
n_hidden = 3   # number of neurons in the hidden layer
n_output = 1   # single output neuron for binary classification

# Initialize weights and biases
W1 = np.random.randn(n_input, n_hidden) * 0.1
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output) * 0.1
b2 = np.zeros((1, n_output))

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# 3. TRAINING SETTINGS

lr = 0.1        # learning rate
epochs = 20000   # number of iterations

# 4. TRAINING LOOP
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_noisy, W1) + b1   # Pre-activation of hidden layer
    A1 = sigmoid(Z1)               # Hidden layer activation
    Z2 = np.dot(A1, W2) + b2       # Pre-activation of output layer
    A2 = sigmoid(Z2)               # Output layer activation

    # Binary Cross-Entropy Loss
    # L = -(1/m) * sum[ y*log(A2) + (1-y)*log(1-A2) ]
    eps = 1e-8  # small epsilon for numerical stability
    loss = -(1/m) * np.sum(y * np.log(A2 + eps) + (1 - y) * np.log(1 - (A2 + eps)))
    
    # Backpropagation
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = (1/m) * np.dot(X_noisy.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    
    # Optional: print progress
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 5. EVALUATE TRAINING PERFORMANCE

preds = (A2 > 0.5).astype(int)
accuracy = np.mean(preds == y)
print(f"Final training accuracy: {accuracy * 100:.2f}%")

# 6. VISUALIZE THE NOISY XOR DATA

plt.figure(figsize=(6,5))
plt.title("Visualization of groudtruth")
# Points with label=0 will be shown in one color, label=1 in another
plt.scatter(X_noisy[:, 0], X_noisy[:, 1], c=y.reshape(-1), cmap="bwr", alpha=0.7)
plt.xlabel("x1")
plt.ylabel("x2")
plt.colorbar(label="Class (0 or 1)")
plt.show()

plt.figure(figsize=(6,5))
plt.title("Visualization of predictions")
# Points with label=0 will be shown in one color, label=1 in another
plt.scatter(X_noisy[:, 0], X_noisy[:, 1], c=preds.reshape(-1), cmap="bwr", alpha=0.7)
plt.xlabel("x1")
plt.ylabel("x2")
plt.colorbar(label="Class (0 or 1)")
plt.show()
