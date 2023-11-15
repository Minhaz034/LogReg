import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time

def sigmoid(z):
    """
    The sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):
    """
    Logistic regression hypothesis.
    """
    return sigmoid(np.dot(X, theta))

def cost_function(X, y, theta):
    """
    Computes the cost for given X and y with current theta.
    """
    m = len(y)
    h = hypothesis(X, theta)
    return -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def batch_gradient_descent(X, y, theta, learning_rate, iterations,cost_ret = False):
    """
    Implementation of batch gradient descent. It takes the whole dataset and maked perediction
    and finally adjusts the parameters through gradient descent. When cost_ret parameter is set as True, 
    it also returns the cost history over the iterations.
    """
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        h = hypothesis(X, theta)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient
        if cost_ret:
            cost = cost_function(X, y, theta)
            cost_history.append(cost)
    return theta, cost_history


def stochastic_gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            hi = hypothesis(xi, theta)
            gradient = np.dot(xi.T, (hi - yi))
            theta -= learning_rate * gradient
    return theta

def mini_batch_gradient_descent(X, y, theta, learning_rate, iterations, batch_size=5):
    m = len(y)
    for _ in range(iterations):
        idx = np.random.permutation(m)
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            hi = hypothesis(xi, theta)
            gradient = np.dot(xi.T, (hi - yi)) / batch_size
            theta -= learning_rate * gradient
    return theta


def predict(X, theta):
    return hypothesis(X, theta) >= 0.5

def accuracy(X, y, theta):
    pred = predict(X, theta)
    return np.mean(pred == y) * 100




train_data = genfromtxt('./project3_train.csv', delimiter=',')
test_data = np.loadtxt('./project3_test.csv', delimiter=',')
# print(train_data)
X_train = train_data[:,:-1]
# Add a column of ones to X to account for the bias term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
y_train = train_data[:,-1] # Binary target values

X_test = test_data[:,:-1]
# Add a column of ones to X to account for the bias term
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
y_test = test_data[:,-1] # Binary target value

# Initial theta
initial_theta = np.random.rand(X_train.shape[1])


#Generating Results for task Q1
learning_rates = [0.001, 0.002, 0.006, 0.01, 0.1]
# Number of iterations for Q1
iterations_a = 5000
plt.figure(figsize=(12, 8))

for lr in learning_rates:
    theta, cost_history = batch_gradient_descent(X_train, y_train, initial_theta, lr, iterations_a,cost_ret=True)
    plt.plot(range(iterations_a), cost_history, label=f'LR={lr}')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Number of Iterations')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('logistic_regression_costs.png')


#Solving Q2
learning_rate = 0.1
iterations = 300000
# Stochastic Gradient Descent
start_time = time.time()
theta_sgd = stochastic_gradient_descent(X_train, y_train, initial_theta, learning_rate, iterations)
time_sgd = time.time() - start_time
accuracy_sgd_train = accuracy(X_train, y_train, theta_sgd)
accuracy_sgd_test = accuracy(X_test, y_test, theta_sgd)

# Mini-Batch Gradient Descent
start_time = time.time()
theta_mbgd = mini_batch_gradient_descent(X_train, y_train, initial_theta, learning_rate, iterations)
time_mbgd = time.time() - start_time
accuracy_mbgd_train = accuracy(X_train, y_train, theta_mbgd)
accuracy_mbgd_test = accuracy(X_test, y_test, theta_mbgd)

# Batch Gradient Descent
start_time = time.time()
theta_bgd,_ = batch_gradient_descent(X_train, y_train, initial_theta, learning_rate, iterations)
time_bgd = time.time() - start_time
accuracy_bgd_train = accuracy(X_train, y_train, theta_bgd)
accuracy_bgd_test = accuracy(X_test, y_test, theta_bgd)


# Define a function to print results in a table format
def print_table(headers, data):
    # Finding the maximum width for each column
    col_width = max(len(word) for row in data for word in row) + 2  # padding
    print("".join(word.ljust(col_width) for word in headers))
    for row in data:
        print("".join(word.ljust(col_width) for word in row))

# Headers for the table
headers = ["Method", "Time (sec)", "Training Acc (%)", "Test Acc (%)"]

# Data for the table
data = [
    ["Stochastic GD", f"{time_sgd:.2f}", f"{accuracy_sgd_train:.2f}", f"{accuracy_sgd_test:.2f}"],
    ["Mini-Batch GD", f"{time_mbgd:.2f}", f"{accuracy_mbgd_train:.2f}", f"{accuracy_mbgd_test:.2f}"],
    ["Batch GD", f"{time_bgd:.2f}", f"{accuracy_bgd_train:.2f}", f"{accuracy_bgd_test:.2f}"]
]

# Print the table
print_table(headers, data)
