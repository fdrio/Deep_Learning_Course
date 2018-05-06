import numpy as np
import matplotlib.pyplot as plt

with open('data_classification_problem5.csv') as file:
    x0_coordinates = []  # list for the points in x
    x1_coordinates = []  # list for the points in y
    labels = []  # list for the labels

    # extracting data from the csv file
    for line in file.readlines():
        point = line.rstrip('\n').split(',')
        point = list(map(float, point))
        x0_coordinates.append(point[0])
        x1_coordinates.append(point[1])
        labels.append(point[2])
    #plt.plot(x_coordinates, y_coordinates,'ro')
    print(len(x0_coordinates))
# gradient descent for linear regression
# We want to create a line such that minimizes the error between the square distance of an arbitray point (x,y) and the the line equation y = Mx +b
# All vectors are going to be column vectors unless told otherwise

# theta will be numpy vector that contains all the parameters of our model
#                  theta1, theta2 of the theta vector
theta = np.matrix([[0.5, 0.5, 0.5]]).T
loss_vector = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log(x):
    return np.log(x)


def create_input_matrix(x, y):
    # can be generalized for the length of theta
    ones_vec = np.matrix(np.ones(100)).T
    b = np.hstack((ones_vec, x))
    # horizontally stack the vectors to form the input matrix X
    X = np.hstack((b, y))
    return X  # where X is the input Matrix in our model


# y_prediction is a scalar value
# y_target is a vector that contains all
def Loss(y, h):
    loss = np.matmul(-y.T, log(h))
    delta = np.matrix(np.ones_like(y)) - y
    delta = delta.T
    delta = np.matmul(delta, log(np.matrix(np.ones_like(y)) - h))
    loss = (loss - delta) / (1.0 * len(x0_coordinates))
    return loss

# y is our target vector, X is the input matrix, theta the parameter vector
# lr is the learning rate for gradient descent and err is our accepted error


def gradient(X, theta, y):
    grad = sigmoid(np.matmul(X, theta) - labels)
    grad = np.matmul(X.T, grad)
    grad = grad / (1.0 * len(x0_coordinates))
    print(grad.shape)
    return grad


# keep track of the dimensions of y, x and theta
def gradient_descent(y_target, X, theta, lr, err):
    y_pred = np.matmul(X, theta)

    # while the error is big continue doing gradient descent
    # while(MSE_Loss(y_target, y_pred) >= err):
    for i in range(50):
        loss_vector.append(MSE_Loss(y_target, y_pred))
        theta = theta - (lr * gradient(y_target, X, theta))
        y_pred = np.matmul(X, theta)
        print ("Y predicted \n" + str(y_pred))
        print('\n')
        print ("Y target \n" + str(y_target))

    return theta  # returns the optimal theta vector




X = create_input_matrix(np.matrix(x0_coordinates).T,np.matrix(x1_coordinates).T)
labels = np.matrix(labels).T
# print(X.shape)
# print(theta)
#print (log(sigmoid(np.matmul(X,theta)))-labels)
# print(np.matmul(X.T,log(sigmoid(np.matmul(X,theta)))-labels))
# print(gradient(labels,X,theta))

# print(Loss(labels.T,sigmoid(np.matmul(X,theta))))
print(gradient(X, theta, labels.T))
# Debugging
# print(create_input_matrix(np.matrix([x_coordinates]).T))
# print(MSE_Loss(np.matrix([x_coordinates]).T,np.matrix([y_coordinates]).T))
# print('\n')
#X = create_input_matrix(np.matrix([x_coordinates]).T)
#print("X :\n" +str(X))
#y_target = np.matrix([y_coordinates]).T
# print('\n')
#print("Y target \n" + str (y_target))
# print('\n')

print("Labels \n" + str(labels.shape))
print('\n')

#print(gradient(y_target, X,theta))

# So far the code seems to work
# We could adjust the parameters in a single go, however we are using gradient descent explicitly

# Testing code for gradient descent
lr = 0.001  # the learning rate that worked for me
err = 0.5
#parameters,y_pred = gradient_descent(labels,X,theta, lr , err)
num = [i for i in range(10)]
