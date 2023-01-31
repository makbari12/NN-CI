import pandas as pd
import numpy as np
import math
import random

# function to load the data
#TODO: add n_test??
def load_data(n_train=100, n_test=100):
    # n_data gives the number of instances to use
    xi = pd.read_csv("./data/xi.csv")
    tau = pd.read_csv("./data/tau.csv", header=None, index_col=False)
    xi = xi.to_numpy()
    tau = tau.to_numpy()
    xi_train = xi[:,0:n_train]
    tau_train = tau[:,0:n_train]
    xi_test = xi[:,n_train:n_train+n_test]
    tau_test = tau[:,n_train:n_train+n_test]
    return xi_train, tau_train, xi_test, tau_test


###################### Function for SGD ############################

# initialize weight vectors with L2-norm of 1
def initialize_weights(k, N, seed=0):
    w = np.zeros([N, k])
    for i in range(k):
        np.random.seed(i+seed)
        w_k = np.random.randint(0, 100, N)
        w_unit = w_k/np.linalg.norm(w_k,2)
        w[:,i] = w_unit
    return w  

def calculate_output(k, xi, w):
    sigma = 0
    for i in range(k):
        sigma += np.tanh(np.dot(w[:,i], xi))    # no v_k because v_k is 1
    return sigma

def calculate_error(sigma, tau):
    return np.mean(1/2 * (sigma - tau) ** 2)

def calculate_test_error(k, xi, tau, w):
    sigma = []
    for i in range(np.shape(xi)[1]):
        output = calculate_output(k, xi[:,i], w)
        sigma.append(output)
    return calculate_error(sigma, tau)

def calculate_gradient(xi,sigma, tau, w):
    gradient = (sigma - tau) * (1 - np.tanh(w.T @ xi.reshape(-1, 1))**2)
    return gradient

def calculate_learning_step(xi_train, tau_train, w, learning_rate):
    # select random example
    idx = random.randint(0, len(xi_train))
    xi = xi_train[:, idx]
    tau = tau_train[0,idx]   
    
    # calculate sigma
    sigma = calculate_output(k=np.shape(w)[1], xi=xi, w=w)
        
    # calculate error
    error = calculate_error(sigma, tau)

    #print("sigma: ", sigma, " tau: ", tau, " error: ", error)

    # calculate gradient
    partial_gradients = calculate_gradient(xi, sigma, tau, w)
    # update weights 
    for i in range(len(partial_gradients)):
        g = partial_gradients[i] * xi
        w[:,i] -= learning_rate * g * error    # *error?
    return w

def sgd_training(xi_train, tau_train, xi_test, tau_test, n_epochs=500, learning_rate=0.05, seed=0, k=2):
    # initialize weights
    w = initialize_weights(k, len(xi_train), seed)
    errors = []
    test_errors = []
    for i in range(n_epochs):
        w = calculate_learning_step(xi_train=xi_train, tau_train=tau_train, w=w, learning_rate=learning_rate)
        # calculate errors each P steps
        if i % np.shape(tau_train)[1] == 0:
            sigma_train = calculate_output(k=2, xi=xi_train, w=w)
            error_train = calculate_error(sigma_train, tau_train)
            errors.append(error_train)
            sigma_test = calculate_output(k=2, xi=xi_test, w=w)
            error = calculate_error(sigma_test, tau_test)
            test_errors.append(error)
    return w, errors, test_errors