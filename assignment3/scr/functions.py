import pandas as pd
import numpy as np
import math
import random

# function to load the data
#TODO: add n_test??
def load_data(n_train=5000):
    # n_data gives the number of instances to use
    xi = pd.read_csv("./data/xi.csv")
    tau = pd.read_csv("./data/tau.csv")
    xi = xi.to_numpy()
    tau = np.array(list(tau))
    xi_train = xi[:,0:n_train]
    tau_train = tau[0:n_train]
    xi_test = xi[:,n_train:]
    tau_test = tau[n_train:]

    return xi_train, tau_train, xi_test, tau_test


###################### Function for SGD ############################

# initialize weight vectors with L2-norm of 1
def initialize_weights(k, N):
    w = np.zeros([N, k])
    for i in range(k):
        w_k = np.random.randint(0, 100, N)
        w_unit = w_k/np.linalg.norm(w_k,2)
        w[:,i] = w_unit
    return w  

def calculate_output(k, xi, w):
    sigma = 0
    for i in range(k):
        sigma += math.tanh(np.dot(w[:,i], xi))    # no v_k because v_k is 1
    return sigma

def calculate_error(sigma, tau):
    return ((sigma - tau)**2)/2

def calculate_learning_step(xi_train, tau_train, w, learning_rate):
    # select random example
    idx = random.randint(0, len(xi_train))
    xi = xi_train[:, idx]
    tau = np.asarray(tau_train[idx], dtype=float)    
    
    # calculate sigma
    sigma = calculate_output(k=np.shape(w)[1], xi=xi, w=w)
        
    # calculate error
    error = calculate_error(sigma, tau)
    
    print("sigma: ", sigma, " tau: ", tau, " error: ", error)
    
    # calculate gradient
    # update weights
    
    return w

def sgd_training(xi_train, tau_train, n_epochs=50, learning_rate=0.05, k=2):
    # initialize weights
    w = initialize_weights(k, len(xi_train))
    
    for i in range(n_epochs):
        w = calculate_learning_step(xi_train=xi_train, tau_train=tau_train, w=w, learning_rate=learning_rate)
        
    return