from scr.functions import *

# load data
xi_train, tau_train, xi_test, tau_test = load_data(n_train=100)

# perform SGD
sgd_training(xi_train=xi_train, tau_train=tau_train, n_epochs=1)