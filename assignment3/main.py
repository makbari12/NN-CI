from scr.functions import *
from scr.settings import *
import matplotlib.pyplot as plt

# for plotss
epoch = 0
x = np.linspace(0, N_EPOCHS, int(N_EPOCHS/100))

###################### General Test of the network ###########################

# load data
xi_train, tau_train, xi_test, tau_test = load_data(n_train=P, n_test=Q)

# perform SGD
errors = []
test_errors = []
print("----------- train and test network -------------")
for i in range(N_RUNS):
    print("---------- run ", i, "---------------")
    w, err, test_err = sgd_training(xi_train=xi_train, tau_train=tau_train, xi_test=xi_test,
                                    tau_test=tau_test, n_epochs=N_EPOCHS, learning_rate=LR, k=K)
    errors.append(err)
    test_errors.append(test_err)

mean_train_error = np.mean(errors, 0)
mean_test_error = np.mean(test_errors,0)

print("train error: ", mean_train_error[len(mean_train_error)-1], "test error: ", 
      mean_test_error[len(mean_test_error)-1])

np.savetxt('w.txt', w) # to load again use w = np.loadtxt('w.txt', delimiter=' ')

# plot
plt.rcParams['figure.figsize'] = (8, 4)
plt.plot(x, mean_train_error, label='Training error')
plt.plot(x, mean_test_error, label='Test error')
plt.legend()
plt.title("Training and test error")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig('traintest.png') 
plt.show()

# ############################ Test different P's ##############################

print('-------------- Test different Ps -------------------')
P_sizes = [20, 50, 200, 500, 1000, 2000]
train_errors = []
test_errors=[]
for n_train in P_sizes:
    print('---------------- P = ', n_train, ' ------------------')
    xi_train, tau_train, xi_test, tau_test = load_data(n_train=n_train, n_test=Q)
    w, err, test_err = sgd_training(xi_train=xi_train, tau_train=tau_train, xi_test=xi_test,
                                    tau_test=tau_test, n_epochs=N_EPOCHS, learning_rate=LR, k=K)
    if n_train is 20:
        train_errors = err
        test_errors = test_err   
    else:
        train_errors = np.vstack([train_errors, err])
        test_errors = np.vstack([test_errors, test_err])
    print("train error: ", err[len(err)-1], "test error: ", test_err[len(test_err)-1])

plt.rcParams['figure.figsize'] = (6, 4)
for i in range(np.shape(train_errors)[0]):    
    plt.plot(x, train_errors[i,:])
plt.legend(['P=20', 'P=50', 'P=200', 'P=500', 'P=1000', 'P=2000'])
plt.title("Training error for different P's")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig('train_differentp.png') 
plt.show()

for i in range(np.shape(test_errors)[0]):    
    plt.plot(x, test_errors[i,:])
plt.legend(['P=20', 'P=50', 'P=200', 'P=500', 'P=1000', 'P=2000'])
plt.title("Test error for different P's")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig('test_differentp.png') 
plt.show()

########################### Test different constants learning rates ###########################

print('-------------- Test different learning rates -------------------')
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
train_errors = []
test_errors=[]
for lr in learning_rates:
    print('---------------- lr = ', lr, ' ------------------')
    xi_train, tau_train, xi_test, tau_test = load_data(n_train=P, n_test=Q)
    w, err, test_err = sgd_training(xi_train=xi_train, tau_train=tau_train, xi_test=xi_test,
                                    tau_test=tau_test, n_epochs=N_EPOCHS, learning_rate=lr, k=K)
    if lr is learning_rates[0]:
        train_errors = err
        test_errors = test_err   
    else:
        train_errors = np.vstack([train_errors, err])
        test_errors = np.vstack([test_errors, test_err])
    print("train error: ", err[len(err)-1], "test error: ", test_err[len(test_err)-1])

for i in range(np.shape(train_errors)[0]):    
    plt.plot(x, train_errors[i,:])
plt.legend(['lr=0.001', 'lr=0.005', 'lr=0.01', 'lr=0.05', 'lr=0.1', 'lr=0.5'])
plt.title("Training error for different learning rates")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig('train_differentlr.png') 
plt.show()

for i in range(np.shape(test_errors)[0]):    
    plt.plot(x, test_errors[i,:])
plt.legend(['lr=0.001', 'lr=0.005', 'lr=0.01', 'lr=0.05', 'lr=0.1', 'lr=0.5'])
plt.title("Test error for different learning rates")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig('test_differentlr.png') 
plt.show()

# add learning_rate(t) = a/(b+t)?
########################### Test learningrate schedule ###########################
print('-------------- Test different Ps -------------------')
P_sizes = [20, 50, 200, 500, 1000, 2000]
train_errors = []
test_errors=[]
for n_train in P_sizes:
    print('---------------- P = ', n_train, ' ------------------')
    xi_train, tau_train, xi_test, tau_test = load_data(n_train=n_train, n_test=Q)
    w, err, test_err = sgd_training(xi_train=xi_train, tau_train=tau_train, xi_test=xi_test,
                                    tau_test=tau_test, n_epochs=N_EPOCHS, learning_rate=LR, k=K, lr_schedule=True)
    if n_train is 20:
        train_errors = err
        test_errors = test_err   
    else:
        train_errors = np.vstack([train_errors, err])
        test_errors = np.vstack([test_errors, test_err])
    print("train error: ", err[len(err)-1], "test error: ", test_err[len(test_err)-1])

plt.rcParams['figure.figsize'] = (6, 4)
for i in range(np.shape(train_errors)[0]):    
    plt.plot(x, train_errors[i,:])
plt.legend(['P=20', 'P=50', 'P=200', 'P=500', 'P=1000', 'P=2000'])
plt.title("Training error for different P's with learning rate schedule")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig('train_differentp_schedule.png') 
plt.show()

for i in range(np.shape(test_errors)[0]):    
    plt.plot(x, test_errors[i,:])
plt.legend(['P=20', 'P=50', 'P=200', 'P=500', 'P=1000', 'P=2000'])
plt.title("Test error for different P's with learning rate schedule")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig('test_differentp_schedule.png') 
plt.show()


##################### Plots weights ########################
# barplot of w1 and w2
w = np.loadtxt('w.txt', delimiter=' ')
plt.bar(np.arange(w[:,0].size), w[:,0])
plt.title("Components of w1 vector")
plt.xlabel("Component")
plt.ylabel("Value")
plt.savefig('w1.png')
plt.show()

plt.bar(np.arange(w[:,1].size), w[:,1])
plt.title("Components of w2 vector")
plt.xlabel("Component")
plt.ylabel("Value")
plt.savefig('w2.png')
plt.show()


##################### More plots ########################

# plt.bar(np.arange(w[:,0].size), w[:,0])
# plt.title("Components of w1 vector")
# plt.xlabel("Component")
# plt.ylabel("Value")
# plt.show()

# plt.bar(np.arange(w[:,1].size), w[:,1])
# plt.title("Components of w2 vector")
# plt.xlabel("Component")
# plt.ylabel("Value")
# plt.show()

# sigma_test = calculate_output(k=np.shape(w)[1], xi=xi_test, w=w)
# plt.plot(tau_test, sigma_test, 'bo')
# plt.xlabel('True values')
# plt.ylabel('Predicted values')
# plt.title('Test set prediction')
# plt.show()
# mse = np.mean((sigma_test - tau_test) ** 2)
# print("Mean squared error: ", mse)



