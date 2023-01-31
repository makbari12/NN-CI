from scr.functions import *
import matplotlib.pyplot as plt

# # load data
# xi_train, tau_train, xi_test, tau_test = load_data(n_train=800, n_test=200)

# # perform SGD
# errors = []
# test_errors = []
# print("----------- train and test network -------------")
# for i in range(10):
#     print("---------- run ", i, "---------------")
#     w, err, test_err = sgd_training(xi_train=xi_train, tau_train=tau_train, xi_test=xi_test,tau_test=tau_test, n_epochs=10000)
#     errors.append(err)
#     test_errors.append(test_err)

# mean_train_error = np.mean(errors, 0)
# mean_test_error = np.mean(test_errors,0)


# print('----------------- plot -----------------')
# plt.plot(mean_train_error, label='Training error')
# plt.plot(mean_test_error, label='Test error')
# plt.legend()
# plt.title("Training error")
# plt.xlabel("Epochs")
# plt.ylabel("Error") 
# plt.show()

print('-------------- Test different Ps -------------------')
P = [20, 50, 200, 500, 1000, 2000]
train_errors = []
test_errors=[]
for n_train in P:
    print('---------------- P = ', n_train, ' ------------------')
    xi_train, tau_train, xi_test, tau_test = load_data(n_train=n_train, n_test=200)
    w, err, test_err = sgd_training(xi_train=xi_train, tau_train=tau_train, xi_test=xi_test,
                                    tau_test=tau_test, n_epochs=10000)
    train_errors.append(err)
    test_errors.append(test_err)
    
plt.plot(train_errors)
plt.legend(['P=20', 'P=50', 'P=50'])
    

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



