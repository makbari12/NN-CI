from scr.functions import *
import matplotlib.pyplot as plt

# load data
xi_train, tau_train, xi_test, tau_test = load_data(n_train=100)

# perform SGD
w, errors = sgd_training(xi_train=xi_train, tau_train=tau_train, n_epochs=500)

plt.plot(errors)
plt.title("Training error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend(["Training error", "Test error"])
plt.show()

plt.bar(np.arange(w[:,0].size), w[:,0])
plt.title("Components of w1 vector")
plt.xlabel("Component")
plt.ylabel("Value")
plt.show()

plt.bar(np.arange(w[:,1].size), w[:,1])
plt.title("Components of w2 vector")
plt.xlabel("Component")
plt.ylabel("Value")
plt.show()

sigma_test = calculate_output(k=np.shape(w)[1], xi=xi_test, w=w)
plt.plot(tau_test, sigma_test, 'bo')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Test set prediction')
plt.show()
mse = np.mean((sigma_test - tau_test) ** 2)
print("Mean squared error: ", mse)



