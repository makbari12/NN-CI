from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('result.pkl', 'rb') as f:
    result: List[Union[float, int, float]] = pickle.load(f)


if __name__ == '__main__':
    # plot the success rate as a function of alpha and p
    alpha = np.unique(result[:, 0])

    for n in np.unique(result[:, 2]):
        plt.plot(alpha, result[result[:, 2] == n, 3],  label=f'n={n}')

    # set the x labels of the plot

    plt.xticks(np.arange(0.75, 3.01, 0.20))
    plt.xlabel('Alpha level')
    plt.ylabel('Success rate')
    plt.legend()
    plt.savefig("success_rate.png")
