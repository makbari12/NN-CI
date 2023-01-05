from typing import List
import pickle
import numpy as np


def generate_data(p,n):
    # Generates P amount of N dimensional feature vectors sampled from a gaussian with -1,1 random labels
    xi = np.random.normal(0.0, 1.0, (p, n))
    labels = np.random.choice([-1, 1], p)

    return xi, labels

def rosenblad(p: int, n: int, max_epochs: int = 100) -> bool:
    # p = number of vectors
    # n = number of dimensions
    # xi = p array of feature vectors
    # labels = randomly sampled -1 , 1 labels
    # Note: D = (xi, labels)
    # epochs = amount of epocs
    # weights = n dimensional weight vector
    # e = output of rosenblad formula
    xi, labels = generate_data(p, n)

    weights = np.zeros(n)

    success_count = 0

    for _ in range(max_epochs):
        for j in range(p):
            e = np.dot(weights, xi[j]) * labels[j] #P: is het niet e = np.dot(weights, xi[j] * labels[j])? of is dat hetzelfde?

            if e <= 0:
                weights = weights + 1/n * (xi[j] * labels[j])
            else:
                success_count += 1

        if success_count == p:
            return True

        success_count = 0

    return False


def run(all_n, alpha, max_epoch=100, n_runs = 50):
    result = []

    for n in all_n:
        print("now doing n = ", n)
        # P is defined as a function of alpha * N
        all_p = np.round(n * alpha)                 

        for p, current_alpha in zip(all_p, alpha):
            successes = []

            for _ in range(n_runs):
                successes.append(rosenblad(int(p), n, max_epoch))

            # Since rosenblad returns a boolean we can calculate the success rate
            success_fraction = np.mean(np.array(successes))
            result.append((current_alpha, p, n, success_fraction))

    return np.array(result)


if __name__ == '__main__':

    all_n = [20, 40, 60, 80, 100]
    # Alpha (0.75, 1 ... 3) as defined in experiment
    alpha = np.arange(0.75, 3.01, 0.10)                 #P: je moet toch alpha berekenen -> alpha = P/N   

    result = run(all_n, alpha, n_runs=100)

    print(result)
    with open('result.pkl', 'wb') as f:
        pickle.dump(result, f)
