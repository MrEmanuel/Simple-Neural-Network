import numpy as np
from nn1 import generate_set_data

p = 500
mean = 0
std = 0.1
noise_rate = 0.1
learning_rate = 0.5

inputs, labels = generate_set_data(2, p, noise_rate)

# Generate the weights

w1 = np.random.normal(mean, std, (2,2))
w2 = np.random.normal(mean, std, (2))
b1 = 0
b2 = 0


def g(a):
    return np.tanh(a)

def g_prim(x):
    return np.cosh(x)**(-2)

# Forward propagation to calculate node states
        

def NN2(inputs, labels, w1, w2, b1, b2, learning_rate):
        
        p = len(inputs)
        a1 = []
        a2 = []

        for index, mu in enumerate(inputs):
                a1.append(np.matmul(w1, mu) - b1)
                h1_out = g(a1)

        for index, mu in enumerate(h1_out):
                a2.append(np.matmul(w2, mu) - b2)
                h2_out = g(a2)

        predictions = np.sign(h2_out)


        # Precision
        precision = np.sum(np.equal(predictions, labels)) / p

        # Energy funcion
        H = np.sum((labels - h2_out)**2)/(2*p)

        delta2 = g_prim(a2) * (labels - h2_out)

        delta1 = []

        for index, d1 in enumerate(delta2):
                delta1.append(np.matmul((d1 * w1), g_prim(a1[index])))

        w2 = w2 + learning_rate * np.matmul(delta2, inputs) / p
        b2 = b2 - learning_rate * np.sum(delta2) / p


        delta_h = []
        for index, delta in enumerate(delta1):
                        delta_h.append(np.matmul(delta, h1_out[index]))


        w1 = w1 + learning_rate * np.sum(delta_h) /p
        b1 = b1 - learning_rate * np.sum(delta1) / p

        
        return w1, w2, b1, b2, precision, H, predictions


for _ in range(500):

        w1, w2, b1, b2, precision, H, predictions = NN2(inputs, labels, w1, w2, b1, b2, learning_rate)
        if _% 100 == 0:
                print(precision, H)

        
