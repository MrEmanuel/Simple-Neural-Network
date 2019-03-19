import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from nn1 import generate_set_data, generate_set1_data, generate_set2_data, generate_set3_data

p = 500
mean = 0
std = 0.1
noise_rate = 0.1
learning_rate = 0.5
batching = 0
plot_data = 1


#inputs, labels = generate_set_data(2, p, noise_rate)
inputs, labels = generate_set1_data(2, p, noise_rate)
#inputs, labels = generate_set2_data(2, p, noise_rate)
#inputs, labels = generate_set3_data(2, p, noise_rate)

# Generate the weights

w1 = np.random.normal(mean, std, (2,2))
w2 = np.random.normal(mean, std, (2))




b1 = 0
b2 = 0
axis = [-1.2, 1.2]
epochs = 300
prec = []
energy = []


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

        for index, d2 in enumerate(delta2):
                delta1.append(np.matmul((d2 * w1), g_prim(a1[index])))
        
        w2 = w2 + (learning_rate * np.matmul(delta2, inputs) / p)
        b2 = b2 - learning_rate * np.sum(delta2) / p


        delta_h = []
        for index, delta in enumerate(delta1):
                        delta_h.append(np.matmul(delta, h1_out[index]))


        w1 = w1 + learning_rate * np.sum(delta_h) /p
        b1 = b1 - learning_rate * np.sum(delta1) / p

        return w1, w2, b1, b2, precision, H, predictions


for epoch in range(epochs):

        if batching:
            # pair inputs and labels together to be able to scramble them
            inputs_labels = zip(inputs, labels)
            
            # Scramble the input and label pairs
            inputs_labels = np.random.permutation(list(inputs_labels)) 

            batches = []
            predictions = []
            # Create 50 batches with 10 entries in each
            for index in range(50):
                start = index*10
                end = start+10
                batches.append(np.array(inputs_labels[start:end]))
            
            
            for batch in batches:
                batch_inputs = np.array([row[0] for row in batch])
                batch_labels = np.array([row[1] for row in batch])
                w1, w2, b1, b2, precision_, H_, predictions_ = NN2(batch_inputs, batch_labels, w1, w2, b1, b2, learning_rate)

        w1, w2, b1, b2, precision, H, predictions = NN2(inputs, labels, w1, w2, b1, b2, learning_rate)

        prec.append(precision)
        energy.append(H)


        if epoch % 100 == 0:
                print(precision, H)

        


plt.figure(figsize=[10,10])
plt.subplot(2,2,1)
plt.title("Sample data")
plt.axis("scaled")

for index, state in enumerate(inputs):
        if labels[index] == 1:
                plt.plot(state[0], state[1], 'r.')
        else:
                plt.plot(state[0], state[1], 'b.')
        if predictions[index] == 1:
                plt.plot(state[0], state[1], 'ko', mfc='none')


x1_span = np.linspace(axis[0], axis[1],100)
for idx, w in enumerate(w1):
        x2_span = (-w[0] * x1_span + b1)/w[1]
        plt.plot(x1_span, x2_span, "k")
        print(f"Slope gradient {idx}: {-w[0]/w[1]}")
        print(f"Crossing y-axis at {b2/w[1]}")


#plot y-axis
plt.plot([0]*100, x1_span, 'k--')
plt.axis([*axis, *axis])


lin = np.linspace(0,1,len(list(energy)))
plt.subplot(2,2,3)
plt.title("Precision")
plt.plot(lin, prec, 'b-')
plt.axis("scaled")


plt.subplot(2,2,4)
plt.title("Energy")
plt.plot(lin, energy, 'r-')
plt.axis("scaled")
plt.show()
plt.plot()
