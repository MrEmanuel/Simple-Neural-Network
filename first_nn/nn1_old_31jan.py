import numpy as np
import matplotlib
import matplotlib.pyplot as plt




def generate_set1_data(input_dim, p, noise_rate):
    # Generate random points between -1 and 1
    inputs_set1 = np.random.rand(p, input_dim)*2-1

    n = (np.array([0.5, 0.5]) , np.array([0.5, 0.5]))
    c = (0.2, -0.2)


    labels_set1=np.ones(p)*(-1)

    for index, state in enumerate(inputs_set1):
        if( np.dot(n[0], state) + c[0] > 0 and np.dot(n[0], state) + c[1] < 0 or np.random.rand() < noise_rate):
            labels_set1[index] = 1

    return inputs_set1, labels_set1

def generate_set2_data(input_dim, p, noise_rate):
    inputs_set2 = np.random.rand(p,input_dim)*2-1
    labels_set2 = np.ones(p)*(-1)

    for index, state in enumerate(inputs_set2):
        if state[0] * state[1] > 0 or np.random.rand() < noise_rate:
            labels_set2[index] = 1
    return inputs_set2, labels_set2


def create_weights(layers_dims, mean, std):
    # Layers_dims is all system layers, including input and output layers.
    weights = []
    for index in range(len(layers_dims)-1):
        left_layer = layers_dims[index]
        right_layer = layers_dims[index + 1]
        layer_weights = []

        for _ in range(right_layer):
            # number of loops equal number of nodes in layer
            # each node has a many weights as previous layer has nodes (fully connected layers)
            layer_weights.append(np.random.normal(mean, std, left_layer))

        weights.append(layer_weights)
    return np.array(weights)

def create_biases(layers_dims):
    system_biases = []
    for _, layer_size in enumerate(layers_dims):
        system_biases.append(np.zeros(layer_size))

    return np.array(system_biases)



def g(x): 
    return np.tanh(x)

def g_prim(x):
    return np.cosh(x)**(-2)



def NN(inputs, labels, weights, biases, layers_dims):
    # propagates inputs through network of neuron layers with specified weights
    # updates the weights according to labeled data
    # # neuron_state = g( [x1, x2] * [w1, w2] - bias )

    p = len(inputs)
    hidden_layers_dims = layers_dims[:][1:-1]
    output_dim = layers_dims[-1]
    # Array where all layers state for each input data's propagation is saved.
    neuron_states = []
    


    for index, input_state in enumerate(inputs):
        # First input for each input is the input. This updates on each layer.
        neuron_inputs = input_state
        neuron_layers_row = []
        
        # Loop through each layer
        for layer, layer_size in enumerate([*hidden_layers_dims, output_dim]):

            neuron_layer_states = []

            # Calculate the state for each neuron in the layer
            for neuron_index in range(layer_size):
                #Calculate one neuron state and append it to a list of neuron states for that layer
                neuron_layer_states.append(np.matmul(neuron_inputs, weights[layer][neuron_index]) - biases[layer][neuron_index])
            
            # set the input for the next layer to be the output of the previous layer
            neuron_inputs = g(neuron_layer_states)
            # Add the row of neuron states for the layer to a list
            neuron_layers_row.append(np.array(neuron_layer_states))
            
        # Add all neuron states in all layers to a list
        neuron_states.append(np.array(neuron_layers_row))

    # Save all out-values in separate list
    Out = [row[-1][0] for row in neuron_states]
    predicted_labels = np.sign(Out)
    
    # Enegy function
    H = np.sum((labels - Out)**2)/(2*p)
    errors = p
    
    for index, label in enumerate(labels):
        if label == predicted_labels[index]:
            errors -=1
    accuracy = 1 - errors/p

    # calculate the first error delta^{(L)}
    output_state = [row[-1][0] for row in neuron_states]


    delta_out = g_prim(np.array(output_state)) * (labels - Out)

    # Update the output node's bias
    biases[-1] = biases[-1] - eta * np.sum(delta_out) / p



    delta_hidden_layers = []
    old_delta = delta_out
    for layer, _ in enumerate(hidden_layers_dims, 2):
        layer = layer*(-1)
        new_delta = []
        for index in range(p):        
            new_delta.append( old_delta[index] * np.matmul(weights[layer+1][0], g_prim(neuron_states[index][layer])))
        
        # Insert to beginning of list to keep weight ordered from left to right of the neural network
        delta_hidden_layers.insert(0, new_delta)
        old_delta = new_delta

    #update weights

    for index, delta in enumerate(delta_hidden_layers):
        biases[index] = biases[index] - eta * np.sum(delta) / p

        for idx, weight in enumerate(weights[index]):
            weights[index][idx] = weight + eta * np.matmul(delta, [row[index][idx] for row in neuron_states]) / p

    return predicted_labels, weights, biases, H, accuracy


#==================================================================

## Build the network. Change these parameters if you want
p = 500
mean = 0
std = 0.5
epochs = 1000
noise_rate = 0
eta = 0.5
input_dim = 2
output_dim = 1
hidden_layers_dims = [3,3]
layers_dims = [input_dim, *hidden_layers_dims, output_dim]  # Input, hidden and output layer dimensions



# neural network epoch function starts here


inputs, labels = generate_set2_data(input_dim , p, noise_rate)
weights = create_weights(layers_dims, mean, std)
biases = create_biases([*hidden_layers_dims, output_dim])
energy = np.zeros(epochs)
accuracy = np.zeros(epochs)
plot_data = 1


# Empty variables to store best values
best_acc = 0
weights = create_weights(layers_dims, mean, std)
biases = create_biases([*hidden_layers_dims, output_dim])

for eon in range(10):
    print(f"Eon no: {eon}")

    for epoch in range(epochs):
        if epoch%100==0:
            print(f"Epoch no: {epoch}")
        predictions, weights, biases, H, acc = NN(inputs, labels, weights, biases, layers_dims)
        energy[epoch] = H
        accuracy[epoch] = acc
        

        if acc > best_acc:
            best_acc = acc
            best_biases = biases
            best_weights = weights
            best_predictions = predictions

if plot_data:
    for index, state in enumerate(inputs):
        if labels[index] == 1:
            plt.plot(state[0], state[1], 'r.')
        else:
            plt.plot(state[0], state[1], 'b.')
        if predictions[index] == 1:
            plt.plot(state[0], state[1], 'ko', mfc='none')
        x1_span = np.linspace(-1,1,100)
        #x2_span = 
    plt.show()

i = list(range(epochs))

#plt.plot(i, energy, 'b.')
#plt.subplot(2,1,1)

#plt.plot(i, accuracy, 'r.')
#plt.subplot(2,1,2)
#plt.show()

plt.figure()
plt.plot(i, accuracy, 'b.')
plt.subplot(1,1,1)

plt.plot(i, energy, 'r.')
#plt.subplot(1,1,2)
plt.show()

if plot_data:
    for index, state in enumerate(inputs):
        if labels[index] == 1:
            plt.plot(state[0], state[1], 'r.')
        else:
            plt.plot(state[0], state[1], 'b.')
        if best_predictions[index] == 1:
            plt.plot(state[0], state[1], 'ko', mfc='none')
        x1_span = np.linspace(-1,1,100)
        #x2_span = 
    plt.show()