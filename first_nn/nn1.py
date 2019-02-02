import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def generate_set0_data(input_dim, p, noise_rate):
    # Generate an XOR data set.
    inputs_set0 = [[0,1], [0,0], [1,0], [1,1]]
    labels_set0 = np.ones(len(inputs_set0))

    for index, inputs in enumerate(inputs_set0):
        if inputs[0] == 0:
            labels_set0[index] = -1

    return inputs_set0, labels_set0

def generate_set1_data(input_dim, p, noise_rate):
    # GEnerate a set of points and assign them to red category if they are within 0.2 units from a -45 degree line in the middle of the set
    inputs_set1 = np.random.rand(p, input_dim)*2-1

    n = (np.array([0.5, 0.5]) , np.array([0.5, 0.5]))
    c = (0.2, -0.2)


    labels_set1=np.ones(p)*(-1)

    for index, state in enumerate(inputs_set1):
        if( np.dot(n[0], state) + c[0] > 0 and np.dot(n[0], state) + c[1] < 0 or np.random.rand() < noise_rate):
            labels_set1[index] = 1

    return inputs_set1, labels_set1

def generate_set2_data(input_dim, p, noise_rate):
    # Data set split into four quadrants where on category belong to quadrant 1 and 3 and the other to 2 and 4.
    inputs_set2 = np.random.rand(p,input_dim)*2-1
    labels_set2 = np.ones(p)*(-1)

    for index, state in enumerate(inputs_set2):
        if state[0] * state[1] > 0 or np.random.rand() < noise_rate:
            labels_set2[index] = 1
    return inputs_set2, labels_set2

def generate_set3_data(input_dim, p, noise_rate):
    inputs_set3 = np.random.rand(p, input_dim)*2-1
    labels_set3 = np.ones(p)*(-1)

    for index, state in enumerate(inputs_set3):
        # if the point is within a circle at the origin with radius r, assign it label 1
        r = 0.8
        if np.sqrt(state[0]**2 + state[1]**2 ) < r or np.random.rand() < noise_rate:
            labels_set3[index] = 1
    return inputs_set3, labels_set3


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
    correct_predictions = 0
    
    for index, label in enumerate(labels):
        if label == predicted_labels[index]:
            correct_predictions +=1
    
    accuracy = correct_predictions/p
   


    # calculate the first error delta^{(L)}
    output_state = [row[-1][0] for row in neuron_states]


    delta_out = g_prim(np.array(output_state)) * (labels - Out)
    

    # Update the output node's bias
    biases[-1] = biases[-1] - eta * np.sum(delta_out) / p



    delta_hidden_layers = []
    old_delta = delta_out
    # Calculate delta for each layer
    input(f"Delta0 is {delta_out.shape}")
    for layer, _ in enumerate(hidden_layers_dims, 2):
        layer = layer*(-1) # Negative index to iterate backwards through the layers
        new_delta = []
        print(f"Our layers are {hidden_layers_dims}, Active layer: {layer}")
        print(F"Weights shape: {weights}")
        print(f"Neuron states: {neuron_states[0]}")
        print(f"Active neurons: {neuron_states[0][layer]}")
        print(f"Active weights: {weights[layer+1][0]}")
        print(f"Active old_delta: {old_delta[0]}")
        input(f"...")
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
batching = 0
epochs = 1
eons = 1
noise_rate = 0
eta = 0.5
input_dim = 2
output_dim = 1
axis = [-1, 1]
hidden_layers_dims = [2, 3, 2]
layers_dims = [input_dim, *hidden_layers_dims, output_dim]  # Input, hidden and output layer dimensions



# neural network epoch function starts here


inputs, labels = generate_set1_data(input_dim , p, noise_rate)
weights = create_weights(layers_dims, mean, std)
biases = create_biases([*hidden_layers_dims, output_dim])
energy = np.zeros(epochs)
accuracy = np.zeros(epochs)
best_acc = 0
lowest_energy = 9999
plot_data = 1
plot_best_data = 0
best_epochs = np.zeros(eons)
best_epochs_acc = np.zeros(eons)
starting_weights = [2]

for eon in range(eons):
    print(f"Eon no: {eon}")
    # Set new weights and biases for each eon.
    weights = create_weights(layers_dims, mean, std)
    biases = create_biases([*hidden_layers_dims, output_dim])
    best_acc_epoch = 0
    lowest_energy_epoch = 0
    starting_weights.append(weights)
    for epoch in range(epochs):
        if epoch%100==0:
            print(f"Epoch no: {epoch}")
        
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
            
            #if(epoch%100==0):
                #print(f"Running {len(batches)} batches of length {len(batches[-1])}")
            
            for batch in batches:
                batch_inputs = np.array([row[0] for row in batch])
                batch_labels = np.array([row[1] for row in batch])
                predictions_, weights, biases, H_, acc_ = NN(batch_inputs, batch_labels, weights, biases, layers_dims)
    
        predictions, weights, biases, H, acc = NN(inputs, labels, weights, biases, layers_dims)
        energy[epoch] = H
        accuracy[epoch] = acc
        
        if acc > best_acc_epoch:
        #if H < lowest_energy_epoch:
            lowest_energy_epoch = H
            best_acc_epoch = acc
            best_epochs[eon] = epoch
            best_epochs_acc[eon] = acc

        if acc > best_acc:
        #if H < lowest_energy:
            lowest_energy = H
            best_acc = acc
            best_biases = biases
            best_weights = weights
            best_predictions = predictions
            best_eon = eon
            best_epoch = epoch
            print(f"Best epoch: {best_epoch}")
            print(f"Best eon: {best_eon}")
            print(f"Accuracy {acc} | Energy {H}")

print(f"Best epochs: {best_epochs}")
print(f"Best ep acc: {best_epochs_acc}")
print(f"Best accuracy of {best_acc} found in eon {best_eon} in epoch {best_epoch}")

for index in range(eons):
    print(f"Eon {index} | Epoch {best_epochs[index]} | Accuracy {best_epochs_acc[index]}")  #| Weights {[w[:] for w in starting_weights[index]]}")
if plot_data:
    for index, state in enumerate(inputs):
        if labels[index] == 1:
            plt.plot(state[0], state[1], 'r.')
        else:
            plt.plot(state[0], state[1], 'b.')
        if predictions[index] == 1:
            plt.plot(state[0], state[1], 'ko', mfc='none')
        x1_span = np.linspace(axis[0], axis[1],100)
    for idx, w in enumerate(weights[0]):
        x2_span = (-w[0] * x1_span + biases[0][idx])/w[1]
        plt.plot(x1_span, x2_span, "k")
        print(f"Slope gradient {idx}: {-w[0]/w[1]}")
        print(f"Crossing y-axis at {biases[0][idx]/w[1]}")
    #plot y-axis
    plt.plot([0]*100, x1_span, 'k--')
    plt.axis([*axis, *axis])
    plt.show()
i = list(range(epochs))

#plt.plot(i, energy, 'b.')
#plt.subplot(2,1,1)

#plt.plot(i, accuracy, 'r.')
#plt.subplot(2,1,2)
#plt.show()

plt.figure()
plt.plot(i, accuracy, 'b-')
plt.subplot(1,1,1)

plt.plot(i, energy, 'r-')
#plt.subplot(1,1,2)
plt.show()

print(f"weights: {weights}")
print(f"biases: {biases}")

#Plot the data for best predicted labels
if plot_best_data:
    for index, state in enumerate(inputs):
        if labels[index] == 1:
            plt.plot(state[0], state[1], 'r.')
        else:
            plt.plot(state[0], state[1], 'b.')

        if best_predictions[index] == 1:
            plt.plot(state[0], state[1], 'ko', mfc='none')
        x1_span = np.linspace(axis[0],axis[1],100)
    for idx, w in enumerate(best_weights[0]):
        x2_span = (-w[0] * x1_span + best_biases[0][idx])/w[1]
        plt.plot(x1_span, x2_span, "k")
        print(f"Slope gradient {idx}: {-w[0]/w[1]} with bias {best_biases[0][idx]}")
        print(f"Crossing y-axis at {best_biases[0][idx]/w[1]}")
    plt.axis([*axis, *axis])
    plt.show()
