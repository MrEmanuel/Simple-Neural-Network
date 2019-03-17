import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def generate_set_data(input_dim, p, noise_rate):
    # Generate states as 2 random numbers between -1 and 1
    n = np.array([ 1 , 1])
    c = 0 
    inputs_set = np.random.rand(p,input_dim)*2-1

    # Assign labels
    labels_set=np.ones(p)*(-1)

    for mu, state in enumerate(inputs_set):
        # if the state obeys the condition, set label to 1, or with a 10% probability to -1 
        if np.dot(n,state) + c > 0  or np.random.rand() < noise_rate :
            labels_set[mu] = 1
    return inputs_set, labels_set


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
    labels_set1=np.ones(p)*(-1)
    
    n = (np.array([0.5, 0.5]) , np.array([0.5, 0.5]))
    c = (0.2, -0.2)

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
    layer_weights = []
    # Create a list of weight matrixes. one entry per layer. 
    # The size of the matrix is nodes in layer n+1 x layer-n
    for index in range(len(layers_dims)-1):
        layer_weights.append(np.random.normal(mean, std, (layers_dims[index+1], layers_dims[index])))

    return layer_weights

def create_biases(layers_dims):
    system_biases = []
    for _, layer_size in enumerate(layers_dims):
        system_biases.append(np.zeros(layer_size))

    return np.array(system_biases)


def g(x): 
    return np.tanh(x)

def g_prim(x):
    return np.cosh(x)**(-2)

import copy
def NN(inputs, labels, weights, biases, training_rate, layers_dims):

    p = len(inputs)
    P=1
    ##############################################################################
    # Forward propagataion. 
    ##############################################################################
    # Propagate the inputs through the first layer's weights, 
    # calculate the state "a" for that layer's output neurons
    # use the calculated state "a" as inputs in the next layer. 
    # Repeat for each layer
    
    # Store the all layers' states "a" in a matrix A.
    # A = [ a0, a1, a2... an] where a is a numpy array
    # an = g( [x1, x2] x [w1, w2] - bias )

    

    A = []
    layer_input = inputs
    W = copy.copy(weights)
    
    layer_dims= [2,2,1]
    W = [np.array([[3, 3],[-2,-2]]), np.array([0, 0])]
    biases = [np.array([-0.25, -0.25]), np.array([0])]
    print("Hardcoded input data!")

    for layer, w in enumerate(W):
        
    
        # Array to hold the layer states for each layer input data
        an = np.zeros((p, layers_dims[layer+1]))
    

        for mu in range(p):
            print(w.shape)
            print(layer_input[0].shape)
            an[mu] = np.matmul(w, layer_input[mu]) - biases[layer]
        A.append(an)
        print(A[-1])
        layer_input = g(A[layer])
    print(1/0)
    

    

    # Set the last neuron's state to be the output.
    Out = g(an)

    # Calculate the predictions by taking the sign function of the outputs
    predictions = np.sign(g(A[-1])).reshape(labels.shape)



    # Compare predictions to labels to get the precision
    precision = np.sum((np.equal(predictions, labels))) / p
    
    ## Calculate loss (energy)
    H = np.sum((labels - np.squeeze(Out))**2)/(2*p)

    
    ###########################################################################
    ## Back propagation
    ###########################################################################
    # Calculate delta for each neuron
    # Start with the first one using the formula
    # d = g_prim(a) * (labels- Out)
    # and add it to a list which we will append other deltas to.
    

    
    D = [np.expand_dims(    g_prim(np.squeeze(A[-1])) * (labels - g( np.squeeze(A[-1]) ) )   , 1)]

    # Iterate through the layers backwards, using the delta from 
    # the previous layer to calculate the current layer's deltas
    for layer, w in enumerate(reversed(W[1:]), 1): 
        print(w.shape)
        d = [] # List to store arrays of delta for each layer and input data.
        layer *= -1
        a = A[layer-1]
        

        for mu in range(p):
            
            delta = np.diag(D[layer][mu])
            
            wa = np.matmul(w, np.diag(g_prim(a[mu])))
            
            d.append(np.sum(np.matmul(delta, wa), axis = 0))
            
            if mu==-1:
                
                print("======================")
                print(f"Layer: {layer}")
                print(f"w: {w.shape}")
                print(f" a: {g_prim(a[mu]).shape}")
                print(f"wa: {wa.shape}")   
                print(f"Delta: {delta.shape}") 
                print(f"delta wa: {np.matmul(delta, wa).shape}")
                print(f"Delta for layer: {np.sum(np.matmul(delta, wa), axis = 0).shape}") 
                
        # Insert d to the first position of D to keep the order consistent with 
        D.insert(0,np.array(d))
        #print(D[-1][0].shape)
    print(1/0)
    ###########################################################################
    ## Update weights and biases
    ###########################################################################
    
    # Update the output node's bias
    biases[-1] = biases[-1] - (training_rate * np.sum(D[-1])) / p

    
    neuron_inputs = inputs
    for layer, w in enumerate(W):

        delta = np.sum(D[layer], axis = 0)
        neuron_inputs = np.sum(neuron_inputs, axis=0)

        for i in range(w.shape[0]):
            
            biases[layer][i] = biases[layer][i] - (training_rate/p) * delta[i]
            
            for j in range(w.shape[1]):
                
                W[layer][i,j] = w[i,j] + (training_rate/p) * delta[i] * neuron_inputs[j]

        neuron_inputs = g(A[layer])
    
    #print(f"Weights diff: {W[0] - weights_in}")
    return predictions, W, biases, H, precision

####################################################################################
"""
#==================================================================

## Build the network. Change these parameters if you want
p = 500
mean = 0
std = 0.5
batching = 1
epochs = 400
eons = 1
noise_rate = 0.1
training_rate = 0.5
input_dim = 2
output_dim = 1
axis = [-1.2, 1.2]
hidden_layers_dims = [3]
layers_dims = [input_dim, *hidden_layers_dims, output_dim]  # Input, hidden and output layer dimensions



# neural network epoch function starts here


inputs, labels = generate_set1_data(input_dim , p, noise_rate)
energy = np.zeros(epochs)
accuracy = np.zeros(epochs)
best_acc = 0
lowest_energy = 9999
plot_data = 1
plot_best_data = 0
plot_change_data = 0
best_epochs = np.zeros(eons)
best_epochs_acc = np.zeros(eons)
saved_weights = []
saved_biases = []


for eon in range(eons):
    if best_acc > 0.92:
        break

    print(f"Eon no: {eon}")
    # Set new weights and biases for each eon.
    weights = create_weights(layers_dims, mean, std) 
    biases = create_biases([*hidden_layers_dims, output_dim])
    


    best_acc_epoch = 0
    lowest_energy_epoch = 0
    
   
    for epoch in range(epochs):

        saved_weights.append(weights[0])
        saved_biases.append(biases[0])
        #print(saved_biases[-1][0]/saved_weights[-1][0][1])
        if best_acc_epoch > 0.95:
            break
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
                predictions_, weights, biases, H_, acc_ = NN(batch_inputs, batch_labels, weights, biases, training_rate, layers_dims)
        predictions, weights, biases, H, acc = NN(inputs, labels, weights, biases, training_rate, layers_dims)

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

        ## Save weights and biases to plot change of boudry layers
        




print(f"Best epochs: {best_epochs}")
print(f"Best ep acc: {best_epochs_acc}")
print(f"Best accuracy of {best_acc} found in eon {best_eon} in epoch {best_epoch}")

for index in range(eons):
    print(f"Eon {index} | Epoch {best_epochs[index]} | Accuracy {best_epochs_acc[index]}")  #| Weights {[w[:] for w in starting_weights[index]]}")



    #====================================================
    if plot_change_data==1:
        x2_span_saved = []
        for index, state in enumerate(inputs):
            if labels[index] == 1:
                plt.plot(state[0], state[1], 'r.')
            else:
                plt.plot(state[0], state[1], 'b.')
            if predictions[index] == 1:
                plt.plot(state[0], state[1], 'ko', mfc='none')
        
        x1_span = np.linspace(axis[0], axis[1],100)
        for ii, W in enumerate(saved_weights):
            print(f"saved weight no: {ii}/{len(saved_weights)}")
            print(f"Length of saved biases: {len(saved_biases)}")
            alph = ii/len(saved_weights)
            for idx, w in enumerate(W):
                
                
                x2_span = (-w[0] * x1_span + saved_biases[ii][idx])/w[1]
                plt.plot(x1_span, x2_span, "k", alpha = alph)
        
        #plot y-axis
        print("====================================")
        plt.plot([0]*100, x1_span, 'k--')
        plt.plot(x1_span, x1_span*0, 'k--')
        plt.title("Boundary changes")
        plt.axis([*axis, *axis])
        plt.show()
    #====================================================





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


###############################################################################################

###############################################################################################


def neuralNet(inputs, labels, weights_layer1, weights_layer2, biases_layer1, biases_layer2, training_rate):

    p = len(inputs)
    mu = inputs[0]
    a1 = np.zeros(np.shape(p))
    a2 = np.zeros(np.shape(p))
    a_out = np.zeros(np.shape(p))
    
    #Forward propagation
    for mu in inputs:
        a1[mu] = g(np.matmul(weights_layer1, mu) - biases_layer1)
        a_out[mu] = g(np.matmul(weights_layer2, a1) - biases_layer2)
    print("test")
    predictions = np.sign(a_out)
    precision = np.sum(np.equal(predictions, labels)) / p
    H = np.sum((labels - a_out)**2) / (2*p)


    delta_out = g_prim(a_out) 

    delta_1 = np.matmul(delta_out, )



    return predictions, weights, biases, H, precision


## Build the network. Change these parameters if you want
p = 500
mean = 0
std = 0.5
batching = 1
epochs = 400
eons = 1
noise_rate = 0.1
training_rate = 0.5

l1 = 2
weights_layer1 = np.random.normal(mean, std, (2,l1))
weights_layer2 = np.random.normal(mean, std, (l1,1))
biases = [0, 0]

### Propagate the neutal network

inputs, labels = generate_set1_data(2, p, noise_rate)
predictions = neuralNet(inputs, labels)
###############################################################################################

###############################################################################################
"""
