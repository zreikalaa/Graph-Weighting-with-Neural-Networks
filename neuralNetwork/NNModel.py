import random
from layersBuilder import *
import numpy
from numpy import exp
import matplotlib.pyplot as plt
from dataTransformation import *
from parametersClass import *
import sys
numpy.set_printoptions(threshold=sys.maxsize)


class NNModel:
    def sigmoid(self, x):
        x = x + 0.0000001
        sigmoid=1 / (1 + exp(-x))
        return sigmoid, x

    def sigmoid_derivative(self, x):
        x = x + 0.0000001
        return self.sigmoid(x)[0] * (1 - self.sigmoid(x)[0])

    def relu(self, x, nodes_type):
        x = x + 0.0000001
        return (x * (x > 0) + 0.000001, x)

    def relu_derivative(self, x):
        x = x + 0.0000001
        x[x <= 0] = 0
        x[x > 0] = 1

        return x

    def tanh(self, x, nodes_type):
        x = x + 0.00001
        tanh = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return (tanh*(nodes_type == 1))+(x*(nodes_type == 0)), x

    def tanh_derivative(self, x, nodes_type):
        x = x + 0.0000001
        return (1 - np.square(self.tanh(x, nodes_type)[0]))*(nodes_type==1) + x*(nodes_type==0)

    def initialize_parameters_deep(self, layers, lb):
        """
        Arguments:
        layers -- python array 2D (list) contain the concepts in each layer layers[0] is the input concepts 'leaf nodes'
        lb -- layersBuilder

        Returns:
        parameters -- python dictionary containing the parameters "W1", "b1", "IdW1", "Idb1", ..., "WL", "bL", "IdWL", "IdbL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
                        IdWl -- identity matrix of shape (layer_dims[l], layer_dims[l-1])
                        Idbl -- identity bias vector of shape (layer_dims[l], 1)
        """

        #for layer in layers:
        #    print(layer)

        parameters = {}
        for layer_index in range(0, len(layers) - 1):

            IdW = np.zeros((len(layers[layer_index + 1]), len(layers[layer_index])))
            Idb = np.zeros((len(layers[layer_index + 1]), 1))
            W = np.random.randn(len(layers[layer_index + 1]), len(layers[layer_index]))
            b = np.zeros((len(layers[layer_index + 1]), 1))

            #update Idb
            for index, node in enumerate(layers[layer_index + 1]):
                if "$$pass" not in node:
                    Idb[index][0] = 1

            for node_index, node in enumerate(layers[layer_index]):
                if "$$pass" in node:
                    node = node.split("$$")[0]

                parent = lb.node_parent(node)

                if node + "$$pass" in layers[layer_index + 1]:
                    pass_index = layers[layer_index + 1].index(node + "$$pass")
                    for temp_index in range(len(layers[layer_index + 1])):
                        if temp_index == pass_index:
                            W[pass_index][node_index] = 1
                            IdW[pass_index][node_index] = 1
                        else:
                            W[temp_index][node_index] = 0

                elif parent in layers[layer_index + 1]:
                    parent_index = layers[layer_index + 1].index(parent)
                    for temp_index in range(len(layers[layer_index + 1])):
                        if temp_index == parent_index:
                            IdW[parent_index][node_index] = 1
                        else:
                            W[temp_index][node_index] = 0

            parameters['IdW' + str(layer_index + 1)] = IdW
            parameters['Idb' + str(layer_index + 1)] = Idb
            parameters['W' + str(layer_index + 1)] = W
            parameters['b' + str(layer_index + 1)] = b

        return parameters

    # GRADED FUNCTION: linear_forward

    def compute_cost(self, AL, Y, outputActivation):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]
        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code)
        if outputActivation == "sigmoid":
            a = np.multiply((Y), np.log(AL))
            b = np.multiply((1 - Y), np.log(1 - AL))
        elif outputActivation == "tanh":
            a = np.multiply((Y + 1) / 2, np.log((AL + 1) / 2))
            b = np.multiply((1 - (Y + 1) / 2), np.log(1 - (AL + 1) / 2))

        cost = -(1 / m) * np.sum(a + b)
        ### END CODE HERE ###

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())

        return cost

    # GRADED FUNCTION: linear_backward

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        ### START CODE HERE ### (≈ 3 lines of code)
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    # GRADED FUNCTION: linear_activation_backward

    def linear_activation_backward(self, dA, cache, activation, nodes_type):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = dA * self.relu_derivative(activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###

        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)

            dZ = dA * self.sigmoid_derivative(activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###

        elif activation == "tanh":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = dA * self.tanh_derivative(activation_cache, nodes_type)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###

        elif activation == "linear":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = dA
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###

        return dA_prev, dW, db

    # GRADED FUNCTION: L_model_backward

    def L_model_backward(self, AL, Y, caches, parameters):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        ### END CODE HERE ###

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL,
                                                                                                               current_cache,
                                                                                                               "sigmoid", parameters["Idb" + str(L)])
        ### END CODE HERE ###

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                             "tanh", parameters["Idb" + str(L)])
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads

    # GRADED FUNCTION: update_parameters
    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 4  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)] * \
                                           parameters["IdW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)] * \
                                           parameters["Idb" + str(l + 1)]

        ### END CODE HERE ###
        return parameters

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W, A) + b
        ### END CODE HERE ###

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, nodes_type, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        linear_cache, activation_cache, A = 0, 0, 0
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = self.relu(Z,nodes_type)
        elif activation == "tanh":
            A, activation_cache = self.tanh(Z,nodes_type)
        elif activation == "linear":
            A, activation_cache = Z, Z

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 4  # number of layers in the neural network

        # Implement [LINEAR -> SIGMOID]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], parameters["Idb" + str(l)],
                                                      "tanh")
            caches.append(cache)

            ### END CODE HERE ###

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], parameters["Idb" + str(l)], "sigmoid")
        caches.append(cache)
        ### END CODE HERE ###

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches
    # GRADED FUNCTION: L_layer_model

    def L_layer_model(self, X, Y, layers, lb, learning_rate=1, num_iterations=1000, print_cost=True):  # lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []  # keep track of cost

        # Parameters initialization. (≈ 1 line of code)
        ### START CODE HERE ###

        filesize = os.path.getsize("parameters.txt")
        if filesize == 0:
            print("parameters initialization")
            parameters = self.initialize_parameters_deep(layers, lb)
            self.save_parameters(parameters)
        else:
            print("Load  parameters...")
            parameters = self.get_parameters()
        ### END CODE HERE ###

        ALs=[] #List containing the AL after each 100 iteration. Used to analyse the convergence.
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = self.L_model_forward(X, parameters)
            ### END CODE HERE ###

            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            cost = self.compute_cost(AL, Y, "sigmoid")
            ### END CODE HERE ###

            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = self.L_model_backward(AL, Y, caches, parameters)
            ### END CODE HERE ###

            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###

            # Print the cost every 100 training example
            if print_cost and i % 20 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 100 == 0:
                costs.append(cost)
            ALs.append(AL)

        """for index, y in enumerate(Y[0]):
            print("class = ",y," ALS=", end=" ")
            for al in ALs:
                print('{0:.3g}'.format(al[0][index]), end=" --> ")
            print(" ")"""

        print(parameters["W5"][0])

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        return parameters

    def get_parameters(self, fileName="parameters.txt"):
        with open(fileName,"rb") as MyFile:
            parametersObject=pickle.load(MyFile)
            return parametersObject.parameters

    def save_parameters(self, parameters):
        with open("parameters.txt", "wb") as MyFile:
            parametersObject = parametersClass(parameters)
            pickle.dump(parametersObject, MyFile)

    def load_data(self, tableName, hors_dusage_Percentage, trainpercentage, lengthMinimum=3, number_of_dimension=-1):

        conn = psycopg2.connect(
            "dbname=BNF port=5432 user=postgres password=Postalaa1")
        cursor = conn.cursor()

        class1 = []
        class2 = []

        query = "SELECT * from " + tableName + " where class=0"
        cursor.execute(query)

        hors_dusage = cursor.rowcount
        communicable = int((100 - hors_dusage_Percentage) * hors_dusage / hors_dusage_Percentage)
        print("hors d'usage = ",hors_dusage)
        print("communicable = ",communicable)


        for row in cursor:
            class2.append(row[:-1])


        query = "SELECT * from " + tableName + " where class=1"
        cursor.execute(query)

        for row in cursor:
            class1.append(row[:-1])

        print("len class1 = ",len(class1))
        print("len class2 = ",len(class2))

        communicable = len(class1)
        hors_dusage = len(class2)

        train_id = []
        test_id = []

        random.shuffle(class1)
        random.shuffle(class2)

        train_communciable = int(communicable * trainpercentage / 100)
        train_hors = int(hors_dusage * trainpercentage / 100)

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for index in range(0, train_communciable):
            train_x.append(class1[index][1:-2])
            train_y.append(class1[index][-1])
            train_id.append(class1[index][0])

        for index in range(train_communciable, len(class1)):
            test_x.append(class1[index][1:-2])
            test_y.append(class1[index][-1])
            test_id.append(class1[index][0])

        for index in range(0, train_hors):
            train_x.append(class2[index][1:-2])
            train_y.append(class2[index][-1])
            train_id.append(class2[index][0])

        for index in range(train_hors, len(class2)):
            test_x.append(class2[index][1:-2])
            test_y.append(class2[index][-1])
            test_id.append(class2[index][0])

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_y = train_y.reshape((len(train_y), 1))

        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_y = test_y.reshape((len(test_y), 1))

        print("train x shape: ", train_x.shape)
        print("train y shape: ", train_y.shape)
        print("test x shape: ", test_x.shape)
        print("test y shape: ", test_y.shape)
        return train_id, test_id, train_x.T, train_y.T, test_x.T, test_y.T

    def predict(self, X, Y, id, parameters, print_result=False):
        AL = self.L_model_forward(X, parameters)[0]

        cc = 0
        ch = 0
        hc = 0
        hh = 0
        accurate = 0
        for index, output in enumerate(AL[0]):
            if output > 0.5:
                output = 1
            else:
                output = 0

            if output == int(Y[0][index]):
                accurate += 1

            if output == 1 and int(Y[0][index]) == 1:
                cc += 1
            elif output == 0 and int(Y[0][index]) == 0:
                hh += 1
            elif output == 0 and int(Y[0][index]) == 1:
                ch += 1
            elif output == 1 and int(Y[0][index]) == 0:
                hc += 1


        return cc, ch, hh, hc, accurate * 100 / len(Y[0])
