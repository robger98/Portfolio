import numpy as np
import random as random
import copy
import matplotlib.pyplot as plt
import sys
from skimage.util.shape import view_as_windows

# referenced  code from https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
# and https://towardsdatascience.com/back-propagation-the-easy-way-part-3-cc1de33e8397
# and https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/


class NeuralNetwork:
    """Implementation of a simple FCFF neural network in python
    
    Attributes:
        layers (`list` of `Layer`)  : Contains the layers of the nn.\n
        learning_rate (`float`)     : The learning rate.\n
        momemtum_factor (`float`)   : The momentum factor.\n
        loss_func   (`function`)    : This holds the loss function, which should accept input in the form (y, y_hat) 
            and return a `float`.\n
        seed (`int` or `None`)      : The random seed which may be None if left unused
        epoch_int (`int`)           : The interval between training metric print outs in terms of epochs 
            (ex. 1 would print out the metrics every epoch)
        
    """
    def __init__(
        self,
        layers,
        loss_func,
        learning_rate=0.05,
        momentum_factor=0.0,
        seed=None,
        epoch_int=100,
    ):
        """Initializes the NN and sets the random see if passed.
    
        Args:
            layers (`list` of `Layer`)          : Contains the layers of the nn.\n
            learning_rate (`float`)             : The learning rate.\n
            momemtum_factor (`float`)           : The momentum factor.\n
            loss_func   (`function`)            : This holds the loss function, which should accept input in 
                the form (y, y_hat) and return a `float`.\n
            seed (`int` or `None`) (optional)   : The random seed which may be None if left unused.\n
            epoch_int (`int`)                   : The interval between training metric print outs in terms of epochs 
                (ex. 1 would print out the metrics every epoch).
        
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.loss_func = loss_func
        self.seed = seed
        self.epoch_int = epoch_int
        if seed != None:
            random.seed(seed)
            np.random.seed(seed)

    def copy(self):
        """Wrapper for copy.deepcopy()"""
        return copy.deepcopy(self)

    def forward_prop(self, input, save=True):
        """Performs forward prop on the input through the network and returns the activations"""
        A = self.layers[0].feedforward(input, save=save)
        for i in range(1, len(self.layers)):
            A = self.layers[i].feedforward(A, save=save)

        return A

    def backprop(self, y_hat, y):
        """Performs backprop on the given y_hat and y through the network and updates the weights.

        Args:
            y_hat (`ndarray`): The prediction labels to compare to the true labels.\n
            y (`ndarray`): The true labels.
        """
        m = y.shape[0]

        dA_prev = Losses.cross_entropy(y, y_hat, deriv=True)
        dA_prev = self.layers[-1].backprop(dA_prev, None)

        for i in range(len(self.layers) - 2, -1, -1):
            dA_prev = self.layers[i].backprop(dA_prev, self.layers[i + 1])

    def divide_chunks(self, l, n):
        """Divides a list like into n equally sized chunks.

        Args:
            l (`list-like`): list-like to be divided.\n
            n (`int`): Number of chunks to be divided into.\n

        Returns:
            `list`: The divided list. 
        """
        # looping till length l
        return [l[i : i + n] for i in range(0, len(l), n)]

    def train(
        self, x, y, epochs, validation_x=None, validation_y=None, batch_size=None
    ):
        """Trains the neural network on data x, which should be of the shape 
        (<num examples>, <feature count>), and the labels y, which currently are 
        one hot encoded.

        Args:
            x (`ndarray`)               : Training examples.\n
            y (`ndarray`)               : Training labels.\n
            epochs (`int`)              : Number of epochs to train for. Epochs = -1 will run forever.\n
            validation_x (`ndarray`)    : Optional. Validation examples.\n
            validation_y (`ndarray`)    : Optional. Validation labels.\n
            batch_size (`int`)          : Optional.
        
        Returns:
            `tuple` of 3 `list` containing accuracy history, loss history, and validation accuracy history
        """
        cost_history = []
        accuracy_history = []
        validation_history = []
        act_hist = []
        i = 0
        while i < epochs or epochs == -1:
            accuracy_this_round = []
            x_y_pair = [(x[i], y[i]) for i in range(len(x))]
            x = np.asarray([pair[0] for pair in x_y_pair])
            y = np.asarray([pair[1] for pair in x_y_pair])

            if batch_size != None:
                x_batches = self.divide_chunks(x, batch_size)
                y_batches = self.divide_chunks(y, batch_size)

                for batch_num in range(len(x_batches)):
                    y_hat = self.forward_prop(x_batches[batch_num])

                    self.backprop(y_hat, y_batches[batch_num])
                    a = x_batches[batch_num]
                    for layer in self.layers:
                        layer.update(a)
                        a = layer.A
            else:
                y_hat = self.forward_prop(x, save=True)
                act_hist.append(self.layers[0].A)
                self.backprop(y_hat, y)
                a = x
                for layer in self.layers:
                    layer.update(a)
                    a = layer.A

            if i % self.epoch_int == 0:
                y_hat = self.forward_prop(x, save=False)
                cost = self.loss_func(y, y_hat)
                accuracy = self.get_accuracy(y, y_hat)

                cost_history.append(cost)
                accuracy_history.append(accuracy)
                print("Epoch", i)
                print("      Training Loss: {:4.5f}".format(np.sum(cost) / len(x)))
                print("  Training Accuracy: {:4.5f}".format(accuracy))
                if type(validation_x) != type(None):
                    vy_hat = self.forward_prop(validation_x, save=False)
                    v_cost = self.loss_func(validation_y, vy_hat)
                    v_accuracy = self.get_accuracy(validation_y, vy_hat)
                    validation_history.append(v_accuracy)
                    print(
                        "    Validation Loss: {:4.5f}".format(
                            np.sum(v_cost) / len(validation_x)
                        )
                    )
                    print("Validation Accuracy: {:4.5f}".format(v_accuracy))
                print()
            i += 1

        return accuracy_history, cost_history, act_hist, validation_history

    def predict(self, x):
        """Return the predicted categorical label of examples

        Args:
            x (`ndarray`): the example to predict

        Returns:
            int: _description_
        """
        y_hat = self.forward_prop(x)
        y_hat_none_one_hot = []
        for i in range(len(y_hat)):
            y_hat_none_one_hot.append(np.argmax(y_hat[i]))

        return y_hat_none_one_hot

    def test(self, x, y):
        """Given a list of examples and their one-hot labels, will return the accuracy

        Args:
            x (`ndarray`): list of examples
            y (`ndarray`): list of true one-hot labels

        Returns:
            `float`: Accuracy of the model on the provided examples
        """
        y_hat = self.forward_prop(x)
        accuracy = self.get_accuracy(y, y_hat)
        return accuracy

    def get_accuracy(self, y, y_hat):
        """Helper function for getting the accuracy between predicitions and true labels

        Args:
            y (`ndarray`): true labels
            y_hat (`ndarray): predicted labels

        Returns:
            `float`: accuracy
        """
        classed = copy.deepcopy(y_hat)

        list_y = list(y)
        correct = 0
        incorrect = 0

        for i in range(len(y)):
            max_ind = np.argmax(y_hat[i])
            if y[i][max_ind] != 1:
                incorrect += 1
            else:
                correct += 1
        return (correct / (incorrect + correct)) * 100

    def add_layer(
        self,
        input_size,
        output_size,
        activation_func,
        bias=0,
        random_init=True,
        is_output=False,
    ):
        """Adds new layer to the network

        Args:
            input_size (`int`): input size
            output_size (`int`): output size
            activation_func (`funtion`): activation function
            bias (int, optional): bias. Defaults to 0.
            random_init (bool, optional): Random weight initialization. Defaults to True.
            is_output (bool, optional): true if this layer is an output layer. Defaults to False.
        """
        self.layers.append(
            Layer(
                input_size,
                output_size,
                activation_func,
                bias=bias,
                random_init=random_init,
                learning_rate=self.learning_rate,
                momentum_factor=self.momentum_factor,
                is_output=is_output,
            )
        )

    @staticmethod
    def create_from_config(filepath):
        """Reads a config file to create a new network.
        
        The config file must follow the following format:
            line 1: <input size>, <output size>\n
            line 2: <num hidden layers>, <size of hidden layer 1>, <size of hidden layer 2>, etc.\n
            line 3: <learning rate>, <momentum factor>\n
            line 4: <random seed>
            line 5: <training epochs>
            line 6: <number of examples>
            lines 7 to 7+number_of_examples contains the training examples
                each example follows the format
                <index>, x_1, x_2, ... , x_n, <class label> where n is the input size
            line -3: <"true" if cross validation>, <k for k-fold cross validation>, <train/validation split ex. .7>
            line -2: reporting interval (ex. 100 would print accuracy every 100 epochs)
            line -1: epsilon (not implemented)

        Args:
            filepath (`string`): The filepath for the config file

        Returns:
            _type_: _description_
        """
        
        # read the file
        with open(filepath) as config:

            temp_in_out = config.readline().split(",")
            num_in = int(temp_in_out[0])
            num_out = int(temp_in_out[1])

            temp_hidden = config.readline().split(",")
            # num_hidden = temp_hidden[0]
            layer_sizes = [int(x) for x in temp_hidden[1:]]

            temp_nu_alpha = config.readline().split(",")
            nu = float(temp_nu_alpha[0])
            alpha = float(temp_nu_alpha[1])
            seed = int(config.readline())

            num_t_and_s = config.readline().split(",")
            num_t = int(num_t_and_s[0])
            num_s = None
            if num_t == 0:
                num_t = -1
                num_s = int(num_t_and_s[1])

            num_examples = int(config.readline())
            examples = []
            classes = []
            for i in range(num_examples):

                ex = config.readline().split(",")
                ex_class = int(ex[1])
                ex_attributes = [int(x) for x in ex[2:]]
                examples.append(ex_attributes)
                classes.append(ex_class)
                
            # convert class labels to one-hot
            one_hot_labels = NeuralNetwork.to_one_hot(classes)

            temp_a_c = config.readline().split(",")
            a = temp_a_c[0] in ["true", "True"]
            c = int(temp_a_c[1])
            v = None
            if a:
                v = float(temp_a_c[2])

            epoch_int = int(config.readline())

            epsilon = float(config.readline())

            new_nn = NeuralNetwork(
                [],
                Losses.cross_entropy,
                learning_rate=nu,
                momentum_factor=alpha,
                seed=seed,
                epoch_int=epoch_int,
            )
            prev_out = num_in
            for ls in layer_sizes:
                new_nn.add_layer(prev_out, ls, Activations.relu)
                prev_out = ls
            new_nn.add_layer(prev_out, num_out, Activations.softmax, is_output=True)
            # nns.append(new_nn)
            return new_nn, examples, one_hot_labels, num_t, num_s, a, c, v, epsilon

    def reinit(self):
        """Reinitializes the network
        """
        for layer in self.layers:
            layer.reinit()

    @staticmethod
    def to_one_hot(classes):
        """Helper method to convert class labels to one-hot vectors

        Args:
            classes (`ndarray`): array of the class labels

        Returns:
            `ndarray`: array of one-hot labels
        """
        one_hot_labels = []
        class_set = set(classes)
        num_classes = len(class_set)
        class_dict = {}
        i = 0
        for cls in class_set:
            class_dict.update({cls: i})
            i += 1
        for cls in classes:
            one_hot_labels.append(
                [1 if class_dict[cls] == i else 0 for i in range(num_classes)]
            )
        return one_hot_labels


class Layer:
    """Implementation of a single FCFF layer
    """
    def __init__(
        self,
        input_size,
        output_size,
        activation_func,
        bias=0,
        random_init=True,
        learning_rate=0.05,
        momentum_factor=0.5,
        is_output=False,
    ):
        """Initializes a new layer

        Args:
            input_size (`int`): input size
            output_size (`int`): output size
            activation_func (`function`): activation function for the layer
            bias (int, optional): bias. Defaults to 0.
            random_init (bool, optional): True for random weight initialization. Defaults to True.
            learning_rate (float, optional): learning rate. Defaults to 0.05.
            momentum_factor (float, optional): momentum factor. Defaults to 0.5.
            is_output (bool, optional): True for an output layer. Defaults to False.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.A = None
        self.activation_func = activation_func
        self.weights = (
            np.random.rand(input_size, output_size)
            if random_init
            else np.zeros(input_size, output_size)
        )
        self.momentum_factor = momentum_factor
        self.previous_weight_deltas = np.zeros((input_size, output_size))
        self.bias = np.random.randn(output_size)
        self.Z = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.is_output = is_output
        self.delta = 0

    def feedforward(self, input_A, save=True):
        """
        Feeds forward through this layer
        """
        if save:
            self.Z = np.dot(input_A, self.weights) + self.bias
            self.A = self.activation_func(self.Z)
            return self.A
        else:
            z = np.dot(input_A, self.weights) + self.bias
            a = self.activation_func(z)
            return a

    def backprop(self, error, right_layer):
        """
        Runs backpropagations through this layer
        """
        if self.is_output:
            self.delta = error
            return self.delta

        self.delta = np.dot(
            right_layer.delta, right_layer.weights.T
        ) * self.activation_func(self.Z, deriv=True)

        return self.delta

    def reinit(self, random_init=True):
        """Reinitializes the weights for the layer

        Args:
            random_init (bool, optional): True for random reinitialization. Defaults to True.
        """
        self.A = None
        self.weights = (
            np.random.rand(self.input_size, self.output_size)
            if random_init
            else np.zeros(self.input_size, self.output_size)
        )
        self.previous_weight_deltas = np.zeros((self.input_size, self.output_size))
        self.bias = np.random.randn(self.output_size)
        self.Z = np.zeros(self.output_size)
        self.delta = 0

    def update(self, left_a):
        """
        Updates the weights and biases for this layer
        """
        dzh = 1
        # if not self.is_output:
        #     dzh = self.activation_func(self.Z, deriv=True)
        ad = np.dot(left_a.T, dzh * self.delta)

        self.previous_weight_deltas = self.learning_rate * ad + (
            self.momentum_factor * self.previous_weight_deltas
        )
        self.weights -= self.previous_weight_deltas
        self.bias -= self.learning_rate * np.sum(self.delta * dzh, axis=0)


class Losses:
    """
    This class contains loss functions (Currently only cross entropy)
    """

    @staticmethod
    def cross_entropy(y, y_hat, deriv=False):
        if deriv:
            return y_hat - y
        return -np.sum(y * np.log(y_hat))


class Activations:
    """
    This class contains different activation functions
    """

    @staticmethod
    def sigmoid(Z, deriv=None):
        if deriv:
            s = Activations.sigmoid(Z)
            return s * (1 - s)
        return 1 / (1 + np.exp(-Z))

    """
    Note: can only be used in the output layer with cross_entropy, as it does not have a derivative implemented
    """

    @staticmethod
    def softmax(Z, deriv=None):

        shiftZ = Z - np.max(Z)
        exps = np.exp(shiftZ)
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def relu(Z, deriv=None):
        if deriv:
            dZ = np.array(Z, copy=True)
            dZ[Z <= 0] = 0
            dZ[Z > 0] = 1

            return dZ
        return np.maximum(0, Z)


class CrossValidationDataset:
    """Converts Dataset into a collection of training and validation set pairs

    Class vars:

    train_valid_pairs --    list containing dicts with keys "train" and "validation" for training
                            and validation Datasets respectively"""

    def __init__(self, dataset, k):
        """Arguments:

        dataset -- the dataset to split

        k -- the number of subsets to make

        seed -- the seed for random shuffling
        """

        random.shuffle(dataset)
        print("Splitting dataset...")
        split_lists = np.array_split(dataset, k)
        print("Creating training validation pairs...")
        self.train_valid_pairs = []
        counter = 1
        for l in split_lists:
            print("Creating pair", counter)
            counter += 1
            train_list = []
            for m in split_lists:
                
                if m[0] not in l:
                    train_list.extend(m)
            self.train_valid_pairs.append(self.TV_pair(train_list, l))

    def get_split_exs_labels(self, i):
        """Returns the training and validation splits 

        Args:
            i (`int`): Which k-fold to get the data for

        Returns:
            `tuple`: Returns 4 lists, the training examples, training labels, validation examples, and validation labels
        """
        train_exs = []
        train_labels = []
        valid_exs = []
        valid_labels = []
        tv = self.train_valid_pairs[i]

        for exl in tv.t:
            train_exs.append(exl.attributes)
            train_labels.append(exl.label)
        for exl in tv.v:
            valid_exs.append(exl.attributes)
            valid_labels.append(exl.label)

        return train_exs, train_labels, valid_exs, valid_labels

    class TV_pair:
        """Data class that holds a training/validation pair
        """
        def __init__(self, train, validation):
            self.t = train
            self.v = validation


class Example:
    """Data class that holds the attributes and labels of a single example"""
    def __init__(self, attributes, label):
        self.attributes = attributes
        self.label = label


def de_one_hot(one_hot):
    """Helper function to convert one_hot into single valued class labels
    """
    return np.argmax(one_hot)

if __name__ == "__main__":
    # create the NN
    (
        nn,
        examples,
        one_hot_labels,
        t,
        s,
        a,
        c,
        v,
        epsilon,
    ) = NeuralNetwork.create_from_config(sys.argv[1])
    
    # if performing cross validation
    if a:
        trained_models = []
        dataset = [
            Example(examples[i], one_hot_labels[i]) for i in range(len(examples))
        ]
        # print(dataset)
        cv_set = CrossValidationDataset(dataset, c)
       
        train_history = []
        validation_history = []
        test_history = []
        
        # for each fold in the dataset
        for i in range(len(cv_set.train_valid_pairs)):
            
            # getting our data
            learn_ex, learn_label, test_ex, test_label = cv_set.get_split_exs_labels(i)
            ex_len = len(learn_ex)
            stop_point = int(v * ex_len)
            train_ex = np.asarray(learn_ex[:stop_point])
            train_label = np.asarray(learn_label[:stop_point])
            valid_ex = np.asarray(learn_ex[stop_point:])
            valid_label = np.asarray(learn_label[stop_point:])
            do_valid = len(valid_ex) != 0
            
            #train the network
            th, lh, act_hist, vh = nn.train(
                train_ex,
                train_label,
                t,
                validation_x=valid_ex if do_valid else None,
                validation_y=valid_label if do_valid else None,
                batch_size=32,
            )
            train_history.append(th)
            validation_history.append(vh)
            
            # Reporting
            test_history.append(nn.test(np.asarray(test_ex), np.asarray(test_label)))
            info = "Fold {:d}:\n\tTrainAcc {:4.5f}\n\tValidAcc {:4.5f}\n\t TestAcc {:4.5f}".format(
                i + 1, th[-1], vh[-1] if do_valid else -1, test_history[-1]
            )
            print(info)
            
            # Save a copy of the model and reinitialize weights
            trained_models.append(nn.copy())
            nn.reinit()

        # ensemble phase testing
        preds = []
        for tm in trained_models:
            preds.append(tm.predict(examples))
        correct = 0
       
        for i in range(len(preds[0])):
            
            votes = [p[i] for p in preds]
            vote_set = set(votes)
            max_vote = max(vote_set, key=lambda x: votes.count(x))
            if max_vote == de_one_hot(one_hot_labels[i]):
                correct += 1

        print("Total ensemble accuracy: {:4.5}".format(correct / len(examples)))
    else: # not performing cross validation or any validation
        trained_models = []
        train_history = []
        validation_history = []
        test_history = []

        for i in range(c):
            np_ex = np.asarray(examples)
            np_lab = np.asarray(one_hot_labels)
            # print(np_ex, np_lab)
            th, lh, act_hist, _ = nn.train(
                np_ex,
                np_lab,
                t,
                batch_size=None,
            )
        
            train_history.append(th)
            
            te = nn.test(np_ex, np_lab)

            if i == 0:

                fig, ax1 = plt.subplots()
                color = "tab:red"
                ax1.set_xlabel("Epoch#/" + str(nn.epoch_int))
                ax1.set_ylabel("Loss", color=color)
                ax1.plot(lh, color=color)
                ax1.tick_params(axis="y", labelcolor=color)

                ax2 = (
                    ax1.twinx()
                )  # instantiate a second axes that shares the same x-axis

                color = "tab:blue"
                ax2.set_ylabel(
                    "Accuracy", color=color
                )  # we already handled the x-label with ax1
                ax2.plot(th, color=color)
                ax2.tick_params(axis="y", labelcolor=color)

                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()

            test_history.append(te)

            info = "Init {:d}:\n\tTrainAcc {:4.5f}\n\tTestAcc {:4.5f}".format(
                i + 1, th[-1], test_history[-1]
            )
            print(info)

            trained_models.append(nn.copy())
            nn.reinit()

        # ensemble phase
        preds = []
        for tm in trained_models:
            preds.append(tm.predict(examples))
        correct = 0
        print(preds)
        for i in range(len(preds[0])):
            print(i)
            votes = [p[i] for p in preds]
            vote_set = set(votes)
            max_vote = max(vote_set, key=lambda x: votes.count(x))
            if max_vote == de_one_hot(one_hot_labels[i]):
                correct += 1

        print("Total ensemble accuracy: {:4.5}".format(correct / len(examples)))
