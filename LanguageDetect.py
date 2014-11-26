# -*- coding: utf-8 -*-
import pickle
import re
from math import *
from random import *


def sigmoid(z):
    return 1 / (1 + exp(-1 * z))


def der_sigmoid(sig):
    return sig * (1 - sig)


def print_confusion_matrix(confusion_matrix):
    print "Confusion matrix:"
    languages = ['en', 'it', 'nl']
    print(''.join(['{:5}'.format(item) for item in [" "] + languages]))
    print('\n'.join([language + ''.join(['{:5}'.format(item) for item in row])
                     for row, language in zip(confusion_matrix, languages)]))

    positive_classified = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
    total = sum([sum(_) for _ in confusion_matrix])
    print "Accuracy:", 100.0 * positive_classified / total, "%"


class Neuron:
    def __init__(self, alpha, no_of_inputs):
        self.alpha = alpha
        self.act = 0.0
        self.weights = [random() for _ in range(no_of_inputs)]

    def feed_forward(self, inputs):
        weighted_sum = sum([x * weight for x, weight in zip(inputs, self.weights)])
        self.act = sigmoid(weighted_sum)
        return self.act


class NeuralNetwork:
    def __init__(self, alpha, num_input_neurons, num_hidden_neurons, classes):

        # +1 for bias weights.
        self.alpha = alpha
        self.hidden = [Neuron(alpha, num_input_neurons + 1) for _ in range(num_hidden_neurons)]
        self.output = [Neuron(alpha, num_hidden_neurons + 1) for _ in range(classes)]

    def query(self, ip):

        ip.append(1)  # bias
        answers = self.forward_propogation(ip)
        # print answers
        language = answers.index(max(answers))
        if language == 0:     return "English"
        elif language == 1:   return "Italian"
        else:                 return "Dutch"

    def forward_propogation(self, example):
        """
        Send this example through the network and return output activations.
        """
        hidden_activations = [n.feed_forward(example) for n in self.hidden]
        output_activations = [o.feed_forward(hidden_activations) for o in self.output]
        return output_activations

    def validate(self, ip, op):

        val_err = 0.0
        for ex in ip: ex.append(1)  # bias

        for i, o in zip(ip, op):
            estimate = self.forward_propogation(i)
            # sum of squared error
            val_err += 0.5 * sum([(actual - comp) ** 2 for actual, comp in zip(o, estimate)])
        return val_err

    def test(self, test_files):

        # get test input and output set
        test_in_set, test_op_set = get_example_set(test_files)

        for inp in test_in_set: inp.append(1)  # bias

        confusion_matrix = [[0, 0, 0] for _ in range(3)]

        for inp, op in zip(test_in_set, test_op_set):

            guess = self.forward_propogation(inp)
            actual_class = op.index(1)
            classified_class = guess.index(max(guess))
            confusion_matrix[actual_class][classified_class] += 1

        print_confusion_matrix(confusion_matrix)

    def back_propogation(self, train_files, val_files, epochs):

        train_in, train_op = get_example_set(train_files)
        val_in, val_op = get_example_set(val_files)
        train_error_list, val_err_list = [], []

        for ip in train_in: ip.append(1)  # bias

        for epoch in range(epochs):
            error = 0.0
            for ip, op in zip(train_in, train_op):

                out_act = self.forward_propogation(ip)
                error_out = [(y - o) * der_sigmoid(o) for o, y in zip(out_act, op)]  # error at output

                # Error at hidden.
                error_hidden = []
                for index, neuron in enumerate(self.hidden):
                    err = sum([out.weights[index] * error_out[j] for j, out in enumerate(self.output)])
                    error_hidden.append(der_sigmoid(neuron.act) * err)

                # Update weights between output and hidden.
                for o, out_neu in enumerate(self.output):
                    for h, hid_neu in enumerate(self.hidden):
                        out_neu.weights[h] += self.alpha * hid_neu.act * error_out[o]

                # Update weights between hidden and input
                for h, hid_neu in enumerate(self.hidden):
                    for i, input in enumerate(ip):
                        hid_neu.weights[i] += self.alpha * input * error_hidden[h]

                # sse
                error += 0.5 * sum([(o.act - y) ** 2 for o, y in zip(self.output, op)])

            # error for this epoch.
            train_error_list.append(error)

            val_err = self.validate(val_in, val_op)
            val_err_list.append(self.validate(val_in, val_op))

            if val_err < 0.5:
                print "Validation threshold reached. Stopping training..."
                break

        # Remove bias from input for possible next iter from random restart.
        for ip in train_in: ip.pop()
        return train_error_list, val_err_list


def eval_text(text):
    words = text.split()
    if len(words) == 0:
        print "Empty text"
        return
    raw_features = [0, 0, 0, 0, 0]
    vowels = ['a', 'e', 'i', 'o', 'u']

    for word in words:

        if len(word) <= 2: continue
        # Ends with vowels for italian.
        if word[-1] in vowels:                      raw_features[0] += 1
        if word[-1] == '.' and word[-2] in vowels:  raw_features[0] += 1

        # Long words in dutch.
        if len(word) > 8:   raw_features[1] += 1

        # Words ending in "ed" for English in past tense.
        if word[-2:] == 'ed': raw_features[4] += 1

    raw_features[1] = text.count('en')  # bigram en.
    # Repeated letters for mostly dutch.
    raw_features[2] += len(re.findall(r'(.)\1', text))

    # ij pretty much NEVER shows up anywhere ever except Dutch!
    raw_features[2] += 4 * text.count('ij')

    # 'th' for english as most frequent bigram
    raw_features[3] = 2 * text.count('th')

    raw_features = [feature * 1.0 / len(words) for feature in raw_features]

    return raw_features


def get_example_set(train_files):

    op = [0, 0, 0]
    combined_set = []
    # open each file an make ip,op pair from examples.
    for index, instance_file in enumerate(train_files):
        try:
            with open(instance_file) as f:
                for line in f:
                    this_example_op = op[:]
                    this_example_op[index] = 1
                    combined_set.append((eval_text(line), this_example_op))
        except IOError:
            print instance_file, "NOT FOUND"

    shuffle(combined_set)

    # Return separate ip and op
    ip, op = [], []
    for i, o in combined_set:
        ip.append(i)
        op.append(o)

    return ip, op


def train(train_files, val_files, alpha, epochs, random_restarts=1):

    best_net = []
    for iteration in range(random_restarts):
        nn = NeuralNetwork(alpha, 5, 5, 3)
        train_err, val_err = nn.back_propogation(train_files, val_files, epochs)

        # plt.plot(range(epochs), train_err)
        # plt.plot(range(epochs), val_err)
        best_net.append(nn)
    # plt.legend(['Training error', 'Validation error'])
    # plt.show()
    # model = raw_input("Which model?")
    return best_net[0]


def test_mode():
    model = get_from_file()

    example = raw_input()
    if len(example) == 0:
        print "No text given"
    else:
        print model.query(eval_text(example))


def train_mode():
    train_files = ['en_train', 'it_train', 'nl_train']
    test_files = ['en_test', 'it_test', 'nl_test']
    val_files = ['en_val', 'it_val', 'nl_val']
    model = train(train_files, val_files, alpha=0.5, epochs=400, random_restarts=1)
    user_input = ''
    print 'Enter \"test\" to test model'
    while user_input is not "quit":

        user_input = raw_input("example? : ")
        if user_input == '': continue

        if user_input == 'test':
            print "space delimited test files in order <en> <it> <nl>"
            print "\"default\" to use default test files"
            user_test_files = raw_input("test files? : ")

            if user_test_files == 'default':
                model.test(test_files)
            else:
                model.test(user_test_files.split())
        elif user_input == 'write':
            save_to_file(model)
        else:
            print model.query(eval_text(user_input))


def main():

    read_from_file_and_test = True

    if read_from_file_and_test:
        test_mode()
    else:
        train_mode()


def save_to_file(neural_net):
    with open("mandeep_model.pickle", 'wb') as f:
        pickle.dump(neural_net, f)
    f.close()


def get_from_file():
    with open("mandeep_model.pickle", 'rb') as f:
        net = pickle.load(f)
    return net


if __name__ == "__main__":
    main()