import numpy
import scipy.special
import matplotlib.pyplot
class neuralNetwork:
    def __init__(self,inputnodes,hidennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hidennodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        self.activation_function = lambda x:scipy.special.expit(x)


    def train(self,inputs_list,target_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0-final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)),numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.2
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

data_file = open("mnist_dateset/mnist_train.csv",'r')
data_list = data_file.readlines()
data_file.close()

epochs = 5
for i in range(epochs):
    for record in data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("mnist_dateset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# print(data_list[0])
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_value = int(all_values[0])
    print(all_values[0])
    image_array = numpy.asfarray(all_values [1:]).reshape((28,28))
    matplotlib.pyplot.imshow( image_array , cmap='Greys' , interpolation='None')
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print(label)
    if (label == correct_value):
        scorecard.append(1)
    else:
        scorecard.append(0)
scorecard_array = numpy.asarray(scorecard)
print(scorecard_array.sum())
print(scorecard_array.size)
print("performance = ",scorecard_array.sum() / scorecard_array.size)


