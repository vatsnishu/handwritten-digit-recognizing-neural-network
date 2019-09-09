#!/usr/bin/python

from scipy.misc import imresize
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from copy import deepcopy
import numpy as np
from sklearn.model_selection import KFold



# Calculate neuron activation for an input
def net(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation


# Train a network for a fixed number of epochs and break when error is very less
def training(network, train):
	n_epochs=1000
	for n in range(n_epochs):
		sum_error=0
		for row in train:
			intermediate_inputs = []
			for j in range(len(network[0])):
				activation = net(network[0][j]['weights'], row)
				network[0][j]['output'] = 1.0 / (1.0 + exp(-activation))
				intermediate_inputs.append(network[0][j]['output'])

			outputs = []
			for j in range(len(network[1])):
				activation = net(network[1][j]['weights'], intermediate_inputs)
				network[1][j]['output'] = 1.0 / (1.0 + exp(-activation))
				outputs.append(network[1][j]['output'])

			hotEncoding = [0 for i in range(n_outputs)]
			hotEncoding[row[-1]] = 1
			for i in range(len(hotEncoding)):
				sum_error += (hotEncoding[i]-outputs[i])**2 

			for j in range(len(network[1])):
				network[1][j]['delta'] = (hotEncoding[j] - network[1][j]['output']) * (network[1][j]['output']* (1.0 - network[1][j]['output']))

			for j in range(len(network[0])):
				error = 0.0
				for i in range(len(network[1])):
					error += (network[1][i]['weights'][j] * network[1][i]['delta'])
				network[0][j]['delta'] = error * (network[0][j]['output']* (1.0 - network[0][j]['output']))
				

			for i in range(len(network[0])):
				for j in range(len(row[:-1])):
					network[0][i]['weights'][j] += l_rate * network[0][i]['delta'] * row[:-1][j]
				network[0][i]['weights'][-1] += l_rate * network[0][i]['delta']

			inputs = [network[0][j]['output'] for j in range(len(network[0]))]
			for i in range(len(network[1])):
				for j in range(len(inputs)):
					network[1][i]['weights'][j] += l_rate * network[1][i]['delta'] * inputs[j]
				network[1][i]['weights'][-1] += l_rate * network[1][i]['delta']
		if(sum_error<4):
			break

def print_network_weight(network):
	for i in range(n_hidden):
		print network[0][i]["weights"]
	for i in range(n_outputs):
		print network[1][i]["weights"]

# Backpropagation Algorithm With Stochastic Gradient Descent
def neural_network(train, test):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)

# Train the network using train set
	training(network, train)

#printing network weights after learning is over
	print_network_weight(network)

#make predictions on test data
	predictions = list()
	for row in test:
		intermediate_inputs = [] #ouputs of hidden layer
		for j in range(len(network[0])):
			activation = net(network[0][j]['weights'], row)
			network[0][j]['output'] = 1.0 / (1.0 + exp(-activation))
			intermediate_inputs.append(network[0][j]['output'])

		outputs = [] #outputs of output layer
		for j in range(len(network[1])):
			activation = net(network[1][j]['weights'], intermediate_inputs)
			network[1][j]['output'] = 1.0 / (1.0 + exp(-activation))
			outputs.append(network[1][j]['output'])

		maxi=0 # predicts the class corresponding to node giving maximum output
		for j in range(len(outputs)):
			if(outputs[j]>=outputs[maxi]):
				maxi=j
		predictions.append(maxi)
	return(predictions)

##############################
##############################
##          MAIN            ##
##############################
##############################

################################################################
#     collects data from file digitises and downscales it      #
################################################################

seed(1)
fname = 'optdigits-orig.tra'
t=0
im=list()
newimg=list()
dataset= list()
j=0
i=0
str1=""
with open(fname, 'r') as file:
	csv_reader = reader(file)
	for row in csv_reader:
		if not row:
			continue
		row=list(row[0].strip())
		row = [(int)(x.strip()) for x in row]
		if t==32:
			if(row[0]==5 or row[0]==1 or row[0]==2):
				im=np.array(im)
				newimg=imresize(im,(8,8)) #Downscaled data
				newimg = np.reshape(newimg, (1,np.product(newimg.shape))) 
				final_img=newimg[0].tolist()
				label=deepcopy(row[0])
				for i in range(len(final_img)): #digitizing data to grayscale
					if(final_img[i]>0):
						final_img[i]=final_img[i]/16
					if(final_img[i]<6):
						final_img[i]=0
					else:
						final_img[i]=1
				final_img.append(label)
				dataset.append(final_img)
			t=0
			im=list()
			newimg=list()
		else:
			im.append(row)
			t=t+1
lengthOfDataset= len(dataset[0])

for column in range(lengthOfDataset-1):
	for row in dataset:
		row[column] = float(row[column])

###################################################
#     running neural network for digits 5,2,1     #
###################################################
for row in dataset:
	if row[lengthOfDataset-1] == 5:
		row[lengthOfDataset-1]=0
	elif row[lengthOfDataset-1]==2:
		row[lengthOfDataset-1]=2
	elif row[lengthOfDataset-1]==1:
		row[lengthOfDataset-1]=1

n_folds = 5
l_rate = 0.3

########################################
#    Running kfold neural network      #
########################################
n_hidden_units = [8,16,24]
for n_hidden in n_hidden_units:
	folds = KFold(n_splits=4)
	scores = list()
	final_accuracy=0
	for train_set , test_set in folds.split(dataset):
		x_tr=[]
		x_test=[]
		for i in test_set:
			x_test.append(dataset[i])
		for i in train_set:
			x_tr.append(dataset[i])
	n_inputs = len(x_tr[0]) - 1
	n_outputs = len(set([row[-1] for row in x_tr]))
	predicted_test = neural_network(x_tr, x_test)
	predicted_train = neural_network(x_tr, x_tr)
	actual_test = [row[-1] for row in x_test]
	actual_train= [row[-1] for row in x_tr]
	correct_test = 0
	correct_train = 0
	for i in range(len(actual_test)):	
		if actual_test[i] == predicted_test[i]:
			correct_test += 1
	accuracy_test= correct_test / float(len(actual_test)) * 100.0

	correct_train = 0
	for i in range(len(actual_train)):	
		if actual_train[i] == predicted_train[i]:
			correct_train += 1
	accuracy_train= correct_train / float(len(actual_train)) * 100.0
	print "Number of Hidden Nodes\t\taccuracy_on_test\t\taccuracy_on_train"
	print "____________________________________________________________________________________"
	print str(n_hidden)+"\t\t\t\t"+str(accuracy_test)+"\t\t\t\t"+str(accuracy_train)

