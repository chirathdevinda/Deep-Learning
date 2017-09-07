import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 

path = "DataSets\MNIST_train.csv"

def preprocessing(path):
	dataset = pd.read_csv(path, header=0)
	data = dataset.values

	X = data[:1000, 1:] / 255.0

	return X

def RelU(z):
	return (z > 0) * z

def Sigmoid(z):
	return 1 / (1 + np.exp(-z))

def Derivative(z, type):
	if type == "RelU":
		return (z > 0)
	if type == "sigmoid":
		return z*(1 - z)

def plot_error(data):
	fig = plt.figure(figsize=(10,6))
	fig.suptitle("Error Plot")
	plt.xlabel("Iteration")
	plt.ylabel("Error")
	plt.plot(data)
	plt.show()

def Autoencorder(xtrain, alpha=0.01, iterations=200):

	np.random.seed(1)

	num_inputs = xtrain.shape[1]
	num_hidden_nodes = 40
	num_outputs = num_inputs
	num_samples = xtrain.shape[0]

	weights_0_1 = 0.01*np.random.randn(num_inputs, num_hidden_nodes)
	weights_1_2 = 0.01*np.random.randn(num_hidden_nodes, num_outputs)

	errors = []

	for iteration in range(iterations):
		error = 0.0
		for i in range(num_samples):
			layer_0 = xtrain[i:i+1]
			layer_1 = RelU(layer_0.dot(weights_0_1))
			layer_2 = layer_1.dot(weights_1_2)
			prediction = Sigmoid(layer_2)

			error += np.sum((prediction - layer_0) ** 2)

			layer_2_delta = (prediction - layer_0) * Derivative(prediction, "sigmoid")
			layer_1_delta = layer_2_delta.dot(weights_1_2.T) * Derivative(layer_1, "RelU")

			weights_1_2 -= layer_1.T.dot(layer_2_delta) * alpha
			weights_0_1 -= layer_0.T.dot(layer_1_delta) * alpha

		error = float(error) / num_samples
		errors.append(error)
		sys.stdout.write("\r" + "Iteration: " + str(iteration+1) + " Error: " + str(error))

	return errors, weights_0_1, weights_1_2

X = preprocessing(path)
error, encorder, decoder = Autoencorder(X)

# Error Plot
plot_error(error)

fig = plt.figure(figsize=(10,6))

for i in range(0, 10, 2):
	original_img = X[i].reshape((28,28))
	encoded_img = X[i:i+1].dot(encorder)
	decoded_img = encoded_img.dot(decoder).reshape((28,28))

	ax = fig.add_subplot(5,2,i+1)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	if i < 1:
		ax.set_title("Before")
	img = plt.imshow(original_img, cmap=plt.get_cmap("gray"))

	ax = fig.add_subplot(5,2,i+2)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	if i < 1:
		ax.set_title("After")
	img = plt.imshow(decoded_img, cmap=plt.get_cmap("gray"))

plt.show()