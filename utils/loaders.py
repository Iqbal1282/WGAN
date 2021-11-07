from tensorflow.keras.datasets import cifar10 , cifar100 
import numpy as np 

def load_cifar(label, num):
	if num == 10: 
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	else: 
		(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode = 'fine')


	train_mask = [y[0] == label for y in y_train]
	test_mask = [y[0] == label for y in y_test]

	x_data = np.concatenate([x_train[train_mask], x_test[test_mask]])
	y_data = np.concatenate([y_train[train_mask], y_test[test_mask]])

	x_data = (x_data.astype('float32')- 127.5)/127.5

	return (x_data, y_data)

	