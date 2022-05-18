import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.compat.v1.disable_eager_execution()

(x_train, y_train), (x_test, y_test) = tf.compat.v1.keras.datasets.boston_housing.load_data(path='boston_housing.npz',
                                                                                            test_split=0.2, seed=113)

print('X Train Shape is : ', x_train.shape)
print('X Train  is : ', x_train[5])
print('---------------------------------------- ')
print('X Test Shape is : ', x_test.shape)
print('X Test  is : ', x_test[5])
print('---------------------------------------- ')
print('y Train Shape is : ', y_train.shape)
print('y Train is : ', y_train[5])
print('---------------------------------------- ')
print('y Test Shape is : ', y_test.shape)
print('y Test  is : ', y_test[5])

(x_train, y_train), (x_test, y_test) = tf.compat.v1.keras.datasets.cifar10.load_data()

print('X Train Shape is : ', x_train.shape)
print('X Train  is : ', x_train[5])
print('---------------------------------------- ')
print('X Test Shape is : ', x_test.shape)
print('X Test  is : ', x_test[5])
print('---------------------------------------- ')
print('y Train Shape is : ', y_train.shape)
print('y Train is : ', y_train[5])
print('---------------------------------------- ')
print('y Test Shape is : ', y_test.shape)
print('y Test  is : ', y_test[5])
print('---------------------------------------- ')
print('All y is : ', np.unique(y_train))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.axis('off')
    plt.title('Class ' + str(y_train[i]))

plt.show()

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST-data", one_hot=True)

# for i in range(10) :

#     batch = mnist.train.next_batch(5)
#     print(batch[0])
#     print(batch[1])
#     print('----------------------')

# next batch is get each 5 images and label and combine it
# one hot is refer label with serial of binary and put it in one hot like
# when you target is 9 refer as [0 0 0 0 0 0 0 0 0 1]
