import tensorflow as tf
mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt
import numpy as np
import random as ran

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# x_train, x_test = x_train / 255.0, x_test / 255.0

# Normalising using keras
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

def display_digit(digit, label):
    image = digit.reshape([28,28])
    plt.title('Label: {}'.format(label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_digit_by_index(index):
    display_digit(x_train[index], y_train[index])

def display_mult_flat(start, stop):
    images = x_train[start].reshape([1,784])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Can also be done as a list of layers as an arg to the model
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test, y_test)

l = model.layers[1].get_weights()
network_weights = l[0]
network_biases = l[1]

output_weights = np.transpose(network_weights)

for i in range(10):
    plt.subplot(2, 5, i+1)
    weights = output_weights[i]
    plt.title(i)
    plt.imshow(weights.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

plt.show()


