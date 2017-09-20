'''
Flip (left-right)
'''
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    '''
    Iterate through the array according to batch_dim. Setting batch_dim=0 means we go through the rows (top to bottom).
    For each item in the iteration
    Slice a second dimension, denoted by seq_dim. Setting seq_dim=1 means we go through the columns (left to right).
    '''
    x = tf.reverse_sequence(x, np.ones((height,)) * width, seq_dim=1, batch_dim=0)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()
