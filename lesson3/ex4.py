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
half = tf.slice(x,[0, 0, 0],[height, width/2, depth])
with tf.Session() as session:
    '''
    Iterate through the array according to batch_dim. Setting batch_dim=0 means we go through the rows (top to bottom).
    For each item in the iteration
    Slice a second dimension, denoted by seq_dim. Setting seq_dim=1 means we go through the columns (left to right).
    '''
    session.run(model)
    half2 = session.run(tf.reverse_sequence(half, np.ones((height,)) * width/2, seq_dim=1, batch_dim=0))
    half1 = session.run(half)
    result = session.run(tf.pack([half1, half2],axis=1))

print(result.shape)
plt.imshow(result)
plt.show()
