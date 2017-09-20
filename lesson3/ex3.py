'''
Flip (top-bottom)
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
    x = tf.reverse_sequence(x, np.ones((width,)) * height, seq_dim=0, batch_dim=1)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()
