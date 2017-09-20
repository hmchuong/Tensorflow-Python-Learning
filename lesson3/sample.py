'''
Read image and tranpose, turning the image 90 degrees counter-clockwise
'''
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
image = mpimg.imread(filename)

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2]) # transpose, swapping the axes 0 and 1 around, axis 2 stays where it is
	session.run(model)
	result = session.run(x)

plt.imshow(result)
plt.show()
