'''
Break the image apart into four "corners", then stitch it back together again.
'''
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np

# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/../lesson3/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)
height, width, depth = raw_image_data.shape

start = tf.placeholder("int32", [3])
size = tf.placeholder("int32", [3])
part = tf.slice(raw_image_data, start, size)

with tf.Session() as session:
    # top left
    top_left = session.run(part, feed_dict={start: [0,0,0], size: [height/2, width/2, depth]})
    plt.imshow(top_left)
    plt.show()

    #top right
    top_right = session.run(part, feed_dict={start: [0,width/2+1,0], size: [height/2, width/2, depth]})
    plt.imshow(top_right)
    plt.show()

    #bottom left
    bottom_left = session.run(part, feed_dict={start: [height/2,0,0], size: [height/2, width/2, depth]})
    plt.imshow(bottom_left)
    plt.show()

    #bottom right
    bottom_right = session.run(part, feed_dict={start: [height/2,width/2+1,0], size: [height/2, width/2, depth]})
    plt.imshow(bottom_right)
    plt.show()

    # stitch
    top_half = np.concatenate([top_left, top_right],axis=1)
    bottom_half = np.concatenate([bottom_left, bottom_right],axis=1)
    whole = np.concatenate([top_half, bottom_half], axis=0)
    plt.imshow(whole)
    plt.show()
