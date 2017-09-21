'''
Convert image to grayscale
'''
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "../lesson3/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder(tf.int32, [None, None, 3])

# Reduce axis 2 by mean (= color)
# i.e. image = [[[r,g,b], ...]]
# out = [[[ grayvalue ], ... ]] where grayvalue = mean(r, g, b)
out = tf.reduce_mean(image, 2, keep_dims=True)

# Associate r,g,b to the same mean value = concat mean on axis 2.
# out = [[[ grayvalu, grayvalue, grayvalue], ...]]
out = tf.concat([out, out, out], 2)
out = tf.cast(out, tf.uint8)

with tf.Session() as session:
    result = session.run(out, feed_dict={image: raw_image_data})

print(result.shape)
plt.imshow(result)
plt.show()
