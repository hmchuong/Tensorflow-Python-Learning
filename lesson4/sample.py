'''
Placeholder is simple a variable that we will assign data to at a later date
It allows us to create our operations and build our computation graph, without needing the data
In Tensorflow, we then feed data into the graph through these placeholders
'''
import tensorflow as tf

# i.e, a place in memory where we will store float value later no
x = tf.placeholder("float", None)
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)
