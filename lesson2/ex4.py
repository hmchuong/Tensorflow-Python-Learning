import tensorflow as tf
import numpy as np

total = tf.Variable(0, name='total')
count = tf.Variable(0, name='count')


model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        count = count + 1
        total = (total + np.random.randint(1000))/count
        print(session.run(total))
