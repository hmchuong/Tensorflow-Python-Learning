import tensorflow as tf
import numpy as np

data = np.random.randint(1000, size=10000)

x = tf.constant(data, name='x')
y = tf.Variable(5*x*x - 3*x + 15, name='y')


model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))
