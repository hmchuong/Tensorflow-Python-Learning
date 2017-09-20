#  1
import tensorflow as tf
# 2
x = tf.constant(35, name='x')               # create a constant value called x, and give it the numerical value 35
# 3
y = tf.Variable(x + 5, name='y')            # create a Variable called y, and define it as being the equation x + 5
# 4
model = tf.global_variables_initializer()   # init the variables
'''
In this step, a graph is created of the dependencies between the variables.
In this case, the variable y depends on the variable x, and that value is transformed by adding 5 to it.
Keep in mind that this value isn’t computed until step 7, as up until then,
only equations and relations are computed.
'''
# 5
with tf.Session() as session:
    # 6
    session.run(model)                      # Run model
    # 7
    print(session.run(y))                   # Run the variable y and print out its current value
