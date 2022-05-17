import tensorflow as tf
tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x

# constant may be single value or tensor_1d or 2d or .....

a = tf.Variable(3)
b = tf.Variable(3,tf.float32)
c = tf.Variable([2,5,6,9,6])
d = tf.Variable([[2,5,6],[8,5,8], [8,8,1]])
e = tf.Variable('I love TensorFlow')

with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())
    print(session.run(a))
    print(session.run(b))
    print(session.run(c))
    print(session.run(d))
    print(session.run(e))