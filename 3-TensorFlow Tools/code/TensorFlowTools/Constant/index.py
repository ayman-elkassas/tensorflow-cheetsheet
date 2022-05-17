import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x

# constant may be single value or tensor_1d or 2d or .....

a = tf.constant(3)
b = tf.constant(3, tf.float32)
c = tf.constant([2, 5, 6, 9, 6])
d = tf.constant([[2, 5, 6],
                 [8, 5, 8],
                 [8, 8, 1]])
e = tf.constant('I love TensorFlow')

with tf.compat.v1.Session() as session:
    print(session.run(a))
    print(session.run(b))
    print(session.run(c))
    print(session.run(d))
    print(session.run(e))
