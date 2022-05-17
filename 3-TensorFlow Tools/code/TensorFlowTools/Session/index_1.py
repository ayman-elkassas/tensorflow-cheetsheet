import tensorflow as tf
tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x

# Launch the graph in a session.
sess = tf.compat.v1.Session()
a = tf.constant(5.0)

print(sess.run(a))
print(a)