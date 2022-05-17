import tensorflow as tf
tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x

a= tf.constant(8)

with tf.compat.v1.Session() as session:
    print(session.run(a))