import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x

# placeholder is a value that is not yet known assign it later

a = tf.compat.v1.placeholder(tf.float32)
b = a*2

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
z = tf.compat.v1.placeholder(tf.float32)
w = tf.compat.v1.placeholder(tf.float32)

n = x*2 + 3*y**4 - 4/z + w-9

p1 = tf.compat.v1.placeholder(tf.float32)
data_target = p1**2

with tf.compat.v1.Session() as session:
    print(session.run(b, feed_dict={a: 3}))
    result = session.run(n, feed_dict={x: 3, y: 4, z: 8, w: 6})
    print(result)
    print(session.run(data_target, {p1: list(range(5, 100))}))

# most of the time we use placeholder to feed data from CSVs or other sources
