import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x

x = tf.compat.v1.constant([5, 6, 7, 8])
y = tf.compat.v1.constant(10)
z = tf.compat.v1.add(x, y)
z1 = tf.multiply(x, y)

# subtract
# divide
# pow
# matmul (for matrix multiplication)
# matrix_transpose

with tf.compat.v1.Session() as session:
    result = session.run(z)
    result2 = session.run(z1)
    print(result, result2)
