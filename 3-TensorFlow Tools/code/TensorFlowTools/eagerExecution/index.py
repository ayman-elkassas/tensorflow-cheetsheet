import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# eager execution for using tensor-flow ops without building a graph with session
# and that is not use in eager execution

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.compat.v1.encode_base64("hello world"))

print(tf.square(2) + tf.square(3))