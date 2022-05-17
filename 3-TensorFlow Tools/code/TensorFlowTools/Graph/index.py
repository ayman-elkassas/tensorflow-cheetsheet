import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x

graph = tf.compat.v1.get_default_graph()
# print(graph.get_operations())

a = tf.compat.v1.constant(5)

b = tf.compat.v1.constant(7)

c = tf.compat.v1.add(a, b)

d = tf.compat.v1.placeholder(tf.float32)
e = tf.compat.v1.Variable(2)

for operation in graph.get_operations():
    print(operation.name)
