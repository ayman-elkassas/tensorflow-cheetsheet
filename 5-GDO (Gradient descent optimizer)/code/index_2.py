import tensorflow as tf

tf.compat.v1.enable_eager_execution()

(train_images, train_labels), (test_images, test_labels) = \
    tf.compat.v1.keras.datasets.mnist.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

sess = tf.compat.v1.InteractiveSession()
x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 784])
y_ = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 10])
w = tf.compat.v1.Variable(tf.compat.v1.zeros([784, 10], tf.compat.v1.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([10], tf.compat.v1.float32))

sess.run(tf.compat.v1.global_variables_initializer())

y = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(x, w) + b)

# prepare cost function
cost_function = tf.compat.v1.reduce_mean(-tf.compat.v1.reduce_sum(y_*tf.compat.v1.log(y),
                                                                  reduction_indices=[1]))

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cost_function)

for i in range(1000):
    batch = list(train_dataset.batch(50).as_numpy_iterator())
    print(batch[0])
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.compat.v1.equal(tf.compat.v1.argmax(y,  1), tf.compat.v1.argmax(y_, 1))
accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32))
acc = accuracy.eval(feed_dict={x: test_images, y_: test_labels})*100
print('Final accuracy is ', acc)




