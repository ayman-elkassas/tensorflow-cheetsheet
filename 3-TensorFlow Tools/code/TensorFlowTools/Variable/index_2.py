import tensorflow as tf
tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x

# tf.Variable can be changed unless constant

price = tf.Variable(20)
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
print('original price is : ',sess.run(price))

price=tf.compat.v1.assign(price, price + 90)
print('original price is : ',sess.run(price))
