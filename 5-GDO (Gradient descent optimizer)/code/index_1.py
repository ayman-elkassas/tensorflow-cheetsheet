import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# the equation is  : 4x+2 = 0
unknownvalue = tf.compat.v1.Variable(0.0) # is x

a = tf.compat.v1.constant(4.0)
b = tf.compat.v1.constant(2.0)

c = tf.compat.v1.multiply(unknownvalue,a)  # 4x
equation  = tf.compat.v1.add(c,b) # 4x+2

zerovalue = tf.compat.v1.constant(0.0)
diff = tf.compat.v1.square(equation-zerovalue) # differnce is : 4x+2  -  0 

solving = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(diff)
# 0.01 is learning rate
init = tf.compat.v1.global_variables_initializer()

session = tf.compat.v1.Session()
session.run(init)

for n in range(1000):
    session.run(solving)

print(session.run(unknownvalue))