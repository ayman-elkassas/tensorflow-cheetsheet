import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split

# tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

# todo: 1- read data

data = pd.DataFrame(pd.read_csv("Car_Data.csv"))
data.head(100)

# print('buying data is : ' , data['buying'].unique())
# print('maint data is : ' , data['maint'].unique())
# print('doors data is : ' , data['doors'].unique())
# print('persons data is : ' , data['persons'].unique())
# print('lug_boot data is : ' , data['lug_boot'].unique())
# print('safety data is : ' , data['safety'].unique())
# print('rating data is : ' , data['rating'].unique())

# todo: 2- dealing with number is faster than string and if we build NN should be numeric so, cont.
# todo: if feature have determine string values so can convert it to number

buying_mapping = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
# note: if there a value vhigh and v high then it will be consider as the same value map to 0
maint_mapping = {'vhigh': 0, 'v high': 0, 'high': 1, 'med': 2, 'low': 3}
doors_mapping = {'2': 0, '3': 1, '4': 2, '5more': 3}
persons_mapping = {'2': 0, '4': 1, 'more': 2}
lug_boot_mapping = {'small': 0, 'med': 1, 'big': 2}
safety_mapping = {'low': 0, 'med': 1, 'high': 2}
rating_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}

data['buying'] = data['buying'].map(buying_mapping)
data['maint'] = data['maint'].map(maint_mapping)
data['doors'] = data['doors'].map(doors_mapping)
data['persons'] = data['persons'].map(persons_mapping)
data['lug_boot'] = data['lug_boot'].map(lug_boot_mapping)
data['safety'] = data['safety'].map(safety_mapping)
data['rating'] = data['rating'].map(rating_mapping)

# print(data.head(30))

# todo: 3- split data

X_train, X_test, y_train, y_test = train_test_split(
    data[["buying", "maint", "doors", "persons", "lug_boot", "safety"]].values, data["rating"].values, random_state=42)

print('X_train shape is ', X_train.shape)
print('X_test shape is ', X_test.shape)
print('y_train shape is ', y_train.shape)
print('y_test shape is ', y_test.shape)

# todo: 4- create DNN model

# todo: 4-1- determine input feature layer, I have 6 feature then I have to create 6 neuron
# dimension of input layer is 6 (for 6 feature)

feature_columns = [tf.compat.v1.contrib.layers.real_valued_column("", dimension=6)]

# todo: 4-2- Build 3 hidden layer DNN with 512, 256, 128 units respectively.
# :and with practice NN number of units in hidden layers should be n^2 from higher to lower count
# I have 4 output values, then number of classes should be 4
# and inject activation function with hyperparameter
classifier = tf.compat.v1.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[512, 256, 128],
                                            n_classes=4,
                                            optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
                                                learning_rate=0.15,
                                                l1_regularization_strength=0.001
                                            ))

# todo: 4-3- train model

# Define the training inputs
def get_train_inputs():
    x = tf.compat.v1.constant(X_train)
    y = tf.compat.v1.constant(y_train)
    return x, y

# Define the test inputs
def get_test_inputs():
    x = tf.compat.v1.constant(X_test)
    y = tf.compat.v1.constant(y_test)

    return x, y

# todo: 4-4- fit model

# Fit model to training data
classifier.fit(input_fn=get_train_inputs, steps=1200)

# todo:5- evaluate model

# Evaluate accuracy to get predictions and compare with test data and get accuracy
accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
print("Test Accuracy: {0:f}".format(accuracy_score))  