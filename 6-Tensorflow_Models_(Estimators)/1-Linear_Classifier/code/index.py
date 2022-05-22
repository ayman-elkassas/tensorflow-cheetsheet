import tensorflow as tf
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


tf.compat.v1.disable_eager_execution()

# todo:1-Read data

cwd = os.getcwd()

census = pd.read_csv(f'{cwd}/data.csv', skiprows=1, header=None)

census.columns = ['age', 'workclass', 'education', 'education_num', 'marital_status',
                  'occupation', 'relationship', 'race', 'gender', 'capital_gain',
                  'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']

# print(census.head(10))

# print(census['income_bracket'].unique()) //to get classes name

# convert income_bracket to label 0 or 1


def label_fix(label):
    if label == ' <=50K':
        return 0
    else:
        return 1


census['income_bracket'] = census['income_bracket'].apply(label_fix)
# print(census['income_bracket'].unique())

# print(census.head(10))

# todo:2-Split data

x_data = census.drop('income_bracket', axis=1)
y_labels = census['income_bracket']

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_labels, test_size=0.3, random_state=101)

# todo:3-Determine type foreach column feature for tensorflow

# types of features (numerical, categorical, boolean)

# numerical features
age = tf.compat.v1.feature_column.numeric_column("age")
education_num = tf.compat.v1.feature_column.numeric_column("education_num")
capital_gain = tf.compat.v1.feature_column.numeric_column("capital_gain")
capital_loss = tf.compat.v1.feature_column.numeric_column("capital_loss")
hours_per_week = tf.compat.v1.feature_column.numeric_column("hours_per_week")

# categorical features with vocabulary list (feature that well know values ex. gender)
gender = tf.compat.v1.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["Female", "Male"])

# categorical features with hash bucket (string feature that can be any value and make range can be variant in like 1000 possible value)
occupation = tf.compat.v1.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
marital_status = tf.compat.v1.feature_column.categorical_column_with_hash_bucket(
    "marital_status", hash_bucket_size=1000)
relationship = tf.compat.v1.feature_column.categorical_column_with_hash_bucket(
    "relationship", hash_bucket_size=1000)
education = tf.compat.v1.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
workclass = tf.compat.v1.feature_column.categorical_column_with_hash_bucket(
    "workclass", hash_bucket_size=1000)
native_country = tf.compat.v1.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

# todo:4-combine all feature in one list

feat_cols = [gender, occupation, marital_status, relationship, education, workclass,
             native_country, age, education_num, capital_gain, capital_loss, hours_per_week]

# todo:5-create input function for inject input_data and input_label(target)

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

# todo:6-create tf model(estimator) object and set model feature columns

model = tf.compat.v1.estimator.LinearClassifier(feature_columns=feat_cols)

# todo:7-start train model and inject input_func (that describe data and target) and num of epochs

model.train(input_fn=input_func, steps=100)

# todo:8-create input function for test data

pred_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_test, batch_size=len(X_test), shuffle=False)

# todo:9-predict for test data and convert to list
predictions = list(model.predict(input_fn=pred_fn))
print(predictions)

# todo:10-get classification report

final_predictions = []
for pred in predictions:
    final_predictions.append(pred['class_ids'][0])

print(classification_report(y_test, final_predictions))

# todo:11-save model
fpr, tpr, thresholds = roc_curve(y_test, final_predictions)
roc_auc = auc(fpr, tpr)
print("ROC AUC Score: {}".format(roc_auc))
plt.figure()
plt.plot(fpr, tpr, color='green', lw=1,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
