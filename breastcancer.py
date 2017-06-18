from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.framework import deprecated

import os
import urllib
import easygui
import numpy as np
import tensorflow as tf

# Error logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Data sets
CANCER_TRAINING = "breastcancertraining4.csv"
CANCER_TEST = "breastcancertesting4.csv"


def main():

  # Load datasets
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=CANCER_TRAINING,
      target_dtype=np.float32,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=CANCER_TEST,
      target_dtype=np.float32,
      features_dtype=np.float32)

  # Specify that all features have real-value data and 9 feature columns
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=9)]

  # Build 2 layer DNN with 10, 20 units respectively and 2 classes
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20],
                                             n_classes=2)

  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y

  # Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=100)

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Get user input from doctor
  msg = "Enter your patient's information"	
  title = "Breast Cancer Predictor"
  fieldNames = ["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin","Normal Nucleoli","Mitoses"]
  fieldValues = []  # we start with blanks for the values
  fieldValues = easygui.multenterbox(msg,title, fieldNames)
  
  # make sure that none of the fields was left blank	
  while 1:
  	if fieldValues == None: break
  	errmsg = ""
  	for i in range(len(fieldNames)):
  	   if fieldValues[i].strip() == "":	
	   	  errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
  	if errmsg == "": break # no problems found
  	fieldValues = easyggui.multenterbox(errmsg, title, fieldNames, fieldValues)

  fieldValues2 = [float(i) for i in fieldValues]

  # Classify the inputted data
  def new_samples2():
    return np.array([fieldValues2], dtype=np.float32)

  predictions2 = list(classifier.predict(input_fn=new_samples2))
  
  diagnosis = ""
  if predictions2[0] == 0: diagnosis = "BENIGN\n"
  if predictions2[0] == 1: diagnosis = "MALIGNANT \n"  
  print("Class Prediction: " +diagnosis)
  display = "The breast mass is: " + diagnosis
  display += "\nTest Accuracy: {0:f}\n".format(accuracy_score)
  easygui.msgbox(display, ok_button="Submit patient data")

if __name__ == "__main__":
	print("Breast Cancer Predictor \nCopyright (c) Breast Cancer Answer Inc. 2017 \n")
	print("Analyzing.................................................................")
	main()
