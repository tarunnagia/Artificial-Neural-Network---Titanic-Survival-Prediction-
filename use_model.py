#importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import numpy
import os
from keras.models import Sequential
#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
#Taking user input
age = input("Enter your age : ")
sex = input("Enter your Gender : ")
sibsp = input("Enter Number of Siblings or Spouse with you : ")
parch = input("Enter Number of Parents or Children with you : ")
pclass = input("Enter Class in Which you were travelling - 1,2 or 3 :")
fare = input("Enter Fare : ")

if sex.lower() == 'male' :
    gender = 1
elif sex.lower() == 'female':
    gender = 0
#Taking user input as dataframe
df = [[gender,age,sibsp,parch,pclass,fare]]
df = pd.DataFrame(df,columns=['Sex','Age','SibSp','Parch','Pclass','Fare'])
#Applying feature Scaling on Input data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df = sc.fit_transform(df)
df = sc.transform(df) 
# Predicting the Test set results using saved ANN model
#loading model
yaml_file = open('titanic_model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("titanic_model.h5")
#Loaded model from disk"

#Final Prediction
y_pred = loaded_model.predict(df)
print("Probability of your survival are: ",y_pred)
y_pred = (y_pred > 0.5)
if y_pred == True:
    print("Yippie!!! you will survive ")
else:
    print("Oh Fuck!!! you will not survive")
