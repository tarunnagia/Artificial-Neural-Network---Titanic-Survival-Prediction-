#importing the libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("titanic_dataset.csv")
#print(dataset)
#print(x)
dataframe = pd.DataFrame(dataset)
print(dataframe)
CSV = dataframe.iloc[:,4:12]
dff = CSV.fillna('24')
#dataframe to new CSV
dff.to_csv(r'final.csv')


