import pandas as pd 
import numpy as np 
import torch.nn as nn

iris_dataset = pd.read_csv('../dataset/IRIS.csv')
print(len(iris_dataset))

iris_features = iris_dataset.drop(columns=['species']) 

#standardixze dataset
iris_features = (iris_features - iris_features.mean()) / iris_features.std()

#convert dataframe to numpy array
iris_features = iris_features.numpy()

print(iris_features)


