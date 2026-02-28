import pandas as pd 
import numpy as np 


def standardize_dataset():
    iris_dataset = pd.read_csv('../dataset/IRIS.csv')
    print(len(iris_dataset))

    iris_features = iris_dataset.drop(columns=['species']) 

    #standardixze dataset
    iris_features = (iris_features - iris_features.mean()) / iris_features.std()

    #convert dataframe to numpy array
    iris_features = iris_features.to_numpy()

    return iris_features, len(iris_features)


if __name__ == "__main__":
    standardize_dataset()
