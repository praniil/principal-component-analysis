from os import wait
import standardize_dataset as sd 
import numpy as np

iris_features, len_iris_dataset = sd.standardize_dataset()
print(len_iris_dataset)

# covariance_matrix
def covariance_matrix(X, n_rows):
    covariance_matrix = sum(X[i].reshape(-1, 1) @ X[i].reshape(1, -1) for i in range(n_rows)) / n_rows
    return covariance_matrix

# derive eigen vectors and eigen values of the covariance covariance_matrix
def compute_eigen_vectors(X, n_rows):
    C = covariance_matrix(X, n_rows)
    eigen_values, eigen_vectors = np.linalg.eig(C)
    return eigen_values, eigen_vectors

eigen_values, eigen_vectors = compute_eigen_vectors(iris_features, len_iris_dataset)

print(eigen_values)
print(eigen_vectors)

# sort the eigen values and eigen vectors
sort = np.argsort(eigen_values)[::-1]

# use sort array to order eigen_vectors
principal_components = eigen_vectors[:, sort]

print(principal_components)



