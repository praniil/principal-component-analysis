import standardize_dataset as sd 
import calcualte_eigen_values as pc

iris_dataset, len_iris_dataset = sd.standardize_dataset()
principal_component = pc.get_principal_components()

print(len_iris_dataset)
print(len(principal_component))


def transform(iris_dataset, principal_component):
    X = iris_dataset.copy()
    X_proj = X.dot(principal_component.T)
    return X_proj

# transforming the iris dataset from four dimensions to three dimensions

X_transform = transform(iris_dataset, principal_component)

print(X_transform)

