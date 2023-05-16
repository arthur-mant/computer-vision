import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# import some data to play with
iris = datasets.load_iris()

iris_df = pd.DataFrame(data = iris.data)
iris_df['target'] = iris.target

iris_df.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']
print ("DESCRIBING PANDAS DF")
print (iris_df.describe())


X_reduced = PCA(n_components=3).fit_transform(iris.data)
X_red =  pd.DataFrame(data = X_reduced)
X_red['target'] = iris_df['target']

x = np.array(X_red.iloc[:,0:3]) #extract our features
y = np.array(X_red['target']) # extract our target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
pca_pred = knn.predict(x_test)
acc_pca = accuracy_score(y_test,pca_pred) * 100
print ("ACCURACY WITH KNN AND PCA", acc_pca)

