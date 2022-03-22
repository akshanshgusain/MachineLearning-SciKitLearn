from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

"""
       0 Iris-Setosa
       1 Iris-Versicolour
       2 Iris-Virginica
"""


def my_classifier_1():
    # Loading the Datasets
    iris = datasets.load_iris()

    features = iris.data
    labels = iris.target

    # Training the classifier
    clf = KNeighborsClassifier()
    clf.fit(features, labels)

    # Making Predictions
    predic = clf.predict([[1, 1, 1, 1]])
    print(predic)


if __name__ == "__main__":
    my_classifier_1()
