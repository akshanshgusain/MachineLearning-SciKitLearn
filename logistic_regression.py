from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np


# Train a Logistic Regression Classifier to pridict whether a flower is Iris-Virginica or not
def my_regressor():
    # Load dataset
    iris = datasets.load_iris()

    x = iris["data"][:, 3:]
    y = (iris["target"] == 2).astype(np.int64)  # Check for IRIS-Virginica

    # Train the LRC
    lrc = LogisticRegression()
    lrc.fit(x, y)

    # test
    prediction_1 = lrc.predict(([[1.6]]))
    print(prediction_1)

    prediction_2 = lrc.predict(([[2.6]]))
    print(prediction_2)

    prediction_3 = lrc.predict(([[3.6]]))
    print(prediction_3)


if __name__ == "__main__":
    my_regressor()