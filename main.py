import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


def my_regression():
    # Loading build-in datasets from scikitlearn
    diabetes = datasets.load_diabetes()

    # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename',
    # 'data_module'])
    # Taking 2nd feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Taking all the features
    diabetes_X = diabetes.data

    # Features:
    diabetes_X_train = diabetes_X[:-30]
    diabetes_X_test = diabetes_X[-30:]

    # Labels:
    diabetes_Y_train = diabetes.target[:-30]
    diabetes_Y_test = diabetes.target[-30:]

    model = linear_model.LinearRegression()

    # Data Fitting
    model.fit(diabetes_X_train, diabetes_Y_train)

    # Prediction
    diabetes_Y_predicted = model.predict(diabetes_X_test)

    # Calculating mean squared Error
    print(f'Mean Squared Error: {mean_squared_error(diabetes_Y_test, diabetes_Y_predicted)}')

    print(f'Model weights/coefficient: {model.coef_}')
    print(f'Model intercept: {model.intercept_}')

    # Plotting
    # plt.scatter(diabetes_X_train, diabetes_Y_train)
    # # Plotting line
    # plt.plot(diabetes_X_test, diabetes_Y_predicted)
    #
    # plt.show()


if __name__ == '__main__':
    my_regression()
