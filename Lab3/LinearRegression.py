import matplotlib.pyplot as plotter
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

filter = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'median_income', 'median_house_value']
dataset = pd.read_csv('C:\\Users\\shiva\\PycharmProjects\\MachineLearningLab\\Lab3\\california_housing.csv',
                      skipinitialspace = True, usecols = filter)
dataset = clean_dataset(dataset)

# Loading the dataset for the feature matrix X
#filter = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'median_income']

# Creating the feature matrix X
X = np.array(dataset.iloc[:, 0:6])
print(X)

#Loading the dataset for the response vector y
#filter = ['median_house_value']

# Creating the response vector Y
y = np.array(dataset.iloc[:, 6])
print(y)

# Splitting X and Y into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Creating a linear regression object
linear_regression_obj = linear_model.LinearRegression()

# Training the model
linear_regression_obj.fit(X_train, y_train)

# Getting the regression coefficients
print('The regression coefficients are')
print(linear_regression_obj.coef_)

# Getting the variance score
print('Variance score: {}'.format(linear_regression_obj.score(X_test, y_test)))

#plotting
reg = linear_regression_obj
plt = plotter
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

## plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

## plotting legend
plt.legend(loc='upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()








