# import pandas as pd
import openpyxl
import numpy as np

# how to install sklearn on m1 https://github.com/scikit-learn/scikit-learn/issues/19137#issuecomment-945890439
from sklearn.linear_model import LinearRegression

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html example
# https://ithelp.ithome.com.tw/articles/10186905

# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = np.dot(X, np.array([1, 2])) + 3
# X = np.array([0, 1, 2])
# y = np.array([1, 2, 3])

temperatures = np.array([0, 1, 2])
iced_tea_sales = [1, 2, 3]  # <- list ot numpy array np.array([1, 2, 3])


lm = LinearRegression()

X = np.reshape(
    temperatures, (len(temperatures), 1)
)  # or list of list [[0], [1], [2]] !!
y = iced_tea_sales  # <- array works. array of array: np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)) works too.

# X2 = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y2 = np.dot(X2, np.array([1, 2])) + 3  # [6, 8, 9, 11] array

# reg = lm.fit(X, y)
reg = lm.fit(
    X,
    y,
)

score = reg.score(X, y)  # 1.0
coef = reg.coef_  # 1
intercept = reg.intercept_  #  1

to_be_predicted = np.array([30])
predicted_sales = lm.predict(
    np.reshape(to_be_predicted, (len(to_be_predicted), 1))
)  # 31, yes

print("done")

# todo list
# 1. read excel
# 2. plot excel
# 3. plot fitting line
