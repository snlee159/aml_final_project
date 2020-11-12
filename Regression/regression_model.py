import sys
from project.global_config import GlobalConfig
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


def train_regression_model(X_train, y_train, X_test, y_test,
                           type=GlobalConfig.LINEAR_REG_TYPE):

    # train model
    print('Train regression model')
    if type == GlobalConfig.LINEAR_REG_TYPE:
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)
    elif type == GlobalConfig.LASSO_REG_TYPE:
        reg_model = Lasso()
        reg_model.fit(X_train, y_train)
    elif type == GlobalConfig.RIDGE_REG_TYPE:
        reg_model = Ridge()
        reg_model.fit(X_train, y_train)
    else:
        print('Enter relevant type for regression model')
        sys.exit(0)

    # evaluate model
    print('Evaluate regression model')
    predictions = reg_model.predict(X_test)
    score = reg_model.score(X_test, y_test)
    print('Score of', type, 'Regression Model: ', score)



