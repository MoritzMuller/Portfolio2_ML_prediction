''' 
Create a ML model to predict the rating of a beer based on the other features. Try various ML models and compare the results. Do some feature engineering and parameter tuning to improve the results.
'''
#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv('data/beer_profile_and_ratings.csv')
data = data[data['number_of_reviews'] >= 5]

# Split the data into X (features) and y (target)
X = data.iloc[:, [8,
                  9,
                  10,
                  11,
                  12,
                  13,
                  14,
                  15,
                  16,
                  17,
                  18]]
y = data['review_overall']

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a function that evaluates the performance of the model and saves the results to performance
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    r2 = 100 * r2_score(y_test, predictions)
    mse=mean_squared_error(y_test, predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('R2 = {:0.2f}%.'.format(r2))
    print('MSE = {:0.2f}.'.format(mse))
    return accuracy, r2, mse



# Create a function that fits the model and evaluates the performance
def fit_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    return evaluate(model, X_test, y_test)

# Create a function that fits the model, evaluates the performance and plots the residuals
def fit_evaluate_plot(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    r2 = 100 * r2_score(y_test, predictions)
    mse=mean_squared_error(y_test, predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('R2 = {:0.2f}%.'.format(r2))
    print('MSE = {:0.2f}.'.format(mse))
    plt.scatter(y_test, errors)
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.show()


## Use various ML models to predict the rating of a beer based on the other features
# Create a dataframe to store the performance of the models
performance = pd.DataFrame(columns=['Accuracy', 'R2', 'MSE'])
# Start with a simple linear regression model
base_model = LinearRegression()
base_plot = fit_evaluate_plot(base_model, X_train, y_train, X_test, y_test)
#save the performance of the model to the performance dataframe
performance.loc['Linear Regression', :] = evaluate(base_model, X_test, y_test)

#Show predicted vs actual values
predictions = base_model.predict(X_test)
pred_true = pd.DataFrame({'True Values': y_test, 'Predictions': predictions})
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# Out of the box, the accuracy of a linear regression model is not bad, 92.44%. The residual plot shows that the model is not perfect, but it is not too bad either. 
# We can see that the model is off at the tail ends of the distribution, which is expected. Given that it is a linear regression model, the model will be biased to reduce the errors in the ratings range with the highest density - after all it aims to reduce the residuals.
# Therefore, predictions are best in the range between 3.5 and 4.

# Let's try a model that is not biased towards the mean - a decision tree
tree = DecisionTreeRegressor()
tree_plot = fit_evaluate_plot(tree, X_train, y_train, X_test, y_test)
# errors are spread more evenly, but the model is not as accurate as the linear regression model
#save the performance of the model to the performance dataframe
performance.loc['Decision Tree', :] = evaluate(tree, X_test, y_test)

# Let's try a random forest model
forest = RandomForestRegressor()
forest_plot = fit_evaluate_plot(forest, X_train, y_train, X_test, y_test)
# we are getting better results, but the model is still biased towards the mean - but results are better than the linear regression model
#save the performance of the model to the performance dataframe
performance.loc['Random Forest', :] = evaluate(forest, X_test, y_test)

# Let's try a KNN model
knn = KNeighborsRegressor()
knn_plot = fit_evaluate_plot(knn, X_train, y_train, X_test, y_test)
# the model is not as accurate as the random forest model, but close
#save the performance of the model to the performance dataframe
performance.loc['KNN', :] = evaluate(knn, X_test, y_test)

# Let's try a SVM model
svm = SVR()
svm_plot = fit_evaluate_plot(svm, X_train, y_train, X_test, y_test) 
# we are getting better. Still, errors are largest at the tail ends of the distribution
#save the performance of the model to the performance dataframe
performance.loc['SVM', :] = evaluate(svm, X_test, y_test)

# Lets try XGBoostRegressor for a last check
import xgboost as xgb
xgb_model = xgb.XGBRegressor()
xgb_plot = fit_evaluate_plot(xgb_model, X_train, y_train, X_test, y_test)
#save the performance of the model to the performance dataframe
performance.loc['XGBoost', :] = evaluate(xgb_model, X_test, y_test)

# create an empty plot with 3 subplots and fill the subplots with the plots of the models: accuracy, R2 and MSE
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
performance['Accuracy'].plot(kind='bar', ax=ax1)
performance['R2'].plot(kind='bar', ax=ax2)
performance['MSE'].plot(kind='bar', ax=ax3)
ax1.set_title('Accuracy')
ax2.set_title('R2')
ax3.set_title('MSE')
plt.show()

# The best model is the SVM model, as it comes with the highest accuracy and R2 and the lowest MSE.

# Let's try to improve the SVM model by tuning the hyperparameters
# Create a function that automatically tunes the hyperparameters of the SVM model
def tune_svm(X_train, y_train, X_test, y_test):
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'epsilon': [0.1, 0.01, 0.001, 0.0001]
    }
    # Create a based model
    svm = SVR()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 3)
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    # Evaluate the model
    evaluate(grid_search, X_test, y_test)
    return grid_search

# create a tuning function that uses randomized search to find the best hyperparameters

param_grid = {
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma' : ['scale', 'auto']
}
# Create a based model
svm = SVR()
# Instantiate the random search model
random_search = RandomizedSearchCV(estimator = svm, param_distributions = param_grid, n_iter = 50, cv = 3, verbose=3, random_state=42, n_jobs = -1, scoring='neg_mean_squared_error')
# Fit the grid search to the data
random_search.fit(X_train, y_train)
# Evaluate the model
evaluate(random_search, X_test, y_test)
# what are the parameters of the best model?
random_search.best_params_ # conveniently, the standard parameters of the SVM model are the best parameters

# Can a neural network do better?
# Create a neural network model to predict the ratings
from keras.models import Sequential 
from keras.layers import Dense

# create a neural network to predict the ratings

model = Sequential()
# add the input layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# add a hidden layer
model.add(Dense(32, activation='relu'))
# add the output layer
model.add(Dense(1, activation='linear'))
# compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# fit the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
# evaluate the model
model.evaluate(X_test, y_test)

# vizualise model architecture
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# calculate model performance manually, because keras format is different from sklearn
predictions = model.predict(X_test)
errors = abs(predictions[:,0] - y_test)
mape = 100 * np.mean(errors / y_test)
accuracy = 100 - mape
r2 = 100 * r2_score(y_test, predictions)
mse=mean_squared_error(y_test, predictions)
print('Model Performance')
print('Average Error: {:0.4f} rating points.'.format(np.mean(errors)))
print('Accuracy = {:0.2f}%.'.format(accuracy))
print('R2 = {:0.2f}%.'.format(r2))
print('MSE = {:0.2f}.'.format(mse))




    



