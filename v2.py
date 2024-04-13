import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('data/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


#create and train model
model = XGBRegressor(n_estimators = 500,early_stopping_rounds = 5, learning_rate = 0.05, n_jobs=5)
model.fit(X_train, y_train,
          eval_set=[(X_valid, y_valid)],
          verbose = False)

prediction = model.predict(X_valid)

print(mean_absolute_error(prediction, y_valid))
