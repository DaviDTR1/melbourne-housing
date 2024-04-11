import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

X_full = pd.read_csv('data/melb_data.csv')

#remove missing target
X_full.dropna(axis=0, subset=['Price'], inplace=True)
y = X_full.Price
X_full.drop(['Price'], axis=1, inplace=True)

#split train data
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2 , random_state=0)

categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                        X_train[cname].dtype == "object"]

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

cols = categorical_cols + numerical_cols
X_train = X_train[cols]
X_valid = X_valid[cols]

#preprocessing numerical data
numerical_transformer = SimpleImputer(strategy='constant')

#preproccesing categorical data
categorical_transformer  = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#bundle preprocessing for numerical ant categorical data    
preproccesor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=0)

my_pipeline = Pipeline(steps=[('preprocessor', preproccesor),
                              ('model', model)
                              ])

# preprocessing of training data 
my_pipeline.fit(X_train,y_train)

pred = my_pipeline.predict(X_valid)
mae = mean_absolute_error(pred, y_valid)

print(mae) 

