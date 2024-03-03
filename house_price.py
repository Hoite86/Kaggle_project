import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 

train_input = pd.read_csv('/home/hoite/Downloads/house_project/train.csv')
train_input.head()
 
test_input = pd.read_csv('/home/hoite/Downloads/house_project/test.csv')
test_input.head()

x_train = train_input.drop(columns=['SalePrice'])
y_train = train_input['SalePrice']


categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder=('passthrough'))

x_train_encoded = preprocess.fit_transform(x_train)

imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train_encoded)

linear_regression = LinearRegression()
linear_regression.fit(x_train_imputed, y_train)

x_test_encoded = preprocess.transform(test_input)
x_test_imputed = imputer.transform(x_test_encoded)

y_pred = linear_regression.predict(x_test_imputed)

submission_df = pd.DataFrame({'Id': test_input['Id'], 'SalePrice': y_pred})


submission_df["SalePrice"] = submission_df["SalePrice"].apply(lambda x: round(x, 2)) 
submission_df['SalePrice'] = submission_df['SalePrice'].apply(lambda x: max(0,x))

submission_df.to_csv('/home/hoite/Downloads/house_project/Submission_linear_regression.csv')  
