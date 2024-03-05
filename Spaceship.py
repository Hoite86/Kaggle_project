
import pandas as pd
import os

import matplotlib.pyplot as plt
#print(plt.style.available)
plt.style.use("seaborn-v0_8-whitegrid")
import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/home/hoite/project/space_ship'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('/home/hoite/project/space_ship/train.csv')
test_data = pd.read_csv('/home/hoite/project/space_ship/test.csv')


#Train data
train_data
train_data.columns
train_data.head(10)
train_data.tail(10)
train_data.describe()
train_data.info()

#average data
train_data.drop(columns=['Name'], inplace=True)
train_data["RoomService"] = train_data['RoomService'].fillna(train_data['RoomService'].mean())
train_data['FoodCourt'] = train_data['FoodCourt'].fillna(train_data['FoodCourt'].mean())
train_data['ShoppingMall'] = train_data['ShoppingMall'].fillna(train_data['ShoppingMall'].mean())
train_data['VRDeck'] = train_data['VRDeck'].fillna(train_data['VRDeck'].mean())
train_data['Spa'] = train_data['Spa'].fillna(train_data['Spa'].median())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

train_data[['VIP', 'CryoSleep', 'Destination', 'HomePlanet']] = train_data[['VIP', 'CryoSleep', 'Destination', 'HomePlanet']].fillna(value = 'Other')

train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', expand=True)
train_data.head()

train_data[['Deck', 'Side']] = train_data[['Deck', 'Side']].fillna(value='X')

train_data.drop(columns=['Cabin', 'Num'], inplace=True)
train_data.head()

df = pd.get_dummies(train_data, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side'], dtype=float)

df.head()



# Normalize 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

columns_to_normalize = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

scaler = MinMaxScaler()

df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df.head()

X = df.drop(columns=['Transported'])
Y = df['Transported']



X.head()

test_data.isnull().sum()

test_data.drop(columns=['Name'], inplace=True)
test_data['RoomService'] = test_data['RoomService'].fillna(test_data['RoomService'].mean())
test_data['FoodCourt'] = test_data['FoodCourt'].fillna(test_data['FoodCourt'].mean())
test_data['ShoppingMall'] = test_data['ShoppingMall'].fillna(test_data['ShoppingMall'].mean())
test_data['VRDeck'] = test_data['VRDeck'].fillna(test_data['VRDeck'].mean())
test_data['Spa'] = test_data['Spa'].fillna(test_data['Spa'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

test_data[['VIP', 'CryoSleep', 'Destination', 'HomePlanet']] = test_data[['VIP', 'CryoSleep', 'Destination', 'HomePlanet']].fillna(value='Other')

test_data[['Deck', 'Num', 'Side']] = test_data['Cabin'].str.split('/', expand=True)
test_data.head()

test_data[['Deck', 'Side']] = test_data[['Deck', 'Side']].fillna(value='X')

test_data.drop(columns=['Cabin', 'Num'], inplace=True)
test_data.head()

df_test = pd.get_dummies(test_data, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side' ], dtype=float)

df_test.head()

from sklearn.preprocessing import MinMaxScaler

# Normalization
columns_to_normalize = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

scaler = MinMaxScaler()


df_test[columns_to_normalize] = scaler.fit_transform(df_test[columns_to_normalize])
df_test.head()


model = RandomForestClassifier(n_estimators=350, max_depth=10, random_state=1)
model.fit(X, Y)  # Make sure 'X' does not contain 'Transported'
predictions = model.predict(df_test)


submission = pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Transported' : predictions})
submission.to_csv('/home/hoite/project/space_ship/submission.csv', index=False)
print("Your Submission is saved!!")