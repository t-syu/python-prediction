import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RF

train_x_first = pd.read_csv("train_x.csv")
train_y_first = pd.read_csv("train_y.csv")

train = pd.merge(train_x_first, train_y_first, on = "お仕事No.", how= "inner")

train = train.dropna(axis = 1).reset_index(drop=True)
train = train.drop(columns=train.select_dtypes(include="object").columns)

y = train["応募数 合計"]
x = train.drop(["お仕事No.", "応募数 合計"], axis=1)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=0)

rfr = RF(random_state=0)

rfr.fit(train_x, train_y)

# pickling the model
pickle_out = open("rfr.pkl", "wb")
pickle.dump(rfr, pickle_out)
pickle_out.close()
