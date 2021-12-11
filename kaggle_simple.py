import pandas as pd
from sklearn.ensemble import RandomForestRegressor


train_data=pd.read_csv('train.csv')
y=train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X=train_data[features]
test_data=pd.read_csv('test.csv')
print(test_data)
X_test=test_data[features]



clf=RandomForestRegressor()
clf.fit(X,y)
preds=clf.predict(X_test)


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds})
output.to_csv('submission.csv', index=False)