import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
train_data=pd.read_csv('train.csv')
print(train_data)

y=train_data.SalePrice
print(y)
'''
0       208500
1       181500
2       223500
3       140000
4       250000
         ...
1455    175000
1456    210000
1457    266500
1458    142125
1459    147500
Name: SalePrice, Length: 1460, dtype: int64

'''
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
train_data=train_data[features]
print(train_data)
'''
 LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
0        8450       2003       856       854         2             3             8
1        9600       1976      1262         0         2             3             6
2       11250       2001       920       866         2             3             6
3        9550       1915       961       756         1             3             7
4       14260       2000      1145      1053         2             4             9
...       ...        ...       ...       ...       ...           ...           ...
1455     7917       1999       953       694         2             3             7
1456    13175       1978      2073         0         2             3             7
1457     9042       1941      1188      1152         2             4             9
1458     9717       1950      1078         0         1             2             5
1459     9937       1965      1256         0         1             3             6

[1460 rows x 7 columns]

'''
test_data=pd.read_csv('test.csv')
test_data=test_data[features]
print(test_data)
'''
LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
0       11622       1961       896         0         1             2             5
1       14267       1958      1329         0         1             3             6
2       13830       1997       928       701         2             3             6
3        9978       1998       926       678         2             3             7
4        5005       1992      1280         0         2             2             5
...       ...        ...       ...       ...       ...           ...           ...
1454     1936       1970       546       546         1             3             5
1455     1894       1970       546       546         1             3             6
1456    20000       1960      1224         0         1             4             7
1457    10441       1992       970         0         1             3             6
1458     9627       1993       996      1004         2             3             9

[1459 rows x 7 columns]

'''


X_train=pd.get_dummies(train_data)
print(X_train)
'''
LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
0        8450       2003       856       854         2             3             8
1        9600       1976      1262         0         2             3             6
2       11250       2001       920       866         2             3             6
3        9550       1915       961       756         1             3             7
4       14260       2000      1145      1053         2             4             9
...       ...        ...       ...       ...       ...           ...           ...
1455     7917       1999       953       694         2             3             7
1456    13175       1978      2073         0         2             3             7
1457     9042       1941      1188      1152         2             4             9
1458     9717       1950      1078         0         1             2             5
1459     9937       1965      1256         0         1             3             6

[1460 rows x 7 columns]

'''

X_test=pd.get_dummies(test_data)
print(X_test)
'''
LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
0       11622       1961       896         0         1             2             5
1       14267       1958      1329         0         1             3             6
2       13830       1997       928       701         2             3             6
3        9978       1998       926       678         2             3             7
4        5005       1992      1280         0         2             2             5
...       ...        ...       ...       ...       ...           ...           ...
1454     1936       1970       546       546         1             3             5
1455     1894       1970       546       546         1             3             6
1456    20000       1960      1224         0         1             4             7
1457    10441       1992       970         0         1             3             6
1458     9627       1993       996      1004         2             3             9

[1459 rows x 7 columns]

'''

train_X, valid_X, train_y, val_y=train_test_split(X_test,y, random_state=1)
print(train_X)
'''
  LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
6       10084       2004      1694         0         2             3             7
807     21384       1923      1072       504         1             3             6
955      7136       1946       979       979         2             4             8
1040    13125       1957      1803         0         2             3             8
701      9600       1969      1164         0         1             3             6
...       ...        ...       ...       ...       ...           ...           ...
715     10140       1974      1350         0         2             3             7
905      9920       1954      1063         0         1             3             6
1096     6882       1914       773       582         1             3             7
235      1680       1971       483       504         1             2             5
1061    18000       1935       894         0         1             2             6

[1095 rows x 7 columns]

'''


print(valid_X)
'''
 LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
258     12435       2001       963       829         2             3             7
267      8400       1939      1052       720         2             4             8
288      9819       1967       900         0         1             3             5
649      1936       1970       630         0         1             1             3
1233    12160       1959      1188         0         1             3             6
...       ...        ...       ...       ...       ...           ...           ...
1017     5814       1984      1360         0         1             1             4
534      9056       2004       707       707         2             3             6
1334     2368       1970       765       600         1             3             7
1369    10635       2003      1668         0         2             3             8
628     11606       1969      1040      1040         1             5             9

[365 rows x 7 columns]
'''


clf=tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(train_X, train_y)
predictions=clf.predict(valid_X)
mean_abs_error=mean_absolute_error(val_y, predictions)

output=pd.DataFrame({
    'id':test_data.Id,
    'SalePrice':predictions
})

output.to_csv('submission.csv',index=False)