import pandas as pd
import numpy as np
df = pd.read_csv("D:/r python/machine learning/data/Bike Share Demand/train.csv",
                 parse_dates=['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday']=df['datetime'].dt.weekday

df['season'] = df['season'].astype('category')
df['weather'] = df['weather'].astype('category')
df.drop(columns=['datetime','casual', 'registered'],inplace=True)

dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.drop('count',axis=1)
y = dum_df['count']

##########################    linear regression  ################
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred[y_pred<0] = 0
print("Root Mean Squared Log Error  = %6.4f" % np.sqrt(mean_squared_log_error(y_test, y_pred)))

###################### SVR defalt  ##############
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVR
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
y_pred[y_pred<0] = 0
print("Root Mean Squared Log Error  = %6.4f" % 
      np.sqrt(mean_squared_log_error(y_test, y_pred)))



######## Tunning ##################
from sklearn.model_selection import GridSearchCV

C_range = np.array([0.01,0.05,0.1,1,1.5,1.7,2,4])

param_grid = dict( C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, train_size=2, test_size=None, random_state=42)
svmGrid = GridSearchCV(SVR(kernel='linear'), 
                       param_grid=param_grid, cv=5)
svmGrid.fit(X, y)

# Best Parameters
print(svmGrid.best_params_)
print(svmGrid.cv_results_)
print(svmGrid.best_score_)

################################ Random Forest ###########
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=2019,
                                  n_estimators=500,oob_score=True)
model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)
y_pred[y_pred<0] = 0
print("Root Mean Squared Log Error  = %6.4f" % 
      np.sqrt(mean_squared_log_error(y_test, y_pred)))

####################### XGboost
from xgboost import XGBClassifier

clf = XGBClassifier()
clf.fit(X_train,y_train,verbose=True)

y_pred = clf.predict(X_test)
