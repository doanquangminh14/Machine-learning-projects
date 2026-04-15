from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

import pandas as pd
df = pd.read_csv('dataset/Marketing.csv')

def split_data():
    X=df[['Adm','Marketing','RnD']]
    y=df.Profit
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    return X_train,X_test,y_train,y_test 



def lire(X_train,X_test,y_train,y_test):
    model_lire = LinearRegression()
    model_lire.fit(X_train,y_train)
    y_pred = model_lire.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    return r2    

def deci_R(X_train,X_test,y_train,y_test):
    r_2 = []
    for i in range(2,20):
        model_deci = DecisionTreeRegressor(max_depth= i)
        model_deci.fit(X_train,y_train)
        y_pred = model_deci.predict(X_test)
        r_2.append(r2_score(y_test,y_pred))
    max_value = r_2.index(max(r_2)) + 2
    model_deci = DecisionTreeRegressor(max_depth= max_value)
    model_deci.fit(X_train,y_train)
    y_pred = model_deci.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    return r2
        
def random_fr_R(X_train,X_test,y_train,y_test):
    rf = RandomForestRegressor()
    rf_space = {
                'max_depth':[5,10,15,20],
                'n_estimators':[10,30,50],
                'criterion':['squared_error','absolute_error']
    }
    model_rf = GridSearchCV(rf, param_grid=rf_space, scoring= 'r2')
    model_rf.fit(X_train,y_train)
    y_pred = model_rf.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    return r2

def knn_R(X_train,X_test,y_train,y_test):
    r_2 = []
    for i in range(2,20):
        model_knn = KNeighborsRegressor(n_neighbors= i)
        model_knn.fit(X_train, y_train)
        y_pred = model_knn.predict(X_test)
        r_2.append(r2_score(y_test,y_pred))
    max_value = r_2.index(max(r_2)) + 2
    model_knn = KNeighborsRegressor(n_neighbors= max_value)
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    return r2

def sup_ve_R(X_train,X_test,y_train,y_test):
    svc_space = {
                 'C':[0.01,0.1,1,10,100],
                 'gamma':[0.01,0.1,1,10,100],
                 'kernel':['linear']
    }
    model_spv = GridSearchCV(SVR(),svc_space)
    model_spv.fit(X_train,y_train)
    y_pred = model_spv.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    return r2

if __name__ == "__main__":
  X_train, X_test, y_train, y_test = split_data()
  print("Linear Regression Accuracy:", lire(X_train, X_test, y_train, y_test))
  print("Decision Tree R Accuracy:", deci_R(X_train, X_test, y_train, y_test))
  print("Random Forest R Accuracy:", random_fr_R(X_train, X_test, y_train, y_test))
  print("KNeighborsRegressor Accuracy:", knn_R(X_train, X_test, y_train, y_test))
 # print("SVR Accuracy:", sup_ve_R(X_train, X_test, y_train, y_test))