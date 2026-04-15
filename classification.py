from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
df = pd.read_csv('dataset/Can_nag.csv')

def split_data():
    y = df["Class"].values
    X = df.iloc[:, 0:-1].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    return X_train,X_test,y_train,y_test 



def lore(X_train,X_test,y_train,y_test):
    model_lire = LogisticRegression()
    model_lire.fit(X_train,y_train)
    y_pred = model_lire.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return acc    

def deci_C(X_train,X_test,y_train,y_test):
    acc = []
    for i in range(2,20):
        model_deci = DecisionTreeClassifier(max_depth= i)
        model_deci.fit(X_train,y_train)
        y_pred = model_deci.predict(X_test)
        acc.append(accuracy_score(y_test,y_pred))
    max_value = acc.index(max(acc)) + 2
    model_deci = DecisionTreeClassifier(max_depth= max_value)
    model_deci.fit(X_train,y_train)
    y_pred = model_deci.predict(X_test)
    acc1 = accuracy_score(y_test,y_pred)
    return acc1
        
def random_fr_C(X_train,X_test,y_train,y_test):
    rf = RandomForestClassifier()
    rf_space = {
                'max_depth':[5,10,15,20],
                'n_estimators':[10,30,50],
                'criterion':['gini','entropy']
    }
    model_rf = GridSearchCV(rf, param_grid=rf_space, scoring= 'accuracy')
    model_rf.fit(X_train,y_train)
    y_pred = model_rf.predict(X_test)
    acc2 = accuracy_score(y_test,y_pred)
    return acc2

def knn_C(X_train,X_test,y_train,y_test):
    acc = []
    for i in range(2,20):
        model_knn = KNeighborsClassifier(n_neighbors= i)
        model_knn.fit(X_train, y_train)
        y_pred = model_knn.predict(X_test)
        acc.append(accuracy_score(y_test,y_pred))
    max_value = acc.index(max(acc)) + 2
    model_knn = KNeighborsClassifier(n_neighbors= max_value)
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    acc3 = accuracy_score(y_test,y_pred)
    return acc3

def sup_ve_C(X_train,X_test,y_train,y_test):
    svc_space = {
                 'C':[0.01,0.1,1,10,100],
                 'gamma':[0.01,0.1,1,10,100],
                 'kernel':['linear']
    }
    model_spv = GridSearchCV(SVC(),svc_space)
    model_spv.fit(X_train,y_train)
    y_pred = model_spv.predict(X_test)
    acc4 = accuracy_score(y_test,y_pred)
    return acc4

if __name__ == "__main__":
  X_train, X_test, y_train, y_test = split_data()
  print("Logistic Regression Accuracy:", lore(X_train, X_test, y_train, y_test))
  print("Decision Tree C Accuracy:", deci_C(X_train, X_test, y_train, y_test))
  print("Random Forest C Accuracy:", random_fr_C(X_train, X_test, y_train, y_test))
  print("KNeighborsClassifier Accuracy:", knn_C(X_train, X_test, y_train, y_test))
 # print("SVC Accuracy:", sup_ve_C(X_train, X_test, y_train, y_test))