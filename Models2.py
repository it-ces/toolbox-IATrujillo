# Models
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import f1_score  #ytrue, ypred
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


def help():
    print('Models with grid search and stratified K-folds...')


# Support Vector Machine
def grid_SVC(X_train, y_train, performance_metric='f1', resultsGrid=False):
    model = SVC()
    C = np.linspace(0.000001 , 1000, 5)
    kernels = ['poly', 'rbf']
    gamma = ['scale']
    grid = dict(C = C, kernel = kernels, gamma = gamma)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring=performance_metric,error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    if resultsGrid==True:
        return grid_result.cv_results_
    else:
        return  grid_result.best_estimator_
    

# To add in pipeline
class GridSearchSVC(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = grid_SVC(X, y)  
        return self

    def hypers(self):
        params = self.model.get_params()  
        return params

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)  






# K-Nearest Neighborhood
def grid_KNN(X_train, y_train, performance_metric='f1', resultsGrid=False):
    model = KNeighborsClassifier()
    n_neighbors = np.arange(1,10,1) # from K to N in steps of z
    weights = ["uniform", "distance"]
    grid = dict(n_neighbors = n_neighbors, weights = weights)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring=performance_metric,error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    if resultsGrid==True:
        return grid_result.cv_results_
    else:
        return  grid_result.best_estimator_


# To add in pipeline
class GridSearchKNN(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        self.model = grid_KNN(X, y)
        return self
    def hypers(self):
        params = self.model.best_estimator_.get_params()
        return params
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)


# Logistic Regression
def grid_lr(X_train, y_train):
    model = LogisticRegression(random_state=666, max_iter=1000)
    solvers = ['liblinear']
    penalty = ['l2','l1']
    c_values = [ 10, 1.0, 0.1, 0.01, 0.001, ]
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring='roc_auc',error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    return  grid_result.best_estimator_



# To add in pipeline
class GridSearchLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = grid_lr(X, y)  
        return self

    def hypers(self):
        params = self.model.get_params()  
        return params

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)



def grid_RandomForest(X_train, y_train):
  model = RandomForestClassifier(random_state=0)
  n_estimators = np.arange(2,10,1)
  criterion = ['gini', 'entropy', 'log_loss']
  min_samples_split = [0.05, 0.1,]
  max_depth = [2,3,4]
  grid = dict(n_estimators = n_estimators, criterion = criterion,  
              min_samples_split = min_samples_split, max_depth = max_depth)
  cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
  grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                            scoring='roc_auc',error_score='raise')
  grid_result = grid_search.fit(X_train, y_train)
  return  grid_result.best_estimator_

class GridSearchRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = grid_RandomForest(X, y)  
        return self

    def hypers(self):
        params = self.model.get_params()  
        return params

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)



def grid_Adaboost(X_train, y_train):
    model = AdaBoostClassifier(random_state=1)
    n_estimators = [2, 15, 35, 50, 70, 100]
    learning_rate = np.linspace(0.01, 1, 10)
    grid = dict(n_estimators=n_estimators, learning_rate=learning_rate)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                               scoring='roc_auc', error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    return grid_result.best_estimator_

class GridSearchAdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = grid_Adaboost(X, y)  
        return self

    def hypers(self):
        params = self.model.get_params()  
        return params

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# Adding specificity....

def especificityF(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

especificity = make_scorer(especificityF, greater_is_better=True)


###################################################################
# Plot in gridSearchCV   
####################################################################


def meet(fix_parameters, array):
    # Return true if the conditions are meet
    counter = 0
    for param in fix_parameters:
        if array[param] != fix_parameters[param]:
            counter = counter + 1
    return not bool(counter)

def meet_indexes(results_, fix_parameters):
    # input results_ of GridsearchCV
    indexes = []
    for ith, array in enumerate(results_['params']):
        if meet(fix_parameters, array):
            indexes.append(ith)
    return indexes



def Plot_parameter_score(results_, fix_parameters, x_axis):
    # results_ is the results of gridsearch....
    # return 
    # How works in fix_parameters you specify in a dictionary what hyperparamters are hold 
    # while change one...
    indices = meet_indexes(results_, fix_parameters)
    X = [i[x_axis] for i in np.array(results_['params'])[indices]]
    return X, results_['mean_test_score'][indices]



def mean_scores_kfold(classifier,X_,y_):
  X =  X_.reset_index(drop=True)
  y = y_.reset_index(drop=True)
  X = X.to_numpy()
  rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
  accuracys,  recalls, f1s =  [],[],[]
  for itrain, itest in rskf.split(X,y):
    classifier.fit(X[itrain], y[itrain])
    preds = classifier.predict(X[itest])
    accuracys.append(accuracy_score(y[itest], preds))
    recalls.append(recall_score(y[itest], preds))
    f1s.append(f1_score(y[itest], preds))
  return np.array(accuracys).mean(), np.array(recalls).mean(), np.array(f1s).mean()
