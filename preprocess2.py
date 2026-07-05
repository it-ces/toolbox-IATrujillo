# Preprocessing Module

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


def reduce_categories(X, min_percent):
    # if the relative share is lesser than min_percent put the label 'Another'
    X_transformed = X.copy()
    for var in X_transformed.columns:
        cats = X_transformed[var].value_counts(normalize=True).to_dict()
        to_replace = [key for key in cats if cats[key]<min_percent and key!='nan']  
         # Convert categories that not appear at least minpercent to <Another>
        X_transformed[var] = X_transformed[var].replace(to_replace=to_replace, value='Another')
    return X_transformed
      
class reduceCategories(BaseEstimator, TransformerMixin):
    def __init__(self, min_percent=0.05):
        self.min_percent = min_percent

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        return reduce_categories(pd.DataFrame(X_transformed), self.min_percent)

 

# To use reduce_categories in a pipeline!!




# Drop columns with NaN's
def drop_colNaN(df_, min_percent):
    # If the column have lesser of threshold complete data 
    # will be drop of the dataset...
    # threshold = 1,indicates that keep only variables without nans...
    df = df_.copy()
    N  = df.shape[0] 
    keep = [var for var in df.columns if df[var].notnull().sum()/N >= min_percent]
    return df[keep].to_numpy()


# Drop columns to uses in a PipeLine!!!
class drop_ColumnsNan(BaseEstimator, TransformerMixin):
    def __init__(self, min_percent=0.8):
        self.min_percent = min_percent

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        return drop_colNaN(pd.DataFrame(X_transformed), self.min_percent)



def val_to_str(val):
    if val==np.nan:
        pass
    else:
        val  = str(val)
    return val

def cats_to_str(X, y=None):
    X_cat = X.copy()
    for col in X_cat: 
        X_cat[col] = X_cat[col].apply(lambda x: val_to_str(x))
    return X_cat


class categ_str(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return cats_to_str(pd.DataFrame(X))


############################ Preprocess...
    
# Algortihm Table One

from scipy import stats

def classify_vars(df):
    categorial, nonormal, normal = [],[],[]
    for t in df.columns:
        if df[t].dtypes=="object" or df[t].dtypes.name=='category':
            categorial.append(t)
        if df[t].dtypes=="int64" or df[t].dtypes=="float64":
            n,p = stats.shapiro(df[t])
            if p<0.05:
                nonormal.append(t)
            else: 
                normal.append(t)
    return categorial, nonormal, normal


#### Adding more functions...


# This could help us to identify where a numerical variable
# is suspect of be categorical...
def is_float(num):
    num = str(num)
    flag  = False
    for index  in range(len(num)-1):
        if num[index]==',' or num[index]=='.' and num[index+1]!='0':
            flag = True
    return flag

def InferCategorical(df,var):
    Flag = True
    floats = df[var].apply(lambda x: is_float(x)).sum()
    range  = df[var].max() - df[var].min()
    if floats>10 and range>10:
        Flag=False
    return Flag


def is_binary(df_, nums):
    variables = []
    for var in nums:
        flag = True
        unique = df_[var].unique()
        for value in unique:
            if value not in [0, 1, np.nan, 0.0, 1.0]:
                flag = False
        if flag == True:
            variables.append(var)
    return variables

