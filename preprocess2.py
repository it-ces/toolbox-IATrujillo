# Preprocessing Module

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np





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




def normal_noNormal(df, numeric):
    df = df[numeric].copy()
    nonormal, normal = [],[]
    for t in df.columns:
            n, p = stats.shapiro(df[t])
            if p<0.05:
                nonormal.append(t)
            else: 
                normal.append(t)
    return normal, nonormal


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




import pandas as pd

def check_dates(date, date_format):
    if pd.isna(date):
        return False

    try:
        pd.to_datetime(date, format=date_format, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def tab_check_dates(df, dates, date_format="%d/%m/%Y"):
    if isinstance(dates, str):
        dates = [dates]

    result = {}

    for col in dates:
        valid = df[col].map(lambda x: check_dates(x, date_format=date_format))
        result[col] = valid.mean()

    return pd.DataFrame.from_dict(
        result,
        orient="index",
        columns=["pct_correct_format"]
    )

class categ_str(BaseEstimator, TransformerMixin):
    def __init__(self, min_percent=0.01):
        self.min_percent = min_percent

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        # MUY IMPORTANTE:
        # pasar a object para evitar error con pandas Categorical
        for col in X_df.columns:
            X_df[col] = X_df[col].astype("object")

            mask = X_df[col].notna()
            X_df.loc[mask, col] = X_df.loc[mask, col].astype(str)

        self.frequent_categories_ = {}

        for col in X_df.columns:
            freqs = X_df[col].value_counts(normalize=True, dropna=True)
            frequent = freqs[freqs >= self.min_percent].index.tolist()
            self.frequent_categories_[col] = set(frequent)

        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        for col in X_df.columns:
            # MUY IMPORTANTE:
            # pasar a object antes de asignar "Another"
            X_df[col] = X_df[col].astype("object")

            # Convertir a string sin convertir NaN en "nan"
            mask = X_df[col].notna()
            X_df.loc[mask, col] = X_df.loc[mask, col].astype(str)

            frequent = self.frequent_categories_[col]

            mask_rare = X_df[col].notna() & ~X_df[col].isin(frequent)
            X_df.loc[mask_rare, col] = "Another"

        return X_df
    
    