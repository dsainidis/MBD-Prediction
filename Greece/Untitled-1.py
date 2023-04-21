# %%
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
import sys

sys.path.insert(0, "E:/Google Drive/NOA/MBD-Prediction/Modules")

pd.set_option('display.max_columns', None)
random.seed(0)
np.random.seed(0)

from utils import *

# %%
NUTS0 = 'GR'
NUTS2 = 'Attica'

# %%
f = open("E:/Google Drive/NOA/MBD-Prediction/Greece/data/GR_WNV_Attica_cases_dates.txt", "r")
print(f.read())

# %%
r_u = 0.05
near_u = 0.25
smote = 1
scaler = 'minmax'
#years_to_remove = [year for year in range(2012, 2018)]
years_to_remove = []
path = "E:/Google Drive/NOA/MBD-Prediction/Greece/data/GR_Attica_WNV_Dataset_2010_2021.csv"

# %%
dataset = read_data(path, exclude_years = years_to_remove)
dataset.head()

# %%
#Check months

dataset = dataset.query('month >= 5 & month <= 10')
dataset.reset_index(inplace=True, drop=True)

# %%
dataset

# %%
dsc = dataframe_describe(dataset)
dsc.head(15)

# %%
#plot_imbalance(dataset)

# %%
coefficients = []
results_train = pd.DataFrame()
results_test = pd.DataFrame()
feature_names = dataset.select_dtypes(exclude=['object']).drop(columns = ['case']).columns

# %%
for train_idx, test_idx in yearCV_split(dataset):
    data_train = dataset.iloc[train_idx]
    data_test = dataset.iloc[test_idx]
    
    data_train.reset_index(inplace = True, drop = True)
    data_test.reset_index(inplace = True, drop = True)
    
    X_train = data_train.select_dtypes(exclude=['object']).drop(columns = ['case'])
    y_train = data_train['case']
    X_test = data_test.select_dtypes(exclude=['object']).drop(columns = ['case'])
    y_test = data_test['case']
    
    X_train, X_test, y_train, y_test, X_train_df = transform_data(X_train, X_test, y_train, y_test, r_u, near_u, smote, scaler_key=scaler)
    
    w0,w1 = calculate_weights(y_train)

    model = LogisticRegression(max_iter = 1000, class_weight = {0.0: w0, 1.0:w1}, n_jobs = -1, random_state = 0)
    
    train, test, coef = train_and_predict(model, X_train_df, data_test, X_train, y_train, X_test, y_test)
    
    results_train = results_train.append(train, ignore_index = True)
    results_test = results_test.append(test, ignore_index = True)
    coefficients.append(coef)
    
    print(f"Year: {test['year'].iloc[0]} || w0:{w0}, w1:{w1} || ##")

# %%
# plot_feature_importance(coefficients, feature_names, top = 15, figure_size = (8, 4))

# # %%
# plot_trend_curve(results_test, list(dsc['case']))

# # %%
# plot_probability_curve(results_test)

# # %%
# plot_roc_curve(results_test)

# # %%
# plot_pr_curve(results_test, plot_fbeta = True)

# %%
operational, cumulative, avg, weighted = evaluate_operational(results_test, k = 5, prob_threshold = -1)
print(f'Average: {avg*100:.2f}%\nWeighted Average: {weighted*100:.2f}%')

# %%
operational

# %%
cumulative

# %%
report_train, report_test = classification_report(results_train, results_test, threshold = 0.2, beta = 4)

# %%
report_test.head(15)

# %%



