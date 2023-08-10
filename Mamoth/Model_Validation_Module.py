# %%
import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import random
import math

# %%
def plot_error_dist(actual,predictions, model_type, case=''):
    """Prints the error distribution plot
    Parameters
    --------
    actual : pd.Series
        A Series with the actual class of each prediction
        
    predictions : pd.Series
        A Series with the predicted class of each prediction
        
    model_type : str
        Could be 'class_regression' or 'mosquito_regression' or 'classification'
    
    case: str, optional
        Title of the plot (default= '')
    """
    error = np.abs(actual-predictions).tolist()
    if model_type == 'classification' or model_type == 'class_regression':
        bins = np.arange(len(actual.unique())) - 0.5
        plt.hist(error, bins)
        plt.xticks(range(len(actual.unique())))
    else:
        plt.hist(error)
    plt.xlabel('abs(error)')
    plt.title('Error Distribution \n' + case)
    plt.show()

# %%
def plot_error_per_class(actual,predictions, case=''):
    """Prints the error distribution per class plot
    Parameters
    --------
    actual : pd.Series
        A Series with the actual class of each prediction
        
    predictions : pd.Series
        A Series with the predicted class of each prediction
        
    model_type : str
        Could be 'class_regression' or 'mosquito_regression' or 'classification'
        
    case: str, optional
        Title of the plot (default= '')
    """
    actual = pd.Series(actual)
    predictions = pd.Series(predictions)
    labels = actual.unique().tolist()
    labels.sort()
    f = []
    mae = mean_absolute_error(actual, predictions)

    for k in labels:
        index = actual.loc[actual==k].index
        actual_p = actual.iloc[index]
        predictions_p = predictions.iloc[index]
        mae_class = mean_absolute_error(actual_p, predictions_p)
        f.append(mae_class)
    labels = [str(int(e)) for e in labels]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(labels,f)
    for i, v in enumerate(f):
        ax.text(i, v, str('%.2f'%(v)),rotation=30)
    plt.xlabel('class')
    plt.ylabel('MAE')
    plt.title('MAE per class \n'+case)
    plt.show()
    
    print('-----------|class error-MAE| difference-----------')
    z = np.abs(f-mean_absolute_error(actual, predictions))
    print('mean:',z.mean())
    print('std:',z.std())
    print('coefficient of variation (std/mean):',z.std()/z.mean())
    
    print()
    
    print('----------normalized difference-------------')
    min_val = min(z)
    max_val = max(z)
    z = (z - min_val) / (max_val-min_val)
    print('mean:',z.mean())
    print('std:',z.std())
    print('coefficient of variation (std/mean):',z.std()/z.mean())

# %%
def plot_error_per_month(df, case=''):
    """Prints the error per month
    Parameters
    --------
    df : dataframe
        A dataframe containing the data
    
    case: str, optional
        Title of the plot (default= '')
    """
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    labels = (df['dt_prediction'].dt.month).unique()
    labels.sort()
    labels = [str(e) for e in labels]
    f = df.groupby(by=[df['dt_prediction'].dt.month])['abs(error)'].mean().values
    ax.bar(labels,f)
    for i, v in enumerate(f):
         ax.text(i, v, str('%.2f'%(v)),rotation=30)
    plt.xlabel('Month')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error per month \n' + case)
    plt.show()

# %%
def scatter_plot_error(actual, predictions, case=''):
    """Prints the error in relation with the distance of point from the train region
    Parameters
    --------
    actual : pd.Series
        A Series with the actual class of each prediction
        
    predictions : pd.Series
        A Series with the predicted class of each prediction
    
    case: str, optional
        Title of the plot (default= '')
    """
    # choose the input and output variables
    plt.scatter(actual, np.abs(actual-predictions))
    plt.xlabel('Mosquito target')
    plt.ylabel('Error')
    plt.title('Scatterplot of error \n' + case)
    plt.show()

# %%
def plot_hist(actual,predictions, model_type, case=''):
    """Prints the histogram of the actual values and the predicted values
    Parameters
    --------
    actual : pd.Series
        A Series with the actual class of each prediction
        
    predictions : pd.Series
        A Series with the predicted class of each prediction
        
    model_type : str
        Could be 'class_regression' or 'mosquito_regression' or 'classification'
    
    case: str, optional
        Title of the plot (default= '')
    """
    plt.figure(figsize=(10,8))
    if model_type == 'classification' or model_type == 'class_regression':
        bins = np.arange(len(actual.unique())+1)-0.5
        plt.hist(actual, bins=bins, alpha=0.5, label='actual')
        plt.hist(predictions, bins=bins, alpha=0.5, label='prediction')
        plt.xticks(range(len(actual.unique())))
    else:
        plt.hist(actual, alpha=0.5, label='actual')
        plt.hist(predictions, alpha=0.5, label='prediction')
    plt.legend()
    plt.title('Histogram of actual vs predicted values \n'+case)
    plt.show()

# %%
def plot_error_per_group(actual,prediction,case=''):
    """Prints the error distribution plot
    Parameters
    --------
    actual : pd.Series
        A Series with the actual class of each prediction
        
    predictions : pd.Series
        A Series with the predicted class of each prediction
        
    model_type : str
        Could be 'class_regression' or 'mosquito_regression' or 'classification'
    
    case: str, optional
        Title of the plot (default= '')

    """
    test = {'mosq_bins(t+1)':actual,'predictions':prediction}
    test = pd.DataFrame(test)
    test['classes'] = pd.cut(x=test['mosq_bins(t+1)'], bins=[-1, 100, 200, 300, 400, 500, np.inf],
                      labels=['0-100', '101-200', '201-300', '301-400', '401-500', '500<'])
    labels = test['classes'].unique().tolist()
    labels.sort()
    f = []
    length = []
    for k in labels:
        cc = test.loc[test['classes']==k]
        length.append(len(cc))
        actual = cc.loc[:,'mosq_bins(t+1)']
        predictions = cc.loc[:,'predictions']
        f.append(mean_absolute_error(actual, predictions))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(labels,f)
    for i, v in enumerate(f):
        ax.text(i, v, str('%.2f'%(v)),rotation=30)
        ax.text(i, v/2,'n = '+ str(length[i]),weight="bold",ha='center')
    plt.xlabel('group')
    plt.ylabel('MAE')
    plt.title('MAE per group \n'+case)
    plt.show()

# %%
def error_cdf(actual,prediction, case=''):
    """Prints the cdf of errors
    Parameters
    --------
    actual : pd.Series
        A Series with the actual class of each prediction
        
    predictions : pd.Series
        A Series with the predicted class of each prediction
        
    case: str, optional
        Title of the plot (default= '')

    """
    error = np.abs(actual-prediction)
    
    a = np.sort(error.unique())
    b = np.array(error)
    cdf = np.zeros(len(a))
    for k, val in enumerate(a):
        mask_d = b <= val
        cdf[k] = mask_d.sum()/ len(b)
    plt.plot(a,cdf)
    plt.grid()
    plt.xlabel('abs(error)')
    plt.ylabel('CDF')
    plt.title('CDF of error \n' + case)
    plt.show() 
    
    b = np.sort(error)
    a = np.arange(1,len(error)+1) 
    cdf = np.zeros(len(a))
    for k, val in enumerate(b):
        cdf[k] = b[k]
    plt.plot(a,cdf)
    plt.grid()
    plt.xlabel('Number of samples')
    plt.ylabel('Error')
    plt.title('CDF of error \n' + case)
    plt.show()

# %%
def metrics(train, test, threshold=3):
    """Calculates the perfomance of the model on train and test set
    Parameters
    --------
    train : Dataframe
        A Dataframe with the actual and the predicted values on the train set
        
    test : Dataframe
        A Dataframe with the actual and the predicted values on the train set
        
    threshold: int, optional
        A threshold to calculate percentage of error < threshold (default= 3)

    """    
    print('MAE on train set: ', mean_absolute_error(train['actual'], train['prediction']))

    print('min prediction:',min(train['prediction']))
    print('max prediction:',max(train['prediction']))
    
    print()

    print('MAE on test set: ', mean_absolute_error(test['actual'], test['prediction']))
    perc = ((np.abs(np.array(test['actual'])-np.array(test['prediction'])) < (threshold+0.5)).mean())*100
    print('Error ⩽ '+str(threshold)+':',"%.2f"%perc,'%')

    print('min prediction:',min(test['prediction']))
    print('max prediction:',max(test['prediction']))

# %%
def validation_plots(test, model_type, case=''):
    """Prints plots about the performance of the model on the test set
    
    Parameters
    --------        
    test : Dataframe
        A Dataframe with the actual and the predicted values on the train set
    
    model_type : str
        Could be 'class_regression' or 'mosquito_regression' or 'classification'
        
    case: str, optional
        Title of the plot (default= '')
    """
    if model_type == 'classification' or model_type == 'class_regression':
        plot_error_per_class(test['actual'],test['prediction'], case)
    else:
        plot_error_per_group(test['actual'],test['prediction'], case)
        error_cdf(test['actual'],test['prediction'], case)
    scatter_plot_error(test['actual'],test['prediction'], case)
    plot_error_dist(test['actual'],test['prediction'], model_type, case)
    plot_hist(test['actual'],test['prediction'], model_type, case)


