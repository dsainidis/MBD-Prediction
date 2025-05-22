# %%
import math
import copy
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import plot_tree
import geopandas as gpd
from Neural_Network_Module import Dataset, transformations_nn, prediction_nn, train_nn
from XGboost_Model_Module import  transformations_xgboost, predict_xgboost, train_xgboost
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import random

# %%
def Average(lst):
    """Calculates the average value of a list
    
    Parameters
    ----------
    lst : list
        A list of numbers
        
    Returns
    ----------
    avg: float
        The average value of the list
    """
    avg = sum(lst) / len(lst)
    return avg

# %%
def plot_error_dist(actual, predictions, model_type, case=''):
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
    if model_type != 'mosquito_regression':
        bins = np.arange(len(actual.unique())) - 0.5
        plt.hist(error, bins)
        plt.xticks(range(len(actual.unique())))
    else:
        plt.hist(error)
    plt.xlabel('abs(error)')
    plt.title('Error Distribution \n' + case)
    plt.show()

# %%
def plot_hist(actual, predictions, model_type, case=''):
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
    if model_type != 'mosquito_regression':
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
def plot_error_per_class(test, case=''):
    """Prints the error distribution per class plot
    Parameters
    --------
    actual : lst
        A list with the actual class of each prediction
        
    case: str, optional
        Title of the plot (Area and mosquito genus) (default= '')
    """
    labels = test.loc[:,'actual'].unique().tolist()
    labels.sort()
    f = []
    length = []
    for k in labels:
        cc = test.loc[test['actual']==k]
        length.append(len(cc))
        actual = cc.loc[:,'actual']
        predictions = cc.loc[:,'prediction']
        mae_class = mean_absolute_error(actual, predictions)
        f.append(mae_class)
    labels = [str(int(e)) for e in labels]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(labels,f)
    for i, v in enumerate(f):
        ax.text(i, v, str('%.2f'%(v)), rotation=30)
        ax.text(i, v/2,'n = '+ str(length[i]), weight="bold", ha='center', rotation=90)
    plt.xlabel('class')
    plt.ylabel('MAE')
    plt.title('MAE per class ' + case)
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

# %%
def plot_error_per_month(df, case=''):
    """Prints the error per month
    Parameters
    --------
    df : dataframe
        A dataframe containing the data
    
    case: str, optional
        Title of the plot (Area and mosquito genus) (default= '')
    """
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    labels = (df['dt_prediction'].dt.month).unique()
    labels.sort()
    labels = [str(e) for e in labels]
    df['abs(error)'] = np.abs(df['actual']-df['prediction'])
    f = df.groupby(by=[df['dt_prediction'].dt.month])['abs(error)'].mean().values
    length = df.groupby(by=[df['dt_prediction'].dt.month])['dt_prediction'].count().values
    ax.bar(labels,f)
    for i, v in enumerate(f):
        ax.text(i, v, str('%.2f'%(v)),rotation=30)
        ax.text(i, v/2,'n = '+ str(length[i]),weight="bold",ha='center', rotation=90)
    plt.xlabel('Month')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error per month ' + case)
    plt.show()

# %%
def scatter_plot_error(actual, prediction, case=''):
    """Prints the error in relation with the distance of point from the train region
    Parameters
    --------
    df : dataframe
        A dataframe containing the data
    
    case: str, optional
        Title of the plot (Area and mosquito genus) (default= '')
    """
    # choose the input and output variables
    x, y = actual, np.abs(actual-prediction)
    plt.scatter(x, y)
    plt.xlabel('Mosquito bins')
    plt.ylabel('Error')
    plt.title('Scatterplot of error ' + case)
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
        
    case: str, optional
        Title of the plot (default= '')

    """
    test = {'mosq_now':actual,'predictions':prediction}
    test = pd.DataFrame(test)
    test['classes'] = pd.cut(x=test['mosq_now'], bins=[-1, 100, 200, 300, 400, 500, np.inf],
                      labels=['0-100', '101-200', '201-300', '301-400', '401-500', '500<'])
    labels = test['classes'].unique().tolist()
    labels.sort()
    f = []
    length = []
    for k in labels:
        cc = test.loc[test['classes']==k]
        length.append(len(cc))
        actual = cc.loc[:,'mosq_now']
        predictions = cc.loc[:,'predictions']
        f.append(mean_absolute_error(actual, predictions))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(labels,f)
    for i, v in enumerate(f):
        ax.text(i, v, str('%.2f'%(v)),rotation=30)
        ax.text(i, v/2,'n = '+ str(length[i]),weight="bold",ha='center', rotation=90)
    plt.xlabel('Mosquito Group')
    plt.ylabel('MAE')
    plt.title('MAE per Mosquito group \n'+case)
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
    plt.figure(figsize=(8,8))
    plt.plot(a,cdf)
    plt.grid()
    plt.xlabel('abs(error)',fontsize=18)
    plt.ylabel('CDF',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title('CDF of error \n' + case)
    plt.show() 
    
    b = np.sort(error)
    a = np.arange(1,len(error)+1) 
    cdf = np.zeros(len(a))
    for k, val in enumerate(b):
        cdf[k] = b[k]
    plt.plot(a,cdf)
    plt.grid()
    plt.xlabel('Number of samples',fontsize=18)
    plt.ylabel('Error',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('CDF of error \n' + case)
    plt.show()

# %%
def metrics(train, test, threshold):
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
    perc = ((np.abs(test['actual']-test['prediction']) < (threshold+0.5)).mean())*100
    print('Error <= '+str(threshold)+':',"%.2f"%perc,'%')

    print('min prediction:',min(test['prediction']))
    print('max prediction:',max(test['prediction']))

# %%
def validation(train, test, model_type, error_buffer, case=''):
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
    if model_type != 'mosquito_regression':
        metrics(train, test, threshold=error_buffer)
        plot_error_per_class(test, case)
    else:
        metrics(train, test, threshold=error_buffer)
        plot_error_per_group(test['actual'],test['prediction'], case)
        error_cdf(test['actual'],test['prediction'], case)
    scatter_plot_error(test['actual'],test['prediction'], case)
    plot_error_dist(test['actual'],test['prediction'], model_type, case)
    plot_hist(test['actual'],test['prediction'], model_type, case)
    plot_error_per_month(test, case)

# %%
def evaluate_nn(model, train, test=None, test_size = 0.2, filepath = '', date_col = 'dt_placement', fi=False, case='', error_buffer=3):
    
    if test is None:
        train, test = train_test_split(train, test_size=0.2, random_state=3)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        
    train = train.drop([date_col],axis=1)    
    date = test[date_col]
    test = test.drop([date_col],axis=1)
        
    mosq_col = train.columns[-1]
    if pd.api.types.is_categorical_dtype(train[mosq_col]):
        max_val = train.iloc[:,-1].cat.codes.max()
    else:
        max_val= train.iloc[:,-1].max()
    
    model_int = copy.deepcopy(model)
    
    features = None
    if fi:
        if model.embedding_data is not None:
            columns = train.drop(columns=model.embedded_data.columns.tolist()).columns[:-1].tolist()
            columns = columns+embedded_data.columns.tolist()
            features = columns
        else:
            features = train.columns[:-1].tolist()
    
    if 'scaling' in model.transformation_list:
        train_X, train_y, test_X, test_y, fitted_scaler = transformations_nn(train, test = test, model_type = model.model_type, 
                                                       embedding_data = model.embedding_data,
                                                       transformation_list = model.transformation_list, test_size=test_size)
    else:
        train_X, train_y, test_X, test_y = transformations_nn(train, test = test, model_type = model.model_type, 
                                                       embedding_data = model.embedding_data,
                                                       transformation_list = model.transformation_list, test_size=test_size)
    training_set = Dataset(train_X, train_y)

    testing_set = Dataset(test_X, test_y)

    results_train, results_test, trained_model = train_nn(model = model_int, train_set = training_set, test_set = testing_set,
                               features=features, max_val = max_val)
    

    test[date_col] = date
    test['dt_prediction'] = test[date_col] + datetime.timedelta(days=15)
    test['prediction'] = results_test['prediction']
    test.loc[test['prediction']<0,'prediction'] = 0
    for col in test.select_dtypes(include='category').columns:
        test[col] = test[col].cat.codes
    test['error'] = test[mosq_col] - test['prediction']
    test['abs(error)'] = np.abs(test[mosq_col] - test['prediction'])
    
    test = test.rename(columns={mosq_col:'actual'})

    validation(results_train, test, model_type=model.model_type, case=case, error_buffer=error_buffer)

    if 'scaling' in model.transformation_list:
        return fitted_scaler, trained_model
    else:
        return None, trained_model

# %%
def give_predictions_nn(model, train, test, env, filepath, date_col='dt_placement', case='', fi=False, export=False):
    """ Trainning of the model
    
    Parameters
    ----------
    model : FeedforwardNeuralNetModel
        A FeedforwardNeuralNetModel model
        
    train : DataFrame
        A Dataframe object with the train set
        
    test : DataFrame
        A Dataframe object with the test set
    
    step : int
        The number of days for prediction
        
    env : boolean
        If true only enviromental features are used
        
    filepath : srt, optional
        The path of the file to export the results
        
    transform_target : boolean, 
        If True, perofrms transformation of the target based on the model_type argument (default = False)
        
    learning_rate : int, optional
        The learning_rate of the training process. (default = None)
        
    epochs : int, optional
        The number of epochs for the training. (default = None)
        
    batch_size : int, optional
        The size of each batch in each iteration. (default = None)
        
    ealry_stop : boolean, optional
        If True, the trainning of the model may stop earlier than the epochs defined. (default = None)
        
    date_col : str, optional
        The name of the date column (default = 'dt_placement')
    
    case : str, optional
        The title of case for the plot (default='')
    
    export : boolean, optional
        Export a csv with the feature importance and a csv with the test data (default=False)
        
    Returns
    ----------        
    output: DataFrame
        A Dataframe containing the actual and the predicted values on the test set

    """    
    date = test[date_col]
    train = train.drop([date_col],axis=1)
    test = test.drop([date_col],axis=1)
    
    mosq_col = train.columns[-1]
    max_val= train.iloc[:,-1].max()
    
    features = None
    if fi:
        if model.embedding_data is not None:
            columns = data.drop(columns=model.embedded_data.columns.tolist()).columns[:-1].tolist()
            columns = columns+embedded_data.columns.tolist()
            features = columns
        else:
            features = data.columns[:-1].tolist()
    
    model_int = copy.deepcopy(model)
    
    train_X, train_y, test_X, test_y = transformations_nn(train, test = test, model_type = model.model_type, 
                                                       embedding_data = model.embedding_data,
                                                       transformation_list = model.transformation_list)

    training_set = Dataset(train_X, train_y)

    testing_set = Dataset(test_X, test_y)
    
    test_predict = prediction_nn(model, training_set, testing_set, features=features, max_val = max_val)
            
            
    test[date_col] = date
    test['dt_prediction'] = test[date_col] + datetime.timedelta(days = 15)
    test['prediction'] = test_predict['prediction']
    test.loc[test['prediction']<0,'prediction'] = 0
    test.loc[test['prediction']>train.iloc[:, -1].max(),'prediction'] = train.iloc[:, -1].max()
    test['entomological_features'] = not(env)
    test['week'] = test['dt_prediction'].dt.isocalendar()['week']
    test = test.sort_values(['dt_prediction'], ascending=True).reset_index(drop=True)
    test = test.drop_duplicates(subset=['week', 'x', 'y'], keep='last').reset_index(drop=True)
    
    classes = test['prediction'].value_counts().sort_index()
    
    print(classes)
    print()
    print('Low risk category stations:',classes[classes.index<2].sum())
    print('Medium risk category stations:',classes[(classes.index>1) & (classes.index<6)].sum())
    print('High risk category stations:',classes[classes.index>5].sum())
    
    output = test[[date_col, 'dt_prediction', 'week', 'x', 'y', 'prediction', 'entomological_features']]
    
    if export:
        csv_name = filepath + case + '.csv'
        output.to_csv(csv_name,index=False)
        geopandas_predictions(output,filepath,case)
    
    return output

# %%
def evaluate_xgboost(model, train, test = None, test_size = 0.2, date_col = 'dt_placement', filepath='', case = '', fi = False, error_buffer=3):
    """Trains a model on random splitted data
    
    Parameters
    --------
    data : dataframe
        A dataframe containing the data
        
    test_df : dataframe
        A dataframe containing the prediction
        
    filepath : srt, optional
        The path of the file to export the results
        
    date_col : str, optional
        The name of the date column (default = 'dt_placement')
        
    case : str, optional
        The title of case for the plot (default='')
    
    """     
    if test is None:
        train, test = train_test_split(train, test_size=0.2, random_state=3)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
    
    del train[date_col]
    date = test[date_col]
    del test[date_col]
    
    if pd.api.types.is_categorical_dtype(train.iloc[:,-1]):
        max_val = train.iloc[:,-1].cat.codes.max()
    else:
        max_val= train.iloc[:,-1].max()
        
    mosq_col = train.columns[-1]
    
    model_int = copy.deepcopy(model)
    
    if 'scaling' in model.transformation_list:
        train_X, train_y, test_X, test_y, fitted_scaler = transformations_xgboost(train, test = test, model_type = model.model_type,
                                                               evaluation=False,
                                                               transformation_list=model.transformation_list,
                                                               embedding_data=model.embedding_data, 
                                                               test_size=test_size)
    else:
        train_X, train_y, test_X, test_y = transformations_xgboost(train, test = test, model_type = model.model_type,
                                                               evaluation=False,
                                                               transformation_list=model.transformation_list,
                                                               embedding_data=model.embedding_data, 
                                                               test_size=test_size)

    results_train, results_test, trained_model = train_xgboost(model_int, train_X, train_y, test_X, test_y, max_val, fi)
        
    
    test[date_col] = date
    test['dt_prediction'] = test[date_col] + datetime.timedelta(days=15)
    test['prediction'] = results_test['prediction']
    test.loc[test['prediction']<0,'prediction'] = 0

    for col in test.select_dtypes(include='category').columns:
        test[col] = test[col].cat.codes
    test['error'] = test[mosq_col] - test['prediction']

    test['abs(error)'] = np.abs(test[mosq_col] - test['prediction'])
    
    test = test.rename(columns={mosq_col:'actual'})
    
    validation(results_train, test, model_type=model.model_type, case=case, error_buffer=error_buffer)
    
    if 'scaling' in model.transformation_list:
        return fitted_scaler, trained_model
    else:
        return None, trained_model

# %%
def give_predictions_xgboost(model, train, test, env, filepath, fi=False, date_col='dt_placement', case='',export=False):
    """Trains a model on random splitted data
    
    Parameters
    --------
    data : dataframe
        A dataframe containing the data
        
    test : dataframe
        A dataframe containing the prediction
    
    features : lst
        A list of features to use
    
    depth : int
        The dpeth of the trees to construct
        
    estimators : int
        The number of trees to construct
        
    env : boolean
        If true only enviromental features are used
        
    filepath : srt, optional
        The path of the file to export the results
        
    date_col : str, optional
        The name of the date column (default = 'dt_placement')
        
    case : str, optional
        The title of case for the plot (default='')
    
    export : boolean, optional
        Export a csv with the feature importance and a csv with the test data (default=False)
    """

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    max_val= train.iloc[:,-1].max()
    mosq_col = train.columns[-1]
#     model_int = copy.deepcopy(model)
    
    del train[date_col]
    date = test[date_col]
    del test[date_col]
    
    
    train_X, train_y, test_X, test_y = transformations_xgboost(train, test = test, model_type = model.model_type,
                                                               evaluation=False,
                                                               transformation_list=model.transformation_list,
                                                               embedding_data=model.embedding_data)

    results_test = predict_xgboost(model, train_X, train_y, test_X, test_y, max_val, fi)
    
    test[date_col] = date
    test['dt_prediction'] = test[date_col] + datetime.timedelta(days=15)
    test['prediction'] = results_test['prediction']
    test['entomological_features'] = not(env)
    test['week'] = test['dt_prediction'].dt.isocalendar()['week']
    test = test.sort_values(['dt_prediction'], ascending=True).reset_index(drop=True)
    test = test.drop_duplicates(subset=['week', 'x', 'y'], keep='last').reset_index(drop=True)
    
    classes = test['prediction'].value_counts().sort_index()
    
    print(classes)
    print()
    print('Low risk category stations:',classes[classes.index<2].sum())
    print('Medium risk category stations:',classes[(classes.index>1) & (classes.index<6)].sum())
    print('High risk category stations:',classes[classes.index>5].sum())
    
    output = test[[date_col, 'dt_prediction', 'week', 'x', 'y', 'prediction', 'entomological_features']]
    
    if export:
        csv_name = filepath + case + '.csv'
        output.to_csv(csv_name,index=False)
        geopandas_predictions(output,filepath,case)
        
    return output

# %%
def merge_predictions(predictions_env, predictions_entom, filepath='',case='', export=False):
    """Integrates the predictions with and without the entomolofical featues
    
    Parameters
    --------
    predictions_env : dataframe
        A dataframe containing the predictions based only on the enviromental features
        
    predictions_env : dataframe
        A dataframe containing the predictions based on the enviromental and entomological features
    
    features : lst
        A list of features to use
        
    filepath : srt, optional
        The path of the file to export the results (default='')
        
    title : str, optional
        The name of the case (default='')
    
    export : boolean, optional
        Exports a csv with the predictions (default=False)
        
    """
    predictions = pd.concat([predictions_entom, predictions_env])
    predictions = predictions.sort_values(['entomological_features'], ascending=False).reset_index(drop=True)
    predictions = predictions.drop_duplicates(subset=['dt_placement', 'x', 'y'], keep='first').reset_index(drop=True)
    predictions = predictions.drop_duplicates(subset=['week', 'x', 'y'], keep='first').reset_index(drop=True)

    # Print risk classes for combined predictions
    classes = predictions['prediction'].value_counts().sort_index()

    print(classes)
    print()
    print('Low risk category stations:', classes[classes.index<2].sum())
    print('Medium risk category stations:', classes[(classes.index>1) & (classes.index<6)].sum())
    print('High risk category stations:', classes[classes.index>5].sum())

    # Save combined predictions to csv
    if export:
        csv_name =  filepath + case + '.csv'
        predictions.to_csv(csv_name,index=False)
        geopandas_predictions(predictions,filepath,case)
        
    return predictions

# %%
def validate_results(predictions_path, data, period=8, radius=1, long_column='x', lat_column='y', error_buffer=3):
    """Checks the accuracy of the predictions of the previous month.
    
    Parameters
    --------
    predictions_path : str
        A path to the file containing the predictions of a month
        
    data : dataframe
        A dataframe containing the dates and the actual classes on each date
        
    period : int, optional
        The period around a prediction that is acceptable to check for error (default = 8)
        
    radius : int or fload, optional
        The distance (in Km) around a point to search for mosquito measurment (default = 1)
        
    long_column : str, optional
            The name of the column with the longitude (default = 'x')
        
    lat_column : str, optional
        The name of the column with the latitude (default = 'y')
        
    error_buffer : int, optional
        The error buffer for cumpiting percentage of error (default = 3)
        
    """
    predictions = pd.read_csv(predictions_path)
    predictions['dt_placement'] = pd.to_datetime(predictions['dt_placement'], format="%Y-%m-%d")
    predictions['dt_prediction'] = pd.to_datetime(predictions['dt_prediction'], format="%Y-%m-%d")
    actual = []
    date_l = []
    
    radians_y = data.loc[:,lat_column].astype(float).apply(math.radians)
    radians_x = data.loc[:,long_column].astype(float).apply(math.radians)
    radians_data = pd.concat([radians_y,radians_x],axis=1)
    radians_y = predictions.loc[:,lat_column].astype(float).apply(math.radians)
    radians_x = predictions.loc[:,long_column].astype(float).apply(math.radians)
    radians_predictions = pd.concat([radians_y,radians_x],axis=1)
    distances = haversine_distances(radians_predictions,radians_data)*6371
    
    for i in range(len(predictions)):
        indexes = np.where(distances[i,:] <= radius)[0].tolist()
        data1 = data.loc[indexes,:].reset_index(drop=True)
        y = np.nan
        d = np.nan
        if len(data1)>0:
            date =  predictions.loc[i,'dt_prediction']
            diff = ((data1['dt_placement'] + datetime.timedelta(days=15)) - date).dt.days
            indexmin = diff.abs().idxmin()
            y = data1.loc[indexmin,'mosq_now']
            d = diff[indexmin]
            diff = diff.drop(indexmin)
            while np.isnan(y) and (len(diff) !=0):
                indexmin = diff.abs().idxmin()
                y = data1.loc[indexmin,'mosq_now']
                d = diff[indexmin]
                diff = diff.drop(indexmin)
        actual.append(y)
        date_l.append(d)
    predictions.loc[:,'actual'] = actual
    predictions.loc[:,'time_diff'] = date_l
    
    predictions = predictions.loc[np.abs(predictions['time_diff'])<period]
    if len(predictions) != 0:
        x = predictions['prediction']-predictions['actual']
        print('Mean time difference in days:',predictions['time_diff'].mean())
        print('-------------------')
        print('Overall MAE:',np.abs(x).mean())
        print('Overall % error <=' +str(error_buffer) +':', np.round(Average(np.abs(x) <= error_buffer)*100,2))
        print('number of observations:', len(x))
        print('-------------------')
        ent = predictions.loc[predictions['entomological_features']==True]
        if len(ent) != 0:
            x = ent['prediction']-ent['actual']
            print('MAE with entomological:',np.abs(x).mean())
            print('% error <='+ str(error_buffer) +' with entomological:', np.round(Average(np.abs(x) <= error_buffer)*100,2))
            print('number of observations:', len(x))
            print('-------------------')
        ent = predictions.loc[predictions['entomological_features']==False]
        if len(ent) != 0:    
            x = ent['prediction']-ent['actual']
            print('MAE without entomological:',np.abs(x).mean())
            print('% error <='+ str(error_buffer) +' without entomological:', np.round(Average(np.abs(x) <= error_buffer)*100,2))
            print('number of observations:', len(x))
            print('-------------------')
    else:
        print('No predictions with difference less than %d days' % period)

# %%
def validate_results2(predictions, data, period=8, radius=1, long_column ='x', lat_column ='y'):
    """Checks the accuracy of the predictions of the previous month.
    
    Parameters
    --------
    predictions_path : str
        A path to the file containing the predictions of a month
        
    data : dataframe
        A dataframe containing the dates and the actual classes on each date
        
    period : int, optional
        The period around a prediction that is acceptable to check for error (default = 8)
        
    period : int or fload, optional
        The distance (in Km) around a point to search for mosquito measurment (default = 1)
        
    long_column : str, optional
            The name of the column with the longitude (default = 'x')
        
    lat_column : str, optional
        The name of the column with the latitude (default = 'y')
        
    """
#     predictions = pd.read_csv(predictions_path)
    predictions['dt_placement'] = pd.to_datetime(predictions['dt_placement'], format="%Y-%m-%d")
    predictions['dt_prediction'] = pd.to_datetime(predictions['dt_prediction'], format="%Y-%m-%d")
    actual = []
    date_l = []
    
    radians_y = data.loc[:,lat_column].astype(float).apply(math.radians)
    radians_x = data.loc[:,long_column].astype(float).apply(math.radians)
    radians_data = pd.concat([radians_y,radians_x],axis=1)
    radians_y = predictions.loc[:,lat_column].astype(float).apply(math.radians)
    radians_x = predictions.loc[:,long_column].astype(float).apply(math.radians)
    radians_predictions = pd.concat([radians_y,radians_x],axis=1)
    distances = haversine_distances(radians_predictions,radians_data)*6371
    

    for i in range(len(predictions)):
#         data1 = data.loc[(data['x'] == predictions.loc[i,'x']) & (data['y'] == predictions.loc[i,'y'])]
        indexes = np.where(distances[i,:] <= radius)[0].tolist()
        data1 = data.loc[indexes,:].reset_index(drop=True)
        y = np.nan
        d = np.nan
        if len(data1)>0:
            date =  predictions.loc[i,'dt_prediction']
            diff = ((data1['dt_placement'] + datetime.timedelta(days=15)) - date).dt.days
            indexmin = diff.abs().idxmin()
            y = data1.loc[indexmin,'mosq_now']
            d = diff[indexmin]
            diff = diff.drop(indexmin)
            while np.isnan(y) and (len(diff) !=0):
                indexmin = diff.abs().idxmin()
                y = data1.loc[indexmin,'mosq_now']
                d = diff[indexmin]
                diff = diff.drop(indexmin)
        actual.append(y)
        date_l.append(d)
    predictions.loc[:,'actual'] = actual
    predictions.loc[:,'time_diff'] = date_l
    
    predictions = pd.merge(predictions, data, how='inner', left_on = [predictions['x'], predictions['y'], predictions['dt_placement']], right_on = [data['x'], data['y'], data['dt_placement']])
    predictions = predictions.drop(columns=['key_0', 'key_1', 'key_2','dt_placement_y', 'x_y', 'y_y']).rename(columns={'x_x':'x', 'y_x':'y', 'dt_placement_x':'dt_placement'})
    return predictions

# %%
def geopandas_predictions(predictions,path,title):

    shapefile_path = '../Datasets/Shapefiles/'+'/'.join(path.split('/')[2:4])+'/'+'_'.join(path.split('/')[2:4])+'_shapefile_2km.shp'
    shapefile_path = shapefile_path.replace('//','/').replace('__','_')
    grid_shp = gpd.read_file(shapefile_path, encoding="utf_8")
    grid_shp['x'] = round(grid_shp['x'], 6)
    grid_shp['y'] = round(grid_shp['y'], 6)

    predictions_grid = pd.merge(predictions[['x','y','prediction']], grid_shp[['x', 'y', 'geometry']], on=['x', 'y'], how='inner')
    predictions_grid = predictions_grid.reset_index(drop=True)
    predictions_grid = gpd.GeoDataFrame(predictions_grid, geometry='geometry')
    csv_name =  path + 'Shapefiles/' + title + '.shp'
    csv_name = csv_name.replace('//','/').replace('__','_')
#     print(csv_name)
    predictions_grid.to_file(csv_name)


