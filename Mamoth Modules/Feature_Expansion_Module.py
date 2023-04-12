# %%
import pandas as pd
import math
import random
from datetime import date
import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# %% [markdown]
# ## Functions

# %%
def imputation_of_mosq_sums(dataframe,del_list = []):
    """Imputes the NaN values of the sum of mosquitoes over the last 30 days and
       the sum of mosquitoes over the running month of previous year for every observation.
       Only arithemetic columns should be included in the process.

        Parameters
        ----------
        dataframe: Dataframe
            The dataframe contaning the number of mosquitoes

        del_list : list, optional
            A list containing the columns to ignore during imputation (e.g. categorical columns). (default = None)

        Returns
        ----------
        dataframe: Dataframe
            An imputed dataframe

        Raises
        ------
        KeyError
        If not some of the columns to ignore are not included in the dataframe
    """
    if len(del_list) != 0:
        for i in del_list:
            if i not in dataframe.columns:
                raise KeyError('Column(s) not in index')
        data2 = dataframe.drop(del_list, axis=1) 
        
    columns = data2.columns
    imputer = IterativeImputer(random_state = 0,  max_iter = 10, min_value = 0)
    imputer.fit(data2.values)
    imputer_df = imputer.transform(data2.values)
    imputer_df = pd.DataFrame(imputer_df)
    imputer_df.columns = columns
    
    dataframe['mosq_month_previousYear'] = imputer_df['mosq_month_previousYear']
    dataframe['mosq_month_sum'] = imputer_df['mosq_month_sum'] 
    dataframe.reset_index(drop=True,inplace =  True)
    return dataframe

# %%
def calculate_mosq_sum(dataframe,long_column ='x',lat_column ='y',date_col='dt_placement',mosq_col='mosq_now',imputation=[]):
    """Calculates the sum of mosquitoes of the last 30 days and
       the sum of mosquitoes in the running month of previous year for every observation

        Parameters
        ----------
        dataframe: Dataframe
            The dataframe contaning the number of mosquitoes

        long_column : str, optional
            The name of the column with the longitude (default = 'x')
        
        lat_column : str, optional
            The name of the column with the latitude (default = 'y')
            
        date_col : str, optional
            The name of the column containing the date of observations (default = 'dt_placement')

        mosq_col : str, optional
            The name of the column containing the number of mosquiotes of observations (default = 'mosq_now')
            
        imputation : lst, optional
            A list of columns to exclude from imputation

        Returns
        ----------
        dataframe: Dataframe
            An expanded by 2 columns dataframe

        Raises
        ------
        KeyError
        If some of the columns are not included in the dataframe
    """
    if (lat_column not in dataframe.columns or long_column not in dataframe.columns or date_col not in dataframe.columns or mosq_col not in dataframe.columns):
        raise KeyError('Column(s) not in index')
        
    culex_month_sum = []
    culex_month_sum_previousYear = []
           
    for i in range(len(dataframe)):
        
        sum_month = np.nan
        sum_month_year = np.nan

        x = dataframe[long_column][i]
        y = dataframe[lat_column][i]
        date = dataframe[date_col][i]

        df = dataframe.loc[(dataframe[long_column] == x) & (dataframe[lat_column] == y)]
        df = df[~(df[date_col] > date)]

        df1 = df[~(df[date_col] < date-datetime.timedelta(days=30))]

        df2 = df[~(df[date_col] < date-datetime.timedelta(days=395))]
        df2 = df2[~(df2[date_col] > date-datetime.timedelta(days=365))]

        if len(df1) !=0:
            sum_month = df1[mosq_col].sum()

        if len(df2) !=0:
            sum_month_year = df2[mosq_col].sum()

        culex_month_sum.append(sum_month)
        culex_month_sum_previousYear.append(sum_month_year)
        
    dataframe['mosq_month_sum'] = culex_month_sum
    dataframe['mosq_month_previousYear'] = culex_month_sum_previousYear
    dataframe = imputation_of_mosq_sums(dataframe,del_list = imputation)
    return dataframe

# %%
def get_season(dt, date_column='dt_placement'):
    """Creates cyclic features based on the season of each observation (sine - cosine transformation)
    
    Parameters
    ----------
    dt : Dataframe
        A datafrane containing the data
        
    date_column : str, optional
        The name of the column with the date of the observations (default = 'dt_placement')
    
    Returns
    ----------
    dt : Dataframe
        An expanded dataframe with two new features
        
    Raise
    ----------
    Keyerror
        If date column name is not in index
    
    """
    if (date_column not in dt.columns):
        raise KeyError('Column(s) not in index')
    lis = []
    Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
               ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
               ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
               ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
               ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]
    for i in range(len(dt)):
        now = dt.loc[i,date_column]
        now = now.replace(year=Y)
        lis.append(next(season for season, (start, end) in seasons
                if start <= now <= end))
    l = {'winter':0,'spring':1,'summer':2,'autumn':3}
    lis = [l[x] for x in lis]
    dt['se_sin'] = np.sin(pd.DataFrame(lis)*(2.*np.pi/4))
    dt['se_cos'] = np.cos(pd.DataFrame(lis)*(2.*np.pi/4))
    return dt

# %%
def get_month(data,date_column = 'dt_placement'):
    """Creates cyclic features based on the month of each observation (sine - cosine transformation)
    
    Parameters
    ----------
    data : Dataframe
         A datafrane containing the data
    
    date_column : str, optional
        The name of the column with the date of the observations (default = 'dt_placement')
    
    Returns
    ----------
    dat : Dataframe
         An expanded dataframe with two new features
         
    Raise
    ----------
    Keyerror
        If date column name is not in index
    
    """
    if (date_column not in data.columns):
        raise KeyError('Column(s) not in index')
    lis = []
    for i in range(len(data)):
        lis = data[date_column].dt.month
    data['mo_sin'] = np.sin(pd.DataFrame(lis)*(2.*np.pi/12))
    data['mo_cos'] = np.cos(pd.DataFrame(lis)*(2.*np.pi/12))
    return data

# %%
def find_distance(data,x=0,y=0, column_x='x', column_y='y'):
    """Calculates the euclidean distance from a specific point for each observation
    
    Parameters
    ----------
    data: Dataframe
        A datafrane containing the data
        
    x : float, optional
       The longitude of the specific point (default = 0)
       
    y : float, optional
       The latitude of the specific point (default = 0)
    
    column_x: str, optional
        The name of the column with the longitude (default = 'x')
        
    column_y: str, optional
        The name of the column with the latitude (default = 'y')
    
    Returns
    ----------
    data: Dataframe
        An expanded dataframe with a new feature
    
    x : float
       The longitude of the specific point
       
    y : float
       The latitude of the specific point
       
    Raise
    ----------
    Keyerror
        If lonfitude or latitude column name is not in index    
    """
    if (column_x not in data.columns or column_y not in data.columns):
        raise KeyError('Column(s) not in index')
    dist = []
    if(x==0):
        x = data.loc[0,column_x]
        y = data.loc[0,column_y]
    for i in range(len(data)):
        distance = math.sqrt( ((x-data.loc[i,column_x])**2)+((y-data.loc[i,column_y])**2) )
        dist.append(distance)
    data['distance'] = dist
    return x,y,data

# %%
def find_days_distance(data, date_column='dt_placement'):
    """Calculates the time distance (days) from a specific date
    
    Parameters
    ----------
    data: Dataframe
        A datafrane containing the data
    
    date_column : str, optional
        The name of the column with the date of the observations (default = 'dt_placement')
    
    Returns
    ----------
    dat: Dataframe
        An expanded dataframe with a new feature
    
    Raise
    ----------
    Keyerror
        If date column name is not in index
    
    """
    if (date_column not in data.columns):
        raise KeyError('Column(s) not in index')
    Y = 2000
    dist = []
    date = datetime.datetime(2020, 1, 1)
    date = date.replace(year=Y)
    for i in range(len(data)):
        now = data.loc[i, date_column]
        now = now.replace(year=Y)
        dist.append((date-now).days)
    data['days_distance'] = dist
    return data

# %%
def calculate_celsius(data,temp_columns=['lst_day','lst_night','lst_jan_mean','lst_feb_mean','lst_mar_mean','lst_apr_mean']):
    """Calculates celcious degrees of each observation
    
    Parameters
    ----------
    data: Dataframe
        A datafrane containing the data
        
    temp_column : list, optional
        The name of the columns with the temperatures to convert (default = ['lst_day','lst_night','lst_jan_mean','lst_feb_mean','lst_mar_mean','lst_apr_mean'])

    Returns
    ----------
    data: Dataframe
        An expanded dataframe with a new feature
    
    Raise
    ----------
    Keyerror
        If temparature column name is not in index
    
    """
    for i in temp_columns:
        if i not in data.columns:
            raise KeyError('Column(s) given in imput_list not in index')
    data['lst_day'] = data['lst_day'] * 0.02-273.15
    data['lst_night'] = data['lst_night'] * 0.02-273.15
    data['lst_jan_mean'] = data['lst_jan_mean'] * 0.02-273.15
    data['lst_feb_mean'] = data['lst_feb_mean'] * 0.02-273.15
    data['lst_mar_mean'] = data['lst_mar_mean'] * 0.02-273.15
    data['lst_apr_mean'] = data['lst_apr_mean'] * 0.02-273.15
    data = data.drop(data[data.lst_night < -30].index)
    data = data.drop(data[data.lst_day < -30].index)
    data = data.reset_index(drop=True)
    data['lst'] = (data['lst_day'] + data['lst_night'])/2
    return data

# %%
def count_summer_days(data, long_column ='x',lat_column ='y', date_col='dt_placement',temp_col='lst'):
    """Counts the days with over 30 degrees celsious, one month prior the running day 
    and from the beggining of the year
    
    Parameters
    ----------
    data : Dataframe
        A Daframe containing the data
        
    long_column : str, optional
        The name of the column with the longitude (default = 'x')
        
    lat_column : str, optional
        The name of the column with the latitude (default = 'y')
    
    date_column : str, optional
        The name of the column with the date of the observations (default = 'dt_placement')
    
    temp_column : str, optional
        The name of the column with the celsius degrees (default = 'lst')
        
    Returns
    ----------
    data: Dataframe
        A  expanded dataframe containing the 2 more columns
    
    Raises
    ---------    
    Keyerror
        If some of the columns not in index
    
    """
    if (long_column not in data.columns or lat_column not in data.columns or date_col not in data.columns or temp_col not in data.columns):
        raise KeyError('Column(s) not in index')
        
    year_sum = []
    month_sum = []
    for i in range(len(data)):
        
        x = data[long_column][i]
        y = data[lat_column][i]
        date = data[date_col][i]

        df = data.loc[(data[long_column] == x) & (data[lat_column] == y)]
        df = df[~(df[date_col] > date)]

        df1 = df[~(df[date_col] < date-datetime.timedelta(days=30))]
        df1 = df1.loc[df1[temp_col] >= 30]

        df2 = df[~(df[date_col] < date-datetime.timedelta(days=365))]
#         df2 = df.loc[df[date_col].dt.year == date.year]
        df2 = df2.loc[df2[temp_col] >= 30]
    
        year_sum.append(len(df2))
        month_sum.append(len(df1))
        
    data['summer_days_year'] = year_sum
    data['summer_days_month'] = month_sum
    return data

# %%
def convert_one_hot(data, columns):
    """Creates one hot encoded features out of a column
    
    Parameters
    ----------
    data: Dataframe
        A datafrane containing the data
    
    columns: lst
        A list containing the columns names to convert to one hot encoded features

    Returns
    ----------
    data: Dataframe
        An expanded dataframe with a new features
        
    Raises
    ----------
    Keyerror
        If column(s) not in index
    """
    
    for i in columns:
        if i not in data.columns:
            raise KeyError('Column(s) not in index')
        one_hot = pd.get_dummies(data[i])
        data = data.drop(i,axis = 1)
        # Join the encoded df
        data = data.join(one_hot)
    return data

# %%
def feature_expansion(data,one_hot = [],imput_list=[],cor=True, temp = True, env=False):
    """Creates new features out of the new ones.
    
    Parameters
    ----------
    data: Dataframe
        A datafrane containing the data
    
    imput_list: lst, optional
        A list containing the columns names to exclude from the imputation process (defalut = [])
        
    one_hot: lst, optional
        A list containing the columns names to convert to one hot encoded features (defalut = [])
    
    cor : boolean, optional
        If true calculate distance out of teh coordinates of each trap site (default = True)
    
    temp : boolean, optional
        If true calculate celsius out of LST (default= True)
    
    env : boolean, optional
        If true calculate new entomological features (default = False)
        
    Returns
    ----------
    data: Dataframe
        An expanded dataframe with a new features
    
    Raises
    ----------
    Keyerror
        If column(s) not in index
    """
    if len(one_hot) != 0:
        for i in one_hot:
            if i not in data.columns:
                raise KeyError('Column(s) given in one_hot not in index')
    if len(imput_list) != 0:
        for i in imput_list:
            if i not in data.columns:
                raise KeyError('Column(s) given in imput_list not in index')
    if cor == True:
        lon,lan,data = find_distance(data)
    data = find_days_distance(data)
    if temp == True:
        data = calculate_celsius(data)
        data = count_summer_days(data)
    data = get_month(data)
    if len(one_hot) != 0:
        data = convert_one_hot(data,one_hot)
    if env==False:
        data = calculate_mosq_sum(data,imputation=imput_list)
    data = data.reset_index(drop=True)
    return data


