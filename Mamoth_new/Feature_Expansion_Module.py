import pandas as pd
import math
import random
import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics.pairwise import haversine_distances
from sklearn.impute import IterativeImputer


def imputation_of_mosq_sums(dataframe, date_col='dt_placement'):
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

    data2 = dataframe.drop([date_col], axis=1)  
    data2 = data2.dropna(axis=1, how='all')
        
    columns = data2.columns
    imputer = IterativeImputer(random_state = 0,  max_iter = 10, min_value=0)
    imputer.fit(data2.values)
    imputer_df = imputer.transform(data2.values)
    imputer_df = pd.DataFrame(imputer_df)
    imputer_df.columns = columns
    
    mosq_cols = ['mosq_sum_month_previous_year', 'mosq_sum_month', 'mosq_sum_year', 'mosq_sum_previous_2weeks']
    for i in mosq_cols:
        if i in imputer_df.columns:
            dataframe[i] = round(imputer_df[i])
            
    dataframe.reset_index(drop=True,inplace =  True)
    return dataframe

def calculate_mosq_features(dataframe, long_column ='x',lat_column ='y', date_col='dt_placement', mosq_col='mosq_now'):
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

        Returns
        ----------
        dataframe: Dataframe
            An expanded by 2 columns dataframe

        Raises
        ------
        KeyError
        If some of the columns are not included in the dataframe
    """
    if (lat_column not in dataframe.columns or long_column not in dataframe.columns or
        date_col not in dataframe.columns or mosq_col not in dataframe.columns):
        raise KeyError('Column(s) not in index')
        
    culex_month_sum = []
    culex_month_sum_previous_year = []
    culex_sum_year = []
    previous_2weeks_sum = []
    previous_measure =[]
    
    radians_y = dataframe.loc[:,lat_column].astype(float).apply(math.radians)
    radians_x = dataframe.loc[:,long_column].astype(float).apply(math.radians)
    radians = pd.concat([radians_y, radians_x],axis=1)
    distances = haversine_distances(radians)*6371
    
    #each date of placement corresponds to the date that the EO data whrere extracted
    #the actual mosquito measurment took place 15 days after the EO data was extracted
    dataframe['mosq_date'] = dataframe[date_col] + datetime.timedelta(days=15)
           
    for i in range(len(dataframe)):        
        date = dataframe[date_col][i]
        
        previous1 = np.nan
        sum_month = np.nan
        sum_month_previous_year = np.nan
        sum_year = np.nan
        previous15 = np.nan
        
        # radius is the size of buffer zone size in km around each point to search for entomological info
        radius = 1
        indexes = np.where(distances[i,:] <= radius)[0].tolist()
        df = dataframe.loc[indexes,:].reset_index(drop=True)
        
        df = df[~(df['mosq_date'] > date)]

        df1 = df[~(df['mosq_date'] < date-datetime.timedelta(days=30))]

        df2 = df[~(df['mosq_date'] > date-datetime.timedelta(days=365))]
        df2 = df2[~(df2['mosq_date'] < date-datetime.timedelta(days=395))]

        df3 = df[~(df['mosq_date'] < pd.to_datetime(str(date.year)+'-01-01'))]

        df4 = df[~(df['mosq_date'] < date-datetime.timedelta(days=15))]
        

        if len(df1) !=0:
            sum_month = df1[mosq_col].sum()

        if len(df2) !=0:
            sum_month_previous_year = df2[mosq_col].sum()
        
        if len(df3) !=0:
            sum_year = df3[mosq_col].sum()

        if len(df4) !=0:
            previous15 = df4[mosq_col].sum()
            df4 = df4.sort_values(by=['mosq_date'], ascending=False).reset_index(drop=True)
            previous1 = df4.loc[0,mosq_col]

        culex_month_sum.append(sum_month)
        culex_month_sum_previous_year.append(sum_month_previous_year)
        culex_sum_year.append(sum_year)
        previous_2weeks_sum.append(previous15)
        previous_measure.append(previous1)
    
    del dataframe['mosq_date']
    dataframe['mosq_sum_month'] = culex_month_sum
    dataframe['mosq_sum_month_previous_year'] = culex_month_sum_previous_year
    dataframe['mosq_sum_year'] = culex_sum_year
    dataframe['mosq_sum_previous_2weeks']= previous_2weeks_sum
    dataframe['previous_mosq_measure']= previous_measure
    dataframe = imputation_of_mosq_sums(dataframe, date_col = date_col)
    return dataframe

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

def get_month(data, date_column = 'dt_placement'):
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
    lis = data[date_column].dt.month
    data['mo_sin'] = np.sin(pd.DataFrame(lis)*(2.*np.pi/12))
    data['mo_cos'] = np.cos(pd.DataFrame(lis)*(2.*np.pi/12))
    data['year'] = data[date_column].dt.year
    return data

def find_distance(data, column_x = 'x', column_y = 'y'):
    """Calculates the euclidean distance from a specific point for each observation
    
    Parameters
    ----------
    data: Dataframe
        A datafrane containing the data
    
    column_x: str, optional
        The name of the column with the longitude (default = 'x')
        
    column_y: str, optional
        The name of the column with the latitude (default = 'y')
    
    Returns
    ----------
    data: Dataframe
        An expanded dataframe with a new feature
       
    Raise
    ----------
    Keyerror
        If lonfitude or latitude column name is not in index    
    """
    if (column_x not in data.columns or column_y not in data.columns):
        raise KeyError('Column(s) not in index')
    data['null_island_distance'] = data.apply(lambda x: haversine_distances([[math.radians(_) for _ in [0,0]], [math.radians(_) for _ in [x[column_y],x[column_x]]]])[0,1]*6371,axis=1)
    data['vert_distance'] = data.apply(lambda x: haversine_distances([[math.radians(_) for _ in [0,x[column_x]]], [math.radians(_) for _ in [x[column_y],x[column_x]]]])[0,1]*6371,axis=1)
    return data

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
    data: Dataframe
        An expanded dataframe with a new feature
    
    Raise
    ----------
    Keyerror
        If date column name is not in index
    
    """
    if (date_column not in data.columns):
        raise KeyError('Column(s) not in index')
    Y = 2000
    date = datetime.datetime(Y, 1, 1)
    data['days_distance'] = data[date_column].apply(lambda x: (x.replace(year = Y)-date).days)
    return data

def calculate_celsius(data, temp_columns=['lst_day','lst_night','lst_jan_day_mean','lst_feb_day_mean',
                                          'lst_mar_day_mean','lst_apr_day_mean','lst_jan_night_mean',
                                          'lst_feb_night_mean', 'lst_mar_night_mean','lst_apr_night_mean']):
    """Calculates celcious degrees of each observation
    
    Parameters
    ----------
    data: Dataframe
        A datafrane containing the data
        
    temp_column : list, optional
        The name of the columns with the temperatures to convert (default = ['lst_day','lst_night','lst_jan_mean',
                                                                        'lst_feb_mean','lst_mar_mean','lst_apr_mean'])

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
    for x in temp_columns:
        data[x] = data[x] * 0.02-273.15
    data = data.drop(data[data.lst_night < -30].index).reset_index(drop=True)
    data = data.drop(data[data.lst_day < -30].index).reset_index(drop=True)
    data['lst'] = (data['lst_day'] + data['lst_night'])/2
    return data

def count_summer_days(data, long_column ='x',lat_column ='y', date_column='dt_placement',temp_col='lst'):
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
    if (long_column not in data.columns or lat_column not in data.columns or date_column not in data.columns or temp_col not in data.columns):
        raise KeyError('Column(s) not in index')
        
    radians_y = data.loc[:,lat_column].astype(float).apply(math.radians)
    radians_x = data.loc[:,long_column].astype(float).apply(math.radians)
    radians = pd.concat([radians_y, radians_x],axis=1)
    distances = haversine_distances(radians)*6371
        
#     year_sum = []
    month_sum = []
    for i in range(len(data)):
        
        date = data[date_column][i]
        
        indexes = np.where(distances[i,:] <= 5)[0].tolist()
        df = data.loc[indexes,:].reset_index(drop=True)
        df = df.groupby(date_column).mean().reset_index()
        df = df[~(df[date_column] > date)]

        df1 = df[~(df[date_column] < date-datetime.timedelta(days=30))]
        df1 = df1.loc[df1[temp_col] >= 30]
        
#         df2 = df[~(df[date_column] < pd.to_datetime(str(date.year)+'-01-01'))]
#         df2 = df2.loc[df2[temp_col] >= 30]
    
#         year_sum.append(len(df2))
        month_sum.append(len(df1))
        
#     data['summer_days_year'] = year_sum
    data['summer_days_month'] = month_sum
    return data

def feature_expansion(data, dist = True, days = True, month_cyc = False, 
                      celcius_conv = False, summer = True, env=False):
    """Creates new features out of the new ones.
    
    Parameters
    ----------
    data: Dataframe
        A datafrane containing the data
    
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
    if (dist == True):
        data = find_distance(data)
    if (days == True):
        data = find_days_distance(data)
    if (month_cyc == True):
        data = get_month(data)
    if (celcius_conv == True):
        data = calculate_celsius(data)
    if (summer == True):
        data = count_summer_days(data)
    if not env:
        data = calculate_mosq_features(data)
    data = data.reset_index(drop=True)
    return data