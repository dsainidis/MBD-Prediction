# %% [markdown]
# # This script preprocess the initial dataset

# %%

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Functions

# %%
def read_data(filepath, encoding = 'utf-8', date_col = 'dt_placement',long_col='x',lat_col='y'):
    """Reads the data out of an input file (.csv or .xls) 
    
    Parameters
    ----------
    filepath : str
        The path of the file
        
    date_col : str, , optional
        The name of the column with the date (default = 'dt_placement')
        
    long_col : str, , optional
        The name of the column with the longitude (default = 'x')
        
    lat_col : str, , optional
        The name of the column with the longitude (default = 'y')
        
    Returns
    ----------
    data: dataframe
        A dataframe created by the input file
    
    Raises
    ------
    NotImplementedError
    If the input file is not .csv or .xls
    
    KeyError
    If there is no column with 'date_col', 'long_col' or 'lat_col' name
    """
    try:
        # reading the file by xlrd (pip install xlrd)
        data = pd.read_excel(filepath, encoding = encoding)
        data = data.replace('<Null>',np.NaN)
    except:
        try:
            # reading as CSV file
            data = pd.read_csv(filepath, encoding = encoding)
            data = data.replace('<Null>',np.NaN)
        except: 
            raise NotImplementedError("Sorry, give me a .csv or .xls file")
    
    try:            
        data[date_col] = pd.to_datetime(data[date_col], format="%Y-%m-%d")
        data[long_col] = round(data[long_col], 6)
        data[lat_col] = round(data[lat_col], 6)
    except: 
        raise KeyError("No date, longitude or latitude column with this name was found")
    print(data.columns)
    return data

# %%
def add_landcover(data, date_col='dt_placement', minimize=False):
    """Finds the most recent landcover of a station.
    
    Parameters
    ----------
    data : dataframe
        A dataframe containing all the EO data
        
    date_col : str, optional
        The name of the column with the date (default = 'dt_placement')
        
    minimize : boolean, optional
        If True minimizes the landcover information given (default = 'False')
        
    Returns
    ----------
    data: dataframe
        A dataframe containing the topological info for each observation
    """

    landcover = []
    years = []
    closest_year = []
    cols = []
    for x in data.columns:
        if 'landcover' in x:
            years.append(int(x[:4]))
            cols.append(x)
    for i in range(len(data)):
        year = data[date_col][i].year
        closest_year.append(min(years, key=lambda x:abs(x-year)))
    for i in range(len(data)):
        landcover.append(data.loc[i,str(closest_year[i])+'_landcover'])
    data['landcover'] = landcover
    data = data.drop(columns = cols)
    if minimize:
        for i in range(len(data)):
            if data['landcover'][i]!=21:
                data['landcover'][i] = data['landcover'][i]//10
    return data

# %%
def add_topological(data,filepath, date_col='dt_placement', 
                    long_column='x',lat_column='y', landcover=False, minimize=False):
    """Adds the topological features of each observation.
    
    Parameters
    ----------
    data : Dataframe
        A dataframe containing all the EO data
    
    filepath : str
        The path of the file with the topological info
        
    long_column : str, , optional
        The name of the column with the longitude (default = 'x')
        
    lat_column : str, , optional
        The name of the column with the latitude (default = 'y')
        
     date_col : str, , optional
        The name of the column with the date (default = 'dt_placement')
        
    landcover : boolean, optional
        If True finds the most recent landcover of each observation (default = 'False')
    
    minimize : boolean, optional
        If True minimizes the landcover information given (default = 'False')
        
    Returns
    ----------
    data: dataframe
        A dataframe containing the topological info for each observation
    
    Raises
    ------
    KeyError
    If the filepath is not valid
    """
    try:
        topological = pd.read_csv(filepath)
    except: 
        raise KeyError("Sorry, give me a .csv file valid path.")
    topological[long_column] = round(topological[long_column], 6)
    topological[lat_column] = round(topological[lat_column], 6)
    topological_cols = topological.columns.tolist()
    topological_cols.remove('x')
    topological_cols.remove('y')
    topological_cols = [e for e in topological_cols if e in data.columns]
    data = data.drop(columns=topological_cols)
    data = pd.merge(data, topological, how='left',left_on = [data[long_column],data[lat_column]],right_on = [topological[long_column],topological[lat_column]])
    data = data.drop(columns=['key_0','key_1', long_column+'_y', lat_column+'_y'])
    data = data.rename(columns={long_column+'_x':long_column, lat_column+'_x':lat_column})
    if landcover:
        data = add_landcover(data,date_col,minimize)
    return data

# %%
def select_columns(dataframe,columns_list=[],columns_names = []):
    """Selects which columns to keep from the dataframe and optionally rename the columns 
    
    Parameters
    ----------
    dataframe: Dataframe
        Dataframe to be transformed
        
    columns_list : list, optional
        A list with the names of the columns to keep (default = a list containing all columns)
    
    columns_names : list, optional
        A list with the new names of the columns (default = a list containing the running names)
        
    Returns
    ----------
    dataframe: Dataframe
        A transformed dataframe
        
    Raises
    ------
    KeyError
    If the length of columns_list and columns_names do not match
    """
    try:
        if len(columns_list) != 0:
            dataframe = dataframe[columns_list]
        if len(columns_names) != 0:
            dataframe.columns = columns_names
    except:
        raise KeyError('The column list and the name list must be of same size')
    return dataframe

# %%
def reshape_dataset(dataframe,dupl_list=['x','y','dt_placement'],group_list=['x','y','dt_placement'],mosq_col='mosq_now'):
    """Removes the duplicates rows and aggragates observations needed
    
    Parameters
    ----------
    dataframe: Dataframe
        Dataframe to be transformed
    
    dupl_list : list
        A list with the names of the columns for removing the duplicates upon them (default=['x','y','dt_placement'])
        
    group_list : list
        A list with the names of the columns for grouping the duplicates upon them (default=['x','y','dt_placement'])
    
    mosq_col : str, optional
        The name of the column with the mosquito number (default = 'mosq_now')
        
    Returns
    ----------
    dataframe: Dataframe
        A transformed dataframe
        
    Raises
    ------    
    KeyError
        If column name(s) given not in index
    """
    
    if (mosq_col not in dataframe.columns):
        raise KeyError('Column(s) not in index')
    if len(dupl_list) != 0:
        for i in dupl_list:
            if i not in dataframe.columns:
                raise KeyError('Column(s) not in index')
    dataframe.drop_duplicates(subset=dupl_list+[mosq_col], keep='first',inplace=True)
    agg_dict = {mosq_col: lambda x: x.sum(min_count=1)}
    col = [e for e in dataframe.columns if e not in [mosq_col]+group_list]
    for i in col:
        agg_dict[i]= 'first'
    dataframe = dataframe.groupby(group_list).agg(agg_dict).reset_index()
    return dataframe

# %%
def fill_data(data, col_list, long_column='x', lat_column='y'):
    """Fills the NaN values of columns based on longitude and latitude column
    
    Parameters
    ----------
    data: Dataframe
        Dataframe to be transformed
    
    col_list : list
        A list with the names of the columnsto complete
        
    long_column : str, , optional
        The name of the column with the longitude (default = 'x')
        
    lat_column : str, , optional
        The name of the column with the latitude (default = 'y')
        
    Returns
    ----------
    data: Dataframe
        A dataframe with filled nan values
        
    Raises
    ------    
    KeyError
        If column name(s) given not in index
    """
    for i in col_list+[long_column,lat_column]:
        if i not in data.columns:
            raise KeyError('Column(s) not in index')
    stations = data[[long_column,lat_column]+col_list].drop_duplicates(subset=[long_column,lat_column])
    data = data.drop(columns=col_list)
    data = pd.merge(data, stations, how='left',left_on = [data[long_column],data[lat_column]],right_on = [stations[long_column],stations[lat_column]])
    data = data.drop(columns=['key_0','key_1',long_column+'_y',lat_column+'_y'])
    data = data.rename(columns={long_column+'_x':long_column, lat_column+'_x':lat_column})
    return data

# %%
def fillna_dataset(dataframe,fill_list):
    """Fills the NaN values of columns specified with spesific values
    
    Parameters
    ----------
    dataframe: Dataframe
        Dataframe to be transformed
    
    dupl_list : dict
        A dictionairy with the names of the columns and the value for NaN to complete
        
    Returns
    ----------
    dataframe: Dataframe
        A transformed dataframe with filled nan values
        
    Raises
    ------    
    KeyError
        If column name(s) given not in index
    """
    for i in list(fill_list.keys()):
        if i not in dataframe.columns:
            raise KeyError('Column(s) not in index')
        else:
            dataframe[i] = dataframe[i].fillna(fill_list[i])
    return dataframe

# %%
def calculate_diff(data,id_column='station_id',date_column='dt_placement'):
    """Creates a list with the time difference between two consecutive observations of each station

    Parameters
    ----------
    data: Dataframe
        The dataframe contaning the number of mosquitoes

    id_column : str, optional
        The name of the column containing the stations' ids (default = 'station_id')

    date_column : str, optional
        The name of the column containing the date of observations (default = 'dt_placement')

    Returns
    ----------
    time_diff: lst
        A list containing the distance of each observation from the next one

    Raises
    ------
    KeyError
    If not some of the columns are included in the dataframe
    """
    if (id_column not in data.columns or date_column not in data.columns):
        raise KeyError('Column(s) not in index')
    time_diff = []
    data = data.dropna(subset=['mosq_now'])
    for x in data[id_column].unique():
        data2 = data.loc[data[id_column]==x]
        data2 = data2.sort_values(by=[date_column], ascending=[True])
        data2.reset_index(drop=True,inplace=True)
        for j in range(len(data2)):
            data3 = data2.loc[data2[date_column].dt.year == data2[date_column][j].year]
            x = data3[date_column][j] < data3[date_column]
            y = x[x==True]
            if len(y) == 0:
                y = np.nan
            else:
                y = x[x==True].idxmin()
                y = np.abs((data2[date_column][j] - data2[date_column][y]).days)
            time_diff.append(y)
    time_diff = [x for x in time_diff if str(x) != 'nan']
    time_diff.sort()
    print('Length: ',len(time_diff))
    return time_diff

# %%
def cdf_plot(d, d_length):
    """Plots the cdf of the vector of days of difference between the observations of each station.
    
    Parameters
    ----------
    d : Vector
        A vector containing the days of differnece

    d_length : int
        The length of the vector d

    Returns
    ----------
    step : lst
        The optimal step in days in order to catch at least 80% of the observations days difference
    """
    a = np.linspace(min(d), max(d), 100)
    cdf = np.zeros(len(a))
    for k, val in enumerate(a):
        mask_d = d < val
        cdf[k] = mask_d.sum()/ d_length

    plt.plot(a,cdf)
    plt.grid()
    plt.xlabel('time difference')
    plt.ylabel('CDF')
    plt.show()
    idx = (np. abs(cdf - 0.8)). argmin()
    return np.round(a[idx])


