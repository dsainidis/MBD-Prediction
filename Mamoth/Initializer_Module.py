# %%
from datetime import date, datetime
import numpy as np
import pandas as pd

# %%
def create_classes(data, q=None, bounds=[], mosq_column='mosq_now',imb=False):
    """Creates the population classes, so as every class is cosnsisting of nearly equal number of observations.
       Either the number of classes or a list of bounds must be given.
    
    Parameters
    ----------
    data : Dataframe
        A Daframe containing the data
        
    q : int, optional
        The number of classes to create (default = None)
    
    bounds : lst, optional
        A list with the bounds of the classes
        
    mosq_column : str, optional
        The name of the column with the mosquito classes (default = 'mosq_now')
        
    imb : bool, optional
        if True performs balance handling by setting zeros as a class itself (default = False)
        
        
    Returns
    ----------
    data: Dataframe
        A dataframe expanded by the column of class
        
    Raises
    ------
    Keyerror
        If the name column of mosquitoes given is not in index
        
    Exception
        If neither of number of classes or the list of the bounds is given
        
    Exception
        If both the number of classes and the list of the bounds is given
    
    """
    if (mosq_column not in data.columns):
        raise KeyError('Column given not in index')
    if q == None and len(bounds) == 0:
        raise Exception('Give the number of classes or a list with the bounds of the class')
    elif q != None and len(bounds) != 0:
        raise Exception('Give only either the number of classes or a list with the bounds of the class')
    if q != None:
        if imb:
            classes = list(range(1,q))
#             classes = list(range(2,q+1))
            dat = data.loc[data[mosq_column]== 0]
            dat['mosq_bins'] = 0
#             dat['mosq_bins'] = 1
            data2 = data.loc[data[mosq_column] != 0]
            bins, bounds = pd.qcut(data2.loc[:,mosq_column],retbins=True,q=q-1,labels=classes)
            data2['mosq_bins'] = bins
            data = pd.concat([dat,data2])
            print('Bounds:',bounds)
        else:
            classes = list(range(q))
#             classes = [x+1 for x in classes]
            bins, bounds = pd.qcut(data.loc[:,mosq_column],retbins=True,q=q,labels=classes)
            data.loc[:,'mosq_bins'] = bins
            print('Bounds:',bounds)
    else:
        classes = list(range(len(bounds)-1))
#         classes = [x+1 for x in classes]
        bins,bounds = pd.cut(data.loc[:,mosq_column],bins=bounds,retbins=True,labels=classes)
        data.loc[:,'mosq_bins'] = bins
        print('Bounds:',bounds)
    return data

# %%
def create_target_variable(data,start,end,step,long_column ='x',lat_column ='y', date_column = 'dt_placement',
                           mosq_column = 'mosq_bins'):
    """Creates the target variable by finding the risk class of step days after the date of observation
    
    Parameters
    ----------
    data : Dataframe
        A Daframe containing the data
        
    start : int
        The lower bound of days to consider in searching for the target variable
        
    end : int
        The upper bound of days to consider in searching for the target variable
        
    step : int
        The optimal distance of days to consider in searching for the target variable
    
    long_column : str, optional
        The name of the column with the longitude (default = 'x')
        
    lat_column : str, optional
        The name of the column with the latitude (default = 'y')
    
    date_column : str, optional
        The name of the column with the date of the observations (default = 'dt_placement')
    
    mosq_column : str, optional
        The name of the column with the mosquito data to create the data from (default = 'mosq_bins')
        
    Returns
    ----------
    data: Dataframe
        A dataframe containing the target varible
    
    Raises
    ------
    Exception
        If the expression start < step < end is not confirmed
    
    Keyerror
        If some of the columns not in index
    """
    
    if (end < start or end < step or step < start):
        raise Exception('start < step < end expression must be confirmed')
    if (long_column not in data.columns or lat_column not in data.columns or date_column not in data.columns or mosq_column not in data.columns):
        raise KeyError('Column(s) not in index')
    dataframe = pd.DataFrame()
    names = list()
#     stations = np.unique(data.loc[:,id_column])
    stations = data.loc[:,[long_column,lat_column]].drop_duplicates().reset_index(drop=True)
    data_mosq = data.dropna(subset=[mosq_column])
    for i in range(len(stations)):
#         data1 = data.loc[data.loc[:,id_column]==x]
        data1 = data.loc[(data[long_column] == stations.loc[i,long_column]) & (data[lat_column] == stations.loc[i,lat_column])]
        data1.reset_index(drop = True,inplace=True)
        data_mosq1 = data_mosq.loc[(data_mosq[long_column] == stations.loc[i,long_column]) & (data_mosq[lat_column] == stations.loc[i,lat_column])]
        data_mosq1.reset_index(drop = True,inplace=True)
        culex = []
        for j in range(len(data1)):
            date =  data1.loc[j,date_column]
#             diff = (data1[date_column] - date).dt.days
            diff = (data_mosq1[date_column] - date).dt.days
            diff = diff[(diff <= end) & (diff >= start)]
            if len(diff) != 0:
                indexmin = diff.sub(step).abs().idxmin()
#                 y = data1.loc[indexmin,mosq_column]
                y = data_mosq1.loc[indexmin,mosq_column]
            else:
                y = np.nan
            culex.append(y)
        data1.loc[:,'mosq_bins(t+1)'] =  culex
        dataframe =  pd.concat([dataframe, data1], axis=0, sort=False)
    names += [('%s' % (j)) for j in data.columns]
    names.append('mosq_bins(t+1)')
    dataframe.columns = names
    return dataframe

# %%
def initializer(data,bound,start,end,step,del_list=[],flag=False, env=False, long_col ='x', lat_col ='y', date_column = 'dt_placement', mosq_column = 'mosq_bins'):
    """Creates the class of abudance and the target variable for each observation,
    and removes all the rows that contain at least one NaN value. Optionally removes columns

        Parameters
        ----------
        dataframe : Dataframe
            The dataframe contaning the number of mosquitoes
        
        bound : int or lst
            The number of classes (if integer is given), or the bounds of the classes (if a list is given)
        
        start : int
            The lower bound of days to consider in searching for the target variable
        
        end : int
            The upper bound of days to consider in searching for the target variable

        step : int
            The optimal distance of days to consider in searching for the target variable

        del_list : list, optional
            A list containing the columns to remove. (default = None)
            
        flag : bool, optional
            if True performs balance handling by setting zeros as a class itself (default = False)
            
        env : bool, optional
            if True removes the entomological features (default = False)
        
        long_col : str, optional
            The name of the column with the longitude (default = 'x')
        
        lat_col : str, optional
            The name of the column with the latitude (default = 'y')
    
        date_column : str, optional
            The name of the column with the date of the observations (default = 'dt_placement')
            
        mosq_column : str, optional
            The name of the column with the mosquito data to create the data from (default = 'mosq_bins')

        Returns
        ----------
        dataframe: Dataframe
            An imputed dataframe

        Raises
        ------
        KeyError
        If not some of the columns to remove are not included in the dataframe
    """
    if isinstance(bound, list):
        bound = np.array(bound)
        data = create_classes(data,bounds=bound)
    else:
        data = create_classes(data,q=bound,imb=flag)
    data = create_target_variable(data,start,end,step, long_column = long_col, lat_column = lat_col, date_column = date_column,mosq_column = mosq_column)
    if len(del_list) != 0:
        for i in del_list:
            if i not in data.columns:
                raise KeyError('Column(s) not in index')
        data = data.drop(del_list, axis = 1)
    data.dropna(inplace = True)
    if env:
        data.drop(columns=['mosq_now','mosq_bins'],inplace=True)
    return data


