from datetime import date, datetime
import numpy as np
import pandas as pd

def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n)])
    f = lambda x: (edges >= x).values.argmax() if not np.isnan(x) else np.nan
    return series.rank(pct=1).apply(f)

def create_classes(data, q, mosq_column='mosq_now', imb=False):
    """Creates the population classes, so as every class is cosnsisting of nearly equal number of observations.
       Either the number of classes or a list of bounds must be given.
    
    Parameters
    ----------
    data : Dataframe
        A Daframe containing the data
        
    q : int or list
        The number of classes to create, or the a list with the bounds of each class 
        
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
    
    """
    if (mosq_column not in data.columns):
        raise KeyError('Column given not in index')
        
    if (not isinstance(q, int)) and (not isinstance(q, list)):
        raise Exception('Give an integer with the number of classes or a list with the bounds of the class')

    if isinstance(q, int):
        if imb:
            classes = list(range(1,q))
            dat = data.loc[data[mosq_column]== 0]
            dat[mosq_column] = 0
            data2 = data.loc[data[mosq_column] != 0]
            bins, bounds = pd.qcut(data2.loc[:, mosq_column], retbins=True, q=q-1, labels=classes)
            data2[mosq_column] = bins
            data = pd.concat([dat, data2])
            data[mosq_column] = pd.Categorical(data[mosq_column])
            print('Bounds:', [0]+bounds.tolist())
        else:
            classes = list(range(q))
            bins, bounds = pd.qcut(data.loc[:, mosq_column], retbins=True, q=q, duplicates='raise', labels=classes)
            data.loc[:,mosq_column] = bins
            print('Bounds:', bounds)
    else:
        classes = list(range(len(q)-1))
        q = np.array(q)
        bins, bounds = pd.cut(data.loc[:, mosq_column], bins=q, retbins=True, labels=classes)
        data.loc[:,mosq_column] = bins
        print('Bounds:', bounds)
        
    data = data.reset_index(drop=True)
    data.loc[:, mosq_column] = data.loc[:, mosq_column].cat.codes
    if -1 not in data.loc[:, mosq_column].cat.categories:
        data.loc[:, mosq_column] = data.loc[:, mosq_column].cat.add_categories([-1])
    data.loc[:, mosq_column] = data.loc[:, mosq_column].fillna(-1)
    data.loc[:, mosq_column] = data.loc[:, mosq_column].astype(int)
    idx = data[data[mosq_column]==-1][mosq_column].index
    data.loc[idx, mosq_column] = None
    
    return data

def initializer(data, model_type, bound=None, flag=False, mosq_column = 'mosq_now', dropna = False):
    """Creates the class of abudance and the target variable for each observation,
    and removes all the rows that contain at least one NaN value. Optionally removes columns

        Parameters
        ----------
        data : Dataframe
            The dataframe contaning the number of mosquitoes
            
        bound : int or list, optional
            The number of classes (if integer is given), or the bounds of the classes (if a list is given)
            (default = None)
            
        flag : bool, optional
            if True performs balance handling by setting zeros as a class itself (default = False)
        
            
        mosq_column : str, optional
            The name of the column with the mosquito data to create the data from (default = 'mosq_bins')

        Returns
        ----------
        dataframe: Dataframe
            An imputed dataframe

        Raises
        ------
        ValueError
        If classes muust be created but bounds are not given
        
        KeyError
        If  mosq column given not in index
    """
    if (mosq_column not in data.columns):
        raise KeyError('Column given not in index')
        
    if model_type != 'mosquito_regression':
        if bound==None:
            raise ValueError('Bound argument must be given (int: number of classes or list: bounds of the classes)')
        else:
            data = create_classes(data, q=bound, imb=flag, mosq_column=mosq_column)
#             data[mosq_column] = pct_rank_qcut(data[mosq_column], bound)
    if dropna:
        data = data.dropna().reset_index(drop=True)
    column_to_move = data.pop(mosq_column)
    data[mosq_column] = column_to_move
    return data