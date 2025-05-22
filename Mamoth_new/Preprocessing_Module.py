import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from scipy.stats import kurtosis, skew

def read_data(filepath, date_col = 'dt_placement', long_col='x', lat_col='y'):
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
        data = pd.read_excel(filepath)
        data = data.replace('<Null>',np.NaN)
    except:
        try:
            # reading as CSV file
            data = pd.read_csv(filepath)
            data = data.replace('<Null>',np.NaN)
        except: 
            raise NotImplementedError("Sorry, give me a .csv or .xls file")
    
    try:            
        data[date_col] = pd.to_datetime(data[date_col], format="%Y-%m-%d")
        data[long_col] = round(data[long_col], 6)
        data[lat_col] = round(data[lat_col], 6)
    except: 
        raise KeyError("No date, longitude or latitude column with this name was found")
    data.columns = data.columns.str.replace(r'_new', '')
    print(data.columns)
    return data

def merge_new_data(new_path, old, date_col='dt_placement', long_col='x', lat_col='y', exp_path=None):
    """Merges the new monthly data with the historical ones.
    
    Parameters
    ----------
    new_path : str
        The file path to the new monthly data
        
    old : dataframe
        A dataframe of the historical data
        
    date_col : str, , optional
        The name of the column with the date (default = 'dt_placement')
        
    long_col : str, , optional
        The name of the column with the longitude (default = 'x')
        
    lat_col : str, , optional
        The name of the column with the longitude (default = 'y')
        
    """
    new = pd.read_csv(new_path)
    new.loc[:,long_col] = round(new.loc[:,long_col], 6)
    new.loc[:,lat_col] = round(new.loc[:,lat_col], 6)
    new.columns = new.columns.str.replace(r'_new', '')
    new[date_col] = pd.to_datetime(new[date_col], format="%Y-%m-%d")
    old = pd.concat([old,new],axis=0).reset_index(drop=True)
    if exp_path!=None:
        old.to_csv(exp_path,index=False)
    return old

def add_topological(data, filepath, long_column='x', lat_column='y', neighbors=1):
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
        raise KeyError("Sorry, give me a .csv valid file path.")
    topological[long_column] = round(topological[long_column], 6)
    topological[lat_column] = round(topological[lat_column], 6)
    topological['x_rad'] = topological[long_column].apply(lambda x: np.deg2rad(x))
    topological['y_rad'] = topological[lat_column].apply(lambda x: np.deg2rad(x))
    data['x_rad'] = data[long_column].apply(lambda x: np.deg2rad(x))
    data['y_rad'] = data[lat_column].apply(lambda x: np.deg2rad(x))
    ball = BallTree(topological[["y_rad", "x_rad"]].values, metric='haversine')
    distances, indices = ball.query(data[["y_rad", "x_rad"]].values, k = 1)
    distances = [(d * 6371).tolist()[0] for d in distances]
    indices = indices.tolist()
    indices = [i[0] for i in indices]
    del data['x_rad']
    del data['y_rad']
    del topological['x_rad']
    del topological['y_rad']
    del topological[long_column]
    del topological[lat_column]
    data['neighbors'] = indices
    data = pd.merge(data, topological, how='left',left_on = [data.neighbors], right_index=True)
    del data['neighbors']
    return data

def add_landcover(data, filepath, long_column='x', lat_column='y', date_column='dt_placement', neighbors=1):
    """Adds the landcover features of each observation.
    
    Parameters
    ----------
    data : Dataframe
        A dataframe containing all the EO data
    
    filepath : str
        The path of the file with the landcover info
        
    long_column : str, , optional
        The name of the column with the longitude (default = 'x')
        
    lat_column : str, , optional
        The name of the column with the latitude (default = 'y')
        
        
    Returns
    ----------
    data: dataframe
        A dataframe containing the landcover info for each observation
    
    Raises
    ------
    KeyError
    If the filepath is not valid
    """
    try:
        landcover = pd.read_csv(filepath)
    except: 
        raise KeyError("Sorry, give me a .csv valid file path.")
    landcover[long_column] = round(landcover[long_column], 6)
    landcover[lat_column] = round(landcover[lat_column], 6)
    landcover['x_rad'] = landcover[long_column].apply(lambda x: np.deg2rad(x))
    landcover['y_rad'] = landcover[lat_column].apply(lambda x: np.deg2rad(x))
    data['x_rad'] = data[long_column].apply(lambda x: np.deg2rad(x))
    data['y_rad'] = data[lat_column].apply(lambda x: np.deg2rad(x))
    ball = BallTree(landcover[["y_rad", "x_rad"]].values, metric='haversine')
    distances, indices = ball.query(data[["y_rad", "x_rad"]].values, k = 1)
    distances = [(d * 6371).tolist()[0] for d in distances]
    indices = indices.tolist()
    indices = [i[0] for i in indices]
    del data['x_rad']
    del data['y_rad']
    del landcover['x_rad']
    del landcover['y_rad']
    del landcover[long_column]
    del landcover[lat_column]
    data['neighbors'] = indices
    #data = pd.merge(data, landcover, how='left',left_on = [data.neighbors], right_index=True)
    data = pd.merge(data, landcover, how='left',left_on = 'neighbors', right_index=True)
    del data['neighbors']
    landcover_cols = landcover.columns.tolist()
    data['landcover'] = data[date_column].apply(lambda x: str(x.year)+'_01_01_LC_Type1' if x.year <=2021 else '2021_01_01_LC_Type1')
    data['landcover'] = data.apply(lambda x: x[x['landcover']], axis=1)
    data = data.drop(columns=landcover_cols)
    return data

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

def fill_data(data, col_list, long_column='x', lat_column='y'):
    """Fills the NaN values of columns based on longitude and latitude column
    
    Parameters
    ----------
    data: Dataframe
        Dataframe to be transformed
    
    col_list : list
        A list with the names of the columns to complete
        
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

def fillna_dataset(dataframe, fill_list):
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

def remove_nan_features(data, test, percentage = 0.3):
    num_rows = len(test)
    nan_features = test.isna().sum()
    nan_features = nan_features/num_rows
    nan_features = nan_features.drop(['mosq_sum_month', 'mosq_sum_month_previous_year', 'mosq_sum_year',
                                     'mosq_sum_previous_2weeks', 'previous_mosq_measure','mosq_now'])
    nan_features = nan_features[nan_features>percentage].index.tolist()
    if len(nan_features) > 0:
        print('Features Dropped:',nan_features)
        data = data.drop(columns=nan_features)
        test = test.drop(columns=nan_features)
    return data, test

def unused_stations(data, test_df, period=None, threshold=None, date_col='dt_placement',):
    """Deletes from test set stations that are not used for a defined period, or that are used less than 
    5 times at the whole dataset to not give predictions for them.
    
    Parameters
    --------
    data : dataframe
        A dataframe containing the data
        
    test_df : dataframe
        A dataframe containing the prediction
    
    period : int
        The number of previous years that a station is not used
        
    date_col : str, optional
        The name of the date column (default = 'dt_placement')
        
    Returns
    --------
    test_df : dataframe
        A dataframe with the most frequent stations
    """ 
    # Calculate the year from which and then the check will be made for the existence of the station
    if period is not None:
        current_year = test_df[date_col].dt.to_period('Y').max().year
        check_years = current_year - period
        station_df = data.loc[data[date_col].dt.to_period('Y')>=str(check_years),['x','y']].drop_duplicates().reset_index(drop=True)
        test_df = pd.merge(test_df, station_df, how='inner',on = ['x','y'])
        
    if threshold is not None:
        count = data.groupby(['x','y'])[date_col].count().reset_index()
        count = count.loc[count[date_col]>=threshold,['x','y']]
        test_df = pd.merge(test_df, count, how='inner',on = ['x','y'])
        
    return test_df

def calculate_diff(data, long_column='x', lat_column='y', date_column='dt_placement'):
    """Creates a list with the time difference between two consecutive observations of each station

    Parameters
    ----------
    data: Dataframe
        The dataframe contaning the number of mosquitoes

    long_column : str, optional
        The name of the column with the longitude (default = 'x')
        
    lat_column : str, optional
        The name of the column with the latitude (default = 'y')

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
    if (long_column not in data.columns or lat_column not in data.columns or date_column not in data.columns):
        raise KeyError('Column(s) not in index')
    time_diff = []
    data = data.dropna(subset=['mosq_now'])
    stations = data.loc[:, [long_column, lat_column]].drop_duplicates().reset_index(drop=True)
    for i in range(len(stations)):
        data2 = data.loc[(data[long_column] == stations.loc[i,long_column]) & (data[lat_column] == stations.loc[i,lat_column])]
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
    #print('Length: ',len(time_diff))
    return time_diff

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
    plt.xlabel('Days Difference')
    plt.ylabel('CDF')
    plt.show()
    idx = (np. abs(cdf - 0.8)). argmin()
    return np.round(a[idx])

def statistics(data, mosq_column='mosq_now', threshold=10, date_column='dt_placement'):
    
    print('Total observatons:', len(data[['x','y','mosq_now']].dropna()))
    print('Number of unique traps:', len(data[['x','y','mosq_now']].dropna().drop_duplicates(subset=['x','y'])))
    print('Start date:', data[['mosq_now','dt_placement']].dropna()['dt_placement'].min())
    print('End date:', data[['mosq_now','dt_placement']].dropna()['dt_placement'].max())
    
    all_obs = 0
    data2 = data.dropna(subset=[mosq_column]).reset_index(drop=True)
    
    x2 = data2[data2[mosq_column] < np.percentile(data2[mosq_column], 95)].reset_index(drop=True)
    plt.hist(x2[mosq_column],density=True)
    plt.xlabel('Mosquito Abundance')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    print("Mean:",data2[mosq_column].mean())
    print("Std:",data2[mosq_column].std())
    print("Skewness:",skew(data2[mosq_column], bias=True))
    print("kurtosis:", kurtosis(data2[mosq_column], bias=True))
    
    max_year = data2['dt_placement'].max().year
    for i in list(range(max_year-2, max_year+1)):
        len_year = len(data2[data2[date_column].dt.year==i])
        print(str(i)+' number of observations:',len_year)
        all_obs = all_obs + len_year
    print('All operational years observations:',all_obs)
    
    count = data[['x','y','mosq_now']].dropna()
    count = count[['x','y']].groupby(['x','y']).size().reset_index(name='frequency')
    count_frequent = len(count.loc[count['frequency']>=threshold])
    count_infrequent = len(count.loc[count['frequency']<threshold])
    print('Number of fixed traps:', count_frequent)
    print('Number of temporal traps:', count_infrequent)
    
    time = calculate_diff(data)
    time = np.array(time)
    time = time[time < np.percentile(time, 95)]
    step = cdf_plot(time, len(time))
    plt.hist(time, density=True)
    plt.ylabel('Frequency')
    plt.ylabel('Days Difference')
    plt.show()
    
    features = ['x', 'y', 'ndvi_mean', 'ndwi_mean', 'ndmi_mean', 'ndbi_mean', 
            'acc_rainfall_1week', 'acc_rainfall_2week', 'acc_rainfall_jan',
            'lst_jan_day_mean', 'lst_feb_day_mean', 'lst_mar_day_mean',
            'lst_apr_day_mean', 'lst_jan_night_mean', 'lst_feb_night_mean',
            'lst_mar_night_mean', 'lst_apr_night_mean', 'DISTANCE_TO_COAST',
            'DISTANCE_TO_RIVER', 'SLOPE_mean_1km', 'ASPECT_mean_200m',
            'ELEVATION_mean_1km', 'HILLSHADE_mean_1km', 'FS_AREA_1km',
            'FLOW_ACCU_200m', 'null_island_distance', 'vert_distance',
            'days_distance', 'lst', 'lst_day', 'lst_night']
    for f in features:
        data_plot = x2[[f, mosq_column]].dropna().reset_index(drop=True)
        plt.figure()
        sns.kdeplot(data=data_plot, x=f, y=mosq_column, shade=True)
        plt.xlabel(f)
        plt.ylabel('Mosquito Abundance')
        plt.grid()
        plt.show()
        
        x = data_plot[f]
        y = data_plot[mosq_column]
        nbins = 20
        n, _ = np.histogram(x, bins=nbins)
        sy, _ = np.histogram(x, bins=nbins, weights=y)
        sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)
        plt.plot(x, y, 'bo')
        plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-')
        plt.xlabel(f)
        plt.ylabel('Mosquito Abundance')
        plt.grid()
        plt.show()
        
#     data_plot = data[['dt_placement', 'lst', 'lst_day', 'lst_night', 'acc_rainfall_1week', 'acc_rainfall_2week', 'acc_rainfall_jan']]
#     data_plot = data_plot.groupby('dt_placement').mean().reset_index()
#     data_plot = data_plot.groupby(data_plot['dt_placement'].dt.to_period('M')).mean().drop(columns=['dt_placement']).reset_index()
#     data_plot['dt_placement'] = data_plot['dt_placement'].dt.to_timestamp()

#     plt.figure()
#     plt.plot(data_plot['dt_placement'], data_plot['lst'], label='Temperature ($^{o}$C)')
#     plt.plot(data_plot['dt_placement'], data_plot['acc_rainfall_2week'], label='Accumulative Rainfall \n 2 weeks(mm)')
#     plt.grid()
#     plt.legend()
#     plt.show()

def temp_rainfall_plot(data, date_column='dt_placement', temp_column='lst', rainfall_column='acc_rainfall_jan'):
    
    temp = data[[date_column,'lst_jan_day_mean',
                 'lst_jan_night_mean', 'lst_feb_day_mean', 'lst_feb_night_mean',
                 'lst_mar_day_mean', 'lst_mar_night_mean', 'lst_apr_day_mean',
                 'lst_apr_night_mean']]

    temp['lst_jan_mean'] = (temp['lst_jan_day_mean'] + temp['lst_jan_night_mean'])/2
    temp['lst_feb_mean'] = (temp['lst_feb_day_mean'] + temp['lst_feb_night_mean'])/2
    temp['lst_mar_mean'] = (temp['lst_mar_day_mean'] + temp['lst_mar_night_mean'])/2
    temp['lst_apr_mean'] = (temp['lst_apr_day_mean'] + temp['lst_apr_night_mean'])/2
    temp = temp.groupby(temp[date_column].dt.to_period('Y')).mean().drop(columns=[date_column]).reset_index()
    temp[date_column] = temp[date_column].dt.to_timestamp()

    temp_jan = temp[[date_column,'lst_jan_mean']].rename(columns={'lst_jan_mean':'lst'})
    temp_jan[date_column] = temp_jan[date_column].apply(lambda dt: dt.replace(month=1))
    temp_feb = temp[[date_column,'lst_feb_mean']].rename(columns={'lst_feb_mean':'lst'})
    temp_feb[date_column] = temp_feb[date_column].apply(lambda dt: dt.replace(month=2))
    temp_mar = temp[[date_column,'lst_mar_mean']].rename(columns={'lst_mar_mean':'lst'})
    temp_mar[date_column] = temp_mar[date_column].apply(lambda dt: dt.replace(month=3))
    temp_apr = temp[[date_column,'lst_apr_mean']].rename(columns={'lst_apr_mean':'lst'})
    temp_apr[date_column] = temp_apr[date_column].apply(lambda dt: dt.replace(month=4))
    temp = pd.concat([temp_jan, temp_feb, temp_mar, temp_apr],axis=0)

    temp_daily = data[[date_column,'lst']]
    temp_daily = temp_daily.groupby(date_column).mean().reset_index()
    temp_daily = temp_daily.groupby(temp_daily[date_column].dt.to_period('M')).mean().drop(columns=[date_column]).reset_index()
    temp_daily[date_column] = temp_daily[date_column].dt.to_timestamp()

    temp = pd.concat([temp,temp_daily],axis=0)
    temp = temp.groupby(temp.dt_placement.dt.month)[temp_column].mean()

    rain = data[[date_column,'acc_rainfall_1week', 'acc_rainfall_2week', 'acc_rainfall_jan']]
    rain = rain[rain[date_column].dt.month!=2].reset_index(drop=True)
    rain = rain.groupby(date_column).mean().reset_index()
    rain = rain.groupby(rain[date_column].dt.to_period('M')).mean().drop(columns=[date_column]).reset_index()
    rain[date_column] = rain[date_column].dt.to_timestamp()

    rain = rain.groupby(rain.dt_placement.dt.month)[rainfall_column].mean()
    
    months = {1: 'January',
              2: 'February',
              3:'March',
              4:'April',
              5:'May',
              6:'June',
              7:'July',
              8:'August',
              9:'September',
              10:'October',
              11:'November'}
    
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.bar(rain.index, rain, label='Rainfall')
    ax2.plot(temp.index, temp, '-o', color='orange', label='Temperature')

    ax1.set_ylabel('Accumulative Rainfall (mm)')
    ax2.set_ylabel('Temperature ($^oC$)')

    labels = [months[item] for item in temp.index]
    ax1.set_xticks(temp.index)
    ax1.set_xticklabels(labels,rotation=30)


    handles, labels = ax2.get_legend_handles_labels()
    colors = {'Rainfall':'#1f77b4'}         
    labels.append('Rainfall')
    handles.append(plt.Rectangle((0,0),1,1, color=colors['Rainfall']))
    plt.legend(handles, labels)
    plt.grid()
    plt.show()
    
    temp = data[[date_column,'lst_jan_day_mean',
       'lst_jan_night_mean', 'lst_feb_day_mean', 'lst_feb_night_mean',
       'lst_mar_day_mean', 'lst_mar_night_mean', 'lst_apr_day_mean',
       'lst_apr_night_mean']]

    temp['lst_jan_mean'] = (temp['lst_jan_day_mean'] + temp['lst_jan_night_mean'])/2
    temp['lst_feb_mean'] = (temp['lst_feb_day_mean'] + temp['lst_feb_night_mean'])/2
    temp['lst_mar_mean'] = (temp['lst_mar_day_mean'] + temp['lst_mar_night_mean'])/2
    temp['lst_apr_mean'] = (temp['lst_apr_day_mean'] + temp['lst_apr_night_mean'])/2
    temp = temp.groupby(temp[date_column].dt.to_period('Y')).mean().drop(columns=[date_column]).reset_index()
    temp[date_column] = temp[date_column].dt.to_timestamp()

    temp_jan = temp[[date_column,'lst_jan_mean']].rename(columns={'lst_jan_mean':'lst'})
    temp_jan[date_column] = temp_jan[date_column].apply(lambda dt: dt.replace(month=1))
    temp_feb = temp[[date_column,'lst_feb_mean']].rename(columns={'lst_feb_mean':'lst'})
    temp_feb[date_column] = temp_feb[date_column].apply(lambda dt: dt.replace(month=2))
    temp_mar = temp[[date_column,'lst_mar_mean']].rename(columns={'lst_mar_mean':'lst'})
    temp_mar[date_column] = temp_mar[date_column].apply(lambda dt: dt.replace(month=3))
    temp_apr = temp[[date_column,'lst_apr_mean']].rename(columns={'lst_apr_mean':'lst'})
    temp_apr[date_column] = temp_apr[date_column].apply(lambda dt: dt.replace(month=4))
    temp = pd.concat([temp_jan, temp_feb, temp_mar, temp_apr],axis=0)

    temp_daily = data[[date_column,'lst']]
    temp_daily = temp_daily.groupby(date_column).mean().reset_index()
    temp_daily = temp_daily.groupby(temp_daily[date_column].dt.to_period('M')).mean().drop(columns=[date_column]).reset_index()
    temp_daily[date_column] = temp_daily[date_column].dt.to_timestamp()
    temp = pd.concat([temp,temp_daily],axis=0)

    temp_2021 = temp[temp[date_column].dt.year==2021]
    temp_2021 = temp_2021.groupby(temp_2021.dt_placement).mean().reset_index()

    temp_2022 = temp[temp[date_column].dt.year==2022]
    temp_2022 = temp_2022.groupby(temp_2022.dt_placement).mean().reset_index()

    temp_2023 = temp[temp[date_column].dt.year==2023]
    temp_2023 = temp_2023.groupby(temp_2023.dt_placement).mean().reset_index()
    temp_2023

    temp = temp[temp[date_column].dt.year<2021]
    temp = temp.groupby(temp.dt_placement.dt.month)[temp_column].mean()


    fig, ax1 = plt.subplots()

    ax1.plot(temp_2021.dt_placement.dt.month, temp_2021[temp_column], '-o', label='2021')
    ax1.plot(temp_2022.dt_placement.dt.month, temp_2022[temp_column], '-o', label='2022')
    ax1.plot(temp_2023.dt_placement.dt.month, temp_2023[temp_column], '-o', label='2023')
    ax1.plot(temp.index, temp, '--', label='2010-2020 mean')


    labels = [months[item] for item in temp.index]
    ax1.set_xticks(temp.index)
    ax1.set_xticklabels(labels,rotation=30)

    plt.ylabel('Temperature ($^oC$)')
    plt.legend()
    plt.grid()
    plt.show()
    
    rain = data[[date_column,'acc_rainfall_1week', 'acc_rainfall_2week', 'acc_rainfall_jan']]
    rain = rain[rain[date_column].dt.month!=2].reset_index(drop=True)
    rain = rain.groupby(date_column).mean().reset_index()
    rain = rain.groupby(rain[date_column].dt.to_period('M')).mean().drop(columns=[date_column]).reset_index()
    rain[date_column] = rain[date_column].dt.to_timestamp()

    rain_2021 = rain[rain[date_column].dt.year==2021]
    rain_2022 = rain[rain[date_column].dt.year==2022]
    rain_2023 = rain[rain[date_column].dt.year==2023]

    rain = rain[rain[date_column].dt.year<2021]
    rain = rain.groupby(rain.dt_placement.dt.month)[rainfall_column].mean()

    fig, ax1 = plt.subplots()

    ax1.plot(rain_2021.dt_placement.dt.month, rain_2021[rainfall_column], '-o', label='2021')
    ax1.plot(rain_2022.dt_placement.dt.month, rain_2022[rainfall_column], '-o', label='2022')
    ax1.plot(rain_2023.dt_placement.dt.month, rain_2023[rainfall_column], '-o', label='2023')
    ax1.plot(rain.index, rain, '--', label='2010-2020 mean')

    labels = [months[item] for item in rain.index]
    ax1.set_xticks(rain.index)
    ax1.set_xticklabels(labels,rotation=30)

    plt.ylabel('Accumulative Rainfall (mm)')
    plt.legend()
    plt.grid()
    plt.show()
    
    data2 = data[data['mosq_now'] < np.percentile(data['mosq_now'].dropna(), 60)].reset_index(drop=True)
    
    mosq = data[[date_column,'mosq_now']]
    mosq = mosq[mosq[date_column].dt.month!=2].reset_index(drop=True)
    mosq = mosq.groupby(date_column).mean().reset_index()
    mosq = mosq.groupby(mosq[date_column].dt.to_period('M')).mean().drop(columns=[date_column]).reset_index()
    mosq[date_column] = mosq[date_column].dt.to_timestamp()
    mosq = mosq.groupby(mosq.dt_placement.dt.month)['mosq_now'].mean()
    
    fig, ax1 = plt.subplots()

    ax1.plot(mosq.index, mosq, '-o', label='mosquito population')

    labels = [months[item] for item in temp.index]
    ax1.set_xticks(temp.index)
    ax1.set_xticklabels(labels,rotation=30)

    plt.ylabel('Mosquito Population')
    plt.legend()
    plt.grid()
    plt.show()
    
    mosq = data[[date_column,'mosq_now']]
    mosq = mosq[mosq[date_column].dt.month!=2].reset_index(drop=True)
    mosq = mosq.groupby(date_column).mean().reset_index()
    mosq = mosq.groupby(mosq[date_column].dt.to_period('M')).mean().drop(columns=[date_column]).reset_index()
    mosq[date_column] = mosq[date_column].dt.to_timestamp()
    
    mosq_2021 = mosq[mosq[date_column].dt.year==2021]
    mosq_2022 = mosq[mosq[date_column].dt.year==2022]
    mosq_2023 = mosq[mosq[date_column].dt.year==2023]

    mosq = mosq[mosq[date_column].dt.year<2021]
    mosq = mosq.groupby(mosq.dt_placement.dt.month)['mosq_now'].mean()
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(mosq_2021.dt_placement.dt.month, mosq_2021['mosq_now'], '-o', label='2021')
    ax1.plot(mosq_2022.dt_placement.dt.month, mosq_2022['mosq_now'], '-o', label='2022')
    ax1.plot(mosq_2023.dt_placement.dt.month, mosq_2023['mosq_now'], '-o', label='2023')
    ax1.plot(mosq.index, mosq, '--', label='2010-2020 mean')

    labels = [months[item] for item in temp.index]
    ax1.set_xticks(temp.index)
    ax1.set_xticklabels(labels,rotation=30)

    plt.ylabel('Mosquito Population')
    plt.legend()
    plt.grid()
    plt.show()

def plot_correlations(data, mosq_column='mosq_now', date_column='dt_placement'):
    
    drop_columns1 = [date_column]
    drop_columns2 = [date_column, 'mosq_sum_month',
                    'mosq_sum_month_previous_year', 'mosq_sum_year',
                    'mosq_sum_previous_2weeks', 'previous_mosq_measure','year']

    plt.figure(figsize=(10,15))
    sns.heatmap(data.drop(columns=drop_columns1).corr()[[mosq_column]].sort_values(by=[mosq_column])[:-1], annot=True)
    plt.show()

    abs_corr = np.abs(data.drop(columns=drop_columns2).corr()[[mosq_column]]).sort_values(by=[mosq_column],ascending=False)[1:]
    zz = data.drop(columns=drop_columns2).groupby(data[date_column].dt.year).corr()[mosq_column]
    for i in abs_corr.index[:3]:
        plt.plot(zz.index.get_level_values(0).unique().sort_values().tolist(), zz.xs(i, level=1, drop_level=False).values, '-o', label=i)
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Correlation')
    plt.grid()
    plt.show()