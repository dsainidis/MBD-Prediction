import shapely
import math
import re
import numpy as np
import pandas as pd
from fiona.crs import from_epsg
import geopandas as gpd
from statistics import mean
import unicodedata
from sklearn.neighbors import BallTree
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial.distance import cdist
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from geopy.distance import geodesic

def create_grid(river_shapes, data_shapes, col_municipal='NAME', col_geometry='geometry', dimension=2):
    """Creates the cell grid for the desired area
    
    Parameters
    --------
    river_shapes : dataframe,
        A GeoDataFrame containing the rivers of the area (to avoid creating cells on them)
                
    data_shapes : dataframe
        A GeoDataFrame containing the polygons (shapes) and the names of municipalities
        
    col_municipal : str, optional
        The column that contains the name for each municipality (default = 'NAME')
        
    col_geometry : string, optional
        The name of the column with the geometry of the river (default = 'geometry')
        
    dimension : int, optional
        The dimension of the square cell in km (eg. 2km x 2km, 1km x 1km) (default = 2)
        
    Returns
    ----------
    cell_df: dataframe
        A dataframe containing the generated cells for each municipality and the centers of each cell
    
    """
    river_shapes_list =  river_shapes[col_geometry].geometry.explode().tolist()
    # Convert cell height dimension to km
    cell_height = (dimension / 6378) * (180 / math.pi)
    
    grid_cells = []
    grid_cells2 = []
    municipal = []
    centers = []
    # Iterate over municipalities to create a grid for each one
    for i in range(data_shapes.shape[0]):
        area_cells = []
        center = []
        # Keep the geometry of municipality
        g_country = data_shapes[col_geometry][i]
        # Extract the bounds of municipality
        xmin, ymin, xmax, ymax = data_shapes[col_geometry][i:i+1].total_bounds
        # Convert cell width dimension to km (dependent to y)
        cell_width  = (dimension / 6378) * (180 / math.pi) / math.cos(ymin  * math.pi/180)
    
        # Iterate to municipality bounds to create the grid's cells
        for x0 in np.arange(xmin, xmax+cell_width, cell_width ):
            for y0 in np.arange(ymin, ymax+cell_height, cell_height):
                y1 = y0+cell_height
                x1 = x0+cell_width
                new_cell = shapely.geometry.box(x0, y0, x1, y1)
                # Check if new cell intersects the municipality
                if new_cell.covered_by(g_country):
                    # Check if new cell intersects previously generated cells
                    delete_cell_boundary = [True for cell in grid_cells2 if new_cell.intersects(cell)]
                    # Check if new cell intersects with rivers
                    delete_cell_river = [True for cell in river_shapes_list if new_cell.intersects(cell)]
                    # Save new cell if it doesn't intersects rivers or previous cells 
                    if not delete_cell_boundary and not delete_cell_river:
                        center.append(new_cell.centroid)
                        area_cells.append(new_cell)
                else:
                    # Continue to next cell generation
                    pass
        # Save the final cells for each municipality
        grid_cells.append(area_cells)
        # Save center of each cell
        centers.append(center)
        # Save municipality name for each generated cell
        municipal.append([data_shapes[col_municipal][i]]*len(area_cells))
        # Save the previous generated cells for intersection check
        grid_cells2 = []
        grid_cells2 = [item for sublist in grid_cells for item in sublist]

    # Flatten lists
    grid_cells_list = [item for sublist in grid_cells for item in sublist]
    centers_list = [item for sublist in centers for item in sublist]
    municipal_list = [item for sublist in municipal for item in sublist]
    
    # Create final GeoDataFrame with municipalities and their cells
    cell_df = gpd.GeoDataFrame(grid_cells_list, columns=[col_geometry], crs=from_epsg(4326))
    cell_df['centers'] = centers_list
    cell_df['NAME'] = municipal_list
    
    x = []
    y = []
    # Extract coordinates from centers
    for index,row in cell_df.iterrows():
        x.append(round(row['centers'].x,5))
        y.append(round(row['centers'].y,5))  
        
    cell_df['x'] = x
    cell_df['y'] = y
    
    return cell_df

def read_river_shapes(col_id, path_rivers, col_geometry='geometry', filter=False, num_region=0, path_region='', *args):
    """Reads rivers shapefile and returns a geopandas dataframe
    
    Parameters
    --------
    col_id : string,
        The name of the column with the id of the river
                
    path_rivers : string
        Path to shapefile of rivers
        
    col_geometry : string, optional
        The name of the column with the geometry of the river (default = 'geometry')
        
    filter : boolean, optional
        If false all rivers will be return (default = False)
        
    num_region : int,optional
        The number of the row of the desired region in GeoDataFrame (default = 0)
        
    path_region : string, optional
        Path to shapefile of region that its' rivers will be kept
        
    data_shapes : dataframe, optional (pass with *args)
        A GeoDataFrame containing the polygons (shapes) and the names of municipalities

    Returns
    ----------
    data: dataframe
        A dataframe containing the desired river shapes
    
    """
    try:
        # reading as shp file
        river_shapes = gpd.read_file(path_rivers, encoding="utf_8")
    except: 
        raise NotImplementedError("Sorry, give me a .shp file.")
    if filter:
        indexes = []
        if path_region:
            try:
                # reading as shp file
                region = gpd.read_file(path_region, encoding="utf_8")
            except: 
                raise NotImplementedError("Sorry, give me a .shp file.")
            for index, row in river_shapes.iterrows():
                if row[col_geometry].intersects(region[col_geometry][num_region]):
                    indexes.append(index)
        else:
            data_shapes=args[0]
            for index, row in river_shapes.iterrows():
                for index2, row2 in data_shapes.iterrows():
                    if row[col_geometry].intersects(row2[col_geometry]):
                        indexes.append(index)
                
        river_shapes = river_shapes.loc[indexes].reset_index(drop=True)
    river_shapes = river_shapes[[col_id, col_geometry]]
    return river_shapes

def remove_accents(input_str):
    """Removes langueage accents from strings
    
    Parameters
    --------
    input_str : string,
        The string with accents
                
    Returns
    ----------
    string: the string without accents 
    """
    
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def process_greek(frame, column):
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    frame[column] = frame[column].apply(lambda x : x.strip())
    frame[column] = frame[column].apply(lambda x : x.lower())
    frame[column] = frame[column].apply(lambda x : x.replace('ά','α'))
    frame[column] = frame[column].apply(lambda x : x.replace('έ','ε'))
    frame[column] = frame[column].apply(lambda x : x.replace('ή','η'))
    frame[column] = frame[column].apply(lambda x : x.replace('ί','ι'))
    frame[column] = frame[column].apply(lambda x : x.replace('ύ','υ'))
    frame[column] = frame[column].apply(lambda x : x.replace('ό','ο'))
    frame[column] = frame[column].apply(lambda x : x.replace('ώ','ω'))
    frame[column] = frame[column].apply(lambda x : x.replace('-',' '))
    frame[column] = frame[column].apply(lambda x : x.replace('–',' '))
    frame[column] = frame[column].apply(lambda x : x.replace('ν. ',''))

    
    frame[column] = frame[column].apply(lambda x : _RE_COMBINE_WHITESPACE.sub(" ", x).strip())

def process_italic(frame, column):
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    frame[column] = frame[column].apply(lambda x : x.lower())
    frame[column] = frame[column].apply(lambda x : _RE_COMBINE_WHITESPACE.sub(" ", x).strip())
    frame[column] = frame[column].apply(lambda x : re.sub(r'\([^)]*\)', '', x))
    frame[column] = frame[column].apply(lambda x : remove_accents(x))
    frame[column] = frame[column].apply(lambda x : x.replace("'", " " ))
    frame[column] = frame[column].apply(lambda x : x.replace("-", " " ))
    frame[column] = frame[column].apply(lambda x : x.replace("’", " " ))
    
    frame[column] = frame[column].apply(lambda x : _RE_COMBINE_WHITESPACE.sub(" ", x).strip())
    frame[column] = frame[column].apply(lambda x : x.replace("s. ", "saint " ))
    frame[column] = frame[column].apply(lambda x : x.replace("san ", "saint " ))
    frame[column] = frame[column].apply(lambda x : x.replace("santa ", "saint " ))
    frame[column] = frame[column].apply(lambda x : x.replace("sant ", "saint " ))
    frame[column] = frame[column].apply(lambda x : x.replace("santo ", "saint " ))

    frame[column] = frame[column].apply(lambda x : x.replace(" d ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" di ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" del ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" dell ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" delle ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" della ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" da ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" de ", " da " ))


def fillna(dataframe,fill_list):
    for i in list(fill_list.keys()):
        if i not in dataframe.columns:
            raise KeyError('Column(s) not in index')
        else:
            dataframe[i] = dataframe[i].fillna(fill_list[i])
    return dataframe

def describe_dataframe(dataframe, rounding_factor = 4, remove_zeros = False):
    df_desc = pd.DataFrame(columns=['Column', 'Missing'])

    for col in dataframe.columns:
        missing_percentage = dataframe[col].isnull().sum() * 100 / len(dataframe[col])
    
        df_desc = df_desc.append({'Column': col, 
                                'Missing': round(missing_percentage, rounding_factor)}, ignore_index = True)
        
        if remove_zeros:
            df_desc = df_desc[df_desc["Missing"] != 0]

    return df_desc

def lau1_mapping(dataframe, x0, y0, lau1_col = 'lau1', map_col = 'lau1_id', dist_col = 'eq_distance'):
    data = dataframe
    data[dist_col] = data.apply(lambda row : round(math.sqrt(math.pow((row['x']-x0) , 2) + math.pow((row['y']-y0) , 2)),5), axis = 1)
    data_sub = data[[lau1_col, dist_col]]
    data_sorted = data_sub.sort_values(dist_col).drop_duplicates(ignore_index = True)
    data_sorted[map_col] = data_sorted.index
    
    mapping = {}
    for index, row in data_sorted.iterrows():
        mapping[row[lau1_col]] = row[map_col]
        
    return mapping

def encode_cyclical(dataframe, col, max_val):
    dataframe[col + '_sin'] = np.sin(2 * np.pi * dataframe[col]/max_val)
    dataframe[col + '_cos'] = np.cos(2 * np.pi * dataframe[col]/max_val)

    return dataframe

def dataset_mamoth_preparation(dataframe, columns, del_months = None, month_col = 'month', mean_lst = True):

    dataset_mamoth = dataframe.copy()
    if mean_lst:
        dataset_mamoth['lst_jan_mean'] = dataset_mamoth.apply(lambda row: mean([row['lst_jan_day_mean'],row['lst_jan_night_mean']]), axis=1)
        dataset_mamoth['lst_feb_mean'] = dataset_mamoth.apply(lambda row: mean([row['lst_feb_day_mean'],row['lst_feb_night_mean']]), axis=1)
        dataset_mamoth['lst_mar_mean'] = dataset_mamoth.apply(lambda row: mean([row['lst_mar_day_mean'],row['lst_mar_night_mean']]), axis=1)
        dataset_mamoth['lst_apr_mean'] = dataset_mamoth.apply(lambda row: mean([row['lst_apr_day_mean'],row['lst_apr_night_mean']]), axis=1)
    dataset_mamoth['mosq_now'] = 0

    if del_months != None:
        dataset_mamoth = dataset_mamoth[~dataset_mamoth[month_col].isin(del_months)]

    dataset_mamoth = dataset_mamoth[columns]

    dataset_mamoth.reset_index(inplace=True, drop=True)

    return dataset_mamoth

def calculate_mosq_previous(dataframe, lau1_col = 'lau1', date_col = 'dt_placement', mosq_prediction_col = 'mosq_pred', result_col = 'mosq_previous'):

    for municipality in dataframe[lau1_col].drop_duplicates():

        data = dataframe.loc[dataframe[lau1_col] == municipality].copy()
        data.sort_values(by=[date_col], inplace = True)
        data.reset_index(inplace = True, drop = True)
        
        for index, row in data.iterrows():
        
            try:
                mosq_minus_2 = data.loc[index-2, mosq_prediction_col]
            except KeyError:
                mosq_minus_2 = 0
        
            try:
                mosq_minus_1 = data.loc[index-1, mosq_prediction_col]
            except KeyError:
                mosq_minus_1 = 0
            
            mosq_sum = mosq_minus_2 + mosq_minus_1
        
            index_at_dataframe = dataframe.index[(dataframe[lau1_col] == row[lau1_col]) & (dataframe[date_col] == row[date_col])].tolist()
            if (len(index_at_dataframe) != 1):
                raise(IndexError("Duplicate Index Found"))
            
            dataframe.at[index_at_dataframe[0], result_col] = mosq_sum

    return dataframe


def convert_temperature(dataframe, result_col = 'lst'):
    temprature_cols = dataframe.filter(regex='^lst_*').columns.tolist()

    dataframe[temprature_cols] = dataframe[temprature_cols].apply(lambda x : x * 0.02 - 273.15)
    dataframe[result_col] = (dataframe['lst_day'] + dataframe['lst_night'])/2

    return dataframe

def convert_multiple_cases(dataframe, target_col = 'case'):

    case_counts = dataframe[target_col].value_counts().index

    for case_count in case_counts:
        if (case_count >= 2):
            dataframe = dataframe.append([dataframe[dataframe[target_col] == case_count]] * (case_count - 1), ignore_index=True)
            
    dataframe[target_col] = dataframe[target_col].apply(lambda x : 1 if(x > 0) else 0)

    return dataframe

def calculate_nearest_topological(data, topological, neighbors=1):
    topological['x_rad'] = topological['x'].apply(lambda x: np.deg2rad(x))
    topological['y_rad'] = topological['y'].apply(lambda x: np.deg2rad(x))
    
    data['x_rad'] = data['x'].apply(lambda x: np.deg2rad(x))
    data['y_rad'] = data['y'].apply(lambda x: np.deg2rad(x))
    
    ball = BallTree(topological[["y_rad", "x_rad"]].values, metric='haversine')
    distances, indices = ball.query(data[["y_rad", "x_rad"]].values, k = neighbors)
    
    if neighbors>1:
        distances = [mean(d) for d in distances]
        distances = [(d * 6371) for d in distances]
    else:
        distances = [(d * 6371).tolist()[0] for d in distances]
        
    indices = indices.tolist()
    indices = [i[0] for i in indices]
    del data['x_rad']
    del data['y_rad']
    del topological['x_rad']
    del topological['y_rad']
    
    return distances, indices

def calculate_nearest_point_temporal(data, topological, date_col = 'dt_placement', neighbors=1):
    topological['x_rad'] = np.deg2rad(topological['x'])
    topological['y_rad'] = np.deg2rad(topological['y'])

    data['x_rad'] = np.deg2rad(data['x'])
    data['y_rad'] = np.deg2rad(data['y'])

    topological['timestamp'] = topological[date_col].view('int64') // 10**9
    data['timestamp'] = data[date_col].view('int64') // 10**9

    coordinates_topological = topological[['y_rad', 'x_rad']].values
    coordinates_data = data[['y_rad', 'x_rad']].values

    distances = cdist(coordinates_data, coordinates_topological, haversine_distance)

    if neighbors > 1:
        distances = np.mean(distances, axis=1)
    else:
        distances = distances[:, 0]

    indices = np.argsort(distances)
    distances = distances[indices]
    indices = indices[:neighbors]

    del data['x_rad']
    del data['y_rad']
    del data['timestamp']
    del topological['x_rad']
    del topological['y_rad']
    del topological['timestamp']

    return distances.tolist(), indices.tolist()

def haversine_distance(coords1, coords2):
    lat1, lon1 = coords1
    lat2, lon2 = coords2

    R = 6371  # Earth's radius in kilometers

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def fuzzy_merge(df1, df2, on, date_column, cutoff, scorer, limit=1):
    df_merged = pd.merge(df1, df2, on=on, how='outer')
    df_merged['Score'] = df_merged.apply(lambda row: scorer(row[on[0]], row[on[1]]), axis=1)
    df_merged = df_merged[df_merged['Score'] >= cutoff]
    df_merged[date_column] = df_merged.apply(lambda row: process.extractOne(row[date_column + '_x'], df2[date_column], scorer=scorer)[0], axis=1)
    df_merged = df_merged.groupby(df_merged.columns.tolist()).apply(lambda x: x.head(limit))
    df_merged.reset_index(drop=True, inplace=True)
    df_merged.drop('Score', axis=1, inplace=True)
    return df_merged

def compute_distance(row, x1 = 'x1', y1 = 'y1', x2 = 'x2', y2 = 'y2'):
    start_point = (row[x1], row[y1])
    end_point = (row[x2], row[y2])
    distance = geodesic(start_point, end_point).meters
    return distance