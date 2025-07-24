# %%
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# %%
def normalizeData(y, min_val=0, max_val=9):
    'Transforms the target data to 0-1 range in order to use the sigmoid function as activation'
    return pd.Series((np.array(y) - min_val) / (max_val-min_val))

def logData(y):
    'Transforms the target variable to log(target) in order to follow a more normal disribution'
    return np.log1p(y)

def expData(y):
    'Calculate the exponential value of the target variable'
    return np.expm1(y)

def denormalizeData(y, min_val=0, max_val=9):
    'Transforms the data from 0-1 range to the initial 0-9 range'
    return pd.Series((np.array(y)*(max_val-min_val))+min_val)

def sigmoid(x,a=1,b=0):
    return 1.0 / (1.0 + np.exp(a*(-x+b)))

# %%
def transformations_xgboost(data, model_type, test=None, evaluation=False, transformation_list=['scaling'],
                    embedding_data=None):
    """Executes scaling, augmentation or label endconding on the dataset 
    
    Parameters
    ----------
    data : dataframe
        A dataframe containing all the data or the data of trainning.
        If test not given, then data will be randomnly slpit in 80-20% train and test set
        
    model_type : str
        The type of the model to be implemented.
        Could be 'class_regression' or 'mosquito_regression' or 'classification'
        
    test : dataframe, optional
        A dataframe containing all the data for testing (default = None)
        
    scaling : boolean, optional
        If True, perofrms scaling on numerical features (default = False)
        
    augment : boolean, optional
        If True, augments the train data with existing observations
        and giving greater weight on the observations with greater target value (default = False)
        
    embedding_data : dataframe, optional
        A datafrane with the categorical features (default = None)
        
    evaluation : boolean, optional
        If True, 20% of the observations of train set will be held for evaluation set (default = False)
        
    Returns
    ----------
    train_X: numpy array
        A numpy array with independent variables for training
        
    train_y: pd.Series
        A array with the dependent variables (target) for training
        
    test_X: numpy array
        A numpy array with independent variables for test
        
    test_y: pd.Series
        A array with the dependent variables (target) for test
    
    """

    if test is None:    
        X, y = data.iloc[:,:-1], data.iloc[:,-1]
        train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=0.20, random_state=1)
    else:
        data = data.sample(frac=1,random_state=1).reset_index(drop=True)
        train_X, train_y = data.iloc[:,:-1], data.iloc[:,-1]
        test_X, test_y = test.iloc[:,:-1], test.iloc[:,-1]
        
    train_X = train_X.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
        
    if evaluation:
        train_X,eval_X,train_y,eval_y = train_test_split(train_X, train_y, test_size=0.20, random_state=1)
        eval_X = eval_X.reset_index(drop=True)
        eval_y = eval_y.reset_index(drop=True)
        
    if model_type == 'mosquito_regression':       
        percentile = round(np.percentile(train_y, 95))
        train_y.loc[train_y >= percentile] = percentile

    if 'augmentation' in transformation_list:
        augment_index = train_y.sample(frac=0.4, weights=train_y, random_state=1, replace=True).index
        train_X = pd.concat([train_X,train_X.iloc[augment_index,:]]).reset_index(drop=True)
        train_y = pd.concat([train_y,train_y[augment_index]]).reset_index(drop=True)
    
    if embedding_data is not None:
        embeddings = embedding_data.columns.tolist()
        
        embedded_columns_train = train_X[embeddings] #categorical columns
        train_X = train_X.drop(columns=embeddings)

        embedded_columns_test = test_X[embeddings] #categorical columns
        test_X = test_X.drop(columns=embeddings)           
            
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(embedded_columns_train)
        if evaluation:
            embedded_columns_eval = eval_X[embeddings] #categorical columns
            eval_X = eval_X.drop(columns=embeddings)
            embedded_columns_eval= encoder.transform(embedded_columns_eval)
        embedded_columns_train = encoder.transform(embedded_columns_train)
        embedded_columns_test = encoder.transform(embedded_columns_test)

        train_X_emb = pd.DataFrame(embedded_columns_train,columns=encoder.get_feature_names_out().tolist())  
        test_X_emb = pd.DataFrame(embedded_columns_test,columns=encoder.get_feature_names_out().tolist())
        if evaluation:
            eval_X_emb = pd.DataFrame(embedded_columns_eval,columns=encoder.get_feature_names_out().tolist())
            
    if 'scaling' in transformation_list:
        scaler = StandardScaler()
        cols = train_X.columns.tolist()
        train_X = scaler.fit_transform(train_X)
        train_X = pd.DataFrame(train_X, columns=cols, dtype='category')
        test_X = scaler.transform(test_X)
        test_X = pd.DataFrame(test_X, columns=cols, dtype='category')
        if evaluation:
            eval_X = scaler.transform(eval_X)
            eval_X = pd.DataFrame(eval_X, columns=cols, dtype='category')
            
    if 'normalization' in transformation_list:
        min_val = train_y.min()
        max_val = train_y.max()
        test_y = normalizeData(test_y, min_val, max_val)
        train_y = normalizeData(train_y, min_val, max_val)
        if evaluation:
            eval_y = normalizeData(eval_y, min_val, max_val)
            
    if 'log' in transformation_list:
        test_y = logData(test_y)
        train_y = logData(train_y)
        if evaluation:
            eval_y = logData(eval_y)
            
    if embedding_data is not None:
        train_X = pd.concat([train_X,train_X_emb],axis=1)
        test_X = pd.concat([test_X, test_X_emb],axis=1)
        if evaluation:
            eval_X = pd.concat([eval_X,eval_X_emb],axis=1)
            
    if evaluation:
        return train_X, train_y, eval_X, eval_y, test_X, test_y
    else:       
        return train_X, train_y, test_X, test_y

# %%
class Xgboost_model:
    def __init__(self, model_type, learning_rate=0.3,
                 embedding_data=None,  transformation_list=[],
                 early_stop = False, l1_weight=0, l2_weight=0, weights=False):
        
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.embedding_data = embedding_data
        self.transformation_list = transformation_list
        self.early_stop = early_stop
        self.weights = weights
        self.estimators = None
        self.depth = None
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.features = None
    
        if self.model_type == 'classification':
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBRegressor()
            
        xgb_params = {'learning_rate': self.learning_rate,
                      'random_state':1,
                      'lambda':self.l2_weight,
                      'alpha':self.l1_weight}
        self.model.set_params(**xgb_params)
        
    def tune_parameters(self, train_X, train_y, flag = False):
        """Returns the depth and the number of estimators with the minimum MAE

        Parameters
        ----------
        data : Dataframe
            A Daframe containing the data

        flag : boolean, otional
            If true plots are printed (default=False)

        transform : boolean, optional
            If True, performs log transformation of the target variable (default = False)

        Returns
        ----------
        depth : int
            The max depth of the trees to construct

        est : int
            The number of trees to construct

        """
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.20, random_state=1)
        
        tr_depth = []
        val_depth = []
        tr_est = []
        val_est = []

        for i in range(1,11):
            xgb_params = {'n_estimators': 5,
                          'max_depth': i}
            self.model.set_params(**xgb_params)
            self.model.fit(train_X, train_y)               
            predictions_train = self.model.predict(train_X)
            predictions_val = self.model.predict(val_X)
    #         if 'log' in self.transform_list :
    #             predictions_train = expData(predictions_train)
    #             predictions_val = expData(predictions_val)
    #         if 'normalization' in self.transform_list :
    #             predictions_train = denormalizeData(predictions_train,max_val=train_y.max())
    #             predictions_val = denormalizeData(predictions_val,max_val=train_y.max())
    #         predictions_train = np.round(predictions_train)
    #         predictions_val = np.round(predictions_val)
            mae_train = mean_absolute_error(train_y, predictions_train)
            mae_val = mean_absolute_error(val_y, predictions_val)
            tr_depth.append(mae_train)
            val_depth.append(mae_val)
        self.depth = val_depth.index(min(val_depth)) + 1


        for i in range(1,36):
            xgb_params = {'n_estimators': i,
                          'max_depth': self.depth}
            self.model.set_params(**xgb_params)
            self.model.fit(train_X, train_y)                
            predictions_train = self.model.predict(train_X)
            predictions_val = self.model.predict(val_X)
    #         if 'log' in self.transform_list :
    #             predictions_train = expData(predictions_train)
    #             predictions_val = expData(predictions_val)
    #         if 'normalization' in self.transform_list :
    #             predictions_train = denormalizeData(predictions_train,max_val=train_y.max())
    #             predictions_val = denormalizeData(predictions_val,max_val=train_y.max())
    #         predictions_train = np.round(predictions_train)
    #         predictions_val = np.round(predictions_val)
            mae_train = mean_absolute_error(train_y, predictions_train)
            mae_val = mean_absolute_error(val_y, predictions_val)
            tr_est.append(mae_train)
            val_est.append(mae_val)
        self.estimators = val_est.index(min(val_est)) + 1

        if flag:
            labels = list(range(1,len(tr_depth)+1))
            fig, ax = plt.subplots()
            plt.grid()
            plt.plot(labels,tr_depth,label='train')
            plt.plot(labels,val_depth,label='validation')
            plt.vlines(x = self.depth,ls='--', ymin = 0, ymax = max(max(tr_depth),max(val_depth)),
                       colors = 'grey', label = 'x = '+str(self.depth))
            ax.set_xticks(np.arange(len(tr_depth)+1))
            plt.legend()
            plt.xlabel('max_depth')
            plt.ylabel('MAE')
            plt.show()

            labels = list(range(1,len(tr_est)+1))
            fig, ax = plt.subplots(figsize=(10,6))
            plt.grid()
            plt.plot(labels,tr_est,label='train')
            plt.plot(labels,val_est,label='validation')
            plt.vlines(x = self.estimators,ls='--', ymin = 0, ymax = max(max(tr_est),max(val_est)),
                       colors = 'grey', label = 'x = '+str(self.estimators))
            ax.set_xticks(np.arange(len(tr_est)+1))
            plt.legend()
            plt.xlabel('n_estimators')
            plt.ylabel('MAE')
            plt.show()

        return self.depth, self.estimators
    
    def select_features(self, train_X, train_y, grid=False):
        """Selects which features to use in the training process.

        Parameters
        --------
        df : dataframe
            A dataframe containing the data

        depth : int
            The dpeth of the trees to construct

        estimators : int
            The number of trees to construct

        grid : boolean, optional
            If True it creates a grid of points around the (depth, estimator) in search of a better combination.
            However it highly increases time complexity (default=False)


        Returns
        --------
        selected_features : lst
            A list containing the features selected

        depth : int
            The depth of the trees to construct

        estimators : int
            The number of trees to constuct
        """   
        if grid:
            r2=-1
            final = None
            dep=0
            est = 0
            for i in range(self.depth-2,self.depth+3):
                for j in range(self.estimators-2,self.estimators+3):
                    xgb_params = {'n_estimators': i,
                                  'max_depth': j}
                    self.model.set_params(**xgb_params)
                    rfe = RFECV(self.model, step=1, cv=10,scoring = 'neg_mean_absolute_error')
                    X_train_rfe = rfe.fit_transform(train_X,train_y)
                    if rfe.score(X_train,y_train) > r2:
                        final = rfe
                        dep = i
                        est = j
                        r2 = rfe.score(X_train,y_train)
            rfe = final
            self.depth = dep
            self.estimators = est
        else:
            xgb_params = {'n_estimators': self.estimators,
                          'max_depth': self.depth}
            self.model.set_params(**xgb_params)
            rfe = RFECV(self.model, step=1, cv=10,scoring = 'neg_mean_absolute_error')
            X_train_rfe = rfe.fit_transform(train_X,train_y)

        #print the features selected
        cols = list(train_X.columns) 
        temp = pd.Series(rfe.support_,index = cols)
        self.features = temp[temp==True].index

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("neg mean squared error")
        plt.plot(range(1,len(rfe.cv_results_['mean_test_score']) + 1), rfe.cv_results_['mean_test_score'])
        plt.vlines(x = np.argmax(rfe.cv_results_['mean_test_score'])+1,
                   ls='--', ymin = np.min(rfe.cv_results_['mean_test_score']),
                   ymax = np.max(rfe.cv_results_['mean_test_score']), colors = 'grey',
                   label = 'x = '+str(np.argmax(rfe.cv_results_['mean_test_score'])+1))
        plt.grid()
        plt.legend()
        plt.show()

        print('max_depth: ', self.depth)
        print('Number of estimators: ', self.estimators)
        print('Selected_features: ', self.features)

        return self.features, self.depth, self.estimators
    
    def plot_feature_importance(self):
        """Prints the plot of feature importance of the model and creates a .csv file with the frature importance

        Parameters
        ----------
        importance : list
            A list containing the importnce of each feature

        names : list
            A list containing the names of the features

        filepath : srt
            The path of the file to export a csv with the importance of the featured

        case: str, optional
            Title of the plot (Area and mosquito genus) (default= '')

        export: boolean, optional
            Exports a csv with the importance of each figure (default = False)
        """
        #Create arrays from feature importance and feature names
        feature_importance = np.array(self.model.feature_importances_)
        feature_names = np.array(self.features)

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

        #Define size of bar plot
        plt.figure(figsize=(10,8))

        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

        #Add chart labels
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Names')
        plt.show()
        
        indicies = ['ndvi', 'ndmi', 'ndwi', 'ndbi',
            'ndvi_mean', 'ndmi_mean', 'ndwi_mean', 'ndbi_mean',
            'ndvi_std','ndmi_std', 'ndwi_std', 'ndbi_std',]

        weather = ['lst', 'lst_day', 'lst_night',
                   'lst_jan_day_mean', 'lst_jan_night_mean',
                   'lst_feb_day_mean', 'lst_feb_night_mean',
                   'lst_mar_day_mean', 'lst_mar_night_mean',
                   'lst_apr_day_mean', 'lst_apr_night_mean',
                   'acc_rainfall_1week', 'acc_rainfall_2week', 'acc_rainfall_jan',]

        geomorphological = ['DISTANCE_TO_COAST', 'DISTANCE_TO_RIVER', 'SLOPE_mean_1km', 'ASPECT_mean_200m',
                            'ELEVATION_mean_1km', 'HILLSHADE_mean_1km', 'FS_AREA_1km', 'FLOW_ACCU_200m', 'landcover']

        spatiotemporal = ['x', 'y', 'year''null_island_distance', 'vert_distance',
                          'days_distance', 'mo_sin', 'mo_cos', 'year',  'summer_days_month']

        entomological = ['mosq_sum_month', 'mosq_sum_month_previous_year', 'mosq_sum_year',
                         'mosq_sum_previous_2weeks', 'previous_mosq_measure',]

        df_indicies = fi_df.loc[fi_df['feature_names'].isin(indicies)]
        df_weather = fi_df.loc[fi_df['feature_names'].isin(weather)]
        df_geomorphological = fi_df.loc[fi_df['feature_names'].isin(geomorphological)]
        df_spatiotemporal = fi_df.loc[fi_df['feature_names'].isin(spatiotemporal)]
        df_entomological = fi_df.loc[fi_df['feature_names'].isin(entomological)]

        categories = {'Category': ['rs_indicies', 'weather', 'geomorphological', 'spatiotemporal', 'entomological'],

                      'Mean_fi': [df_indicies.feature_importance.sum(),
                                  df_weather.feature_importance.sum(),
                                  df_geomorphological.feature_importance.sum(),
                                  df_spatiotemporal.feature_importance.sum(),
                                  df_entomological.feature_importance.sum()]}

        categories_df = pd.DataFrame.from_dict(categories)

        categories_df = categories_df.loc[(categories_df!=0).all(axis=1)].reset_index(drop=True)
        categories_df = categories_df.dropna().reset_index(drop=True)

        categories_df.sort_values(by = 'Mean_fi', ascending=False)


        #Sort the DataFrame in order decreasing feature importance
        categories_df.sort_values(by=['Mean_fi'], ascending=False,inplace=True)

        #Define size of bar plot
        plt.figure(figsize=(10,8))

        #Plot Searborn bar chart
        sns.barplot(x=categories_df['Mean_fi'], y=categories_df['Category'])

        #Add chart labels
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Category')
        plt.show()
        
        categories = {'Category': ['rs_indicies', 'weather', 'geomorphological', 'spatiotemporal', 'entomological'],

                      'Mean_fi': [df_indicies.feature_importance.mean(),
                                  df_weather.feature_importance.mean(),
                                  df_geomorphological.feature_importance.mean(),
                                  df_spatiotemporal.feature_importance.mean(),
                                  df_entomological.feature_importance.mean()]}

        categories_df = pd.DataFrame.from_dict(categories)

        categories_df = categories_df.loc[(categories_df!=0).all(axis=1)].reset_index(drop=True)
        categories_df = categories_df.dropna().reset_index(drop=True)

        categories_df.sort_values(by = 'Mean_fi', ascending=False)


        #Sort the DataFrame in order decreasing feature importance
        categories_df.sort_values(by=['Mean_fi'], ascending=False,inplace=True)

        #Define size of bar plot
        plt.figure(figsize=(10,8))

        #Plot Searborn bar chart
        sns.barplot(x=categories_df['Mean_fi'], y=categories_df['Category'])

        #Add chart labels
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Category')
        plt.show()

# %%
def train_xgboost(xg_model, train_X, train_y, test_X, test_y, max_val, fi=True):
    """ Trainning of the model
    
    Parameters
    ----------
    model : FeedforwardNeuralNetModel
        A FeedforwardNeuralNetModel model
        
    train_set : Dataset
        A Dataset object with the train set
        
    test_set : Dataset
        A Dataset object with the test set
        
    learning_rate : int, optional
        The learning_rate of the training process. (default = None)
        
    epochs : int, optional
        The number of epochs for the training. (default = None)
        
    batch_size : int, optional
        The size of each batch in each iteration. (default = None)
        
    ealry_stop : tuple, optional
        Set the (tolerance,min_dleta), the trainning of the model may stop earlier than the epochs defined. (default = None)
        
    Returns
    ----------
    results_train: DataFrame
        A Dataframe containing the actual and the predicted values on the train set
        
    results_test: DataFrame
        A Dataframe containing the actual and the predicted values on the test set
        
    model : FeedforwardNeuralNetModel
        A trained FeedforwardNeuralNetModel model 
    """
    weights=None
    if xg_model.weights:
        b_val = train_y.mean().item()
        weights = [sigmoid(train_y[i],0.005,b=b_val) for i in range(int(train_X.shape[0]))]
    
    if xg_model.features is None:
        xg_model.tune_parameters(train_X, train_y, flag=True)
        xg_model.select_features(train_X, train_y, grid=False)
    
    train_X = train_X[xg_model.features]
    test_X = test_X[xg_model.features]

    xgb_params = {'n_estimators': xg_model.estimators,
                  'max_depth': xg_model.depth}
    xg_model.model.set_params(**xgb_params)
#     eval_set = [(test_X, test_y)]
    xg_model.model.fit(train_X, train_y, sample_weight = weights)
#                        ,early_stopping_rounds=10, eval_metric="mae", eval_set=eval_set)    
    predictions = xg_model.model.predict(test_X)
    predictions_train = xg_model.model.predict(train_X)
    
    if 'normalization' in xg_model.transformation_list:
        predictions = denormalizeData(predictions, max_val=max_val)
        predictions_train = denormalizeData(predictions_train, max_val=max_val)
        train_y = denormalizeData(train_y, max_val=max_val)
        test_y = denormalizeData(test_y, max_val=max_val)
        
    if 'log' in xg_model.transformation_list:
        predictions = expData(predictions)
        predictions_train = expData(predictions_train)
        train_y = expData(train_y)
        test_y = expData(test_y)
        
    predictions =  np.round(predictions)  
    predictions_train = np.round(predictions_train)
        
    results_train = {'actual': train_y, 'prediction': predictions_train}
    results_train = pd.DataFrame.from_dict(results_train)
    
    results_test = {'actual': test_y, 'prediction': predictions}
    results_test = pd.DataFrame.from_dict(results_test)
    
    results_test.loc[results_test['prediction'] < 0,'prediction'] = 0
    results_test.loc[results_test['prediction'] > max_val,'prediction'] = max_val
    results_train.loc[results_train['prediction'] < 0,'prediction'] = 0
    results_train.loc[results_train['prediction'] > max_val,'prediction'] = max_val
    
    if fi:
        xg_model.plot_feature_importance()
    
    return results_train, results_test

# %%
def predict_xgboost(xg_model, train_X, train_y, test_X, test_y, max_val, fi=True):
    """ Trainning of the model
    
    Parameters
    ----------
    model : FeedforwardNeuralNetModel
        A FeedforwardNeuralNetModel model
        
    Returns
    ----------
    results_train: DataFrame
        A Dataframe containing the actual and the predicted values on the train set
        
    results_test: DataFrame
        A Dataframe containing the actual and the predicted values on the test set
        
    model : FeedforwardNeuralNetModel
        A trained FeedforwardNeuralNetModel model 
    """
    
    weights=None
    if xg_model.weights:
        b_val = train_y.mean().item()
        weights = [sigmoid(train_y[i],0.005,b=b_val) for i in range(int(train_X.shape[0]))]
        
    if xg_model.features is None:
        xg_model.tune_parameters(train_X, train_y, flag=True)
        xg_model.select_features(train_X, train_y, grid=False)
    
    train_X = train_X[xg_model.features]
    test_X = test_X[xg_model.features]

    xg_model.model.fit(train_X, train_y, sample_weight = weights)
    predictions = xg_model.model.predict(test_X)
    
    if 'normalization' in xg_model.transformation_list:
        predictions = denormalizeData(predictions, max_val=max_val)

    if 'log' in xg_model.transformation_list:
        predictions = expData(predictions)

    predictions =  np.round(predictions)  

    results_test = {'prediction': predictions}
    results_test = pd.DataFrame.from_dict(results_test)
    
    results_test.loc[results_test['prediction'] < 0,'prediction'] = 0
    results_test.loc[results_test['prediction'] > max_val,'prediction'] = max_val
    
    if fi:
        xg_model.plot_feature_importance()
   
    return results_test


