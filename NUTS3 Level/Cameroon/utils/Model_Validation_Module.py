import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

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

def plot_hist_plotly(actual, predictions, case='', f_height=500, f_width=800):
    """
    Plots overlapping histograms of actual and predicted values using Plotly.

    Parameters
    ----------
    actual : pd.Series or array-like
        Series with the actual values.

    predictions : pd.Series or array-like
        Series with the predicted values.

    case : str, optional
        Title of the plot.
    """

    all_values = np.concatenate([actual, predictions])
    min_val = int(np.floor(all_values.min()))
    max_val = int(np.ceil(all_values.max()))


    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=actual,
        name='Actual',
        opacity=0.6,
        marker_color='#8fbad8',
        xbins=dict(start=min_val - 0.5, end=max_val + 0.5, size=1)
    ))

    fig.add_trace(go.Histogram(
        x=predictions,
        name='Prediction',
        opacity=0.6,
        marker_color='#febd89',
        xbins=dict(start=min_val - 0.5, end=max_val + 0.5, size=1)
    ))

    fig.update_layout(
        title=f'Histogram of Actual vs Predicted Values<br>{case}',
        xaxis_title='Value',
        yaxis_title='Count',
        barmode='overlay',
        width=f_width,
        height=f_height,
        bargap=0.1,
        legend=dict(x=0.8, y=0.95)
    )

    fig.update_xaxes(
        tickmode='linear',
        tick0=min_val,
        dtick=1
    )

    fig.show()

def plot_error_by_group(
    df,
    y_true_col='actual',
    y_pred_col='prediction',
    date_col=None,
    group_by='class',  # 'class', 'month', or 'year'
    metric='mae',      # 'mae' or 'mse'
    title='',
    x_axis_label=None,
    y_axis_label=None,
    f_width=800,
    f_height=500
):
    """
    Plots the mean error (MAE or MSE) per group using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and predicted values.

    y_true_col : str
        Column name of ground truth labels.

    y_pred_col : str
        Column name of predicted values.

    date_col : str or None
        Column with datetime values. Required if group_by is 'month' or 'year'.

    group_by : str
        Grouping strategy: 'class', 'month', or 'year'.

    metric : str
        Error metric to compute: 'mae' (mean absolute error) or 'mse' (mean squared error).

    title : str
        Plot title.

    x_axis_label : str or None
        X-axis label. Defaults to group_by capitalized.

    y_axis_label : str or None
        Y-axis label. Defaults to 'MAE' or 'MSE'.
    """

    # Choose error function
    if metric == 'mae':
        error_func = mean_absolute_error
        error_label = 'MAE'
    elif metric == 'mse':
        error_func = mean_squared_error
        error_label = 'MSE'
    else:
        raise ValueError("Metric must be 'mae' or 'mse'.")

    # Determine groups
    if group_by == 'class':
        groups = sorted(df[y_true_col].unique())
        group_values = [str(g) for g in groups]
        grouped = [df[df[y_true_col] == g] for g in groups]
    elif group_by in ['month', 'year']:
        if date_col is None:
            raise ValueError("You must provide 'date_col' when grouping by month or year.")
        date_series = df[date_col]

        # Handle PeriodDtype
        if pd.api.types.is_period_dtype(date_series):
            date_series = date_series.dt.to_timestamp()

        date_series = pd.to_datetime(date_series, errors='coerce')

        if group_by == 'month':
            df['_group'] = date_series.dt.month
        else:  # year
            df['_group'] = date_series.dt.year

        groups = sorted(df['_group'].dropna().unique())
        group_values = [str(g) for g in groups]
        grouped = [df[df['_group'] == g] for g in groups]
    else:
        raise ValueError("group_by must be 'class', 'month', or 'year'.")

    # Compute error per group
    errors = []
    counts = []
    for gdf in grouped:
        y_true = gdf[y_true_col]
        y_pred = gdf[y_pred_col]
        # Handle categorical case
        if pd.api.types.is_categorical_dtype(y_true):
            y_true = y_true.cat.codes
        error = error_func(y_true, y_pred)
        errors.append(error)
        counts.append(len(gdf))

    # Add margin above tallest bar
    max_error = max(errors)
    y_max = max_error * 1.15

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=group_values,
        y=errors,
        text=[f"{e:.2f}<br>n={n}" for e, n in zip(errors, counts)],
        textposition='outside',
        marker_color='steelblue'
    ))

    fig.update_layout(
        title=title or f"{error_label} per {group_by}",
        xaxis_title=x_axis_label or group_by.capitalize(),
        yaxis_title=y_axis_label or error_label,
        width=f_width,
        height=f_height,
        margin=dict(t=60, b=60),
        bargap=0.3,
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    fig.update_yaxes(range=[0, y_max])
    fig.show()

    # Clean up temporary column
    if '_group' in df.columns:
        df.drop(columns=['_group'], inplace=True)

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
def metrics(train, test, threshold = None):
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
    print(f"MAE on train set: {mean_absolute_error(train['actual'], train['prediction']):.4f}")
    print(f"MSE on train set: {mean_squared_error(train['actual'], train['prediction']):.4f}")

    print('min prediction:',min(train['prediction']))
    print('max prediction:',max(train['prediction']))
    
    print()

    print(f"MAE on test set: {mean_absolute_error(test['actual'], test['prediction']):.4f}")
    print(f"MSE on test set: {mean_squared_error(test['actual'], test['prediction']):.4f}")

    if threshold is not None:
        is_cat = pd.api.types.is_categorical_dtype(test['actual'])
        if is_cat:
            perc = ((np.abs(test['actual'].cat.codes-test['prediction']) < (threshold+0.5)).mean())*100
        else:
            perc = ((np.abs(test['actual']-test['prediction']) < (threshold+0.5)).mean())*100

        print('Error <= '+str(threshold)+':',"%.2f"%perc,'%')

    print('min prediction:',min(test['prediction']))
    print('max prediction:',max(test['prediction']))
    
class CategoricalKFold:
    def __init__(self, n_splits=None, col=None):
        """
        Parameters:
        -----------
        n_splits : int or None
            Number of unique categories (optional, inferred from data if None).
        col : str
            Column name in the DataFrame to split on.
        """
        self.n_splits = n_splits
        self.col = col

    def split(self, X, y=None):
        """
        Parameters:
        -----------
        X : pd.DataFrame
            The input data containing the column to split on.
        y : Ignored (included for compatibility with scikit-learn).
        
        Yields:
        -------
        train_idx : np.array
            Indices for the training set.
        test_idx : np.array
            Indices for the test set.
        """
        if self.col is None:
            raise ValueError("You must specify the column to split on with 'col' parameter.")

        if self.col not in X.columns:
            raise ValueError(f"Column '{self.col}' not found in input DataFrame.")

        unique_values = pd.Series(X[self.col].unique()).sort_values().tolist()
        if self.n_splits is not None and self.n_splits != len(unique_values):
            raise ValueError("Provided n_splits does not match the number of unique categories.")

        for val in unique_values:
            test_mask = X[self.col] == val
            test_idx = X[test_mask].index.to_numpy()
            train_idx = X[~test_mask].index.to_numpy()
            yield train_idx, test_idx

