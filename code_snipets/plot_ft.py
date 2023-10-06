from utils import *
import numpy as np
import pandas as pd

ft_importance = pd.read_csv('ft_csv.csv')

ft_importance.sort_values(by=['Loss'], ascending=True, inplace=True)

losses = ft_importance['Loss']

losses = losses.to_numpy()

losses = losses.reshape(1,62)

plot_feature_importance(losses, ft_importance['Feature'])
