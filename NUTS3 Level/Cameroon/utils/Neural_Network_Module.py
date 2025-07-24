# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shap
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# %%
def vectorise(target,num):
    """Creates an one hot encoded vector based on the class of each observation.
    
    Parameters
    ----------
    target : pd.Series
        The series containing the target variable to be transformed to vector
        
    num_classes : int
        The number of different classes
        
    Returns
    ----------
    vectors : pd.Series
        A pd.Series object with one hot encoded vectors

    """
    enc = OneHotEncoder(sparse=False,categories = [list(range(num))], handle_unknown='ignore')
    enc.fit(np.array(target).reshape(-1, 1))
    target= target.apply(lambda x: enc.transform(np.array(x).reshape(1, -1))[0])
    target = np.stack(target)
    return target

# %%
def transformations_nn(data, model_type, test=None, evaluation=False, transformation_list=['scaling'],
                    embedding_data=None, test_size=0.2):
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
        train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=test_size, random_state=1)
    else:
        data = data.sample(frac=1,random_state=1).reset_index(drop=True)
        train_X, train_y = data.iloc[:,:-1], data.iloc[:,-1]
        test_X, test_y = test.iloc[:,:-1], test.iloc[:,-1]
        
    train_X = train_X.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
        
    if evaluation:
        train_X,eval_X,train_y,eval_y = train_test_split(train_X, train_y, test_size=test_size, random_state=1)
        eval_X = eval_X.reset_index(drop=True)
        eval_y = eval_y.reset_index(drop=True)
        
    if model_type == 'mosquito_regression':       
        percentile = round(np.percentile(train_y, 95))
        train_y.loc[train_y >= percentile] = percentile

    if 'augmentation' in transformation_list:
        augment_index = train_y.sample(frac=0.4, weights=train_y, random_state=1, replace=True).index
        train_X = pd.concat([train_X,train_X.iloc[augment_index,:]]).reset_index(drop=True)
        train_y = pd.concat([train_y,train_y[augment_index]]).reset_index(drop=True)
    
    if model_type == 'classification':
        num_classes = len(pd.concat([train_y,test_y]).unique())
        if evaluation:
            num_classes = len(pd.concat([train_y, test_y, eval_y]).unique())
            eval_y = vectorise(eval_y,num_classes)               
        train_y = vectorise(train_y,num_classes)
        test_y = vectorise(test_y,num_classes)

    if embedding_data is not None:
        embeddings = embedding_data.columns.tolist()
        embedded_columns_train = train_X.loc[:,embeddings] #categorical columns
        train_X = train_X.drop(columns=embeddings)

        embedded_columns_test = test_X.loc[:,embeddings] #categorical columns
        test_X = test_X.drop(columns=embeddings)
        
        if evaluation:
            embedded_columns_eval = eval_X.loc[:,embeddings] #categorical columns
            eval_X = eval_X.drop(columns=embeddings)
            
        label_encoder = LabelEncoder()
        for col in embeddings:
            label_encoder.fit(pd.concat([embedded_columns_train[col],embedded_columns_test[col]],axis=0))
            if evaluation:
                label_encoder.fit(pd.concat([embedded_columns_train[col],embedded_columns_test[col],embedded_columns_eval[col]],axis=0))
                embedded_columns_eval[col] = label_encoder.transform(embedded_columns_eval[col])
            embedded_columns_train[col] = label_encoder.transform(embedded_columns_train[col])
            embedded_columns_test[col] = label_encoder.transform(embedded_columns_test[col])

        train_X_emb = embedded_columns_train.values
        test_X_emb = embedded_columns_test.values
        if evaluation:
            eval_X_emb = embedded_columns_eval.values
            
    if 'scaling' in transformation_list:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        if evaluation:
            eval_X = scaler.transform(eval_X)
            
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
        train_X = [train_X,train_X_emb]
        test_X = [test_X, test_X_emb]
        if evaluation:
            eval_X = [eval_X,eval_X_emb]
    else:
        train_X = [train_X]
        test_X = [test_X]
        if evaluation:
            eval_X = [eval_X]

            
    if evaluation:
        if 'scaling' in transformation_list:
            return train_X, train_y, eval_X, eval_y, test_X, test_y, scaler
        else:
            return train_X, train_y, eval_X, eval_y, test_X, test_y
    else:
        if 'scaling' in transformation_list:
            return train_X, train_y, test_X, test_y, scaler
        else:          
            return train_X, train_y, test_X, test_y

# %%
class Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        """ Initialization of Dataset for the Neural Network.
        
        Parameters
        ----------
        X : numpy array
            An array with the independent variables
            
        y : numpy array
            An array with the dependent variables
        
        """
        
        self.emb = False
        
        self.X = X[0]
        self.X = torch.tensor(self.X, dtype=torch.float)
        
        if len(X)==2:
            self.X_emb = X[1] #categorical columns
            self.X_emb = torch.tensor(self.X_emb, dtype=torch.int)
            self.emb = True
        
        self.y = y
        if not (isinstance(y[0], list) or isinstance(y[0], np.ndarray)):
            self.y = [[e] for e in self.y]
            
        self.y = torch.tensor(self.y, dtype=torch.float)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        if self.emb:
            X1 = self.X[index]
            X2 = self.X_emb[index]
            y = self.y[index]
            return X1, X2, y
        else:
            X = self.X[index]
            y = self.y[index]
            return X, y

# %%
def normalizeData(y, min_val=0, max_val=9):
    'Transforms the target data to 0-1 range in order to use the sigmoid function as activation'
    return ((np.array(y) - min_val) / (max_val-min_val))

def logData(y):
    'Transforms the target variable to log(target) in order to follow a more normal disribution'
    return np.log1p(y)

def expData(data):
    'Calculate the exponential value of the target variable'
    return(np.expm1(data))

def denormalizeData(data, min_val=0, max_val=9):
    'Transforms the data from 0-1 range to the initial range'
    return((np.array(data)*(max_val-min_val))+min_val).tolist()

def sigmoid(x, a=1, b=0):
    return 1.0 / (1.0 + np.exp(a*(-x+b)))

# %%
class EarlyStopping():
    'Early stopping is a form of regularization used to avoid overfitting on the training dataset.'
    def __init__(self, tolerance=5, min_delta=0):
        """Set the tolerance and the min_delta for the early stopping 
        
        Parameters
        ----------
        tolerance : int, optinal
            The number of how many epochs to wait after validation score - training score
            is greater than min delta. (default = 5)
            
        min delta : int, optinal
            The threshold after which the difference of  validation score - training score
            is critical (default = 0)        
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop_flag = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop_flag = True

# %%
def set_embedding_sizes(data):
    """Calculates the size of the input and the output of the embedding layers.
    
    Parameters
    ----------
    data : numpy array
         An array containg the categorical features
         
     Returns
     ----------
     embedding_sizes: list
         A list of tuples containg the input and output size for each categorical feature
         example: [(input_size,output_size),...,(input_size,output_size)]
    """
    embedding_sizes = []
    for column in data.T:
        n_categories = len(np.unique(column))
        embedding_sizes.append(((n_categories, min(50, (n_categories+1)//2))))
    return embedding_sizes

# %%
class FeedforwardNeuralNetModel(nn.Module):
    'Definition of a neural Network'
    def __init__(self, num_features, num_class, hidden_layers, model_type, learning_rate,
                 epochs, batch_size, embedding_data=None, dropout=None, transformation_list=None,
                 early_stop = None, l1_weight=0, l2_weight=0, weights=False):
        """Initilization of the layers of the neural network
    
        Parameters
        ----------
        num_features : int
            The number of features for input
            
        num_class : int
            The number of outputs of the model
            
        hidden_layers : list
            A list of int with the neurons of each layer

        model_type : str
            The type odf the model to be implemented.
            Could be 'class_regression' or 'mosquito_regression' or 'classification'
            
        learning_rate : int
            The learning_rate of the training process.
        
        epochs : int
            The number of epochs for the training.

        batch_size : int
            The size of each batch in each iteration

        embedding_data : dataframe, optional
            A datafrane with the categorical features (default = None)
            
        dropout : list or float, optional
            If it is float, then creates dropout layers with p=dropout with lenght equal to the lenght of hidden layers
            If it is list, a list of float must be given with lenght equal to the lenght of hidden layers
            (default = None)
            
        transformation_list : list, optional
            If True, sets as activation function the Sigmoid function 
            and giving greater weight on the observations with greater target value (default = None)
            
        l1_weight: int or float, optional
        Weight parameter for L1 regularization (default=0)
    
        l2_weight: int or float, optional
            Weight parameter for L2 regularization (default=0)

        weights: boolean, optional
            If True, each sample is weighted based on the target value (default =  False)
        """
        super(FeedforwardNeuralNetModel, self).__init__()
        
        torch.manual_seed(0)
                
        if dropout != None and embedding_data is None:
            if not isinstance(dropout, float) and len(dropout)!=len(hidden_layers):
                raise ValueError('Dropout list and hidden_layers list must be of the same size')
                
        if dropout != None and embedding_data is not None:
            if not isinstance(dropout, float) and len(dropout)!=len(hidden_layers)+1:
                raise ValueError('Dropout list must be one element greater than the hidden layers list')
                
        self.linear_layers = nn.ModuleList()
        self.batchNorm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.embeddings_layers = nn.ModuleList()
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.transformation_list = transformation_list
        self.weights = weights
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.early_stop = early_stop
        self.embedding_data = embedding_data
        
        self.criterion = nn.MSELoss()
        if self.model_type == 'classification':
            self.classification_activation =  nn.Softmax()
            
        if isinstance(early_stop, tuple):
            self.early_stop = EarlyStopping(tolerance=early_stop[0], min_delta=early_stop[1])
        
        self.hidden_layers = [num_features] + hidden_layers + [num_class]
        
        if embedding_data is not None:
            embedding_sizes = set_embedding_sizes(embedding_data.values)
            num_features = num_features - len(embedding_sizes)
            self.embeddings_layers = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
            n_emb = sum(e.embedding_dim for e in self.embeddings_layers) #length of all embeddings combined
    
            # substract from the first layer input the number of categorical  features,
            # and add the sum of the output of the embeddings layers
            self.hidden_layers[0] = self.hidden_layers[0]-len(embedding_sizes) + n_emb
            
        if dropout != None:
            if isinstance(dropout, float):
                if dropout < 0 or dropout > 1:
                    raise ValueError('Dropout rate must be in [0,1]')
                if embedding_data is not None:
                    for i in range(len(hidden_layers)+1):
                        self.dropout_layers.append(nn.Dropout(p=dropout))
                else:
                    for i in range(len(hidden_layers)):
                        self.dropout_layers.append(nn.Dropout(p=dropout))
            else:       
                for i in range(len(dropout)):
                    self.dropout_layers.append(nn.Dropout(p=dropout[i]))
            

        for i in range(len(self.hidden_layers)-1):
            self.linear_layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))

        for i in self.hidden_layers[1:-1]:
            self.batchNorm_layers.append(nn.BatchNorm1d(i))

        if 'normalization' in self.transformation_list:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        'Initialization of the weights'
        if isinstance(module, nn.Linear):
#             torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.kaiming_normal_(module.weight)
            module.bias.data.fill_(0)
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.fill_(0)

    def forward(self, x_cont, x_cat=None):
        """Initialization of the architecture of the neural network
        
        Parameters
        ----------
        x_cont : DataLoader
            A DataLoader object of the arithemetic features
            
        x_cat : DataLoader
            A DataLoader object of the categorical features
        """
        dropout_layers = self.dropout_layers
        if len(self.embeddings_layers) != 0: 
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings_layers)]
            x = torch.cat(x, 1)
            if len(self.dropout_layers) != 0:
                x = dropout_layers[0](x)
                dropout_layers = dropout_layers[1:] 
            x_cont = torch.cat([x, x_cont], 1)
        
        out = self.linear_layers[0](x_cont)
        out = self.batchNorm_layers[0](out)
        out = self.activation(out)
        if len(self.dropout_layers) != 0:
            out = dropout_layers[0](out)

        for i in range(1,len(self.hidden_layers)-2):
            out = self.linear_layers[i](out)
            out = self.batchNorm_layers[i](out)
            out = self.activation(out)
            if len(self.dropout_layers) != 0:
                out = dropout_layers[i](out)

        out = self.linear_layers[-1](out)
        if self.model_type != 'classification':
            out = self.activation(out)
        else: 
            out = self.classification_activation(out)

        return out

# %%
def my_plot(epochs, train, evals, ylabel):
    """Prints the plot of evaluation
    
    Parameters
    ----------
    epochs : list
        A list with the epochs
        
    train : list
        A list with the prediction score on the train set
        
    evals : list
        A list with the prediction score on the test set
        
    y_label : str
        The label of the y-axis
    """
    for i in range(len(train)):
        if evals[i] > train[i]:
            break
    plt.plot(epochs, train, label='Train')
    plt.plot(epochs, evals, label='Eval')
    plt.vlines(x = i+1,ls='--', ymin = 0, ymax = max(max(train),max(evals)), colors = 'grey', label = 'x = '+str(i+1))
    plt.xlabel('epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# %%
def multi_acc(y_pred, y_test, model_type):
    """Calculates the MAE of each epoch
    Parameters
    ----------
    y_pred : list
        The predictions of the model
        
    y_test : list
        The actual values      
        
    model_type : str
        The type odf the model to be implemented.
        Could be 'class_regression' or 'mosquito_regression' or 'classification'
        
    Returns
    ----------
    acc : float
        The MAE of the predictions
    """
    if model_type =='classification':
        _, y_pred = torch.max(y_pred, dim=1)
        _, labels = torch.max(y_test, dim=1)
        acc = mean_absolute_error(labels, y_pred)
    else:
        y_pred = torch.round(y_pred)
        acc = mean_absolute_error(y_test, y_pred.detach().numpy())
    return acc

# %%
def find_NN_features(model, training_set, testing_set, X_train_minmax, X_test_minmax, features, delete=False, *args):
    
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=X_train_minmax.shape[0])
    testing_generator = torch.utils.data.DataLoader(testing_set, batch_size=X_test_minmax.shape[0])
    batch = next(iter(training_generator))
    train, _ = batch
    batch = next(iter(testing_generator))
    test, _ = batch
    
    e = shap.DeepExplainer(model, train)
    shap_values = e.shap_values(test)
    shap.summary_plot(shap_values, features=test, feature_names=features, plot_type='bar')
    
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    if delete:
        mean_shap = np.delete(mean_shap, args[0])

    weights = mean_shap / mean_shap.sum()

# %%
def train_nn(model, train_set, test_set, learning_rate=None, epochs=None, batch_size=None, 
             early_stop=None, features=None, max_val=None):
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
    loss_stats = {
    "train": [],
    "val": []
    }
    accuracy_stats = {
        "train": [],
        "val": []
    }
    
    if epochs is None:
        epochs = model.epochs
    
    if learning_rate is None:
        learning_rate = model.learning_rate
        
    if batch_size is None:
        batch_size = model.batch_size
        
    if isinstance(early_stop, tuple):
        model.early_stop = EarlyStopping(tolerance=early_stop[0], min_delta=early_stop[1])

    
    criterion = model.criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if model.weights:
        b_val = train_set.y.mean().item()
        weights = [sigmoid(train_set.y[i],0.005,b=b_val) for i in range(int(train_set.X.shape[0]))]
        weighted_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(train_set.X.shape[0]), replacement=True)
        training_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size,sampler=weighted_sampler)
    else:
        training_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    
    testing_generator = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
       
    for epoch in range(epochs):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        if train_set.emb:
            for X, X_emb, y in training_generator:
                optimizer.zero_grad()

                y_train_pred = model(X ,X_emb)

                train_loss = criterion(y_train_pred, y)
                train_acc = multi_acc(y_train_pred, y, model.model_type)
                
                l1_penalty = model.l1_weight * sum([p.abs().sum() for p in model.parameters()])
                l2_penalty = model.l2_weight * sum([(p**2).sum() for p in model.parameters()])
                train_loss = train_loss + l1_penalty + l2_penalty

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
        else:
            for X, y in training_generator:
                optimizer.zero_grad()
                y_train_pred = model(X)
                train_loss = criterion(y_train_pred, y)
                train_acc = multi_acc(y_train_pred, y, model.model_type)
                
                l1_penalty = model.l1_weight * sum([p.abs().sum() for p in model.parameters()])
                l2_penalty = model.l2_weight * sum([(p**2).sum() for p in model.parameters()])
                train_loss = train_loss + l1_penalty + l2_penalty

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
        

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            if test_set.emb:
                for X, X_emb, y in testing_generator:

                    y_val_pred = model(X, X_emb)

                    val_loss = criterion(y_val_pred, y)
                    val_acc = multi_acc(y_val_pred, y, model.model_type)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()
                    
            else:
                for X, y in testing_generator:

                    y_val_pred = model(X)

                    val_loss = criterion(y_val_pred, y)
                    val_acc = multi_acc(y_val_pred, y, model.model_type)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()
                    

        loss_stats['train'].append(train_epoch_loss / len(training_generator))
        loss_stats['val'].append(val_epoch_loss / len(testing_generator))

        accuracy_stats['train'].append(train_epoch_acc / len(training_generator))
        accuracy_stats['val'].append(val_epoch_acc / len(testing_generator))

        print(
            f'Epoch {epoch+1 + 0:03}: | Train Loss: {train_epoch_loss / len(training_generator):.5f} | Val Loss: {val_epoch_loss / len(testing_generator):.5f} | Train Acc: {train_epoch_acc / len(training_generator):.3f}| Val Acc: {val_epoch_acc / len(testing_generator):.3f}')

        if model.early_stop is not None:
            model.early_stop(train_epoch_loss / len(training_generator), val_epoch_loss / len(testing_generator))
            if model.early_stop.early_stop_flag:
                print("We are at epoch:", epoch+1)
                break

    my_plot(np.linspace(1, epoch+1, epoch+1).astype(int), loss_stats['train'], loss_stats['val'],'MSE Loss')
    my_plot(np.linspace(1, epoch+1, epoch+1).astype(int), accuracy_stats['train'], accuracy_stats['val'],'MAE')
    
    if train_set.emb:
        train_predict = model(train_set.X, train_set.X_emb)
        test_predict = model(test_set.X, test_set.X_emb)
    else:
        train_predict = model(train_set.X)
        test_predict = model(test_set.X)
       
    
    if model.model_type == 'classification':
        _, train_predict = torch.max(train_predict, dim=1)
        _, test_predict = torch.max(test_predict, dim=1)
        
        _, train_y = torch.max(train_set.y, dim=1)
        _, test_y = torch.max(test_set.y, dim=1)
        
        train_y = train_y.tolist()
        test_y = test_y.tolist()

    train_predict = train_predict.tolist()
    test_predict = test_predict.tolist()
    
    if model.model_type != 'classification':
        if 'normalization' in model.transformation_list:
            train_y = denormalizeData([e[0].item() for e in train_set.y], max_val=max_val)
            train_predict = [round(x) for x in denormalizeData([e[0] for e in train_predict], max_val=max_val)]

            test_y = denormalizeData([e[0].item() for e in test_set.y], max_val=max_val)
            test_predict = [round(x) for x in denormalizeData([e[0] for e in test_predict], max_val=max_val)]

        elif 'log' in model.transformation_list :
            train_y = expData([e[0].item() for e in train_set.y])
            train_predict = [round(x) for x in expData([e[0] for e in train_predict])]

            test_y = expData([e[0].item() for e in test_set.y])
            test_predict = [round(x) for x in expData([e[0] for e in test_predict])]
            
        else:
            train_y = [e[0].item() for e in train_set.y]
            train_predict = [round(e[0]) for e in train_predict]

            test_y = [e[0].item() for e in test_set.y]
            test_predict = [round(e[0]) for e in test_predict]

        
        
    results_train = {'actual': train_y, 'prediction': train_predict}
    results_train = pd.DataFrame.from_dict(results_train)
    
    results_test = {'actual': test_y, 'prediction': test_predict}
    results_test = pd.DataFrame.from_dict(results_test)
    
    results_test.loc[results_test['prediction'] < 0,'prediction'] = 0
    results_test.loc[results_test['prediction'] > max_val,'prediction'] = max_val
    results_train.loc[results_train['prediction'] < 0,'prediction'] = 0
    results_train.loc[results_train['prediction'] > max_val,'prediction'] = max_val
    
    if features is not None:
        find_NN_features(model, train_set, test_set, train_X, test_X, features)
    
    return results_train, results_test, model

# %%
def prediction_nn(model, train_set, test_set, learning_rate=None, epochs=None, 
               batch_size=None, early_stop=None, features=None, max_val=None):
    
    if epochs is None:
        epochs = model.epochs
    
    if learning_rate is None:
        learning_rate = model.learning_rate
        
    if batch_size is None:
        batch_size = model.batch_size
        
    if isinstance(early_stop, tuple):
        early_stop = EarlyStopping(tolerance=early_stop[0], min_delta=early_stop[1])
    else:
        early_stop = model.early_stop  
    
    criterion = model.criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if model.weights:
        weights = [sigmoid(train_set.y[i],0.005,1) for i in range(int(train_set.X.shape[0]))]
        weighted_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(train_set.X.shape[0]), replacement=True)
        training_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size,sampler=weighted_sampler)
    else:
        training_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    
    testing_generator = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
       
    for epoch in range(epochs):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        if train_set.emb:
            for X, X_emb, y in training_generator:
                optimizer.zero_grad()

                y_train_pred = model(X ,X_emb)

                train_loss = criterion(y_train_pred, y)
                train_acc = multi_acc(y_train_pred, y, model.model_type)
                
                l1_penalty = model.l1_weight * sum([p.abs().sum() for p in model.parameters()])
                l2_penalty = model.l2_weight * sum([(p**2).sum() for p in model.parameters()])
                train_loss = train_loss + l1_penalty + l2_penalty

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
        else:
            for X, y in training_generator:
                optimizer.zero_grad()
                y_train_pred = model(X)
                train_loss = criterion(y_train_pred, y)
                train_acc = multi_acc(y_train_pred, y, model.model_type)
                
                l1_penalty = model.l1_weight * sum([p.abs().sum() for p in model.parameters()])
                l2_penalty = model.l2_weight * sum([(p**2).sum() for p in model.parameters()])
                train_loss = train_loss + l1_penalty + l2_penalty

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

        if early_stop:
            early_stopping(train_epoch_loss / len(training_generator), val_epoch_loss / len(testing_generator))
            if early_stopping.early_stop:
                print("Stopped at epoch:", epoch+1)
                break
    
    if train_set.emb:
        test_predict = model(test_set.X, test_set.X_emb)
    else:
        test_predict = model(test_set.X)
       
    
    if model.model_type == 'classification':
        _, test_predict = torch.max(test_predict, dim=1)
        _, test_y = torch.max(test_set.y, dim=1)
        test_y = test_y.tolist()

    test_predict = test_predict.tolist()
    
    if model.model_type != 'classification':
        if 'normalization' in model.transformation_list:
            test_y = denormalizeData([e[0].item() for e in test_set.y], max_val=max_val)
            test_predict = [round(x) for x in denormalizeData([e[0] for e in test_predict], max_val=max_val)]

        elif 'log' in model.transformation_list :
            test_y = expData([e[0].item() for e in test_set.y])
            test_predict = [round(x) for x in expData([e[0] for e in test_predict])]
        else:
            test_y = [e[0].item() for e in test_set.y]
            test_predict = [round(e[0]) for e in test_predict]
            
    results_test = {'actual': test_y, 'prediction': test_predict}
    results_test = pd.DataFrame.from_dict(results_test)
    results_test.loc[results_test['prediction'] < 0,'prediction'] = 0
    results_test.loc[results_test['prediction'] > max_val,'prediction'] = max_val
    
    if features is not None:
        find_NN_features(model, train_set, test_set, train_X, test_X, features)
    
    return results_test


