# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# %%
def vectorise(target,num_classes):
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
    target = target.apply(lambda x: int(x))
    vectors = []
    for i in range(len(target)):
        y = [0] * num_classes
        vctr = np.array(y) 
        vctr[target[i]] = 1
        vectors.append(vctr.tolist())
    vectors = pd.Series(vectors)
    return vectors

# %%
def transformations(data, model_type, split = 0.2, scaling=True, augment=False, embeddings=None):
    """Executes scaling, augmentation or label endconding on the dataset 
    
    Parameters
    ----------
    data : dataframe
        A dataframe containing all the data
        
    model_type : str
        The type odf the model to be implemented.
        Could be 'class_regression' or 'mosquito_regression' or 'classification'
        
    scaling : boolean, optional
        If True, perofrms scaling on numerical features (default = False)
        
    augment : boolean, optional
        If True, augments the train data with existing observations
        and giving greater weight on the observations with greater target value (default = False)
        
    embeddings : list, optional
        A list of columns with categorical features in order to be label encoded
        
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
    X, y = data.iloc[:,:-1], data.iloc[:,-1]
    
    train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=split, random_state=1)
    
    train_X = train_X.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    
    if augment:
        augment_index = train_y.sample(frac=0.4, weights=train_y, random_state=1, replace=True).index
        train_X = pd.concat([train_X,train_X.iloc[augment_index,:]]).reset_index(drop=True)
        train_y = pd.concat([train_y,train_y[augment_index]]).reset_index(drop=True)
    
    if model_type == 'classification':
        num_classes = len(pd.concat([train_y,test_y]).unique())
        train_y = vectorise(train_y,num_classes)
        test_y = vectorise(test_y,num_classes)

    if embeddings != None:        
        embedded_columns_train = train_X.loc[:,embeddings] #categorical columns
        train_X = train_X.drop(columns=embeddings)

        embedded_columns_test = test_X.loc[:,embeddings] #categorical columns
        test_X = test_X.drop(columns=embeddings)

        for col in embeddings:
            label_encoder = LabelEncoder()
            label_encoder.fit(pd.concat([embedded_columns_train[col],embedded_columns_test[col]],axis=0))
            embedded_columns_train[col] = label_encoder.transform(embedded_columns_train[col])
            embedded_columns_test[col] = label_encoder.transform(embedded_columns_test[col])

        embedded_columns_train = embedded_columns_train.values
        embedded_columns_test = embedded_columns_test.values

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        train_X = np.hstack((train_X,embedded_columns_train))
        test_X = np.hstack((test_X,embedded_columns_test))
    else:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
                
    return train_X, train_y, test_X, test_y, scaler

# %%
class Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y, model_type, embedded_col_names=None, transform=False):
        """ Initialization of Dataset for the Neural Network.
        
        Parameters
        ----------
        X : numpy array
            An array if the dependent variables

        model_type : str
            The type odf the model to be implemented.
            Could be 'class_regression' or 'mosquito_regression' or 'classification'
        
        embedded_col_names : list, optional
            A list of columns with categorical features (default = None)

        transform : boolean, optional
            If True, perofrms stransformation of the target based on the model_type argument (default = False)
        
        """
        
        if transform:
            if model_type != 'class_regression' and model_type != 'mosquito_regression':
                raise KeyError('model_type argumnet must be given! \n values:{"class_regression" or "mosquito_regression"}')
               
        self.transform = transform
        self.model_type = model_type    
        self.emb = False
        
        if embedded_col_names!=None:
            self.X_emb = X[:,-(len(embedded_col_names)):] #categorical columns
            self.X = X[:,:-(len(embedded_col_names))] #numerical columns
            self.X_emb = torch.tensor(self.X_emb, dtype=torch.int)
            self.X = torch.tensor(self.X, dtype=torch.float)
            self.emb = True
        else:
            self.X = X
            self.X = torch.tensor(self.X, dtype=torch.float)
        
        self.y = y
        if model_type != 'classification':
            self.y = [[e] for e in self.y]
        
        if self.transform:
            if self.model_type == 'class_regression':
                self.y = self.__normalizeData__()
            elif self.model_type == 'mosquito_regression':
                self.y = self.__logData__()
            
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
    
    def __normalizeData__(self):
        'Transforms the target data to 0-1 range in order to use the sigmoid function as activation'
        min_val = 0
        max_val = 9
        return ((np.array(self.y) - min_val) / (max_val-min_val))
    
    def __logData__(self):
        'Transforms the target variable to log(target) in order to follow a more normal disribution'
        self.y = [[np.log(0.0000000001)] if e[0]==0 else [np.log(e[0])] for e in self.y]
        return self.y

# %%
def expData(data):
    'Calculate the exponential of the target variable'
    return(np.exp(data))

# %%
def denormalizeData(data):
    'Transforms the data from 0-1 range to the initial 0-9 range'
    min_val = 0
    max_val = 9
    return((np.array(data)*(max_val-min_val))+min_val).tolist()

# %%
class EarlyStopping():
    'Early stopping is a form of regularization used to avoid overfitting on the training dataset.'
    def __init__(self, tolerance=5, min_delta=0):
        """Set the tolerance and the min_delta for the early stopping 
        
        Parameters
        ----------
        tolerance : int, optinal
            The number of how many epochs to wait after validation score - training score
            is greater than min delta . (default = 5)
            
        min delta : int, optinal
            The threshold after which the difference of  validation score - training score
            is critical (default = 0)        
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

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
    def __init__(self, num_features, num_class, hidden_layers, model_type, embedding_data=None, dropout=None, transform=False):
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

        embedding_data : numpy array, optional
            An array with the categorical features (default = None)
            
        dropout : list or float, optional
            If it is float, then creates dropout layers with p=dropout with lenght equal to the lenght of hidden layers
            If it is list, a list of float must be given with lenght equal to the lenght of hidden layers
            (default = None)
            
        transform : boolean, optional
            If True, sets as activation function the Sigmoid function 
            and giving greater weight on the observations with greater target value (default = False)
        """
        super(FeedforwardNeuralNetModel, self).__init__()
                
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
        
        self.hidden_layers = [num_features] + hidden_layers + [num_class]
        
        if embedding_data is not None:
            embedding_sizes = set_embedding_sizes(embedding_data)
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

        if transform and model_type == 'class_regression':
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
def train_nn(model, learning_rate, epochs, batch_size, train_set, test_set, early_stop):
    """ Trainning of the model
    
    Parameters
    ----------
    model : FeedforwardNeuralNetModel
        A FeedforwardNeuralNetModel model 
        
    learning_rate : int
        The learning_rate of the training process.
        
    epochs : int
        The number of epochs for the training.
        
    batch_size : int
        The size of each batch in each iteration
        
    train_set : Dataset
        A Dataset object with the train set
        
    test_set : Dataset
        A Dataset object with the test set
        
    ealry_stop : boolean, optional
        If True, the trainning of the model may stop earlier than the epochs defined
        
    Returns
    ----------
    results_train: DataFrame
        A Dataframe containing the actual and the predicted values on the train set
        
    results_test: DataFrame
        A Dataframe containing the actual and the predicted values on the test set
    """    
    loss_stats = {
    "train": [],
    "val": []
    }
    accuracy_stats = {
        "train": [],
        "val": []
    }
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(tolerance=1, min_delta=5)
    
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
                train_acc = multi_acc(y_train_pred, y, train_set.model_type)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
        else:
            for X, y in training_generator:
                optimizer.zero_grad()
                y_train_pred = model(X)
                train_loss = criterion(y_train_pred, y)
                train_acc = multi_acc(y_train_pred, y, train_set.model_type)

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
                    val_acc = multi_acc(y_val_pred, y, test_set.model_type)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()
                    
            else:
                for X, y in testing_generator:

                    y_val_pred = model(X)

                    val_loss = criterion(y_val_pred, y)
                    val_acc = multi_acc(y_val_pred, y, test_set.model_type)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()
                    

        loss_stats['train'].append(train_epoch_loss / len(training_generator))
        loss_stats['val'].append(val_epoch_loss / len(testing_generator))

        accuracy_stats['train'].append(train_epoch_acc / len(training_generator))
        accuracy_stats['val'].append(val_epoch_acc / len(testing_generator))

        print(
            f'Epoch {epoch+1 + 0:03}: | Train Loss: {train_epoch_loss / len(training_generator):.5f} | Val Loss: {val_epoch_loss / len(testing_generator):.5f} | Train Acc: {train_epoch_acc / len(training_generator):.3f}| Val Acc: {val_epoch_acc / len(testing_generator):.3f}')

        if early_stop:
            early_stopping(train_epoch_loss, val_epoch_loss)
            if early_stopping.early_stop:
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
       
    
    if train_set.model_type == 'classification':
        _, train_predict = torch.max(train_predict, dim=1)
        _, test_predict = torch.max(test_predict, dim=1)
        
        _, train_y = torch.max(train_set.y, dim=1)
        _, test_y = torch.max(test_set.y, dim=1)
        
        train_y = train_y.tolist()
        test_y = test_y.tolist()
        train_predict = train_predict.tolist()
        test_predict = test_predict.tolist()
    else:
        train_predict = train_predict.tolist()
        test_predict = test_predict.tolist()

    if train_set.model_type != 'classification':
        if train_set.transform:
            if train_set.model_type == 'class_regression':
                train_y = denormalizeData([e[0].item() for e in train_set.y])
                train_predict = [round(x) for x in denormalizeData([e[0] for e in train_predict])]

                test_y = denormalizeData([e[0].item() for e in test_set.y])
                test_predict = [round(x) for x in denormalizeData([e[0] for e in test_predict])]

            elif train_set.model_type == 'mosquito_regression' :
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
    
    return results_train, results_test, model

# %%
def give_predictions(model, test_set):
    """ Returns predictions of a nn model on a set of features.
    
    Parameters
    ----------
    model : FeedforwardNeuralNetModel
        A trained FeedforwardNeuralNetModel model 
        
    test_set : Dataset
        A Dataset object with the test set
        
    Returns
    ----------
    test_predict: list
        A list of the predictions for a test set given
        
    results_test: DataFrame
        A Dataframe containing the actual and the predicted values on the test set
    """    
    
    if test_set.emb:
        test_predict = model(test_set.X, test_set.X_emb)
    else:
        test_predict = model(test_set.X)
       
    if test_set.model_type == 'classification':
        _, test_predict = torch.max(test_predict, dim=1)

    test_predict = test_predict.tolist()

    if test_set.model_type != 'classification':
        if test_set.transform:
            if test_set.model_type == 'class_regression':
                test_predict = [round(x) for x in denormalizeData([e[0] for e in test_predict])]

            elif test_set.model_type == 'mosquito_regression' :
                test_predict = [round(x) for x in expData([e[0] for e in test_predict])]
        else:
            test_predict = [round(e[0]) for e in test_predict]
    
    return test_predict


