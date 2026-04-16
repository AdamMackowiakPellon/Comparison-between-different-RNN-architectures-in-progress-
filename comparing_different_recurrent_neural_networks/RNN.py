import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

#These are to use the MNIST dataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


#-------------------------------------------------------------------------------
# Neural Network
# ------------------------------------------------------------------------------

class vanilla_RNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers = 1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        #This is to build the hidden layer
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                batch_first= True)
        #output layer
        self.layer_output = torch.nn.Linear(hidden_size,output_size)

    def forward(self, x):
        """
        x: input of shape (batch_size, seq_length, input_size)
        """
        # RNN returns (output, h_n)
        rnn_out, h_n = self.rnn(x)

        
        # Take the output from the last time step (that is why the -1)
        last_output = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through output layer
        output = self.layer_output(last_output)
        
        return output
    
class LSTM_neural_net(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers = 1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        #This is to build the hidden layer
        self.rnn = torch.nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                batch_first= True)
        #output layer
        self.layer_output = torch.nn.Linear(hidden_size,output_size)

    def forward(self, x):
        """
        x: input of shape (batch_size, seq_length, input_size)
        """
        # RNN returns (output, h_n)
        rnn_out, h_n = self.rnn(x)

        
        # Take the output from the last time step (that is why the -1)
        last_output = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through output layer
        output = self.layer_output(last_output)
        
        return output
    
class GRU_neural_net(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers = 1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        #This is to build the hidden layer
        self.rnn = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                batch_first= True)
        #output layer
        self.layer_output = torch.nn.Linear(hidden_size,output_size)

    def forward(self, x):
        """
        x: input of shape (batch_size, seq_length, input_size)
        """
        # RNN returns (output, h_n)
        rnn_out, h_n = self.rnn(x)

        
        # Take the output from the last time step (that is why the -1)
        last_output = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through output layer
        output = self.layer_output(last_output)
        
        return output

class ffnet(torch.nn.Module):
    def __init__(self, input_size,layer_size,output_size=1):
        super().__init__()

        self.layer1 = torch.nn.Linear(input_size,layer_size)
        self.activation1 = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(layer_size,output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        return x 
    

# ------------------------------------------------------------------------------
# In progress...
# ------------------------------------------------------------------------------



    


class vanilla_RNN_from_scratch(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1):
        super().__init__()

        self.input_layer = torch.nn.Linear(input_size,hidden_size) #input weights
        self.hidden_layer = torch.nn.Linear(hidden_size,hidden_size) #hidden weights
        self.output_layer = torch.nn.Linear(input_size,output_size) #output weights


    def forward(self, x, h_last):
        """
        x (batch_size, input_size)
        h_last (batch_size, hidden_size)
        """

        h_t = torch.tanh(self.input_layer(x) + self.hidden_layer(h_last))
        
        x = self.output_layer(h_t)
        # we don't apply x = self.softmax(x) because the CrossEntropyLoss() already does that for us
        return x, h_t 
    



def create_dataset(data, window_size, p):
    X, Y = [], []
    for i in range(len(data) - window_size - p + 1):
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size + p - 1])
    return np.array(X), np.array(Y)

def simulation(df,p,dataset_size_washout,dataset_size_training,dataset_size_testing, jump_dataset,name,params):

    # ==========================================================================
    # Unpacking parameters and creation of datasets
    # ==========================================================================
    
    
    input_size = params["input_size"]
    output_size = params["output_size"]
    hidden_size = params["hidden_size"]
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    epochs = params["epochs"]
    learning_rate = params["learning_rate"]
    window_size = params["window_size"] # The story we use to the predicition
    #p is the number of steps ahead we want our prediction to be
    
    X, Y = create_dataset(df, window_size, p)
    #print(X.shape)
    #print(Y.shape)
    #print(X)
    #print(Y)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)   # shape: (n_samples, 1, 1) -> After will be (batch_size=32,seq_length=window_size,input_size=1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)    # shape: (n_samples, 1) -> After will be (batch_size = 32, output_size = 1)
    # The seq_length = window_size is basically the same as the number of steps ahead you want to predict (you need the whole story between t and t+tau)
    X_train = X[dataset_size_washout:dataset_size_washout+dataset_size_training]
    Y_train = Y[dataset_size_washout:dataset_size_washout+dataset_size_training]
    X_test = X[dataset_size_washout+dataset_size_training+jump_dataset:dataset_size_washout+dataset_size_training+jump_dataset+dataset_size_testing]
    Y_test = Y[dataset_size_washout+dataset_size_training+jump_dataset:dataset_size_washout+dataset_size_training+jump_dataset+dataset_size_testing]

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,          # shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True        # speeds up GPU transfer if using GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # no need to shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )



    # ==============================================================================
    # Hyperparameters of the RNN and loss function
    # ==============================================================================
    import torch.optim as optim
    
    if name == 'vanilla_RNN':
        model = vanilla_RNN(input_size=input_size,hidden_size=hidden_size,output_size=output_size)
    elif name == 'ffnet':
        model = ffnet(input_size=input_size,layer_size=hidden_size,output_size=output_size)
    elif name == 'LSTM':
        model = LSTM_neural_net(input_size=input_size,hidden_size=hidden_size,output_size=output_size)
    elif name == 'GRU':
        model = GRU_neural_net(input_size=input_size,hidden_size=hidden_size,output_size=output_size)
        



    # Create an Adam optimizer
    # We pass model.parameters() to tell the optimizer which tensors it should manage.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # We also use a pre-built loss function from torch.nn
    loss_fn = torch.nn.MSELoss() # Mean Squared error

    # ==============================================================================
    # Training loop
    # ==============================================================================
    for epoch in range(epochs):
        model.train() #Set the model to training mode
        total_loss = 0

    for batch_x, batch_y in train_loader:
        # 1. Zero the gradients from the previous iteration
        optimizer.zero_grad()   
        #print("batch_x shape:", batch_x.shape)   # should be (32, 1, 1)

        if name == 'ffnet':
        # Flatten: remove the last singleton dimension
            batch_x = batch_x.view(batch_x.size(0), -1)   # shape: (batch, window_size)

        
        #Forward pass
        outputs = model(batch_x)# batch_x already has shape (batch, 1, 1)
        #compute loss
        loss = loss_fn(outputs, batch_y)
        # 2. Compute gradients for this iteration
        loss.backward()
        # 3. Update the parameters
        optimizer.step()

        total_loss +=loss.item() #accumulates for reporting
    
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


    model.eval() #Here is not necessary but it is when you are adding BatchNormLayers or Dropout Layers when training

    with torch.no_grad(): #This is to not actualize the parameters! We do not want to compute the gradients anymore

        output_list = []

        for batch_x,batch_y in test_loader:

            if name == 'ffnet':
                # Flatten: remove the last singleton dimension
                batch_x = batch_x.view(batch_x.size(0), -1)   # shape: (batch, window_size)
            #Forward pass
            outputs = model(batch_x)

            output_list.append(outputs)

        output_list = torch.cat(output_list, dim = 0)
        output_list = output_list.numpy()
        Y_test = Y_test.numpy()
        output_list = output_list.squeeze()
        Y_test = Y_test.squeeze()


        #print(output_list.shape)
        #print(Y_test.shape)


        #print(output_list)
        MSE = (1/len(output_list))*(np.sum((output_list - Y_test)**2))
        NMSE = MSE*(1/np.var(Y_test))
        NRMSE = np.sqrt(NMSE)

        print(f'MSE = {MSE}')
        print(f'NMSE = {NMSE}')
        print(f'NRMSE = {NRMSE}')

