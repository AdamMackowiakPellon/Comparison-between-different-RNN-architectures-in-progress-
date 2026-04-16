import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import delay_reservoir_computing_1_var as drc1
import delay_reservoir_computing_2_var as drc2
import RNN
import optuna

#As a starting point I'm using a total of 500 neurons for each model

# ==============================================================================
# Neural networks
# ==============================================================================
def objective(trial):
    def reservoir_params(name):
        if name == 'mackey_glass':
            #Mackey_glass here theta 0.2 and tau = 100

            theta = trial.suggest_float('theta',0.1,0.5)
            N_nodes = 100
            tau = theta*N_nodes
            mask_array = (2*np.random.rand(N_nodes) - 1)
            regularization = trial.suggest_float('regularization',1e-12,1e-1, log = True)
            params = {
                "eta": trial.suggest_float('eta',0.1,5),
                "tau": tau,
                "gamma": trial.suggest_float('gamma',0.0001,5),
            }
            return params, theta, N_nodes, mask_array, regularization      
        if name == 'lang-kobayashi':
            theta = trial.suggest_float('theta',1,80) * 1e-12
            N_nodes = 80 #Because it is complex, half the nodes
            tau = theta*N_nodes
            mask_array = (2*np.random.rand(N_nodes) - 1) 
            regularization = trial.suggest_float('regularization',1e-12,1e-1, log = True)
            #For now diff_freq = 0
            diff_freq = trial.suggest_float('diff_freq',-60,60) * 1e9
            #Here we set phi = 0
            params = {
                "alpha": 3,
                "tau_ph":2*1e-12,
                "tau_s":2*1e-9,
                "diff_gain":12*1e3,
                "N_0":1.5*1e8,
                "I_th":15.37*1e-3,
                "k_inj":trial.suggest_float('k_inj',0.2,4),
                "k_fb": trial.suggest_float('k_fb',0.05,0.25),
                "tau_in":1e-11,
                "tau": tau,
                "E_inj_0": 100,
                "diff_freq": diff_freq
            }  
            return params, theta, N_nodes, mask_array, regularization    

    def neural_networks():
            params = {
                "input_size":1,
                "output_size":1,
                "hidden_size": 100,
                "batch_size":32,
                "num_workers":1,
                "epochs":100,
                "learning_rate":0.001,
                "window_size": 1
            }
            return params


    # ==============================================================================
    # Santa fe (we have to import yet the dataset and create all the arrays) (Maybe create a function for each dataset)
    # ==============================================================================


    #name = 'mackey_glass'
    name = 'lang-kobayashi'
    #name = 'vanilla_RNN'
    #name = 'ffnet'
    #name = 'LSTM'
    #name = 'GRU'

    if name == 'mackey_glass' or name == 'lang-kobayashi':
        params, theta, N_nodes, mask_array, regularization = reservoir_params(name)
    elif name == 'vanilla_RNN' or name == 'ffnet' or name == 'LSTM' or name == 'GRU':
        params = neural_networks()



    df = pd.read_csv('/home/adam/vscode/python/comparing_different_recurrent_neural_networks/santa_fe_time_series_a2.csv')
    df = np.array(df).flatten() 
    df = (df - min(df))/(max(df) - min(df))


    dataset_size_washout = 20
    dataset_size_training = 4000
    dataset_size_testing = 4000
    jump_dataset = 1000

    input_washout = df[0:dataset_size_washout]
    input_train = df[dataset_size_washout:dataset_size_washout + dataset_size_training]
    input_test = df[jump_dataset + dataset_size_washout + dataset_size_training : jump_dataset + dataset_size_washout + dataset_size_training + dataset_size_testing ]

    p = 1 #time step prediciton
    target_train = df[dataset_size_washout + p :dataset_size_washout + dataset_size_training + p]
    target_test = df[jump_dataset + dataset_size_washout + dataset_size_training + p : jump_dataset + dataset_size_washout + dataset_size_training + dataset_size_testing + p ]



    if name == 'mackey_glass':
        X_states_test, Y, NMSE = drc1.simulation( theta, N_nodes,
                                    input_washout, input_train, input_test,
                                    target_train, target_test, mask_array, regularization, name, params)
    elif name == 'lang-kobayashi':
        X_states_test, Y, NMSE = drc2.simulation( theta, N_nodes,
                                input_washout, input_train, input_test,
                                target_train, target_test, mask_array, regularization, name, params)
    elif name == 'vanilla_RNN' or name == 'ffnet' or name == 'LSTM' or name == 'GRU':
        RNN.simulation(df,p,dataset_size_washout,dataset_size_training,dataset_size_testing, jump_dataset,name,params)
    return NMSE

#By default, optuna uses TPESampler

#sampler = optuna.samplers.GPSampler(
#    n_startup_trials=10,   # random trials before GP kicks in
#    seed=42
#)

#study = optuna.create_study(direction='minimize', sampler= sampler)
#study.optimize(objective, n_trials=500)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)


# The best parameters are already a dictionary
best_params = study.best_params
print(best_params)


#print(NMSE)






#plt.figure()
#plt.plot(np.arange(len(Y)), Y, label = 'prediction')
#plt.plot(np.arange(len(Y)), target_test, label = 'targer')
#plt.show()
#
##plt.figure()
##for i in range(25):
##    plt.plot(np.arange(len(Y)),X_states_test[:,i])
##plt.show()
#
#
#plt.figure()
#for i in range(25):
#    plt.plot(np.arange(N_nodes),X_states_test[i,:])
#plt.show()