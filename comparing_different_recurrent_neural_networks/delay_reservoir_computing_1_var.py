import numpy as np
import scipy as scy
from numba import njit
from sklearn.linear_model import Ridge

# ==============================================================================
# Delay reservoir computing reservoirs to test (maybe also include here another echo state networks)
# ==============================================================================

#Here J is already the injected current*mask. Maybe we should consider separate them.
class mackey_glass(): #Real_valued_oscillator
    def __init__(self,eta,tau,gamma):
        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.h = 0.1 #CHECK!!!   
    def dx_dt(self, x, J, header):
        return -x[header-1] + (self.eta * (x[header] + self.gamma * J ))/(1 + (x[header] + self.gamma * J ))
    

 
#@njit
def RK4(oscillator, time, header, h_half,h, X, J, oscillator_type):
    if oscillator_type == 0:
        #In Mackey-glass there is no time dependence, so I wont' update the time variable
        k_X_1 = oscillator.dx_dt(X,J,header)
        k_X_2 = oscillator.dx_dt(X + h_half * k_X_1,J,header)
        k_X_3 = oscillator.dx_dt(X + h_half * k_X_2,J,header)
        k_X_4 = oscillator.dx_dt(X + h * k_X_3,J,header)
        return X[header-1] + (h/6)*(k_X_1 + 2*(k_X_2 + k_X_3) + k_X_4)

def euler(oscillator, time, header, h_half,h, X, J, oscillator_type):
    if oscillator_type == 0:
        #In Mackey-glass there is no time dependence, so I wont' update the time variable
        k_X_1 = oscillator.dx_dt(X,J,header)
        return X[header-1] + h*(k_X_1)



# ==============================================================================
# Washout, train and test stage
# ==============================================================================

#@njit
def washout_stage(oscillator, oscillator_type, time, header, limit_header, h, h_half, X,
            input_washout, mask_array, N_nodes, steps_per_node):
    input_washout_length = len(input_washout)
    for i in range(input_washout_length):
        input = input_washout[i]
        for k in range(N_nodes):
            mask = mask_array[k]
            J = mask*input
            for _ in range(steps_per_node):
                #X[header] = RK4(oscillator,time,header,h_half,h,X,J,oscillator_type)
                X[header] = euler(oscillator,time,header,h_half,h,X,J,oscillator_type)
                header = header + 1
                if header == limit_header:
                    header = int(0)
                time = time + h
    return X, time, header
                    

#@njit
def train_stage(oscillator, oscillator_type, time, header, limit_header, h, h_half, X,
            input_train, X_states_train, mask_array, N_nodes, steps_per_node):
    input_train_length = len(input_train)
    for i in range(input_train_length):
        input = input_train[i]
        for k in range(N_nodes):
            mask = mask_array[k]
            J = mask*input
            for _ in range(steps_per_node):
                #X[header] = RK4(oscillator,time,header,h_half,h,X,J,oscillator_type)
                X[header] = euler(oscillator,time,header,h_half,h,X,J,oscillator_type)
                header = header + 1
                if header == limit_header:
                    header = int(0)
                time = time + h #I'm aware I'm also doing the same inside RK4 and I know that i could optimize it.
            X_states_train[i,k] = X[header-1]
    return X, time, header, X_states_train
    
                    

#@njit
def test_stage(oscillator, oscillator_type, time, header, limit_header, h, h_half, X,
            input_test, X_states_test, mask_array, N_nodes, steps_per_node):
    input_test_length = len(input_test)
    for i in range(input_test_length):
        input = input_test[i]
        for k in range(N_nodes):
            mask = mask_array[k]
            J = mask*input
            for _ in range(steps_per_node):
                #X[header] = RK4(oscillator,time,header,h_half,h,X,J,oscillator_type)
                X[header] = euler(oscillator,time,header,h_half,h,X,J,oscillator_type)
                header = header + 1
                if header == limit_header:
                    header = int(0)
                time = time + h
            X_states_test[i,k] = X[header-1]
    return X_states_test
    


        
    


    
#Maybe we should include the circular buffer inside the classes? Or maybe as a separate function outside of the classes?
#Maybe we should a program for delay reservoir computing and another one for echo state networks.






"""
Here we will create the X_states_train, X_states_test, X_states_train
"""
def simulation(theta, N_nodes,
                            input_washout, input_train, input_test,
                            target_train, target_test, mask_array, regularization, name, params):
    if name == "mackey_glass":
        oscillator = mackey_glass(**params)
        X_states_train = np.ones((len(input_train), N_nodes)) 
        X_states_test = np.ones((len(input_test), N_nodes))
        X = np.random.randn(int(oscillator.tau / oscillator.h + 1))
        oscillator_type = 0
    else:
        raise ValueError("Unknown oscillator")  
    
    steps_per_node = int(theta/oscillator.h)
    h = oscillator.h
    h_half = oscillator.h/2
    time = 0
    header = int(0)
    limit_header = len(X)

    X[:], time, header = washout_stage(oscillator=oscillator, oscillator_type=oscillator_type, time=time, header=header, limit_header=limit_header, 
                                       h=h, h_half=h_half, X=X, input_washout=input_washout, mask_array=mask_array, N_nodes=N_nodes, steps_per_node=steps_per_node)
    X[:], time, header, X_states_train[:] = train_stage(oscillator=oscillator, oscillator_type=oscillator_type, time=time, header=header, limit_header=limit_header, 
                                       h=h, h_half=h_half, X=X, X_states_train=X_states_train, input_train=input_train, mask_array=mask_array, N_nodes=N_nodes, steps_per_node=steps_per_node)
    
    model = Ridge(alpha=regularization)
    model.fit(X_states_train, target_train)

    W = model.coef_
    b = model.intercept_
    
    X_states_test[:] = test_stage(oscillator=oscillator, oscillator_type=oscillator_type, time=time, header=header, limit_header=limit_header, 
                                    h=h, h_half=h_half, X=X, X_states_test=X_states_test, input_test=input_test, mask_array=mask_array, N_nodes=N_nodes, 
                                    steps_per_node=steps_per_node)
    #Idk if this is the correct dimensionality.
    Y = X_states_test @ W.T + b
    Y = np.asarray(Y).ravel()
    #target_test = np.asarray(target_test).ravel()
    var_y = np.var(target_test)

    #print(oscillator.eta)
    #print(oscillator.tau)
    #print(oscillator.gamma)
    #print(W.shape)
    #print(X_states_test.shape)
    #print(b.shape)
    #print("---")
    #print(var_y)
    #print(1/len(target_test))
#
#
    #print(Y.shape)
    #print(sum(Y-target_test))
    #print(target_test.shape)

    MSE = np.mean((Y - target_test)**2)
    NMSE =  MSE / var_y
    NRMSE = np.sqrt(NMSE)

    print(f'MSE = {MSE}')
    print(f'NMSE = {NMSE}')
    print(f'NRMSE = {NRMSE}')

    
    return X_states_test, Y, NMSE

    







    
   