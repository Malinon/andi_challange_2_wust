from andi_datasets.models_phenom import models_phenom
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


T = 48
N = 100000
L = 1.5*128 # it mimics a typical experimental setup

def generate_data(traj_length, data_size):
    trajs, labels = models_phenom().single_state(N = data_size, 
                                            L = L,
                                            T = traj_length,
                                            alphas=[1,2],
                                            Ds=[1.0 ,3.0])
    old_format_ds = np.zeros((data_size, traj_length * 2))
    for i in range(data_size):
        for step in range(traj_length):
            old_format_ds[i][step * 2 ] = trajs[step][i][0]
            old_format_ds[i][step * 2 + 1] = trajs[step][i][1]
    return old_format_ds, labels[0,:,0]

def data_prepare(X,Y,N,traj_length,dimension):                # regularize trajectories for training
    import numpy as np 
    thr=1e-10
    r = np.array(X).reshape(N,dimension,traj_length)              
    r = np.diff(r,axis=2)
    x = np.zeros((N,0))
    for dim in range(dimension):
        y = r[:,dim,:]
        sy = np.std(y,axis=1)
        y = (y-np.mean(y,axis=1).reshape(len(y),1)) / np.where(sy>thr,sy,1).reshape(len(y),1)   # normalize x data
        y = np.concatenate((y,np.zeros((N,1))),axis=1)
        x = np.concatenate((x,y),axis=1)                   # merge dimensions
    x = np.transpose(x.reshape(N,dimension,traj_length),axes = [0,2,1])
    
    label = Y
    
    return(x, label)
    


def create_randi_net(dimension):
    model_inference = Sequential()
    
    block_size = 4*dimension                                   # Size of the blocks of data points

    model_inference.add(LSTM(250,                              # first layer: LSTM of dimension 250
                         return_sequences=True,            # return sequences for the second LSTM layer            
                         recurrent_dropout=0.2,            # recurrent dropout for preventing overtraining
                         input_shape=(None, block_size)))  # input shape
                                                           
    model_inference.add(LSTM(50,                               # second layer: LSTM of dimension 50
                        dropout=0,
                        recurrent_dropout=0.2))

    model_inference.add(Dense(1))                              # output 

    model_inference.compile(optimizer='adam',
                        loss='mse', 
                        metrics=['mae'])
    return model_inference

def train_model(model, traj_length, N, dimension=2):
    batch_sizes = [32, 128, 512, 2048]
    dataset_used = [1, 4, 5, 20]
    number_epochs = [5, 4, 3, 2]
    block_size = 4*dimension

    for batch in range(len(batch_sizes)):
        print('Batch size: ', batch_sizes[batch])  
        for repeat in range(dataset_used[batch]):
            X, Y = generate_data(traj_length, N)
            x, label = data_prepare(X,Y,N,traj_length,dimension)
            model.fit(x.reshape(N,int((traj_length * dimension)/block_size),block_size),
                            label, 
                            epochs=number_epochs[batch], 
                            batch_size=batch_sizes[batch],
                            validation_split=0.1,
                            shuffle=True)


model = create_randi_net(2)
train_model(model, T, N)
model.save('randi.h5')






