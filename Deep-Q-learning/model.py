from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, activations

def q_nn(input_size, output_size):
    model = Sequential()
<<<<<<< HEAD
    model.add(Dense(32, input_dim=input_size, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
=======
    model.add(Dense(32, input_dim=input_size, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(16, activation='elu'))
>>>>>>> 039a49674d88c33262ebe067bb4d3690b04dcbb6
    #softmax because actions are one-hot-encoded
    #max(model output) will gize us the best guess
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam())
    return model
