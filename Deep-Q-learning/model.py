from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def q_nn(input_size, output_size):
    model = Sequential()
    model.add(Dense(32, input_dim=input_size, activation='relu'))
    model.add(Dense(16), activation='relu')
    model.add(Dense(16), activation='relu')
    #softmax because actions are one-hot-encoded
    #max(model output) will gize us the best guess
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='mse', optimizer=optimizers.Adam())
    return model