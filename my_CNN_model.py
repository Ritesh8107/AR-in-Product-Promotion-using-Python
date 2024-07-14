# Used header files as per the requirements.
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
#CNN model architecture.
def CNN_model_architecture():
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), input_shape=(96, 96, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
# Model_Architecture that conatins available data.
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
# Model_Architecture for different parameters.
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
# Model_Architecture for different parameters.
    model.add(Convolution2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
# Model_Architecture for different parameters.
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(30))
    # Compile the model that sync with my_model.py
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
    return model

def create_compile_train_save_model(X_train, y_train, model_filename):
    model = CNN_model_architecture()
    # Train CNN_Model
    history = model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)
    # CNN_Model saved.
    model.save(model_filename + '.h5')
    return model, history
# Loading CNN_Model.
def load_my_CNN_model(fileName):
    return load_model(fileName + '.h5')
