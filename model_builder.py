from utils import load_data
from my_CNN_model import CNN_model_architecture

# Load training set.
X_train, y_train = load_data()

# Create an instance of the CNN model
my_model = CNNModel()

# Compile the CNN model with an appropriate optimizer, loss, and metrics.
my_model.compile_model(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
training_history = my_model.train_model(X_train, y_train)

# Save the trained model
my_model.save_model('my_model')
