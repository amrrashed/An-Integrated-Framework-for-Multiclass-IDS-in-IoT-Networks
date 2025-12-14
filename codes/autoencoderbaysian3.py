import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization
from matplotlib import pyplot

# Load your dataset
df = pd.read_csv('D:/new researches/Security/dataset/processed win dataset/windows10_dataset.csv')

# Extract features and labels
X = df.drop(columns=['label', 'type'])
y = df['type']

# Apply label encoding to y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check for string values in X and encode them if exist
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        print(f"Encoded column: {column}")

# Number of input columns
n_inputs = X.shape[1]

# Split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

# Define encoder
visible = Input(shape=(n_inputs,))
# Encoder level 1
e = Dense(n_inputs * 2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# Encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# Bottleneck
n_bottleneck = round(float(n_inputs) / 2)
bottleneck = Dense(n_bottleneck)(e)
# Define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# Decoder level 2
d = Dense(n_inputs * 2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# Output layer
output = Dense(n_inputs, activation='linear')(d)
# Define autoencoder model
model = Model(inputs=visible, outputs=output)

# Compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# Define the search space for Bayesian optimization
pbounds = {
    'epochs': (50, 200),
    'batch_size': (8, 32),
    'n_bottleneck': (2, 7),  # Adjust the range as needed
}

# Objective function to minimize (negative validation loss)
def objective(epochs, batch_size, n_bottleneck):
    # Ensure parameters are integers
    epochs, batch_size, n_bottleneck = int(epochs), int(batch_size), int(n_bottleneck)

    # Set bottleneck size
    bottleneck_size = round(float(n_inputs) / n_bottleneck)

    # Define the model with the current parameters
    visible = Input(shape=(n_inputs,))
    e = Dense(n_inputs * 2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    bottleneck = Dense(bottleneck_size)(e)
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    d = Dense(n_inputs * 2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    output = Dense(n_inputs, activation='linear')(d)
    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, X_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    # Return the negative validation loss (Bayesian Optimization seeks to minimize)
    return -history.history['val_loss'][-1]

# Bayesian Optimization
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

# Maximize the objective function (minimize negative validation loss)
optimizer.maximize(init_points=5, n_iter=10)

# Get the best parameters
best_params = optimizer.max['params']
best_epochs = int(best_params['epochs'])
best_batch_size = int(best_params['batch_size'])
best_n_bottleneck = int(best_params['n_bottleneck'])

# Retrain the model with the best hyperparameters
history = model.fit(X_train, X_train, epochs=best_epochs, batch_size=best_batch_size, verbose=2, validation_data=(X_test, X_test))

# Plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'encoder_compress3.png', show_shapes=True)

# Save the encoder to file
encoder.save('encoder3.h5')
