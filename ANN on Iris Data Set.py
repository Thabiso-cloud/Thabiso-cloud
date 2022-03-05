import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

iris_data = load_iris()
features = iris_data.data
labels = iris_data.target.reshape(-1, 1)

# We have 3 classes so the labels have 3 values (0, 1, 2)
# First class: (1, 0, 0) 2nd Class: (0, 1, 0) 3rd class: (0, 0, 1)
encoder = OneHotEncoder()
targets = encoder.fit_transform(labels).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=4, activation="sigmoid"))
model.add(Dense(3, activation="softmax"))

# Define the loss function MSE or negative log likelihood
# Optimizer will find the right adjustments for the weights: SGD, Adagrad, Adam
optimizer = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=10000, batch_size=20, verbose=2)
results = model.evaluate(test_features, test_targets, use_multiprocessing=True)

print((model.predict(test_features)).astype("float64"))
print("Training is finished... The loss and accuracy values are:")

print(results)