from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

credit_data = pd.read_csv("D:\\Python Machine Learning Course\\PythonMachineLearning\\Datasets\\"
                          "Datasets\\credit_data.csv")

features = credit_data[['income', 'age', 'loan']]
y = np.array(credit_data.default).reshape(-1, 1)

# Coding default column
encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

model = Sequential()

model.add(Dense(10, input_dim=3, activation='sigmoid'))
model.add(Dense(2, activation="softmax"))

optimizer = Adam(learning_rate = 0.001)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=1000, verbose=2)
results = model.evaluate(test_features, test_targets, use_multiprocessing=True)

print('Training is finished... The loss and accuracy values are:')
print(results)