from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

directory = "D:\\Python Machine Learning Course\\PythonMachineLearning\\PythonMachineLearning (4)\\Datasets" \
            "\\Datasets\\smiles_dataset\\training_set\\"

pixel_intensities = []

# one-hot encoding: happy (1, 0) and sad (0, 1)
labels = []

for filename in os.listdir(directory):
    image = Image.open(directory+filename).convert('1')
    pixel_intensities.append(list(image.getdata()))
    if filename[0:5] == 'happy':
        labels.append([1, 0])
    elif filename[0:3] == 'sad':
        labels.append([0, 1])

pixel_intensities = np.array(pixel_intensities)
labels = np.array(labels)

# Normalize the arrays
pixel_intensities = pixel_intensities/ 255.0

# Create the model
model = Sequential()
model.add(Dense(1024, input_dim=1024, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

optimizer = Adam(learning_rate=0.005)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(pixel_intensities, labels, epochs=1000, batch_size=20, verbose=2)

# Testing the model
print("Testing the neural network ......")
test_pixel_intensities = []

test_image1 = Image.open('D:\\Python Machine Learning Course\\PythonMachineLearning\\PythonMachineLearning (4)\\Datasets\\Datasets'
                         '\\smiles_dataset\\test_set\\happy_test.png').convert('1')

test_pixel_intensities.append(list(test_image1.getdata()))
test_pixel_intensities = np.array(test_pixel_intensities)/255.0

print(model.predict(test_pixel_intensities).round())
