import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import cv2

from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths and hyperparameters
path = "Dataset"  # Folder containing dataset
labelFile = 'labels.csv'  # CSV file with labels
batch_size_val = 32  # Batch size
epochs_val = 30 # Number of epochs
imageDimesions = (32, 32, 3)  # Image dimensions
testRatio = 0.2  # 20% of data used for testing
validationRatio = 0.2  # 20% of remaining data used for validation

# Load dataset and labels
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes...")

# Reading images and their labels
for count in range(len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")

images = np.array(images)
classNo = np.array(classNo)

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Data shapes
print("Data Shapes")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)

# Read CSV file with labels
data = pd.read_csv(labelFile)
print("Data shape", data.shape, type(data))

# Preprocessing functions
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

# def preprocessing(img):
#     img = grayscale(img)
#     img = equalize(img)
#     img = img / 255  # Normalize between 0 and 1
#     return img

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = cv2.resize(img, (32, 32))
    return img


# Preprocess the images
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Reshape the images to add a depth of 1 (for grayscale)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Image Data Generator for augmenting training images
dataGen = ImageDataGenerator(
    width_shift_range=0.1,   # Horizontal shift
    height_shift_range=0.1,  # Vertical shift
    zoom_range=0.2,          # Zoom
    shear_range=0.1,         # Shear
    rotation_range=10        # Rotation
)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Model Architecture
def myModel():
    model = Sequential()
    
    # First Conv Block
    model.add(Conv2D(60, (5, 5), activation='relu', input_shape=(imageDimesions[0], imageDimesions[1], 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dropout(0.3))  # instead of 0.5

    
    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(noOfClasses, activation='softmax'))
    
    # Compile Model
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = myModel()
print(model.summary())

# Train the model
history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=len(X_train) // batch_size_val,
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=True
)

# Plotting the training and validation loss
plt.figure(1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plotting the training and validation accuracy
plt.figure(2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Show the plots
plt.show()

# Evaluate the model on test data
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Save the trained model
model.save("model.h5")
