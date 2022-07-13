import tensorflow
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model, save, save_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report


import cv2
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from PIL import ImagePath


# Load from the MNIST dataset. MNIST returns (trainX, trainY, testX, testY)
# Reshape training data into (# of featuraes, 28x28 pixels, 1 color channel(grayscale))

def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    
    print(trainX.shape)
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    
    return trainX, trainY, testX, testY

# Display image by transversing through each image in TrainX
# TrainY contains the labeled data for trainX, use as title
# Only displays 25 images, 5 each row and column

def load_images(trainX, trainY, num):

    for i in range(25):
        plt.subplot(5, 5, i+1), plt.imshow(trainX[i], cmap='gray')
        plt.title(trainY[i], color="WHITE")
        plt.xticks([]), plt.yticks([])
    
    plt.tight_layout()
    plt.show()


# Data must be type.float
# To reduce image noise and make each data have a value between [0,1] we normalize by dividing the 
# X data by 255.0 (the highest number color value)
# y data is categorized with 10 features (0-9) where the index represents the number

def reshape_normalize():
    trainX, trainY, testX, testY = load_dataset()
    #load_images(trainX, trainY)
    
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX = trainX / 255.0
    testX = testX / 255.0
    
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)
    
    return trainX, trainY, testX, testY


def create_model(trainX, trainY, testX, testY):
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(trainX, trainY, epochs=10, batch_size=128, validation_data=(testX, testY), verbose=1)
    
    model.save(filepath)



# Function to load model from file_path that it is saved in.
def load_saved_model():
    file_path = r'***'
    return load_model(file_path)


def load_image(filepath):
    img = load_img(filepath, color_mode='grayscale', target_size=(28, 28))
    img_pixel = img.load()
   
    amt = 0
    for x in range(28):
        if x > 10 and x < 20:
           if img_pixel[x, 14] != 0:
               amt += 1 
    
    if amt == 0:
        return '0'
    
    # Convert to array
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    
    # Prepare Pixel Data
    img = img.astype('float32')
    img = img / 255.0
    

    return img


# When a number is saved on file, it gets passed into this function and returns the predicted number
def evaluate(digit):
    trainX, trainY, testX, testY = reshape_normalize()
    model = load_saved_model()

    
    img = load_image(digit)
    
    if img == '0':
        return img
    
    pred = model.predict(img)
    answer = pred.argmax()

    return answer
    #print(answer)
    


