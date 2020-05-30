import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from glob import glob
import requests
from io import BytesIO

def preprocess_image(pil_image):
    '''
    Preprocesses or prepares a PIL image for the model.
    '''
    return np.array(pil_image.convert("RGB").resize((200,200)))/255.0

def convert_image_format(directory_from:str, directory_to:str, format:str="jpg", prefix:str="img"):
    '''
    Takes a directory full of images and converts them to another format.

    Parameters
    ----------
    directory_from : str
        the name of the folder containing your images
    directory_to : str
        the name of the folder to place converted images into
    format : str
        the type of image file to convert to (one of png, jpg)
    prefix : str
        the name of each created file, which is followed by a number
    '''
    filenames = glob(directory_from + "/*")

    i = 1
    for filename in filenames:
        img = Image.open(filename)
        img.save(directory_to + "/%s-%d.%s"%(prefix, i, format))
        i += 1

def resize_all(directory_from:str, directory_to:str, size=(200,200), format:str="jpg", prefix:str="img"):
    '''
    Takes a directory full of images and resizes them all to the same size.

    Parameters
    ----------
    directory_from : str
        the name of the folder containing your images
    directory_to : str
        the name of the folder to place resized images into
    format : str
        the type of image file to convert to (one of png, jpg)
    prefix : str
        the name of each created file, which is followed by a number
    '''
    filenames = glob(directory_from + "/*")

    i = 1
    for filename in filenames:
        img = Image.open(filename).resize(size)
        img.save(directory_to + "/%s-%d.%s"%(prefix, i, format))
        i += 1

def load_data():
    print("Loading data...")
    filenames_apple = glob("apples-jpg-resized/*.jpg")
    filenames_orange = glob("oranges-jpg-resized/*.jpg")
    
    # INPUT
    # array of images (each in the form of a numpy array)
    train_apple_x = np.array([preprocess_image(Image.open(filename)) for filename in filenames_apple])
    train_orange_x = np.array([preprocess_image(Image.open(filename)) for filename in filenames_orange])

    # LABELS
    # 0: apple 1: orange
    train_apple_y = np.array([0 for img in train_apple_x])
    train_orange_y = np.array([1 for img in train_orange_x])

    # combine the arrays together
    train_x = np.array(np.append(train_apple_x, train_orange_x, axis=0))
    train_y = np.array(np.append(train_apple_y, train_orange_y, axis=0))

    return train_x, train_y

model_filename = "model.h5"
model = load_model(model_filename)

def classify(img_url):
    try:
        print("Retrieving image...")
        response = requests.get(img_url)
        print("Preprocessing image...")
        img = preprocess_image(Image.open(BytesIO(response.content))).reshape(1,200,200,3)
        print("Classifying image...")
        prediction = model.predict(img)[0,0]

        return prediction
    except FileNotFoundError:
        print("File not found.")

def retrain():
    '''
    Overwrite the model with a new one and train it again.
    '''
    train_x, train_y = load_data()

    print("Creating model...")
    model = Sequential()
    model.add(Conv2D(32, (4,4), activation="relu", input_shape=(200,200,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (2,2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    print("Compiling model...")
    model.compile(optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"])

    print("Training model...")
    model.fit(train_x, train_y, epochs=40)

    print("Saving model to %s..." % model_filename)
    model.save(model_filename)

def train():
    '''
    Continue training the model without overwriting it.
    '''
    train_x, train_y = load_data()

    print("Training model...")
    model.fit(train_x, train_y, epochs=40)

    print("Saving model to %s..." % model_filename)
    model.save(model_filename)

def interpret(prediction):
    '''
    Returns a string that says what the prediction means.
    '''
    return "I am %.2f%% sure that it's an %s." %(prediction*100 if prediction > 0.5 else (1-prediction)*100, "orange" if prediction >= 0.5 else "apple")

def perform_query():
    query = input("Would you like to train a new model? (y/n) ")

    if query.lower()=='y':
        retrain()

    query = input("Please enter the url of the image file to classify (\"exit\" to exit). ")
    if query == "exit":
        return

    prediction = classify(query)

    print(interpret(prediction))