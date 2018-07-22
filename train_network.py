# USAGE
# python train_network.py --dataset images --model santa_not_santa.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenetnew import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath,0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 20)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    if label == '0':
        label = 0
    elif label == '1':
        label = 1
    elif label == '2':
        label = 2
    elif label == '3':
        label = 3
    elif label == '4':
        label = 4
    elif label == '5':
        label = 5
    elif label == '6':
        label = 6
    elif label == '7':
        label = 7
    elif label == '8':
        label = 8
    elif label == '9':
        label = 9
    elif label == 'a':
        label = 10
    elif label == 'b':
        label = 11
    elif label == 'c':
        label = 12
    elif label == 'd':
        label = 13
    elif label == 'e':
        label = 14
    elif label == 'f':
        label = 15
    elif label == 'g':
        label = 16
    elif label == 'h':
        label = 17
    elif label == 'i':
        label = 18
    elif label == 'j':
        label = 19
    elif label == 'k':
        label = 20
    elif label == 'l':
        label = 21
    elif label == 'm':
        label = 22
    elif label == 'n':
        label = 23
    elif label == 'o':
        label = 24
    elif label == 'p':
        label = 25
    elif label == 'q':
        label = 26
    elif label == 'r':
        label = 27
    elif label == 's':
        label = 28
    elif label == 't':
        label = 29
    elif label == 'u':
        label = 30
    elif label == 'v':
        label = 31
    elif label == 'w':
        label = 32
    elif label == 'x':
        label = 33
    elif label == 'y':
        label = 34
    elif label == 'z':
        label = 35
    elif label == 'A':
        label = 36
    elif label == 'B':
        label = 37
    elif label == 'C':
        label = 38
    elif label == 'D':
        label = 39
    elif label == 'E':
        label = 40
    elif label == 'F':
        label = 41
    elif label == 'G':
        label = 42
    elif label == 'H':
        label = 43
    elif label == 'I':
        label = 44
    elif label == 'J':
        label = 45
    elif label == 'K':
        label = 46
    elif label == 'L':
        label = 47
    elif label == 'M':
        label = 48
    elif label == 'N':
        label = 49
    elif label == 'O':
        label = 50
    elif label == 'P':
        label = 51
    elif label == 'Q':
        label = 52
    elif label == 'R':
        label = 53
    elif label == 'S':
        label = 54
    elif label == 'T':
        label = 55
    elif label == 'U':
        label = 56
    elif label == 'V':
        label = 57
    elif label == 'W':
        label = 58
    elif label == 'X':
        label = 59
    elif label == 'Y':
        label = 60
    elif label == 'Z':
        label = 61
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=62)
testY = to_categorical(testY, num_classes=62)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=62)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Characters")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])