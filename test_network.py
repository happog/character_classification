# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])


def img_preference(img):
    sliced = img[:-4]
    point1 = sliced.index('_')
    sliced1 = sliced[0:point1]
    sliced2 = sliced[point1 + 1:]
    point2 = sliced2.index('_')
    sliced3 = sliced2[0:point2]
    sliced4 = sliced2[point2 + 1:]
    line = int(sliced1)
    word = int(sliced3)
    letter = int(sliced4)
    return line*1000000 + word*1000 + letter*10

file = open('output.txt','w')
# load the image
for imageitem in sorted(os.listdir(args["image"]),key=img_preference):
    image = cv2.imread(args["image"] + '/' + imageitem,0)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)



    # classify the input image
    #(digit0, digit1, digit2,digit3,digit4,digit5,digit6,digit7,digit8,digit9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z) = model.predict(image)[0]
    label = model.predict(image)[0]
    # build the label
    '''label = "Santa" if santa > notSanta else "Not Santa"
    proba = santa if santa > notSanta else notSanta
    label = "{}: {:.2f}%".format(label, proba * 100)'''

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    #cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    maxi = max(label)
    index = label.tolist().index(maxi)
    real_classes = [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f','g','h','i','j','k','l','m',
                    'n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E',
                    'F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W',
                    'X','Y','Z']
    sliced = imageitem[:-4]
    point1 = sliced.index('_')
    sliced1 = sliced[0:point1]
    sliced2 = sliced[point1 + 1:]
    point2 = sliced2.index('_')
    sliced3 = sliced2[0:point2]
    sliced4 = sliced2[point2 + 1:]
    line = int(sliced1)
    word = int(sliced3)
    letter = int(sliced4)
    if line > 1 and word == 1 and letter == 1:
        file.write('\n')
    if word >1 and letter == 1:
        file.write(' ')
    file.write(str(real_classes[index]))
