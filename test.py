import cv2

img = cv2.imread('Fnt/0/img001-00013.png')
#img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 20)
#img = cv2.resize(img, (28, 28))
print(img.shape)
cv2.imshow('img',img)
cv2.waitKey(0)