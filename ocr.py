import os, sys
import cv2
import pytesseract
import numpy as np
from PIL import Image

def denoise(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def removeBorders(image, original):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        mask = np.zeros_like(original)
        border = sorted(contours, key=cv2.contourArea)[-1]
        border.reshape(4, 2)
        cv2.drawContours(mask, [border], 0, (255,255,255), -1)
        return cv2.bitwise_and(original, mask)
    return original

def highlightTarget(footage, original):
    image = original
    image = cv2.GaussianBlur(image, (11, 11), 0)
    footage_edged = cv2.Canny(image, 0, 200)
    footage_edged = cv2.dilate(footage_edged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    contours, hierarchy = cv2.findContours(footage_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)[::-1]
    for contour in contours:
        epsilon = 0.02*cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, epsilon, True)
        if len(approximation) == 4:
            break
    
    corners = np.concatenate(approximation).tolist()
    sortedCorners = np.zeros((4, 2), "float32")
    sortedCorners[0] = corners[np.argmin(np.sum(corners, axis=1))]
    sortedCorners[1] = corners[np.argmin(np.diff(corners, axis=1))]
    sortedCorners[2] = corners[np.argmax(np.sum(corners, axis=1))]
    sortedCorners[3] = corners[np.argmax(np.diff(corners, axis=1))]

    for index, c in enumerate(corners):
        character = chr(65 + index)
        cv2.putText(image, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    x1 = np.sqrt(np.dot(sortedCorners[1], sortedCorners[2]))
    x2 = np.sqrt(np.dot(sortedCorners[0], sortedCorners[3]))
    maxWidth = np.uint(max(int(x1), int(x2)))

    y1 = np.sqrt(np.dot(sortedCorners[0], sortedCorners[1]))
    y2 = np.sqrt(np.dot(sortedCorners[2], sortedCorners[3]))
    maxHeight = np.uint(max(int(y1), int(y2)))

    destinationCorners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    M = cv2.getPerspectiveTransform(np.float32(sortedCorners), np.float32(destinationCorners))
    final = cv2.warpPerspective(original, M, (maxWidth, maxHeight))

    return final


vid = cv2.VideoCapture(0)

while(True):
    # Get video frame by frame
    ret, footage = vid.read()

    footage_grayscale = cv2.cvtColor(footage, cv2.COLOR_BGR2GRAY)
    footage_threshold = cv2.adaptiveThreshold(footage_grayscale, 180, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 10)
    footage_threshold = denoise(footage_threshold)
    footage_borderless = highlightTarget(footage_threshold, footage)
  
    # Display the resulting frame
    cv2.imshow('frame', footage)
    cv2.imshow('borderless', footage_borderless)
    cv2.imshow('denoised', footage_threshold)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()