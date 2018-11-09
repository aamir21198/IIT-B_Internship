#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:57:42 2018

@author: ruturaj
"""

import cv2
from matplotlib import pyplot as plt

# Read the image in BGR mode
img = cv2.imread('gridding practice.PNG', 1)

# Convert image from BGR to GRAY
img2 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

# Perform Canny edge detection
edges = cv2.Canny(img2, 100, 110)

plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# Find all the contours in the image after performing edge detection
im2, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Loop through each contour in the list
for c in contours:
    # Find the moments of the contour
    M = cv2.moments(c)
    # Check if the moments are not zero
    if M["m10"] and M["m00"] and M["m01"]:
        # Find the center of the contour
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Plot a circle with the above co-ordinates as the center
        cv2.circle(img, (cX, cY), 5, (255, 255, 255), 1, -1)

plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Gridded Image'), plt.xticks([]), plt.yticks([])
plt.show()
