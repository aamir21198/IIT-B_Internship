import numpy as np
import cv2


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def apply_clahe(img, clipLimit=2.0, tileGridSize=(8,8)):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def uniform_illuminate(img, gamma=1.0, clipLimit=2.0, tileGridSize=(8,8)):
    img2 = adjust_gamma(img, gamma=gamma)
    final = apply_clahe(img2, clipLimit=clipLimit, tileGridSize=tileGridSize)
    return final

img = cv2.imread('grain.png')
cl1 = uniform_illuminate(img, gamma=1.5, clipLimit=3.0, tileGridSize=(8,8))
# res = np.hstack((img,cl1))
res = np.hstack((img, uniform_illuminate(img, gamma=1.5, clipLimit=1.5, tileGridSize=(8,8)),
    uniform_illuminate(img, gamma=2, clipLimit=1.5, tileGridSize=(8,8))))
cv2.imshow('result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
