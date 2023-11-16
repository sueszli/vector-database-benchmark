from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

def subContourns(src):
    if False:
        while True:
            i = 10
    (width, height) = src.shape[:2]
    src = cv.resize(src, (int(height * 150 / 100), int(width * 150 / 100)))
    cinza = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    (_, canny_output) = cv.threshold(cinza, 127, 255, cv.THRESH_BINARY)
    (contours, _) = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return len(contours)

def filter(src):
    if False:
        i = 10
        return i + 15
    (width, height) = src.shape[:2]
    src = cv.resize(src, (int(height * 150 / 100), int(width * 150 / 100)))
    cinza = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    canny_output = cv.Canny(cinza, 200, 200 * 2)
    cv.imshow('Contours', canny_output)
    cv.waitKey(0)
    (contours, _) = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for (i, c) in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
    i = 0
    for c in contours:
        perimetro = cv.arcLength(c, True)
        if perimetro > 200:
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            (x, y, alt, lar) = cv.boundingRect(c)
            crop = src[y:y + lar, x:x + alt]
            cv.rectangle(src, (x, y), (x + alt, y + lar), color, 2)
            scale = crop.shape[1] / float(crop.shape[0])
            if scale > 2.5 and scale < 3.5:
                subContorn = subContourns(crop)
                if subContorn > 10 and subContorn < 50:
                    cv.imshow('Contours', crop)
                    cv.waitKey(0)
            i += 1
if __name__ == '__main__':
    src = cv.imread('img1.png')
    filter(src)