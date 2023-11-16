import cv2
import webcolors
import numpy as np
from matplotlib import pyplot as plt
import time

img = cv2.imread("../FIND-CONTORNS/roi.png")
# Redimensiona imagem para acelerar  trabalho
# img = cv2.resize(img, (100, 50))

cv2.imshow("img", img)
cv2.waitKey(0)
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([50, 50, 50])
upper_blue = np.array([130, 255, 255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)
# Plot image
cv2.imshow("res", res)
cv2.waitKey()

# obtem uma lista de RBG da imagem
b, g, r = cv2.split(res)
rgb = cv2.merge([r, g, b])

t1 = time.time()
# Percorre os rgb pra verificar a ocorrrencia de azul
azul = []
for i in range(len(rgb)):
    for k in range(len(rgb[0])):
        corAtual = webcolors.rgb_to_hex(
            (((rgb[i])[k][0]), ((rgb[i])[k][1]), ((rgb[i])[k][2]))
        )
        # invalida cor preta e busca por ocorrencia de azul = "#00"
        # invalida cor contorno do recorte 00ff00
        if corAtual != "#000000" and corAtual != "#00ff00" and corAtual[:3] == "#00":
            print(corAtual)
            azul.append(corAtual.upper())
t2 = time.time()
print(t2 - t1, len(azul))
