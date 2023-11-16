import cv2
import numpy as np

def acimage(image):
    #image=cv2.imread("2.png",flags=0)

    #cv2.waitKey()
    #cv2.destroyAllWindows(0)
    image=np.array(image)
    #print(image.shape)

    w=image.shape[0]
    h=image.shape[1]

    #生成积分图像
    ac_image=np.arange(w*h).reshape(w,h)

    for i in range(w):
        for ii in range(h):
            if (ii==0 and i==0):
                ac_image[i][ii]=image[i][ii]
            else:
                if(i==0 and ii!=0):
                    ac_image[i][ii]=ac_image[i][ii-1]+image[i][ii]
                else:
                    if(i!=0 and ii==0):
                        ac_image[i][ii]=ac_image[i-1][ii]+image[i][ii]
                    else:
                        ac_image[i][ii]=image[i][ii]+ac_image[i-1][ii]+ac_image[i][ii-1]-ac_image[i-1][ii-1]

    #print(image[0][0],image[0][1],image[0][2])
    #print(image[1][0],image[1][1],image[1][2])
    #print(image[2][0],image[2][1],image[2][2])
    #print(ac_image[0][0],ac_image[0][1],ac_image[0][2])
    #print(ac_image[1][0],ac_image[1][1],ac_image[1][2])
    #print(ac_image[2][0],ac_image[2][1],ac_image[2][2])
    #cv2.imshow('src1',ac_image)
    #cv2.waitKey()
    #cv2.imshow('src',image)
    #cv2.waitKey()
    #cv2.destroyAllWindows(0)
    return ac_image
#平直矩形x,y,w,h
def ac_rec(image,x,y,w,h):
    i=acimage(image)
    RectSum=i[x-1][y-1]+i[x+w-1][y+h-1]-i[x-1][y+h-1]-i[x+w-1][y-1]
