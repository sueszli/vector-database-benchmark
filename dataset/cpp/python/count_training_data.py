import os

path1 = "../data/finish-line/bmps/train/neg/"
path2 = "../data/finish-line/bmps/train/pos/"

neg = len(os.listdir(path1))
pos = len(os.listdir(path2))

print("Pos: " + str(pos))
print("Neg: " + str(neg))
print("Total: " + str(pos+neg))

