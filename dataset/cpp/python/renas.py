#coding=utf-8
from PIL import Image
import os
import os.path
import glob

def rename(rename_path,outer_path,folderlist):
#列举文件夹
    for folder in folderlist:
        if os.path.basename(folder)=='bus':
            foldnum = 0
        elif os.path.basename(folder)=='taxi':
            foldnum = 1
        elif os.path.basename(folder)=='truck':
            foldnum = 2
        elif os.path.basename(folder)=='family sedan':
            foldnum = 3
        elif os.path.basename(folder)=='minibus':
            foldnum = 4
        elif os.path.basename(folder)=='jeep':
            foldnum = 5
        elif os.path.basename(folder)=='SUV':
            foldnum = 6
        elif os.path.basename(folder)=='heavy truck':
            foldnum = 7
        elif os.path.basename(folder)=='racing car':
            foldnum = 8
        elif os.path.basename(folder)=='fire engine':
            foldnum = 9
        inner_path = os.path.join(outer_path, folder)
        total_num_folder = len(folderlist)       #文件夹的总数
        # print 'total have %d folders' % (total_num_folder)   #打印文件夹的总数
        filelist = os.listdir(inner_path)        #列举图片
        i = 0
        for item in filelist:
            total_num_file = len(filelist)       #单个文件夹内图片的总数
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(inner_path), item)           #原图的地址
                dst = os.path.join(os.path.abspath(rename_path), str(foldnum) + '_' + str(i) + '.jpg')        #新图的地址（这里可以把str(folder) + '_' + str(i) + '.jpg'改成你想改的名称）
            try:
                os.rename(src, dst)
                # print 'converting %s to %s ...' % (src, dst)
                i += 1
            except:
                continue
#训练集
rename_path1 = 'G:/Deeplearn/signal_and_system/renametrain'
outer_path1 = 'G:/Deeplearn/signal_and_system/train'
folderlist1 = os.listdir(r"G:/Deeplearn/signal_and_system/train")
rename(rename_path1,outer_path1,folderlist1)
print("train totally rename ! ! !")
#测试集
rename_path2 = 'G:/Deeplearn/signal_and_system/renametest'
outer_path2 = 'G:/Deeplearn/signal_and_system/val'
folderlist2 = os.listdir(r"G:/Deeplearn/signal_and_system/val")
rename(rename_path2,outer_path2,folderlist2)
print("test totally rename ! ! !")

#修改图片尺寸
def convertjpg(jpgfile,outdir,width=32,height=32):
    img=Image.open(jpgfile)
    img=img.convert('RGB')
    img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    new_img=img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
#训练集
for jpgfile in glob.glob("G:/Deeplearn/signal_and_system/renametrain/*.jpg"):
    convertjpg(jpgfile,"G:/Deeplearn/signal_and_system/data")
print("train totally resize ! ! !")
#测试集
for jpgfile in glob.glob("G:/Deeplearn/signal_and_system/renametest/*.jpg"):
    convertjpg(jpgfile,"G:/Deeplearn/signal_and_system/test")

print("test totally resize ! ! !")
