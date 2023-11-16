from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
from Common.BuildToolError import *
from Common.StringUtils import GetLineNo
from Common.Misc import PathClass
from Common.LongFilePathSupport import LongFilePath
import re
import os
from Common.GlobalData import gIdentifierPattern
from .UniClassObject import StripComments
IMAGE_TOKEN = re.compile('IMAGE_TOKEN *\\(([A-Z0-9_]+) *\\)', re.MULTILINE | re.UNICODE)
EFI_HII_IIBT_END = 0
EFI_HII_IIBT_IMAGE_1BIT = 16
EFI_HII_IIBT_IMAGE_1BIT_TRANS = 17
EFI_HII_IIBT_IMAGE_4BIT = 18
EFI_HII_IIBT_IMAGE_4BIT_TRANS = 19
EFI_HII_IIBT_IMAGE_8BIT = 20
EFI_HII_IIBT_IMAGE_8BIT_TRANS = 21
EFI_HII_IIBT_IMAGE_24BIT = 22
EFI_HII_IIBT_IMAGE_24BIT_TRANS = 23
EFI_HII_IIBT_IMAGE_JPEG = 24
EFI_HII_IIBT_IMAGE_PNG = 25
EFI_HII_IIBT_DUPLICATE = 32
EFI_HII_IIBT_SKIP2 = 33
EFI_HII_IIBT_SKIP1 = 34
EFI_HII_IIBT_EXT1 = 48
EFI_HII_IIBT_EXT2 = 49
EFI_HII_IIBT_EXT4 = 50
EFI_HII_PACKAGE_TYPE_ALL = 0
EFI_HII_PACKAGE_TYPE_GUID = 1
EFI_HII_PACKAGE_FORMS = 2
EFI_HII_PACKAGE_STRINGS = 4
EFI_HII_PACKAGE_FONTS = 5
EFI_HII_PACKAGE_IMAGES = 6
EFI_HII_PACKAGE_SIMPLE_FONTS = 7
EFI_HII_PACKAGE_DEVICE_PATH = 8
EFI_HII_PACKAGE_KEYBOARD_LAYOUT = 9
EFI_HII_PACKAGE_ANIMATIONS = 10
EFI_HII_PACKAGE_END = 223
EFI_HII_PACKAGE_TYPE_SYSTEM_BEGIN = 224
EFI_HII_PACKAGE_TYPE_SYSTEM_END = 255

class IdfFileClassObject(object):

    def __init__(self, FileList=[]):
        if False:
            for i in range(10):
                print('nop')
        self.ImageFilesDict = {}
        self.ImageIDList = []
        for File in FileList:
            if File is None:
                EdkLogger.error('Image Definition File Parser', PARSER_ERROR, 'No Image definition file is given.')
            try:
                IdfFile = open(LongFilePath(File.Path), mode='r')
                FileIn = IdfFile.read()
                IdfFile.close()
            except:
                EdkLogger.error('build', FILE_OPEN_FAILURE, ExtraData=File)
            ImageFileList = []
            for Line in FileIn.splitlines():
                Line = Line.strip()
                Line = StripComments(Line)
                if len(Line) == 0:
                    continue
                LineNo = GetLineNo(FileIn, Line, False)
                if not Line.startswith('#image '):
                    EdkLogger.error('Image Definition File Parser', PARSER_ERROR, 'The %s in Line %s of File %s is invalid.' % (Line, LineNo, File.Path))
                if Line.find('#image ') >= 0:
                    LineDetails = Line.split()
                    Len = len(LineDetails)
                    if Len != 3 and Len != 4:
                        EdkLogger.error('Image Definition File Parser', PARSER_ERROR, 'The format is not match #image IMAGE_ID [TRANSPARENT] ImageFileName in Line %s of File %s.' % (LineNo, File.Path))
                    if Len == 4 and LineDetails[2] != 'TRANSPARENT':
                        EdkLogger.error('Image Definition File Parser', PARSER_ERROR, 'Please use the keyword "TRANSPARENT" to describe the transparency setting in Line %s of File %s.' % (LineNo, File.Path))
                    MatchString = gIdentifierPattern.match(LineDetails[1])
                    if MatchString is None:
                        EdkLogger.error('Image Definition  File Parser', FORMAT_INVALID, 'The Image token name %s defined in Idf file %s contains the invalid character.' % (LineDetails[1], File.Path))
                    if LineDetails[1] not in self.ImageIDList:
                        self.ImageIDList.append(LineDetails[1])
                    else:
                        EdkLogger.error('Image Definition File Parser', PARSER_ERROR, 'The %s in Line %s of File %s is already defined.' % (LineDetails[1], LineNo, File.Path))
                    if Len == 4:
                        ImageFile = ImageFileObject(LineDetails[Len - 1], LineDetails[1], True)
                    else:
                        ImageFile = ImageFileObject(LineDetails[Len - 1], LineDetails[1], False)
                    ImageFileList.append(ImageFile)
            if ImageFileList:
                self.ImageFilesDict[File] = ImageFileList

def SearchImageID(ImageFileObject, FileList):
    if False:
        print('Hello World!')
    if FileList == []:
        return ImageFileObject
    for File in FileList:
        if os.path.isfile(File):
            Lines = open(File, 'r')
            for Line in Lines:
                ImageIdList = IMAGE_TOKEN.findall(Line)
                for ID in ImageIdList:
                    EdkLogger.debug(EdkLogger.DEBUG_5, 'Found ImageID identifier: ' + ID)
                    ImageFileObject.SetImageIDReferenced(ID)

class ImageFileObject(object):

    def __init__(self, FileName, ImageID, TransParent=False):
        if False:
            for i in range(10):
                print('nop')
        self.FileName = FileName
        self.File = ''
        self.ImageID = ImageID
        self.TransParent = TransParent
        self.Referenced = False

    def SetImageIDReferenced(self, ImageID):
        if False:
            return 10
        if ImageID == self.ImageID:
            self.Referenced = True