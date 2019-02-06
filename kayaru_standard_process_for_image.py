###############################################################
#
# version
#  1.0 2018/11/01
#
###############################################################

from PIL import Image
import numpy as np
import kayaru_standard_process as kstd
import os
import glob

class DataImage():
    def __init__(self,image_path):
        self.image_path = ""

        if kstd.checkPathExist(image_path) == kstd.NORMAL_CODE:
            self.image_path = image_path
            self.image      = Image.open(image_path)


    def isImageOpend(self):
        if self.image_path == "":
            return False
        return True

    def getRawImage(self,dtoNT_image):
        dtoNT_image.addNpArray(np.array(self.image))
        return kstd.NORMAL_CODE

    def getGrayConvertedImage(self,dtoNT_gray_image):
        dtoNT_gray_image.addNpArray(np.array(self.image.convert('L')))
        return kstd.NORMAL_CODE

    def getFlatList(self,dtoNL_flatten_image):
        dtoNL_flatten_image.add(np.array(self.image).flatten())
        return kstd.NORMAL_CODE


    def getShape(self):
        return np.array(self.image).shape

    def getFlatShape(self):
        return np.array(self.image).flatten().shape

    def getHeight(self):
        wigth, height = self.image.size
        return height

    def getWigth(self):
        wigth, height = self.image.size
        return wigth

    def checkImageOpend(self):
        if self.image_path == "":
            return kstd.ERROR_CODE
        return kstd.NORMAL_CODE 


























##############################################################################
##############################################################################
##############################################################################

def getImageListsInTargetDir(dir_path,dto_np_table_flat_image):


    image_path_list = glob.glob( dir_path + '\\*') 
    exit_code = kstd.checkDirExist(dir_path)


    if not dto_flat_image_lists.isInitialized():
        image_path = image_path_list[0]
        image_data = DataImage(image_path)
        dto_flat_image_lists.initialize(image_data.getHeight(),image_data.getWigth())

    for image_path in image_path_list:
        image_data = DataImage(image_path)
        exit_code  = max(exit_code , dto_flat_image_lists.addList(image_data.getFlatList()))

    return exit_code

def checkImageSize(height,wigth,flat_image_nplist):
    base_length  = height * wigth
    given_length = flat_image_nplist.shape[0]

    if base_length == given_length:
        return True
    else:
        return False

