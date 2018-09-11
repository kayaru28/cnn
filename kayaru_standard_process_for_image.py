from PIL import Image
import numpy as np
import kayaru_standard_process as kstd
import os

def convGrayImage2NpList(image_file_path):

    if not os.path.exists(image_file_path):
        kstd.echoNotExistThatFile(image_file_path)

    gray_image = np.array(Image.open(image_file_path).convert('L'))

    return gray_image

def convImageNpList2d2Flat(image_nplist):
    flat_image_nplist = image_nplist.flatten()

    return flat_image_nplist   

def checkImageSize(height,wigth,flat_image_nplist):
    base_length  = height * wigth
    given_length = flat_image_nplist.shape[0]

    if base_length == given_length:
        return True
    else:
        return False

def getHeightFromImage(image_file_path):
    if kstd.isNull(image_file_path):
        kstd.echoNullOfAValue(image_file_path,locals())
        return 0
    if not os.path.exists(image_file_path):
        kstd.echoNotExistThatFile(image_file_path)
        return 0
    im = Image.open(image_file_path)
    w, h = im.size
    return h

def getWightFromImage(image_file_path):
    if kstd.isNull(image_file_path):
        kstd.echoNullOfAValue(image_file_path,locals())
        return 0
    if not os.path.exists(image_file_path):
        kstd.echoNotExistThatFile(image_file_path)
        return 0
    im   = Image.open(image_file_path)
    w, h = im.size
    return w

class DtoFlatImageDataNplistForTf():
    def __init__(self):
        self.height               = 0
        self.wigth                = 0
        self.list_size            = 0

    def firstlization(self,height,wigth):
        self.height               = height
        self.wigth                = wigth
        self.list_size            = self.wigth * self.height
        self.flat_image_data_list = np.empty((0,self.list_size))
        self.list_tmp             = np.zeros((1,self.list_size))
        
    def addList(self,flat_image_nplist):
        if not kstd.compareType(self.flat_image_data_list,flat_image_nplist):
            print("wrong image list!!!!!")
            return kstd.ERROR_CODE

        if not checkImageSize(self.height,self.wigth,flat_image_nplist):
            print("not same image size!!!!!!")
            if( self.list_size == 0 ):
                print("not firstlization height or wigth")
            return kstd.ERROR_CODE

        self.list_tmp[0]          = flat_image_nplist
        self.flat_image_data_list = np.append(self.flat_image_data_list,self.list_tmp,axis = 0)
        return kstd.NORMAL_CODE

    def clearList(self):
        self.flat_image_data_list = np.empty((0,self.list_size))

    def valCheck(self):
        kstd.echoBlank()
        print("image height : " + str(self.height) ) 
        print("image wigth  : " + str(self.wigth) ) 

    def getList(self):
        return self.flat_image_data_list
