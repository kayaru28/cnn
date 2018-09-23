import csv
import os
import sys
import numpy as np
import pandas as pd
import datetime
import time
import inspect
import ntpath
import shutil


ERROR_CODE  = 100
NORMAL_CODE = 0

def cp(file_path,copied_dir_path):
    if not os.path.exists(file_path):
        print("FUNC:cp in kstd\t:the file is not existing (" + file_path + ")")
        return ERROR_CODE
    else:
        file_name     = ntpath.basename(file_path)
        file_path_new = os.path.join(copied_dir_path,file_name) 
        shutil.copyfile(file_path, file_path_new)
        return NORMAL_CODE

def mkdir(path):
    if os.path.exists(path):
        print("FUNC:mkdir in kstd\t:the file is existing (" + path + ")")
        return ERROR_CODE
    else:
        os.mkdir(path)
        return NORMAL_CODE


def getPaddingString(x_str,length,padding=" "):

    if len(x_str) >= length:
        return x_str
    else:
        for i in range(length - len(x_str) ):
            x_str = x_str + padding
        return x_str

def getKeyboadInput():
    ans = input()
    return ans

def execSleep(sec):
    time.sleep(sec)

def left(str, amount):
    return str[:amount]

def right(str, amount):
    return str[-amount:]

def mid(str, offset, amount):
    return str[offset:offset+amount]

def max(a,b):
    ans = a
    if( a < b ):
        ans = b
    return ans

def min(a,b):
    ans = a
    if( a > b ):
        ans = b
    return ans

def cutStrBeforeKey(key,str):
    no  = patternMatch(key,str)
    ans = left( str , no - 1) 
    return ans

def cutStrAfterKey(key,str):
    no  = patternMatch(key,str)
    no  = no + ( len(key) - 1 )
    ans = right( str , len(str) - no )
    return ans

def exit():
    sys.exit()


def patternMatch(key,str):
    # 一致する　：＞0
    # 一致しない：＝0
    ans = str.find(key) + 1
    return ans

def judgeError(exit_code):
    if exit_code == ERROR_CODE:
        print("!!!!ERROR OCCURED!!!!11!!")
        sys.exit()

def convA2BinWord(word,a,b):
    ans = word.replace(a, b) 
    return ans

def getScriptDir():
    return os.path.abspath(os.path.dirname(__file__))

def getFileNameFromPath(file_path):
    return os.path.basename(file_path)

def getDateyyyymmdd():
    return str(datetime.date.today())

def getTimeyyyymmddhhmmss():
    return str(datetime.datetime.now())

def getTime():
    return time.time()

def getElapsedTime(base_time,unit="m"):
    elapsed_time = time.time() - base_time
    if unit == "m":
        elapsed_time = elapsed_time / 60
    elif unit == "h":
        elapsed_time = elapsed_time / 60 / 60
    return elapsed_time

###########################################################
#
# varidation
#
###########################################################

def isNotNull(str):
    ans = True

    if(str == None):
        ans = False
    elif(str == ""):
        ans = False
    return ans

def isNull(str):
    ans = False
    if(str == None):
        ans = True
    elif(str == ""):
        ans = True
    return ans

def isInt(val):
    if type(val) is int:
        return True
    else:
        return False

def isStr(val):
    if type(val) is str:
        return True
    else:
        return False

def isTuple(target):
    if isinstance(target, tuple):
        return True
    else:
        return False

def isList(target):
    if isinstance(target, list):
        return True
    else:
        return False

def isEvenNumber(val):

    if not type(val) is int:
        return False
    elif ( val % 2 == 0 ):
        return True
    else:
        return False

###########################################################
#
# read and write for csv
#
###########################################################

class CsvWriter():
    def __init__(self):
        self.file = ""

    def openFile(self,file_path):
        if isNull(file_path):
            echoNullOfAValue(file_path,locals())
            return ERROR_CODE
        
        self.file = open( file_path , 'w')

        return NORMAL_CODE

    def openFileForAdd(self,file_path):
        if isNull(file_path):
            echoNullOfAValue(file_path,locals())
            return ERROR_CODE
        if not os.path.exists(file_path):
            echoNotExistThatFile(file_path)
            return ERROR_CODE

        self.file = open( file_path , 'a')

        return NORMAL_CODE

    def closeFile(self):
        if isNull(self.file):
            echoOpenAnyFile()
            return ERROR_CODE

        self.file.close()

    def writeOfVal(self,val):
        if isNull(self.file):
            echoOpenAnyFile()
            return ERROR_CODE
        self.val_list = []
        self.val_list.append(val)
        self.writer = csv.writer(self.file, lineterminator='\n')
        self.writer.writerow(self.val_list)
        return NORMAL_CODE

    def writeOfList(self,var_list):
        if isNull(self.file):
            echoOpenAnyFile()
            return ERROR_CODE
        self.writer = csv.writer(self.file, lineterminator='\n')
        self.writer.writerow(var_list)
        return NORMAL_CODE

    def writeOfArray2d(self,array_2d):
        if isNull(self.file):
            echoOpenAnyFile()
            return ERROR_CODE
        self.writer = csv.writer(self.file, lineterminator='\n')
        self.writer.writerows(array_2d)
        return NORMAL_CODE

class CsvReader():

    def __init__(self):
        self.file = ""
        self.data = [[]]

    def openFile(self,file_path):
        if isNull(file_path):
            echoNullOfAValue(file_path,locals())
            return ERROR_CODE

        if not os.path.exists(file_path):
            echoNotExistThatFile(file_path)
            return ERROR_CODE

        self.file = open( file_path , "r")

        return NORMAL_CODE

    def closeFile(self):
        if isNull(self.file):
            echoOpenAnyFile()
            return ERROR_CODE

        self.file.close()

    def readFile(self):

        if isNull(self.file):
            echoOpenAnyFile()
            return ERROR_CODE

        self.data_list = csv.reader(self.file)

        for self.data_tmp in self.data_list:
            self.data.append(self.data_tmp)

        del self.data[0]

        return NORMAL_CODE

    def getData(self):
        return self.data

class CsvReaderViaNp():

    #def __init__(self):

    def readFile(self,file_path):
        if isNull(self.file):
            echoOpenAnyFile()
            return ERROR_CODE

        self.data = np.genfromtxt(file_path,dtype=None,delimiter=",")
        return NORMAL_CODE

    def getData(self):
        return self.data


def getVarName( var, symboltable=locals(), error=None ) :
    ans = "("
    for key in symboltable.keys():
        # in consideration of exsisting paires of same id variable
        if id(symboltable[key]) == id(var) :
            ans = ans + " "  + key
    ans = ans + " )"
    return ans

def compareType(var1,var2):
    if type(var1) == type(var2):
        return True
    else:
        return False

def compareNpListSize(var1,var2,axis):
    size1 = var1.shape[axis]
    size2 = var2.shape[axis]
    if size1 == size2:
        return True
    else:
        return False

###########################################################
#
# messages
#
###########################################################

#### layer 1 messages

def echoOpenAnyFile():
    print(" open any file ")

def echoNotExistThatFile(file_path):
    print(" not exist that file :" + file_path)

def echoNullOfAValue(var,symboltable=locals()):
    print(" a value is null :" + getVarName(var,symboltable) )

def echoBlank():
    print("")

def echoStart(process=""):
    print(str(getTimeyyyymmddhhmmss()) + "\tstart process : " + process)

def echoIsAlready(process=""):
    print(str(getTimeyyyymmddhhmmss()) + "\t" + process + "\tis already")

def echoBar(length="50",mark="*"):
    
    if not (isInt(length)):
        length = 50

    bar = ""
    for i in range(length):
        bar = bar + mark
    print(bar)

def echoList1d(x_list):
    for row in x_list:
        print(row)


def echoIsSetting(process="",var=""):
    print(str(process) + "\t: " + str(var) + "\tis setting")

def echoErrorCodeIs(error_code=""):
    print("ERROR_CODE is : " + str(error_code) )

def echoAisB(name,var):
    print( str(name) + "\tis\t" + str(var) )


#### layer 2 messages
def echoBlanks(num):
    for i in range(num):
        echoBlank()

def echoErrorOccured(detail=""):
    echoBlank()
    echoBar()
    print("error is occured !!!!!!!!")
    if(detail!=""):
        print("\t(detail) " + detail)
    echoBar()
    echoBlank()


