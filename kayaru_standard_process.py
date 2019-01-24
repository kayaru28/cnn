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

from abc import ABCMeta,abstractmethod
import kayaru_standard_messages as kstd_m

ERROR_CODE  = 100
NORMAL_CODE = 0

EC_NOT_FILE_EXIST = 101
EC_FILE_EXIST     = 102
EC_NOT_DIR_EXIST  = 103
EC_DIR_EXIST      = 104
EC_NOT_PATH_EXIST = 105


###########################################################
#
# rapper unit
#
###########################################################

def exit():
    sys.exit()

def joinDirPathAndName(dir_path,file_name):
    file_path = os.path.join(dir_path,file_name)
    return file_path

def checkPathExist(file_path):
    if os.path.exists(file_path):
        return NORMAL_CODE
    else:
        return EC_NOT_PATH_EXIST

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

def isNpList(val):
    val_type = type(val)
    np_list = np.array([])
    if val_type == type( np_list ):
        return True
    else:
        return False

###########################################################
#
# IO function set
#
###########################################################

def cpExec(file_path,copied_dir_path):
    file_name     = ntpath.basename(file_path)
    file_path_new = joinDirPathAndName(copied_dir_path,file_name) 
    shutil.copyfile(file_path, file_path_new)

def cpCheck(file_path,copied_dir_path):

    file_name     = ntpath.basename(file_path)
    file_path_new = joinDirPathAndName(copied_dir_path,file_name) 
    exit_code = checkPathExist(file_path_new)
    if exit_code == NORMAL_CODE:
        return EC_FILE_EXIST

    exit_code = checkPathExist(copied_dir_path)
    if not exit_code == NORMAL_CODE:
        return EC_NOT_DIR_EXIST

    exit_code = checkPathExist(file_path)
    if not exit_code == NORMAL_CODE:
        return EC_NOT_FILE_EXIST

    return NORMAL_CODE

def cp(file_path,copied_dir_path):
    exit_code = cpCheck(file_path,copied_dir_path)
    if exit_code == NORMAL_CODE:
        cpExec(file_path,copied_dir_path)
    return NORMAL_CODE


def mkdirExec(path):
    os.mkdir(path)

def mkdirCheck(path):
    if os.path.exists(path):
        return EC_DIR_EXIST
    return NORMAL_CODE

def getKeyboadInput():
    ans = input()
    return ans



#######################################################
# np unit
#######################################################

class AbstractDtoNp(metaclass=ABCMeta):
    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def getVariable(self):
        pass

    @abstractmethod
    def add(self):
        pass


class DtoNpList(AbstractDtoNp):
    def __init__(self,x_dtype="float64"):
        self.np_list = np.array([],dtype = x_dtype)

    def getVariable(self):
        return self.np_list

    def clear(self):
        self.np_list = np.array([])

    def add(self,x_val):
        self.np_list = np.append(self.np_list,x_val)

    def getListVal(self,index):
        return self.np_list[index]

    def getAttrLength(self):
        return self.np_list.shape[0]


class DtoNpTable(AbstractDtoNp):
    def __init__(self,col_length):
        self.col_length = col_length
        self.clear()

    def getVariable(self):
        return self.np_table

    def clear(self):
        self.np_table = np.empty((0,self.col_length))

    def add(self,dto_np):
        self.np_table = npGetListInsertedLists(self.np_table , dto_np.getVariable())

    def addList(self,dtoNL):
        self.add(dtoNL)

    def addTable(self,dtoNT):
        self.add(dtoNT)

    def addNpArray(self,np_array):
        self.np_table = npGetListInsertedLists(self.np_table , np_array)

    def getAttrRowLength(self):
        return self.np_table.shape[0]

    def getAttrColLength(self):
        return self.np_table.shape[1]   

def npNomalizaiton(x, axis=None):
    min = 0
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def npNomalizaitonMinMax(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def npGetListInsertedLists(np_lists,x_list):
    np_lists = np.insert(np_lists,np_lists.shape[0],x_list,axis = 0)
    return np_lists



#######################################################
# compare unit
#######################################################

def compareNpList(np_list1,np_list2):

    judge_same = True

    size1 = np_list1.shape[0]
    size2 = np_list2.shape[0]

    if size1 != size2:
        judge_same = False
        return judge_same

    for si in range(size1):
        if np_list1[si] != np_list2[si]:
            judge_same = False

    return judge_same

def compareDtoNpList(dtoNL1,dtoNL2):

    np_list1 = dtoNL1.getVariable()
    np_list2 = dtoNL2.getVariable()
    return compareNpList(np_list1,np_list2)

def compareNpTable(np_table1,np_table2):
    ndim1 = np_table1.ndim
    ndim2 = np_table2.ndim

    if not ndim1 == 2:
        return False

    if not ndim2 == 2:
        return False

    for di in range(ndim1):
        size1 = np_table1.shape[di]
        size2 = np_table2.shape[di]

        if not size1 == size2:
            return False

    size10 = np_table1.shape[0]
    size11 = np_table1.shape[1]

    for s0i in range(size10):
        for s1i in range(size11):
            if np_table1[s0i][s1i] != np_table2[s0i][s1i]:
                return False
    return True


def compareDtoNpTable(dtoNT1,dtoNT2):
    np_table1 = dtoNT1.getVariable()
    np_table2 = dtoNT2.getVariable()
    return compareNpTable(np_table1,np_table2)


def matchRateOfNpList(np_list1,np_list2):
    size        = np_list1.shape[0]
    count_match = 0.0

    for si in range(size):
        if np_list1[si] == np_list2[si]:
            count_match = count_match + 1.0

    return count_match / size

def matchRateOfDtoNpList(dtoNL1,dtoNL2):
    np_list1 = dtoNL1.getVariable()
    np_list2 = dtoNL2.getVariable()

    return matchRateOfNpList(np_list1,np_list2)



#######################################################
# label unit
#######################################################


def createStaticLabelTable(dtoNT,lists_depth,label):

    list_size = dtoNT.getAttrColLength()

    for ldi in range(lists_depth):
        dtoNL = DtoNpList()
        createStaticLabelList(dtoNL,list_size,label)
        dtoNT.addList(dtoNL)

    return NORMAL_CODE

def createStaticLabelList(dtoNL,list_size,label):
    for lsi in range(list_size):
        if lsi == label:
            flag = 1.0
        else:
            flag = 0.0
        dtoNL.add(flag)

    return NORMAL_CODE


#######################################################
# csv unit
#######################################################
def getCsvDataTable(file_path):

    np_table   = np.loadtxt(file_path,delimiter=",")

    if np_table.ndim == 1:
        col_length = 1
        dtoNT = DtoNpTable(col_length)
        for data in np_table:
            dtoNT.addNpArray(data)
    else:
        col_length = np_table.shape[1]
        dtoNT = DtoNpTable(col_length)
        dtoNT.addNpArray(np_table)

    return dtoNT

def writeNewCsvDataList(file_path,dtoNL):
    open_mode = 'w'
    _writeCsvDataList(file_path,dtoNL,open_mode)

def writeAddCsvDataList(file_path,dtoNL):
    open_mode = 'a'
    _writeCsvDataList(file_path,dtoNL,open_mode)

def writeNewCsvDataTable(file_path,dtoNT):
    open_mode = 'w'
    _writeCsvDataTable(file_path,dtoNL,open_mode)

def writeAddCsvDataTable(file_path,dtoNT):
    open_mode = 'a'
    _writeCsvDataTable(file_path,dtoNL,open_mode)

def _writeCsvDataList(file_path,dtoNL,open_mode):
    file = open( file_path , open_mode)
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(dtoNL.getVariable())
    file.close()

def _writeCsvDataTable(file_path,dtoNT,open_mode):
    file = open( file_path , open_mode)
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(dtoNT.getVariable())
    file.close()



















def sleepExec(sec):
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
def readCsvFile(file_path):
    data = np.genfromtxt(file_path,dtype=None,delimiter=",")
    return data

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

def echoFinish(process=""):
    print(str(getTimeyyyymmddhhmmss()) + "\tfinish process : " + process)

def echoIsAlready(process=""):
    print(str(getTimeyyyymmddhhmmss()) + "\t" + process + "\tis already")

def echoBar(length=50,mark="*"):
    
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


