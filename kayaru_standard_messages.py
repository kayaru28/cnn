import numpy as np

class MessQueue():

    _error_mess = "no more message"

    def __init__(self):
        self._count_in  = 0
        self._count_out = 0
        self._queue     = np.array([]) 

    def setMess(self,mess):
        self._queue    = np.append(self._queue,mess)
        self._count_in = self._count_in + 1

    def getMess(self):
        if self._count_in > self._count_out:
            index_out       = self._count_out
            self._count_out = self._count_out + 1
            return self._queue[index_out]
        else:
            return _error_mess

def messDoesNotExist(name):
    message = "%s does not exit" % ( name )

def messExists(name):
    message = "%s exits" % ( name )


def messInFonc(func_name):
    message = "in func : %s" % ( func_name ) 
    return message

def messErrorOccured():
    message = ""
    message = message + "!!!!!!!!!!!!!111111!!!!11!!\n"
    message = message + "!!!!!!ERROR OCCURED!!!!11!!\n"
    message = message + "!!!!!!!!!!!!!111111!!!!11!!"
    return message

def messGetXisNotInt(name):
    message = str(name) + " is not int"
    return message

def messGetXisNotStr(name):
    message = str(name) + " is not string"
    return message

def messGetXisNotList(name):
    message = str(name) + " is not list"
    return message

def messIsExecuting(process="X"):
    message = process + " is executing..... "
    return message

def messStartProcess(process="X"):
    message = "start process : %s " % ( process )
    return message

def messIsReady(process="X"):
    message = "%s is ready" % ( process )
    return message

def messPaddingMessage(mess,length,padding=" "):

    if len(mess) < length:
        for i in range(length - len(mess) ):
            mess = mess + padding
    return mess

