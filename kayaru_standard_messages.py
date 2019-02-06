
import kayaru_standard_process as kstd
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


#######################################################
# message parts
#######################################################

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

def messOpenAnyFile():
    message = " open any file "
    return message

def messNotExistThatFile(file_path):
    message = "not exist that file :%s" % ( file_path )
    return message

def messStartProcess(process):
    message = "start process : %s" % process
    return message

def messFinishProcess(process):
    message = "finish process : %s" % process
    return message

def messIsAlready(process):
    message = "%s\tis already" % process
    return message

def messIsSetting(process,var):
    message = "%s\t: %s \tis setting" % ( str(process), str(var) )
    return message
#######################################################
# message echors
#######################################################

def _print(message):
    print(message)

def echoOpenAnyFile():
    _print(messOpenAnyFile)

def echoNotExistThatFile(file_path):
    _print(messNotExistThatFile(file_path))

def echoNullOfAValue(var,symboltable=locals()):
    _print(" a value is null :" + getVarName(var,symboltable) )

def echoBlank():
    _print("")

def echoStart(process=""):
    _print("%s\t%s" % ( str( getTimeyyyymmddhhmmss()),messStartProcess(process) ) )

def echoFinish(process=""):
    _print("%s\t%s" % ( str( getTimeyyyymmddhhmmss()),messFinishProcess(process) ) )

def echoIsAlready(process=""):
    _print( "%s\t%s" % ( str( getTimeyyyymmddhhmmss()),messIsAlready(process) ) ) 

def echoIsSetting(process="",var=""):
    _print( messIsSetting(process,var) )

def echoBar(length=50,mark="*"):
    
    if not (isInt(length)):
        length = 50

    bar = ""
    for i in range(length):
        bar = bar + mark
    _print(bar)

def echoList1d(x_list):
    for row in x_list:
        _print(row)

def echoAisB(name,var):
    _print( str(name) + "\tis\t" + str(var) )

#### layer 2 messages
def echoBlanks(num):
    for i in range(num):
        echoBlank()

def echoErrorOccured(detail=""):
    echoBlank()
    echoBar()
    _print(messErrorOccured())
    if(detail!=""):
        _print("\t(detail) " + detail)
    echoBar()
    echoBlank()


