

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

