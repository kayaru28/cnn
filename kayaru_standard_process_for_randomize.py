import random
import kayaru_standard_process as kstd


def getVarInt(start,end):
    
    if not ( kstd.isInt(start) ):
        message = "start is not int"
        kstd.echoErrorOccured(message)
        return None
    elif not ( kstd.isInt(end) ):
        message = "end is not int"
        kstd.echoErrorOccured(message)
        return None

    if (start > end):
        message = "start is bigger than end"
        kstd.echoErrorOccured(message)
        return None

    step    = 1
    end     = end + 1
    var_int = random.randrange(start,end,step)
    return var_int

def getVarEvenInt(start,end):
    
    if not ( kstd.isEvenNumber(start) ):
        message = "start is not even number"
        kstd.echoErrorOccured(message)
        return None
    elif not ( kstd.isEvenNumber(end) ):
        message = "end is not even number"
        kstd.echoErrorOccured(message)
        return None

    if (start > end):
        message = "start is bigger than end"
        kstd.echoErrorOccured(message)
        return None

    step = 2
    var_even_number = random.randrange(start,end,step)
    return var_even_number




