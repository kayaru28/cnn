import kayaru_standard_process  as kstd
import kayaru_standard_messages as kstd_m

def assertionCheck(result,message):
    assert result,message

def assertionCheckIsInt(var,var_name=""):
    result = kstd.isInt(var)
    if not ( result ):
        name    = kstd.getVarName(var)
        if var_name != "":
            message = kstd_m.getXisNotInt(var_name)
        else:
            message = kstd_m.getXisNotInt(name)
        assertionCheck(result,message)

def assertionCheckIsStr(var,var_name=""):
    result = kstd.isStr(var)
    if not ( result ):
        name    = kstd.getVarName(var)
        if var_name != "":
            message = kstd_m.getXisNotStr(var_name)
        else:
            message = kstd_m.getXisNotStr(name)
        assertionCheck(result,message)

def assertionCheckIsList(var,var_name=""):
    result = kstd.isList(var)
    if not ( result ):
        name    = kstd.getVarName(var)
        if var_name != "":
            message = kstd_m.getXisNotList(var_name)
        else:
            message = kstd_m.getXisNotList(name)
        assertionCheck(result,message)


