import kayaru_standard_process as kstd

class filePath():
    _common_dir = kstd.getScriptDir() + "\\sample_file"
    def __init__(self,case,dir=_common_dir):
        if not kstd.isInt(case):
            kstd.echoErrorOccured("case setting in filePath class")
            kstd.judgeError(kstd.ERROR_CODE)

        self._case       = case
        
        self.image_list       = self._common_dir + "\\sample_list_{0:04d}.csv".format(self._case)
        self.label_list       = self._common_dir + "\\label_list_{0:04d}.csv".format(self._case)
        self.learned_param    = self._common_dir + "\\_learned_param_{0:04d}.ckpt".format(self._case)
