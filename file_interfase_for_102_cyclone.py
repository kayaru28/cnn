import kayaru_standard_process as kstd
import os


def getSaveDirName(case):
    if not kstd.isInt(case):
        kstd.echoErrorOccured("case setting in fi func getSaveDirName")
        kstd.judgeError(kstd.ERROR_CODE)

    return "case{0:04d}".format(case)


class filePath():
    _common_dir = kstd.getScriptDir() + "\\sample_file"
    def __init__(self,case,common_dir=_common_dir):
        if not kstd.isInt(case):
            kstd.echoErrorOccured("case setting in filePath class")
            kstd.judgeError(kstd.ERROR_CODE)

        self._case       = case

        self._common_dir = common_dir
        self._input_dir  = self._common_dir
        self.output_dir  = self._common_dir
        self._updatePathes()

        self.test_image_list = self._input_dir + "\\test_sample_list_{0:04d}.csv".format(self._case)
        self.prop = kstd.getScriptDir() + "\\properties_for_102_cyclone.py"


    def _updatePathes(self):
        self.image_list = self._input_dir + "\\sample_list_{0:04d}.csv".format(self._case)
        self.label_list = self._input_dir + "\\label_list_{0:04d}.csv".format(self._case)

        self.learned_param   = self.output_dir + "\\_learned_param_{0:04d}.ckpt".format(self._case)
        self.predicted_value = self.output_dir + "\\_predicted_value_{0:04d}.csv".format(self._case)
        self.predicted_label = self.output_dir + "\\_predicted_label_{0:04d}.csv".format(self._case)


    def updateCase(self,case):
        if not kstd.isInt(case):
            kstd.echoErrorOccured("case setting in filePath class")
            kstd.judgeError(kstd.ERROR_CODE)
            return kstd.ERROR_CODE

        self._case = case
        self._updatePathes()
        return kstd.NORMAL_CODE

    def updateOutputDir(self,output_dir):
        if not os.path.exists(output_dir):
            kstd.echoErrorOccured("output dir is not exists in filePath class : " + output_dir)
            kstd.judgeError(kstd.ERROR_CODE)
            return kstd.ERROR_CODE

        self.output_dir = output_dir
        self._updatePathes()
        return kstd.NORMAL_CODE

