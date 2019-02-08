#coding: UTF-8

# ver.2.0 tutorial
# ver.3.0 we get architecture from resnet

import kayaru_standard_process as kstd
import kayaru_standard_messages as kstd_m
import numpy as np

import tensorflow as tf

PADDING_SAME  = "SAME"
PADDING_VALID = "VALID"

STATIC_1 = 1
EPSILON = 1e-3


AXIS_X_N = 0
AXIS_X_W = 1
AXIS_X_H = 2
AXIS_X_C = 3

################################################################
#
# general
#
################################################################

def crateNLLabelFromValue(dto_data_set):

    row_length = dto_data_set.dtoNT_value.getAttrRowLength()
    col_length = dto_data_set.dtoNT_value.getAttrColLength()
    NT_value   = dto_data_set.dtoNT_value.getVariable()

    dtoNT_label = kstd.DtoNpTable(col_length)
    for ri in range(row_length):

        var_list = NT_value[ri]
        var_max  = var_list.max()

        dtoNL_label = kstd.DtoNpList()
    
        for ci in range(col_length):
            if var_max == var_list[ci]:
                label = 1
            else:
                label = 0

            dtoNL_label.add(label)
        dtoNT_label.addList(dtoNL_label)

    dto_data_set.addLabelTable(dtoNT_label)

    return kstd.NORMAL_CODE

def getX2dFeatureVolume(x_2d):
    num_wigth   = x_2d.shape[AXIS_X_W]
    num_height  = x_2d.shape[AXIS_X_H]
    num_channel = x_2d.shape[AXIS_X_C]

    volume = int(num_wigth * num_height * num_channel)

    return volume



################################################################
#
# formater
#
################################################################
def getBiasVar0InConv(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def getFilter0InConv(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 
    return tf.Variable(initial)

def getWeight0InFC(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 
    return tf.Variable(initial)



################################################################
#
# convolution
#
################################################################

class __Conv2d():

    def __init__(self):
        self.index_filter_width  = 0
        self.index_filter_height = 1
        self.index_in_channels   = 2
        self.index_out_channels  = 3
        self.index_stride_w      = 4
        self.index_stride_h      = 5
        self.max_index           = 5 + 1

        self.parameters = np.zeros(self.max_index, dtype="int32")

    def setFilterH(self,val):
        self.parameters[self.index_filter_height] = val

    def setFilterW(self,val):
        self.parameters[self.index_filter_width] = val

    def setFilterInChannels(self,val):
        self.parameters[self.index_in_channels] = val

    def setOutChannels(self,val):
        self.parameters[self.index_out_channels] = val

    def setStrideW(self,val):
        self.parameters[self.index_stride_w] = val

    def setStrideH(self,val):
        self.parameters[self.index_stride_h] = val

    def isSetParameters(self):
        if np.prod(self.parameters) == 0:
            return False
        else:
            return True

    def createW0(self):
        f_w    = self.parameters[self.index_filter_width]
        f_h    = self.parameters[self.index_filter_height]
        in_ch  = self.parameters[self.index_in_channels]
        out_ch = self.parameters[self.index_out_channels]
        W      = getFilter0InConv([f_w, f_h, in_ch, out_ch])
        return W

    def createStride(self):
        str_w = self.parameters[self.index_stride_w]
        str_h = self.parameters[self.index_stride_h]

        stride = [STATIC_1, str_w, str_h, STATIC_1]
        return stride

    def createB0(self):
        b_h = self.parameters[self.index_out_channels]
        b = getBiasVar0InConv([b_h])
        return b

    def calculate(self,x):
        pass


class Conv2dSame(__Conv2d):
    def __init__(self):
        super().__init__()

    def calculate(self,x,W):
        stride = self.createStride()
        return tf.nn.conv2d(x, W, stride, padding=PADDING_SAME)

class Conv2dValid(__Conv2d):
    def __init__(self):
        super().__init__()

    def calculate(self,x,W):
        stride = self.createStride()
        return tf.nn.conv2d(x, W, stride, padding=PADDING_VALID)


################################################################
#
# pooling
#
################################################################

class __Pool():

    def __init__(self):
        self.index_filter_width  = 0
        self.index_filter_height = 1
        self.index_stride_w      = 2
        self.index_stride_h      = 3
        self.max_index           = 3 + 1

        self.parameters = np.zeros(self.max_index, dtype="int32")

    def setFilterH(self,val):
        self.parameters[self.index_filter_height] = val

    def setFilterW(self,val):
        self.parameters[self.index_filter_width] = val

    def setStrideW(self,val):
        self.parameters[self.index_stride_w] = val

    def setStrideH(self,val):
        self.parameters[self.index_stride_h] = val

    def isSetParameters(self):
        if np.prod(self.parameters) == 0:
            return False
        else:
            return True

    def createFilter(self):
        f_w    = self.parameters[self.index_filter_width]
        f_h    = self.parameters[self.index_filter_height]
        f      = [STATIC_1, f_w, f_h, STATIC_1]
        return f

    def createStride(self):
        str_w = self.parameters[self.index_stride_w]
        str_h = self.parameters[self.index_stride_h]

        stride = [STATIC_1, str_w, str_h, STATIC_1]
        return stride

    def calculate(self,x):
        pass

class MaxPoolSame(__Pool):
    def __init__(self):
        super().__init__()

    def calculate(self,x):
        f      = self.createFilter()
        stride = self.createStride()
        return tf.nn.max_pool(x, f, stride, padding=PADDING_SAME)

class MaxPoolValid(__Pool):
    def __init__(self):
        super().__init__()

    def calculate(self,x):
        f      = self.createFilter()
        stride = self.createStride()
        return tf.nn.max_pool(x, f,stride, padding=PADDING_VALID)

class AvgPoolSame(__Pool):
    def __init__(self):
        super().__init__()

    def calculate(self,x):
        f      = self.createFilter()
        stride = self.createStride()
        return tf.nn.avg_pool(x, f,stride, padding=PADDING_SAME)

class AvgPoolValid(__Pool):
    def __init__(self):
        super().__init__()

    def calculate(self,x):
        f      = self.createFilter()
        stride = self.createStride()
        return tf.nn.avg_pool(x, f,stride, padding=PADDING_VALID)

def globalAveragePool(x):
    for _ in range(2):
        x = tf.reduce_mean(x, axis=1)
    return x

################################################################
#
# full connected
#
################################################################

class FullConnected():

    def __init__(self):
        self.index_input_size  = 0
        self.index_output_size = 1
        self.max_index         = 1 + 1
        self.parameters = np.zeros(self.max_index, dtype="int32")

    def setInputSize(self,val):
        self.parameters[self.index_input_size] = val

    def setOutputSize(self,val):
        self.parameters[self.index_output_size] = val

    def isSetParameters(self):
        if np.prod(self.parameters) == 0:
            return False
        else:
            return True

    def createW0(self):
        W_i    = self.parameters[self.index_input_size]
        W_o    = self.parameters[self.index_output_size]
        W      = getFilter0InConv([W_i, W_o])
        return W

    def createB0(self):
        b_h = self.parameters[self.index_out_channels]
        b = getBiasVar0InConv([bh])
        return b

    def calculate(self,x,W,keep_prob):
        x  = tf.nn.dropout(x, keep_prob)
        return tf.matmul(x, W)

################################################################
#
# batch normalization
#
################################################################

def batchNormWrapper(inputs, is_training, decay = 0.999):
    #Implementing Batch Normalization in Tensorflow - R2RT
    scale    = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta     = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var  = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        batch_mean = tf.cast(batch_mean,tf.float32)
        batch_var  = tf.cast(batch_var,tf.float32)
        train_mean = tf.assign(pop_mean,
                                tf.add(tf.multiply(pop_mean, decay),
                                        tf.multiply(batch_mean, tf.subtract(1.0, decay))))
        train_var = tf.assign(pop_var,
                                tf.add(tf.multiply(pop_var, decay),
                                        tf.multiply(batch_var, tf.subtract(1.0, decay))))
        with tf.control_dependencies([train_mean, train_var]):

            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, EPSILON)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, EPSILON)


################################################################
#
# activation function
#
################################################################
def activationFuncRelu(x):
    return tf.nn.relu(x)

def activationFuncSigmoid(x):
    return tf.nn.sigmoid(x)

################################################################
#
# residual module
#
################################################################
class ResidualResNet():
    def __init__(self):
        self.conv = Conv2dSame()

    def isSetParameters(self):
        if self.conv.isSetParameters():
            return True
        return False

    def createW0(self):
        W1 = self.conv.createW0()
        W2 = self.conv.createW0()
        return W1,W2

    def calculate(self,x,W1,W2,is_training=False):
        fx = x
        fx = self.conv.calculate(fx,W1)
        fx = batchNormWrapper(fx,is_training)
        fx = activationFuncRelu(fx)
        fx = self.conv.calculate(fx,W2)
        fx = batchNormWrapper(fx,is_training)
        fx = tf.add(fx, x)
        fx = activationFuncRelu(fx)
        return fx        

class ResidualResNetStochasticDepth():
    def __init__(self,p_l):
        self.p_l  = p_l
        self.conv = Conv2dSame()

    def isSetParameters(self):
        if self.conv.isSetParameters():
            return True
        return False

    def createW0(self):
        W1 = self.conv.createW0()
        W2 = self.conv.createW0()
        return W1,W2

    def calculate(self,x,W1,W2,is_training=False):
        fx = x
        fx = self.conv.calculate(fx,W1)
        fx = batchNormWrapper(fx,is_training)
        fx = activationFuncRelu(fx)
        fx = self.conv.calculate(fx,W2)
        fx = batchNormWrapper(fx,is_training)
        if self.p_l > np.random.rand():
            fx = tf.add(fx, x)
        else:
            fx = x
        fx = activationFuncRelu(fx)
        return fx        

################################################################
#
# output function
#
################################################################

class OutputForTFCNN():
    def __init__(self):
        self.learned_parameter_file_path = kstd.joinDirPathAndName(kstd.getScriptDir(),"_param.ckpt")
        self.learning_log_file_path      = kstd.joinDirPathAndName(kstd.getScriptDir(),"log.txt")
        self.predicted_value_file_path   = kstd.joinDirPathAndName(kstd.getScriptDir(),"_value.csv")
        self.predicted_label_file_path   = kstd.joinDirPathAndName(kstd.getScriptDir(),"_label.csv")
        self.validation_file_path        = kstd.joinDirPathAndName(kstd.getScriptDir(),"_validation.csv")
        self.summary_dir_path            = kstd.getScriptDir()

    def setLearnedParameterFilePath(self,file_path):
        self.learned_parameter_file_path = file_path
    
    def setLearningLogFilePath(self,file_path):
        self.learning_log_file_path = file_path

    def setPredictedValueFilePath(self,file_path):
        self.predicted_value_file_path   = file_path

    def setPredictedLabelFilePath(self,file_path):
        self.predicted_label_file_path   = file_path

    def setValidatinoFilePath(self,file_path):
        self.validation_file_path = file_path

    def setSummaryDirPath(self,dir_path):
        self.summary_dir_path            = dir_path

    def logOutput(self,msg):
        print(msg)
        kstd.writeAddCsvDataVal(self.learning_log_file_path, msg)

    def validationOutput(self,validation_value):
        kstd.writeAddCsvDataVal(self.validation_file_path,validation_value)

    def PredictionResultSave(self,dto_data_set):
        kstd.writeNewCsvDataTable(self.predicted_value_file_path,dto_data_set.dtoNT_value)
        kstd.writeNewCsvDataTable(self.predicted_label_file_path,dto_data_set.dtoNT_label)

################################################################
#
# data set
#
################################################################

class DtoDataSetForTFCNN():
    def __init__(self,wigth,height,num_of_label_kind):
        self.initializingImage(wigth,height)
        self.initializingLabel(num_of_label_kind)

    # format func #########################################################################

    def initializingImage(self,wigth,height):
        self.wigth  = wigth
        self.height = height

        self.pixel_size     = wigth * height
        self.dtoNT_flat_img = kstd.DtoNpTable(self.pixel_size)

    def initializingLabel(self,num_of_label_kind):
        self.num_of_label_kind = num_of_label_kind
        self.dtoNT_label       = kstd.DtoNpTable(num_of_label_kind)
        self.dtoNT_value       = kstd.DtoNpTable(num_of_label_kind)

    def clearFlatImageTable(self):
        self.dtoNT_flat_img.clear()

    # output func #########################################################################

    def fixDataSet(self):
        self.num_data = self.dtoNT_flat_img.getAttrRowLength()
        self.index_list = np.arange(self.num_data)
        np.random.shuffle(self.index_list)

        dtoNT_shuffle_flat_img = kstd.DtoNpTable(self.pixel_size)
        dtoNT_shuffle_label    = kstd.DtoNpTable(self.num_of_label_kind)

        NT_flat_img = self.dtoNT_flat_img.getVariable()
        NT_label    = self.dtoNT_label.getVariable()
        
        self.NT_flat_img = NT_flat_img[self.index_list]
        self.NT_label    = NT_label[self.index_list]

        self.count_out = 0

    def getBatchSet(self,batch_size):

        index_s = self.count_out

        if self.count_out + batch_size > self.num_data:
            index_e = self.num_data
            self.count_out = 0
            np.random.shuffle(self.index_list)
        else:    
            index_e = index_s + batch_size
            self.count_out = self.count_out + batch_size
            if self.count_out == self.num_data:
                self.count_out = 0
                np.random.shuffle(self.index_list)

        return self.NT_flat_img[index_s:index_e],self.NT_label[index_s:index_e]

    # add func #############################################################################

    def _addTable(self,dtoNT,dtoNT_added,target_name):

        having_col_size = dtoNT.getAttrColLength()
        given_col_size  = dtoNT_added.getAttrColLength()
        if not having_col_size == given_col_size:
            kstd.echoAisB("having size",having_col_size)
            kstd.echoAisB("nplist size",given_col_size)
            return kstd.ERROR_CODE

        dtoNT.addTable(dtoNT_added)
        return kstd.NORMAL_CODE

    def addFlatImageTable(self,dtoNT_flat_img):
        exit_code = self._addTable(self.dtoNT_flat_img,dtoNT_flat_img,"image_list")
        return exit_code

    def addLabelTable(self,dtoNT_label):
        exit_code = self._addTable(self.dtoNT_label,dtoNT_label,"label_list")
        return exit_code

    def addValueTable(self,dtoNT_value):
        exit_code = self._addTable(self.dtoNT_value,dtoNT_value,"value_list")
        return exit_code




















