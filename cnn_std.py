#coding: UTF-8

import kayaru_standard_process as kstd
import kayaru_standard_process_for_randomize as rand
import kayaru_standard_messages as kstd_m
import numpy as np
import sys

import tensorflow as tf

AXIS_X_IMAGE_NUM     = 0
AXIS_X_IMAGE_WIGTH   = 1
AXIS_X_IMAGE_HEIGHT  = 2
AXIS_X_IMAGE_CHANNEL = 3

MODE_LEARNING    = "Learning"
MODE_RE_LEARNING = "re-Learning"
MODE_PREDICTION  = "Prediction"

FLAG_MAX_POOL = "max_pool"
FLAG_AVG_POOL = "average_pool"


class DtoOutputPathForTFCNN():
    def __init__(self):
        self.learned_parameter_file_path = kstd.joinDirPathAndName(kstd.getScriptDir(),"_param.ckpt")
        self.learning_log_file_path      = kstd.joinDirPathAndName(kstd.getScriptDir(),"log.txt")
        self.predicted_value_file_path   = kstd.joinDirPathAndName(kstd.getScriptDir(),"_value.csv")
        self.predicted_label_file_path   = kstd.joinDirPathAndName(kstd.getScriptDir(),"_label.csv")
        self.summary_dir_path            = kstd.getScriptDir()

    def setLearnedParameterFilePath(self,file_path):
        self.learned_parameter_file_path = file_path
    
    def setLearningLogFilePath(self,file_path):
        self.learning_log_file_path = file_path

    def setPredictedValueFilePath(self,file_path):
        self.predicted_value_file_path   = file_path

    def setPredictedLabelFilePath(self,file_path):
        self.predicted_label_file_path   = file_path

    def setSummaryDirPath(self,dir_path):
        self.summary_dir_path            = dir_path

class DtoDataSetForTFCNN():
    def __init__(self,wigth,height,num_of_label_kind):
        self.firstlizationImage(wigth,height)
        self.firstlizationLabel(num_of_label_kind)


    # format func #########################################################################

    def firstlizationImage(self,wigth,height):
        self.height                = height
        self.wigth                 = wigth
        self.image_list_size       = self.wigth * self.height
        self.dtoNT_flat_image      = kstd.DtoNpTable(self.image_list_size)

    def firstlizationLabel(self,num_of_label_kind):
        self.num_of_label_kind      = num_of_label_kind
        self.dtoNT_label            = kstd.DtoNpTable(self.num_of_label_kind)
        self.dtoNT_value = kstd.DtoNpTable(self.num_of_label_kind)

    def clearFlatImageTable(self):
        self.dtoNT_flat_image.clear()

    # add func #############################################################################

    def _addTable(self,dtoNT,dtoNT_added,target_name):

        having_col_size = dtoNT.getAttrColLength()
        given_col_size  = dtoNT_added.getAttrColLength()
        if not having_col_size == given_col_size:
            echoXisNotSameSize(target_name)
            kstd.echoAisB("having size",having_col_size)
            kstd.echoAisB("nplist size",given_col_size)
            return kstd.ERROR_CODE

        dtoNT.addTable(dtoNT_added)
        return kstd.NORMAL_CODE

    def addFlatImageTable(self,dtoNT_flat_image):
        exit_code = self._addTable(self.dtoNT_flat_image,dtoNT_flat_image,"image_list")
        return exit_code

    def addLabelTable(self,dtoNT_label):
        exit_code = self._addTable(self.dtoNT_label,dtoNT_label,"label_list")
        return exit_code

    def addValueTable(self,dtoNT_value):
        exit_code = self._addTable(self.dtoNT_value,dtoNT_value,"value_list")
        return exit_code

class DtoHyperParameterForTFCNN():


    def __init__(self):
    
        NUM_OF_IN_CH_1      = 1

        self.keep_rate           = 1.0
        self.num_of_conv_layer   = 0
        self.num_of_in_ch        = NUM_OF_IN_CH_1
        self.num_of_out_ch       = []
        self.filter_wigth        = []   
        self.filter_height       = [] 
        self.stride_conv         = [] 
        self.stride_pool         = [] 
        self.shape_pool          = [] 
        self.num_of_hidden_layer = 0
        self.learning_rate_min   = 0
        self.learning_rate_max   = 0
        self.learning_iteration  = 0
        self.batch_size          = 0
        self.flag_pool           = FLAG_MAX_POOL
        
        self.step_log_output     = 5
        self.step_para_output    = 10
        
    def setKeepRate(self,keep_rate):
        self.keep_rate = keep_rate
        return kstd.NORMAL_CODE

    def setNumOfInCh(self,num_of_in_ch):
        self.num_of_in_ch = num_of_in_ch
        return kstd.NORMAL_CODE

    def setNumOfHiddenLayer(self,num_of_hidden_layer):
        self.num_of_hidden_layer = num_of_hidden_layer
        return kstd.NORMAL_CODE

    def setNumOfConvLayer(self,num_of_conv_layer):
        self.num_of_conv_layer = num_of_conv_layer
        return kstd.NORMAL_CODE

    def setLearningRate(self,var_min,var_max):
        self.learning_rate_min = var_min
        self.learning_rate_max = var_max
        if var_min > var_max:
            return kstd.ERROR_CODE
        return kstd.NORMAL_CODE

    def setLearningIteration(self,learning_iteration):
        self.learning_iteration = learning_iteration
        return kstd.NORMAL_CODE

    def setBatchSize(self,batch_size):
        self.batch_size = batch_size
        return kstd.NORMAL_CODE

    def setFlagPool(self,flag):
        if flag == FLAG_MAX_POOL:
            self.flag_pool = flag
            return kstd.NORMAL_CODE

        elif flag == FLAG_AVG_POOL:
            self.flag_pool = flag
            return kstd.NORMAL_CODE
        else:
            self.flag_pool = FLAG_MAX_POOL
            return kstd.ERROR_CODE

    def setStepLogOutput(self,step):
        self.step_log_output = step

    def setStepParaOutput(self,step):
        self.step_para_output = step

    def addFilterWigth(self,var):
        self.filter_wigth.append(var)
        return kstd.NORMAL_CODE

    def addFilterHeight(self,var):
        self.filter_height.append(var)
        return kstd.NORMAL_CODE
   
    def addNumOfOutCh(self,var):
        self.num_of_out_ch.append(var)
        return kstd.NORMAL_CODE

    def addStrideConv(self,var_list):
        self.stride_conv.append(var_list)
        return kstd.NORMAL_CODE

    def addStridePool(self,var_list):
        self.stride_pool.append(var_list)
        return kstd.NORMAL_CODE

    def addShapePool(self,var_list):
        self.shape_pool.append(var_list)
        return kstd.NORMAL_CODE

    def varCheck(self):
        size_filter_wigth  = len(self.filter_wigth)
        size_filter_height = len(self.filter_height)
        size_out_ch        = len(self.num_of_out_ch)
        size_stride_conv   = len(self.stride_conv)
        size_stride_pool   = len(self.stride_pool)
        size_shape_pool    = len(self.shape_pool)
    
        if size_filter_height != size_filter_wigth:
            return kstd.ERROR_CODE
        elif size_filter_height != size_out_ch:
            return kstd.ERROR_CODE
        elif size_filter_height != size_stride_conv:
            return kstd.ERROR_CODE
        elif size_filter_height != size_stride_pool:
            return kstd.ERROR_CODE
        elif size_filter_height != size_shape_pool:
            return kstd.ERROR_CODE
        elif not self.num_of_conv_layer > 0:
            return kstd.ERROR_CODE
        elif not self.num_of_hidden_layer > 0:
            return kstd.ERROR_CODE
        elif not self.learning_rate_max > 0:
            return kstd.ERROR_CODE
        elif not self.learning_rate_min > 0:
            return kstd.ERROR_CODE
        elif not self.learning_iteration > 0:
            return kstd.ERROR_CODE
        elif not self.batch_size > 0:
            return kstd.ERROR_CODE
        elif not self.step_log_output > 0:
            return kstd.ERROR_CODE
        elif not self.step_para_output > 0:
            return kstd.ERROR_CODE

        return kstd.NORMAL_CODE

############################################################################
#
# messages
#
############################################################################

def echoWrongList():
    print("wrong image list!!!!")

def echoXisNotSameSize(X):
    print( str(X) + " is not same size!!!!!!" )

def echoNotFirstlization():
    print("not firstlization height or wigth")


############################################################################
#
# standard functions
#
############################################################################
def createRandomIndexList(dtoNL_index,dtoNT_source,batch_size):
    index_0     = 0

    lists_length = dtoNT_source.getAttrRowLength()
    index_n      = lists_length - 1

    for bi in range(batch_size):
        index = rand.getVarInt(index_0,index_n)
        dtoNL_index.add(index)

    return kstd.NORMAL_CODE

def createBatchSample(dtoNL_index,dtoNT_source,dtoNT_batch):

    NT_source = dtoNT_source.getVariable()

    for index in dtoNL_index.getVariable():
        dtoNT_batch.addNpArray(NT_source[index])

    return kstd.NORMAL_CODE

def resultSave(result_lists,file_path):
    csv_writer = kstd.CsvWriter()
    csv_writer.openFile(file_path)
    csv_writer.writeOfArray2d(result_lists)
    csv_writer.closeFile()

def convValueToLabel(value_nplist):
    if(value_nplist.ndim == 1):
        max_value = 0
        for v in value_nplist:
            if(max_value > v):
                max_value = v

        label_nplist = np.empty(len(value_nplist))

        for vi in len(value_nplist):
            if( value_nplist[vi] == max_value ):
                label_nplist[vi] =  1
            else:
                label_nplist[vi] =  0
       

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

def matchRateOfLabel(dto_data_set_predict,dto_data_set_answer):

    row_length = dto_data_set_predict.dtoNT_label.getAttrRowLength()
    count_match = 0.0

    NT_label_predicted = dto_data_set_predict.dtoNT_label.getVariable()
    NT_label_answer    = dto_data_set_answer.dtoNT_label.getVariable()

    for ri in range(row_length):
        NL_label_predicted = NT_label_predicted[ri]
        NL_label_answer    = NT_label_answer[ri]

        res_compare = kstd.compareNpList(NL_label_predicted,NL_label_answer)
        if res_compare:
            count_match = count_match + 1.0

    return count_match / size

def matchRateOfDtoNpList(dto_np_list1,dto_np_list2):
    np_list1 = dto_np_list1.getVariable()
    np_list2 = dto_np_list2.getVariable()

    return matchRateOfNpList(np_list1,np_list2)



############################################################################
#
# part functions for cnn
#
############################################################################
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, stride, padding='SAME')

def max_pool_2x2(x,shape,stride):
    return tf.nn.max_pool(x, shape,stride, padding='SAME')

def avg_pool_2x2(x,shape,stride):
    return tf.nn.avg_pool(x, shape,stride, padding='SAME')

def pool_2x2(x,shape,stride,flag=FLAG_MAX_POOL):
    if flag == FLAG_MAX_POOL:
        return max_pool_2x2(x,shape,stride)
    elif flag == FLAG_AVG_POOL:
        return avg_pool_2x2(x,shape,stride)

def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def cnnLearningExecuter(dto_data_set_train,dto_hyper_param,dto_output_path):
    mode         = MODE_LEARNING
    dto_data_set = dto_data_set_train
    cnnExecuter(mode,dto_data_set,dto_hyper_param,dto_output_path)

def cnnReLearningExecuter(dto_data_set_train,dto_hyper_param,dto_output_path):
    mode         = MODE_RE_LEARNING
    dto_data_set = dto_data_set_train
    cnnExecuter(mode,dto_data_set,dto_hyper_param,dto_output_path)

def cnnPredictionExecuter(dto_data_set_predict,dto_hyper_param,dto_output_path):
    mode         = MODE_PREDICTION
    dto_data_set = dto_data_set_predict
    cnnExecuter(mode,dto_data_set,dto_hyper_param,dto_output_path)

def bondActivationFunc(x):
    return tf.nn.relu(x)
    #return tf.nn.sigmoid(x)

def convActivationFunc(x):
    return tf.nn.relu(x)
    #return tf.nn.sigmoid(x)


def initialSettingXY(dto_data_set):
    x_size = dto_data_set.image_list_size
    x      = tf.placeholder(tf.float32, shape=[ None , x_size ])
    y_size = dto_data_set.num_of_label_kind
    y_     = tf.placeholder(tf.float32, shape=[ None , y_size ])
    
    return x , y_

def initialSettingXImage(dto_data_set,x):
    image_wigth  = dto_data_set.wigth
    image_height = dto_data_set.height
    x_image      = tf.reshape(x, [-1, image_wigth, image_height, 1])

    return x_image

def initialSettingConvPara(dto_hyper_param):
    x_first_value = ""
    W_conv = [x_first_value] * dto_hyper_param.num_of_conv_layer
    b_conv = [x_first_value] * dto_hyper_param.num_of_conv_layer
    h_conv = [x_first_value] * dto_hyper_param.num_of_conv_layer
    h_pool = [x_first_value] * dto_hyper_param.num_of_conv_layer

    return W_conv,b_conv,h_conv,h_pool

def initialSettingFilter(dto_hyper_param,layer_i):
    filter_wigth  = dto_hyper_param.filter_wigth[layer_i]
    filter_height = dto_hyper_param.filter_height[layer_i]
    return filter_wigth,filter_height

def initialSettingConv(dto_hyper_param,layer_i):
    num_of_out_ch = dto_hyper_param.num_of_out_ch[layer_i]
    stride_conv   = dto_hyper_param.stride_conv[layer_i]
    return num_of_out_ch,stride_conv

def initialSettingPool(dto_hyper_param,layer_i):
    stride_pool   = dto_hyper_param.stride_pool[layer_i]
    shape_pool    = dto_hyper_param.shape_pool[layer_i]
    return stride_pool,shape_pool

def initial_setting_batch_dto(dto_data_set):
    x_size = dto_data_set.image_list_size
    dtoNT_batch_x = kstd.DtoNpTable(x_size)
    y_size = dto_data_set.num_of_label_kind
    dtoNT_batch_y = kstd.DtoNpTable(y_size)

def createBatchSampleList(dtoNL_index,dtoNT_source):
    col_length  = dtoNT_source.getAttrColLength()
    dtoNT_batch = kstd.DtoNpTable(col_length)
    createBatchSample(dtoNL_index,dtoNT_source,dtoNT_batch)
    return dtoNT_batch.getVariable()

def histogram_to_TensorBoard(name,tf_gragh):
    tf.summary.histogram(name, tf_gragh)

def outputConvLayerParaToTensorBoard(W_conv,b_conv,h_conv,h_pool,li):

    tf.summary.histogram('No%02d01_W_conv' % (li), W_conv[li])
    tf.summary.histogram('No%02d02_b_conv' % (li), b_conv[li])
    tf.summary.histogram('No%02d03_h_conv' % (li), h_conv[li])
    tf.summary.histogram('No%02d04_h_pool' % (li), h_pool[li])

def outputBondLayerParaToTensorBoard(W_bond,b_bond,h_bond,W_bond2,b_bond2):

    tf.summary.histogram('No0101_W_bond', W_bond)
    tf.summary.histogram('No0102_b_bond', b_bond)
    tf.summary.histogram('No0103_h_bond', h_bond)
    tf.summary.histogram('No0201_W_bond', W_bond2)
    tf.summary.histogram('No0202_b_bond', b_bond2)


def createUpdatedLearningRate(dto_hyper_param,lr_update_count):
    lr_update_weight     = tf.subtract( 1.0 , tf.cast( tf.divide(lr_update_count,dto_hyper_param.learning_iteration), tf.float32 ) )
    lr_update_base       = tf.subtract( dto_hyper_param.learning_rate_max , dto_hyper_param.learning_rate_min )
    
    d_lr                 = tf.multiply( lr_update_base , lr_update_weight ) 
    learning_rate_new    = tf.add(dto_hyper_param.learning_rate_min , d_lr)
    return learning_rate_new



def cnnExecuter(mode,dto_data_set,dto_hyper_param,dto_output_path):



    #####################################################
    # variables
    #####################################################
    
    x , y_  = initialSettingXY(dto_data_set)
    x_image = initialSettingXImage(dto_data_set,x)

    #####################################################
    # convolution layer
    #####################################################
    kstd.echoStart("convolution layer setting")

    W_conv,b_conv,h_conv,h_pool = initialSettingConvPara(dto_hyper_param)

    with tf.name_scope('convolution_layer') as scope:

        num_of_in_ch  = dto_hyper_param.num_of_in_ch

        for li in range(dto_hyper_param.num_of_conv_layer):
            process_name = "No." + str(li) + " layer convolution"
            kstd.echoStart(process_name)

            filter_wigth  ,filter_height = initialSettingFilter(dto_hyper_param,li)
            num_of_out_ch ,stride_conv   = initialSettingConv(dto_hyper_param,li)
            stride_pool   ,shape_pool    = initialSettingPool(dto_hyper_param,li)

            W_conv[li] = weightVariable([filter_wigth,filter_height, num_of_in_ch, num_of_out_ch])
            b_conv[li] = biasVariable([num_of_out_ch])
            h_conv[li] = convActivationFunc( conv2d(x_image, W_conv[li], stride_conv) + b_conv[li] )
            h_pool[li] = pool_2x2(h_conv[li], shape_pool, stride_pool, dto_hyper_param.flag_pool)

            x_image      = h_pool[li]
            num_of_in_ch = num_of_out_ch
        
            # output for tensorboard
            outputConvLayerParaToTensorBoard(W_conv,b_conv,h_conv,h_pool,li)

            kstd.echoIsAlready(process_name)

    #####################################################
    # bonding layer
    #####################################################
    
    with tf.name_scope('bonding_layer') as scope:

        process_name = "bonding layer setting"
        kstd.echoStart(process_name)

        num_wigth   = x_image.shape[AXIS_X_IMAGE_WIGTH]
        num_height  = x_image.shape[AXIS_X_IMAGE_HEIGHT]
        num_channel = x_image.shape[AXIS_X_IMAGE_CHANNEL]

        total_image_pixels = int(num_wigth * num_height * num_channel)

        num_of_hidden_layer = dto_hyper_param.num_of_hidden_layer

        W_bond       = weightVariable([total_image_pixels, num_of_hidden_layer])
        b_bond       = biasVariable([num_of_hidden_layer])
        
        x_image_flat = tf.reshape(x_image, [-1, total_image_pixels])
        h_bond       = bondActivationFunc(tf.matmul(x_image_flat, W_bond) + b_bond)
        
        keep_prob    = tf.placeholder(tf.float32)
        h_bond_drop  = tf.nn.dropout(h_bond, keep_prob)

        W_bond2 = weightVariable([num_of_hidden_layer,dto_data_set.num_of_label_kind])
        b_bond2 = biasVariable([dto_data_set.num_of_label_kind])

        y_cnn = tf.matmul(h_bond_drop, W_bond2) + b_bond2

        # output for tensorboard

        outputBondLayerParaToTensorBoard(W_bond,b_bond,h_bond,W_bond2,b_bond2)

        kstd.echoIsAlready(process_name)


    #***************************************************
    # defining : learing rate 
    #***************************************************
    with tf.name_scope('learning_rate_updater') as scope:
        lr_update_count      = tf.placeholder(tf.int32)
        learning_rate        = tf.Variable(tf.constant(dto_hyper_param.learning_rate_max))

        learning_rate_new    = createUpdatedLearningRate(dto_hyper_param,lr_update_count)
        learning_rate_update = tf.assign(learning_rate , learning_rate_new )
        
        tf.summary.scalar("learning_rate" , learning_rate)


    #***************************************************
    # defining : functions
    #***************************************************
    with tf.name_scope('functions___loss_train_accuracy') as scope:
        kstd.echoBlanks(5)
        cross_entropy      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_cnn))
        #cross_entropy      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_cnn))
       
        train_step         = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_cnn, 1), tf.argmax(y_, 1))
        accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        kstd.echoBlanks(5)

        tf.summary.scalar("accuracy" , accuracy)

    #***************************************************
    # learning or prediction
    #***************************************************
    process_name = mode
    kstd.echoStart(process_name)
    kstd.echoBlanks(2)
    base_time = kstd.getTime()
    bef_time  = kstd.getTime()  
    with tf.Session() as sess:

        summary_merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(dto_output_path.summary_dir_path , sess.graph)

        #***********************************************
        #    variable setting
        #***********************************************

        #***********************************************
        # learning
        if( mode == MODE_LEARNING ):

            sess.run(tf.global_variables_initializer())

        #***********************************************
        # Prediction or re-learning
        elif( mode == MODE_PREDICTION or mode == MODE_RE_LEARNING ):

            saver = tf.train.Saver()
            saver.restore(sess, dto_output_path.learned_parameter_file_path)


        #***********************************************
        #    gragh processing
        #***********************************************

        #***********************************************
        # learning or re-learning
        if( mode == MODE_LEARNING or mode == MODE_RE_LEARNING ):

            iteration    = dto_hyper_param.learning_iteration
            dtoNT_source = dto_data_set.dtoNT_flat_image

            for ii in range(iteration):
                batch_size     = dto_hyper_param.batch_size
                dtoNL_index  = kstd.DtoNpList("int64")
                createRandomIndexList(dtoNL_index,dtoNT_source,batch_size)

                batch_x = createBatchSampleList(dtoNL_index,dto_data_set.dtoNT_flat_image)
                batch_y = createBatchSampleList(dtoNL_index,dto_data_set.dtoNT_label)

                train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: dto_hyper_param.keep_rate})
                sess.run(learning_rate_update , feed_dict={lr_update_count:ii} )

                if (ii + 1 ) % dto_hyper_param.step_log_output == 0:
                    elapsed_time_1 = kstd.getElapsedTime(bef_time,"s")
                    elapsed_time_n = kstd.getElapsedTime(base_time,"m")
                    bef_time = kstd.getTime()  

                    train_accuracy = accuracy.eval(feed_dict={ x: batch_x, y_: batch_y, keep_prob: 1.0})
                    train_entropy  = cross_entropy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: dto_hyper_param.keep_rate})

                    print('step %4d/%d,\taccuracy %0.2g,\tentropy %0.2g \t(%ds/%dm) '
                           % (ii + 1, iteration ,train_accuracy,train_entropy,elapsed_time_1,elapsed_time_n))

                    message = ('step,%4d/%d,\taccuracy,%0.2g,\tentropy,%0.2g,(%ds/%dm)'
                           % (ii + 1, iteration ,train_accuracy,train_entropy,elapsed_time_1,elapsed_time_n))
                    kstd.writeAddCsvDataVal(dto_output_path.learning_log_file_path,message)

                    summary = sess.run(summary_merged , feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, lr_update_count:ii} )
                    summary_writer.add_summary(summary , ii)

                    if (ii + 1 ) % dto_hyper_param.step_para_output == 0:
                        saver = tf.train.Saver()
                        saver.save(sess, dto_output_path.learned_parameter_file_path)


            saver = tf.train.Saver()
            saver.save(sess, dto_output_path.learned_parameter_file_path)

        #***********************************************
        # Prediction
        elif( mode == MODE_PREDICTION ):
            saver = tf.train.Saver()
            saver.restore(sess, dto_output_path.learned_parameter_file_path)

            x_target    = dto_data_set.dtoNT_flat_image.getVariable()
            y_predicted = y_cnn.eval(feed_dict={x: x_target, keep_prob: 1.0} )
            
            dtoNT = kstd.DtoNpTable(dto_data_set.num_of_label_kind)
            dtoNT.addNpArray(y_predicted)
            dto_data_set.addValueTable(dtoNT)
            resultSave(y_predicted,dto_output_path.predicted_value_file_path)

        sess.close()

        kstd.echoBlanks(2)
    kstd.echoIsAlready(process_name)


