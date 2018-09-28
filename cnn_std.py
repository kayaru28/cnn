#coding: UTF-8

import kayaru_standard_process as kstd
import kayaru_standard_process_for_image as image
import kayaru_standard_process_for_randomize as rand
import kayaru_standard_process_for_error_handling as error_handler
import kayaru_standard_messages as kstd_m
import numpy as np
import os

import tensorflow as tf

AXIS_X_IMAGE_NUM     = 0
AXIS_X_IMAGE_WIGTH   = 1
AXIS_X_IMAGE_HEIGHT  = 2
AXIS_X_IMAGE_CHANNEL = 3
        
MODE_LEARNING   = "Learning"
MODE_PREDICTION = "Prediction"

class DtoCaseMetaForTFCNN():
    def __init__(self):
        self.learned_parameter_file_path = os.path.join( kstd.getScriptDir(),"_param.ckpt")
        self.predicted_value_file_path   = os.path.join( kstd.getScriptDir(),"_label.csv")
        self.summary_dir_path            = kstd.getScriptDir()

    def setLearnedParameterFilePath(self,file_path):
        self.learned_parameter_file_path = file_path
        
    def setPredictedValueFilePath(self,file_path):
        self.predicted_value_file_path   = file_path

    def setSummaryDirPath(self,dir_path):
        self.summary_dir_path            = dir_path

class DtoDataSetForTFCNN():
    def __init__(self):
        self.height               = 0
        self.wigth                = 0
        self.image_list_size      = 0
        self.num_of_label_kind    = 0

        self.ERROR_CODE_BY_WIGTH      = 101
        self.ERROR_CODE_BY_HEIGHT     = 102
        self.ERROR_CODE_BY_IMAGE_LIST = 103
        self.ERROR_CODE_BY_LABEL_LIST = 104
        self.ERROR_CODE_BY_LABEL_KIND = 105
        self.ERROR_CODE_BY_TEST_LIST  = 106

    def firstlizationImage(self,wigth,height):
        self.height               = height
        self.wigth                = wigth
        self.image_list_size      = self.wigth * self.height
        self.flat_image_nplists   = np.empty((0,self.image_list_size))
        self.t_flat_image_nplists = np.empty((0,self.image_list_size))

    def firstlizationLabel(self,num_of_label_kind):
        self.num_of_label_kind = num_of_label_kind
        self.label_nplists     = np.empty((0,self.num_of_label_kind))
        self.t_label_nplists   = np.empty((0,self.num_of_label_kind))
        self.t_value_nplists   = np.empty((0,self.num_of_label_kind))

    def clearFlatImageList(self):
        self.flat_image_nplist = np.empty((0,self.image_list_size))

    def _varCheckAddFlatImageList(self,x_image_nplist):
        if not image.checkImageSize(self.height,self.wigth,x_image_nplist):
            echoXisNotSameSize("image_list")
            if( self.image_list_size == 0 ):
                echoNotFirstlization()
            else:
                kstd.echoAisB("height",self.height)
                kstd.echoAisB("wigth",self.wigth)
                kstd.echoAisB("nplist size",x_image_nplist.shape[1])
            return kstd.ERROR_CODE

        return kstd.NORMAL_CODE

    def addFlatImageList(self,flat_image_nplist):
        exit_code = self._varCheckAddFlatImageList(flat_image_nplist)
        if exit_code==kstd.ERROR_CODE:
            return exit_code
        self.flat_image_nplists = np.insert(self.flat_image_nplists,0,flat_image_nplist,axis = 0)
        return kstd.NORMAL_CODE

    def addTestFlatImageList(self,flat_image_nplist):
        exit_code = self._varCheckAddFlatImageList(flat_image_nplist)
        if (exit_code==kstd.ERROR_CODE) :
            return exit_code
        self.t_flat_image_nplists = np.insert(self.t_flat_image_nplists,0,flat_image_nplist,axis = 0)
        return kstd.NORMAL_CODE

    def addLabelList(self,label_nplist):
        if not self.checkLabelSize(self.num_of_label_kind,label_nplist):
            return kstd.ERROR_CODE

        self.label_nplists = np.insert(self.label_nplists,0,label_nplist,axis = 0)
        return kstd.NORMAL_CODE

    def addTestLabelList(self,label_nplist):
        if not self.checkLabelSize(self.num_of_label_kind,label_nplist):
            return kstd.ERROR_CODE

        self.t_label_nplists = np.insert(self.t_label_nplists,0,label_nplist,axis = 0)
        return kstd.NORMAL_CODE

    def addTestValueList(self,value_nplist):
        if not self.checkLabelSize(self.num_of_label_kind,value_nplist):
            return kstd.ERROR_CODE

        self.t_value_nplists = np.insert(self.t_value_nplists,0,value_nplist,axis = 0)
        return kstd.NORMAL_CODE

    def checkLabelSize(self,num_of_label_kind,label_list):
        if kstd.isList(label_list):
            lengeth = len(label_list)
        else:
            lengeth = label_list.shape[1]

        if num_of_label_kind == lengeth:
            return True
        else:
            return False

    def varCheck(self):
        kstd.echoBlank()

        if not self.height > 0:
            return self.ERROR_CODE_BY_HEIGHT
        elif not self.wigth > 0:
            return self.ERROR_CODE_BY_WIGTH
        elif not self.num_of_label_kind > 0:
            return self.ERROR_CODE_BY_LABEL_KIND
        elif kstd.compareNpListSize(self.label_nplists,self.flat_image_nplists,1):
            return self.ERROR_CODE_BY_IMAGE_LIST
        #elif kstd.compareNpListSize(self.t_label_nplists,self.t_flat_image_nplists,1):
        #    return self.ERROR_CODE_BY_TEST_LIST

        return kstd.NORMAL_CODE

    def getBatchSample(self,sample_nplists,batch_size):

        self.lists_length = sample_nplists.shape[0]
        self.list_size    = sample_nplists.shape[1]

        self.ans_nplists = np.empty((0,self.list_size))

        self.index_0     = 0
        self.index_n     = self.lists_length - 1

        for bi in range(batch_size):
            self.index       = rand.getVarInt(self.index_0,self.index_n)
            self.ans_nplists = np.insert(self.ans_nplists,0,sample_nplists[self.index],axis = 0)

        return self.ans_nplists
        
class DtoHyperParameterForTFCNN():
    def __init__(self):
        self.NUM_OF_IN_CH_1      = 1
        self.keep_rate           = 1.0
        self.num_of_conv_layer   = 0
        self.num_of_in_ch        = self.NUM_OF_IN_CH_1
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
        
        self.ERROR_CODE_BY_F_W           = 101
        self.ERROR_CODE_BY_F_H           = 102
        self.ERROR_CODE_BY_OUT_CH        = 103
        self.ERROR_CODE_BY_STRIDE_C      = 104
        self.ERROR_CODE_BY_STRIDE_P      = 105
        self.ERROR_CODE_BY_SHAPE_P       = 106
        self.ERROR_CODE_BY_NUM_C_LAYER   = 107
        self.ERROR_CODE_BY_NUM_H_LAYER   = 108
        self.ERROR_CODE_BY_LEARNING_RATE = 109
        self.ERROR_CODE_BY_LEARNING_ITER = 110
        self.ERROR_CODE_BY_BATCH_SIZE    = 111


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

    def addFilterWigth(self,var):
        error_handler.assertionCheckIsInt(var,"var for filter wigth")
        self.filter_wigth.append(var)
        return kstd.NORMAL_CODE

    def addFilterHeight(self,var):
        error_handler.assertionCheckIsInt(var,"var for filter height")
        self.filter_height.append(var)
        return kstd.NORMAL_CODE
   
    def addNumOfOutCh(self,var):
        error_handler.assertionCheckIsInt(var,"var for channel out number")
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
        self.filter_wigth_size  = len(self.filter_wigth)
        self.filter_height_size = len(self.filter_height)
        self.out_ch_size        = len(self.num_of_out_ch)
        self.stride_conv_size   = len(self.stride_conv)
        self.stride_pool_size   = len(self.stride_pool)
        self.shape_pool_size   = len(self.shape_pool)
    
        if self.filter_height_size   != self.filter_wigth_size:
            return self.ERROR_CODE_BY_WIGTH
        elif self.filter_height_size != self.out_ch_size:
            return self.ERROR_CODE_BY_OUT_CH
        elif self.filter_height_size != self.stride_conv_size:
            return self.ERROR_CODE_BY_STRIDE_C
        elif self.filter_height_size != self.stride_pool_size:
            return self.ERROR_CODE_BY_STRIDE_P
        elif self.filter_height_size != self.shape_pool_size:
            return self.ERROR_CODE_BY_SHAPE_P
        elif not self.num_of_conv_layer > 0:
            return self.ERROR_CODE_BY_NUM_C_LAYER
        elif not self.num_of_hidden_layer > 0:
            return self.ERROR_CODE_BY_NUM_H_LAYER
        elif not self.learning_rate_max > 0:
            return self.ERROR_CODE_BY_LEARNING_RATE
        elif not self.learning_rate_min > 0:
            return self.ERROR_CODE_BY_LEARNING_RATE
        elif not self.learning_iteration > 0:
            return self.ERROR_CODE_BY_LEARNING_ITER
        elif not self.batch_size > 0:
            return self.ERROR_CODE_BY_BATCH_SIZE

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

############################################################################
#
# part functions for cnn
#
############################################################################
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, stride, padding='SAME')

def max_pool_2x2(x,shape,stride):
    return tf.nn.max_pool(x, shape,stride, padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def cnnLearningExecuter(dto_data_set,dto_hyper_param,dto_case_meta):
    mode = MODE_LEARNING
    cnnExecuter(mode,dto_data_set,dto_hyper_param,dto_case_meta)

def cnnPredictionExecuter(dto_data_set,dto_hyper_param,dto_case_meta):
    mode = MODE_PREDICTION
    cnnExecuter(mode,dto_data_set,dto_hyper_param,dto_case_meta)


def cnnExecuter(mode,dto_data_set,dto_hyper_param,dto_case_meta):

    #####################################################
    # variables
    #####################################################
    
    x_size = dto_data_set.image_list_size
    x      = tf.placeholder(tf.float32, shape=[ None , x_size ])
    y_size = dto_data_set.num_of_label_kind
    y_     = tf.placeholder(tf.float32, shape=[ None , y_size ])
    
    image_wigth  = dto_data_set.wigth
    image_height = dto_data_set.height
    x_image      = tf.reshape(x, [-1, image_wigth, image_height, 1])

    # W = tf.Variable(tf.zeros([num_of_image_pixels, num_of_answer_kind]))
    # b = tf.Variable(tf.zeros([NUM_OF_ANSWER_KIND]))

    #####################################################
    # convolution layer
    #####################################################
    kstd.echoStart("convolution layer setting")

    x_first_value = ""
    W_conv = [x_first_value] * dto_hyper_param.num_of_conv_layer
    b_conv = [x_first_value] * dto_hyper_param.num_of_conv_layer
    h_conv = [x_first_value] * dto_hyper_param.num_of_conv_layer
    h_pool = [x_first_value] * dto_hyper_param.num_of_conv_layer
    
    with tf.name_scope('convolution_layer') as scope:
        for li in range(dto_hyper_param.num_of_conv_layer):
            process_name = "No." + str(li) + " layer convolution"
            kstd.echoStart(process_name)

            filter_wigth  = dto_hyper_param.filter_wigth[li]
            filter_height = dto_hyper_param.filter_height[li]
            num_of_in_ch  = dto_hyper_param.num_of_in_ch
            num_of_out_ch = dto_hyper_param.num_of_out_ch[li]
            stride_conv   = dto_hyper_param.stride_conv[li]
            stride_pool   = dto_hyper_param.stride_pool[li]
            shape_pool    = dto_hyper_param.shape_pool[li]

            W_conv[li] = weight_variable([filter_wigth,filter_height, num_of_in_ch, num_of_out_ch])
            b_conv[li] = bias_variable([num_of_out_ch])
            h_conv[li] = tf.nn.relu(conv2d(x_image, W_conv[li], stride_conv) + b_conv[li])
            h_pool[li] = max_pool_2x2(h_conv[li], shape_pool, stride_pool)

            x_image = h_pool[li]

            dto_hyper_param.setNumOfInCh(num_of_out_ch)
        
            kstd.echoIsAlready(process_name)

            tf.summary.histogram('No%02d01_W_conv' % (li), W_conv[li])
            tf.summary.histogram('No%02d01_b_conv' % (li), b_conv[li])


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

        W_bond       = weight_variable([total_image_pixels, dto_hyper_param.num_of_hidden_layer])
        b_bond       = bias_variable([dto_hyper_param.num_of_hidden_layer])
        
        x_image_flat = tf.reshape(x_image, [-1, total_image_pixels])
        h_bond       = tf.nn.relu(tf.matmul(x_image_flat, W_bond) + b_bond)
        
        keep_prob    = tf.placeholder(tf.float32)
        h_bond_drop  = tf.nn.dropout(h_bond, keep_prob)

        W_bond2 = weight_variable([dto_hyper_param.num_of_hidden_layer,dto_data_set.num_of_label_kind])
        b_bond2 = bias_variable([dto_data_set.num_of_label_kind])

        y_cnn = tf.matmul(h_bond_drop, W_bond2) + b_bond2

        kstd.echoIsAlready(process_name)

        tf.summary.histogram('No%02d01_W_bond' % (11), W_bond)
        tf.summary.histogram('No%02d02_b_bond' % (11), b_bond)
        tf.summary.histogram('No%02d01_W_bond' % (12), W_bond2)
        tf.summary.histogram('No%02d02_b_bond' % (12), b_bond2)
    

    #***************************************************
    # defining : learing rate 
    #***************************************************
    with tf.name_scope('learning_rate_updater') as scope:
        lr_update_count      = tf.placeholder(tf.int32)
        lr_update_weight     = tf.subtract( 1.0 , tf.cast( tf.divide(lr_update_count,dto_hyper_param.learning_iteration), tf.float32 ) )
        lr_update_base       = tf.subtract( dto_hyper_param.learning_rate_max , dto_hyper_param.learning_rate_min )
        
        d_lr                 =  tf.multiply( lr_update_base , lr_update_weight ) 
        
        learning_rate        = tf.Variable(tf.constant(dto_hyper_param.learning_rate_max))
        learning_rate_new    = tf.add(dto_hyper_param.learning_rate_min , d_lr)
        learning_rate_update = tf.assign(learning_rate , learning_rate_new )
        
        tf.summary.scalar("learning_rate" , learning_rate)


    #***************************************************
    # defining : functions
    #***************************************************
    with tf.name_scope('functions___loss_train_accuracy') as scope:
        kstd.echoBlanks(5)
        #cross_entropy      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_cnn))
        cross_entropy      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_cnn))
       
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
        summary_writer = tf.summary.FileWriter(dto_case_meta.summary_dir_path , sess.graph)

        sess.run(tf.global_variables_initializer())

        # learning
        if( mode == MODE_LEARNING ):
            iteration = dto_hyper_param.learning_iteration
            for ii in range(iteration):
                batch_size     = dto_hyper_param.batch_size
                
                sample_nplists = dto_data_set.flat_image_nplists
                batch_x        = dto_data_set.getBatchSample(sample_nplists,batch_size)

                sample_nplists = dto_data_set.label_nplists
                batch_y        = dto_data_set.getBatchSample(sample_nplists,batch_size)

                train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: dto_hyper_param.keep_rate})

                train_accuracy = accuracy.eval(feed_dict={ x: batch_x, y_: batch_y, keep_prob: 1.0})
                train_entropy  = cross_entropy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: dto_hyper_param.keep_rate})

                sess.run(learning_rate_update , feed_dict={lr_update_count:ii} )

                elapsed_time_1 = kstd.getElapsedTime(bef_time,"s")
                elapsed_time_n = kstd.getElapsedTime(base_time,"m")
                bef_time = kstd.getTime()  

                print('step %4d/%d,\taccuracy %0.2g,\tentropy %0.2g \t(%ds/%dm) '
                       % (ii + 1, iteration ,train_accuracy,train_entropy,elapsed_time_1,elapsed_time_n))

                summary = sess.run(summary_merged , feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, lr_update_count:ii} )
                summary_writer.add_summary(summary , ii)

            kstd.echoBlanks(2)
            #y_predicted = y_cnn.eval(feed_dict={x: test_x, keep_prob: 1.0} )
            #resultSave(y_predicted,dto_case_meta.predicted_value_file_path)

            saver = tf.train.Saver()
            saver.save(sess, dto_case_meta.learned_parameter_file_path)

        # Prediction
        elif( mode == MODE_PREDICTION ):
            saver = tf.train.Saver()
            saver.restore(sess, dto_case_meta.learned_parameter_file_path)

            test_x = dto_data_set.t_flat_image_nplists
            y_predicted = y_cnn.eval(feed_dict={x: test_x, keep_prob: 1.0} )
            dto_data_set.addTestValueList(y_predicted)
            resultSave(y_predicted,dto_case_meta.predicted_value_file_path)

    kstd.echoBlanks(2)
    kstd.echoIsAlready(process_name)

if __name__ == "__main__":

    echoHyperParameterSetting()

    dto_hyper_param = DtoHyperParameterForTFCNN()
    
    dto_hyper_param.addFilterWigth(2)
    dto_hyper_param.addFilterHeight(2)
    dto_hyper_param.addNumOfOutCh(2)

    dto_hyper_param.addFilterWigth(3)
    dto_hyper_param.addFilterHeight(3)
    dto_hyper_param.addNumOfOutCh(3)

    dto_data_set = DtoDataSetForTFCNN()



    is_preparation_OK = dto_data_set.varCheck() * dto_hyper_param.varCheck()
    
    if(is_preparation_OK):
        print("cnn start")
        cnnExecuter(dto_data_set,dto_hyper_param)

    else:
        print("NG")



