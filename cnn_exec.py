#coding: UTF-8

import kayaru_standard_process as kstd
import kayaru_standard_process_for_image as image
import kayaru_standard_process_for_randomize as rand
import kayaru_standard_messages as kstd_m
import numpy as np
import properties_for_cnn as prop
import file_interfase_for_cnn as fi
import sys
import cnn_std as cnn

import tensorflow as tf

MODE_LEARNING    = "Learning"
MODE_RE_LEARNING = "re-Learning"
MODE_VALIDATION  = "Validation"
MODE_PREDICTION  = "Prediction"

#########################################################
# modeling
#########################################################

def __cnnModel(x_2d):

    num_out_channels = 1


    #####################################################
    # convolution layer
    #####################################################
    process="convolution layer"

    conv1 = cnn.Conv2dSame()
    parameterSettingConv1(conv1,process,num_out_channels)

    W_c1 = conv1.createW0()
    x_2d = conv1.calculate(x_2d,W_c1)

    kstd.echoIsAlready(process)

    num_out_channels = prop.conv1_out_channels

    #####################################################
    # max pool layer
    #####################################################
    process="max pool layer"

    pool1 = cnn.MaxPoolSame()
    parameterSettingPool1(pool1,process)

    x_2d = pool1.calculate(x_2d)

    kstd.echoIsAlready(process)

    #####################################################
    # full connected layer preparation
    #####################################################

    fc_input_size = cnn.getX2dFeatureVolume(x_2d)
    y_cnn         = tf.reshape(x_2d, [-1, fc_input_size ])
    y_acc         = y_cnn
 
    #####################################################
    # full connected layer
    #####################################################
    process = "full connected layer"
    kstd.echoStart(process)

    fc1 = cnn.FullConnected()
    fc1.setInputSize(fc_input_size)
    fc1.setOutputSize(prop.fc1_out)
    fc_input_size = prop.fc1_out

    if fc1.isSetParameters():
        W_f1     = fc1.createW0()
        y_cnn    = fc1.calculate(y_cnn,W_f1,prop.fc1_keep_prop)
        y_acc    = fc1.calculate(y_acc,W_f1,prop.KEEP_PROP_ALL)
    else:
        kstd.echoErrorOccured(process)
        kstd.exit()
    kstd.echoFinish(process)
    
    return y_cnn,y_acc

def judgeParameterSettingError(model,process):
    if not model.isSetParameters():
        kstd.echoErrorOccured(process)
        kstd.exit()

def parameterSettingConv1(conv1,process,num_preunit_out_channels):
    conv1.setFilterW(prop.conv1_filter_w)
    conv1.setFilterH(prop.conv1_filter_h)
    conv1.setOutChannels(prop.conv1_out_channels)
    conv1.setFilterInChannels(num_preunit_out_channels)
    conv1.setStrideW(prop.conv1_stride_w)
    conv1.setStrideH(prop.conv1_stride_h)

    judgeParameterSettingError(conv1,process)

def parameterSettingPool1(pool1,process):
    pool1.setFilterW(prop.pool1_stride_w)
    pool1.setFilterH(prop.pool1_stride_h)
    pool1.setStrideW(prop.pool1_stride_w)
    pool1.setStrideH(prop.pool1_stride_h)
    judgeParameterSettingError(pool1,process)




#########################################################
# trainer
#########################################################
def getProgressMessage(ii,train_accuracy,train_entropy,timer):
    return ('step %4d/%d\taccuracy\t%0.2g\tentropy\t%0.2g\t(%ds/%dm)'
           % (ii + 1, prop.LEARNING_ITERATION,train_accuracy,train_entropy,timer.getLap("s"),timer.getElapsed("m")))

def __cnnTrainingUnit(y_ans,y_cnn,y_acc):
    cross_entropy      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_ans, logits=y_cnn))
    train_step         = tf.train.AdamOptimizer(prop.LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_acc, 1), tf.argmax(y_ans, 1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step,cross_entropy,accuracy

def __cnnLearningUnit(dto_data_set,outputer,x_1d,y_ans,train_step,cross_entropy,accuracy):

    process = "learning unit"
    kstd.echoBlanks(5)
    kstd.echoStart(process)

    timer   = kstd.timeCalculater()
    dto_data_set.fixDataSet()
    for ii in range(prop.LEARNING_ITERATION):

        batch_x,batch_y = dto_data_set.getBatchSet(prop.BATCH_SIZE)
        train_step.run(feed_dict={x_1d: batch_x, y_ans: batch_y})

        if (ii + 1 ) % prop.CYCLE_LOG_OUTPUT == 0:
            train_accuracy = accuracy.eval(feed_dict={ x_1d: batch_x, y_ans: batch_y})
            train_entropy  = cross_entropy.eval(feed_dict={x_1d: batch_x, y_ans: batch_y})
            outputer.logOutput(getProgressMessage(ii,train_accuracy,train_entropy,timer))

            timer.lap()

            #if (ii + 1 ) % prop.CYCLE_PARA_OUTPUT == 0:
            #    saver.save(sess, outputer.learned_parameter_file_path)

    kstd.echoFinish(process)

def cnnLearning(dto_data_set,outputer,x_1d,y_ans):

    process = "cnnLearning"
    kstd.echoStart(process)
    kstd.echoBlanks(2)

    x_2d   = tf.reshape(x_1d, [-1, dto_data_set.wigth, dto_data_set.height, 1])

    #####################################################
    # model learning
    #####################################################
    y_cnn,y_acc                       = __cnnModel(x_2d)    
    train_step,cross_entropy,accuracy = __cnnTrainingUnit(y_ans,y_cnn,y_acc)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        __cnnLearningUnit(dto_data_set,outputer,x_1d,y_ans,train_step,cross_entropy,accuracy)

        saver = tf.train.Saver()
        saver.save(sess, outputer.learned_parameter_file_path)

        sess.close()

def cnnReLearning(dto_data_set,outputer,x_1d,y_ans):

    process = "cnnReLearning"
    kstd.echoStart(process)
    kstd.echoBlanks(2)

    x_2d   = tf.reshape(x_1d, [-1, dto_data_set.wigth, dto_data_set.height, 1])

    #####################################################
    # model learning
    #####################################################
    y_cnn,y_acc                       = __cnnModel(x_2d)    
    train_step,cross_entropy,accuracy = __cnnTrainingUnit(y_ans,y_cnn,y_acc)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, outputer.learned_parameter_file_path)

        __cnnLearningUnit(dto_data_set,outputer,x_1d,y_ans,train_step,cross_entropy,accuracy)

        saver.save(sess, outputer.learned_parameter_file_path)

        sess.close()


def cnnValidation(dto_data_set,outputer,x_1d,y_ans):

    process = "cnnValidation"
    kstd.echoStart(process)
    kstd.echoBlanks(2)

    x_2d   = tf.reshape(x_1d, [-1, dto_data_set.wigth, dto_data_set.height, 1])

    #####################################################
    # model learning
    #####################################################
    y_cnn,y_acc                       = __cnnModel(x_2d)    
    train_step,cross_entropy,accuracy = __cnnTrainingUnit(y_ans,y_cnn,y_acc)

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, outputer.learned_parameter_file_path)

        valid_x = dto_data_set.dtoNT_flat_img.getVariable()
        valid_y = dto_data_set.dtoNT_label.getVariable()

        train_accuracy = accuracy.eval(feed_dict={ x_1d: valid_x, y_ans: valid_y})
        print("validation : %0.2f" % train_accuracy)

        sess.close()

    outputer.validationOutput(train_accuracy)


def cnnPrediction(dto_data_set,outputer,x_1d,y_ans):

    process = "cnnPrediction"
    kstd.echoStart(process)
    kstd.echoBlanks(2)

    x_2d   = tf.reshape(x_1d, [-1, dto_data_set.wigth, dto_data_set.height, 1])

    #####################################################
    # model learning
    #####################################################
    y_cnn,y_acc = __cnnModel(x_2d)    

    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, outputer.learned_parameter_file_path)

        x_target    = dto_data_set.dtoNT_flat_img.getVariable()
        y_predicted = y_cnn.eval(feed_dict={x_1d: x_target} )
        
        dtoNT = kstd.DtoNpTable(dto_data_set.num_of_label_kind)
        dtoNT.addNpArray(y_predicted)
        dto_data_set.addValueTable(dtoNT)

        cnn.crateNLLabelFromValue(dto_data_set)
        outputer.PredictionResultSave(dto_data_set)

        sess.close()



def cnnExecuter(mode,dto_data_set,outputer):

    #####################################################
    # variables
    #####################################################
    
    kstd.echoStart(mode)
    kstd.echoBlanks(2)

    x_size = dto_data_set.pixel_size
    x_1d   = tf.placeholder(tf.float32, shape=[ None , x_size ])

    y_size = dto_data_set.num_of_label_kind
    y_ans  = tf.placeholder(tf.float32, shape=[ None , y_size ])

    if mode == MODE_LEARNING:
        cnnLearning(dto_data_set,outputer,x_1d,y_ans)
    elif mode == MODE_RE_LEARNING:
        cnnReLearning(dto_data_set,outputer,x_1d,y_ans)
    if mode == MODE_VALIDATION:
        cnnValidation(dto_data_set,outputer,x_1d,y_ans)
    if mode == MODE_PREDICTION:
        cnnPrediction(dto_data_set,outputer,x_1d,y_ans)

    kstd.echoBlanks(2)
    kstd.echoIsAlready(mode)

if __name__ == "__main__":

    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数

    if not argc > 0:
        kstd.echoBlanks(2)
        kstd.echoBar()
        print("input  1:Learing 2:re-Learning 3:Validation 4:Prediction")

    elif argc > 1:

        outputer = cnn.OutputForTFCNN()

        case = 0
        file_path = fi.filePath(case)

        path = file_path.learned_param
        outputer.setLearnedParameterFilePath(path)

        path = file_path.predicted_value
        outputer.setPredictedValueFilePath(path)    

        path = file_path.output_dir
        outputer.setSummaryDirPath(path)


        #######################################################
        # each data settings
        #######################################################

        from tensorflow.examples.tutorials.mnist import input_data
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        kstd.echoBar()
        kstd.echoBlank()
        print("data attr")
        kstd.echoBlank()
        print(type(x_train)) # numpy.ndarray
        print(type(y_train)) # numpy.ndarray
        print(type(x_test))  # numpy.ndarray
        print(type(y_test))  # numpy.ndarray
        print(x_train.shape) # (60000,28,28)
        print(y_train.shape) # (60000,)
        print(x_test.shape)  # (10000,28,28)
        print(y_test.shape)  # (10000,)
        print(np.max(x_train)) # 255
        print(np.min(x_train)) # 0
        print(np.max(y_train)) # 9
        print(np.min(y_train)) # 0
        kstd.echoBlank()
        kstd.echoBar()

        wight_x  = 28
        height_x = 28
        num_y    = 10
        dto_data_set_train   = cnn.DtoDataSetForTFCNN(wight_x,height_x,num_y)
        dto_data_set_predict = cnn.DtoDataSetForTFCNN(wight_x,height_x,num_y)
        dto_data_set_answer  = cnn.DtoDataSetForTFCNN(wight_x,height_x,num_y)


        data_size = 1000 * 1

        # setting x_train to data_set
        target_data = x_train
        x_train = []
        dto_np_table = kstd.DtoNpTable(wight_x * height_x)
        count = 0
        for di in range(data_size):
            xi = target_data[di]
            x_tmp = xi.flatten()
            x_tmp = kstd.npNomalizaiton(x_tmp)
            dto_np_table.addNpArray(x_tmp)
            count = count + 1
            if count % 1000 == 0:
                print("setting x_train : %05d" % count)
            # kstd.exit()
        print(dto_np_table.getAttrRowLength())

        dto_data_set_train.addFlatImageTable(dto_np_table)


        # setting x_test to data_set
        target_data = x_test
        x_test = []
        dto_np_table = kstd.DtoNpTable(wight_x * height_x)
        count = 0
        for di in range(data_size):
            xi = target_data[di]
            x_tmp = xi.flatten()
            x_tmp = kstd.npNomalizaiton(x_tmp)
            dto_np_table.addNpArray(x_tmp)
            count = count + 1
            if count % 1000 == 0:
                print("setting x_test  : %05d" % count)
            # kstd.exit()
        print(dto_np_table.getAttrRowLength())
        dto_data_set_predict.addFlatImageTable(dto_np_table)
        dto_data_set_answer.addFlatImageTable(dto_np_table)

        # setting y_train to data_set
        target_data = y_train
        y_train = []
        dto_np_table = kstd.DtoNpTable(num_y)
        count = 0
        for di in range(data_size):
            yi = target_data[di]
            dto_np_list = kstd.DtoNpList()
            kstd.createStaticLabelList(dto_np_list,num_y,yi)
            dto_np_table.addList(dto_np_list)
            count = count + 1
            if count % 1000 == 0:
                print("setting y_train : %05d" % count)
            # kstd.exit()
        print(dto_np_table.getAttrRowLength())
        dto_data_set_train.addLabelTable(dto_np_table)

        # setting y_test to data_set
        target_data = y_test
        y_test = []
        dto_np_table = kstd.DtoNpTable(num_y)
        count = 0
        for di in range(data_size):
            yi = target_data[di]
            dto_np_list = kstd.DtoNpList()
            kstd.createStaticLabelList(dto_np_list,num_y,yi)
            dto_np_table.addList(dto_np_list)
            count = count + 1
            if count % 1000 == 0:
                print("setting y_train : %05d" % count)
            # kstd.exit()
        print(dto_np_table.getAttrRowLength())
        dto_data_set_answer.addLabelTable(dto_np_table)

        selected_value = argvs[1]
        print("your selection : %s" % selected_value)       

        if selected_value == "1":
            mode = MODE_LEARNING
        elif selected_value == "2":
            mode = MODE_RE_LEARNING
        elif selected_value == "3":
            mode = MODE_VALIDATION
        elif selected_value == "4":
            mode = MODE_PREDICTION

        cnnExecuter(mode,dto_data_set_train,outputer)




