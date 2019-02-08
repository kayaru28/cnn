#coding: UTF-8

import kayaru_standard_process as kstd
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

def pl(l,L):
    return 1 - l / ( 2.0 * L )

def __cnnModel(x_2d,is_training):

    num_out_channels = 1

    num_L = 3.0 * 3.0 * 2

    #####################################################
    # convolution layer
    process = kstd_m.messPaddingMessage("convolution layer",20)
    conv1   = cnn.Conv2dSame()
    x_2d    =  makeConvLayer(x_2d, conv1, prop.conv1, process, num_out_channels)
    num_out_channels = prop.conv1.out_channels

    #####################################################
    # max pool layer
    process = kstd_m.messPaddingMessage("max pooling layer",20)
    pool1   = cnn.MaxPoolSame()
    x_2d    = makePoolLayer(x_2d,pool1,prop.pool1,process)

    #####################################################
    # residual resnet
    process = kstd_m.messPaddingMessage("resnet layer",20)
    resnet1_1 = cnn.ResidualResNetStochasticDepth(pl(1,num_L))
    x_2d      = makeResNetLayer(x_2d,resnet1_1,prop.resnet1,process,num_out_channels,is_training)
    resnet1_2 = cnn.ResidualResNetStochasticDepth(pl(2,num_L))
    x_2d      = makeResNetLayer(x_2d,resnet1_2,prop.resnet1,process,num_out_channels,is_training)
    resnet1_3 = cnn.ResidualResNetStochasticDepth(pl(3,num_L))
    x_2d      = makeResNetLayer(x_2d,resnet1_3,prop.resnet1,process,num_out_channels,is_training)

    #####################################################
    # convolution layer
    process = kstd_m.messPaddingMessage("convolution layer",20)
    conv3   = cnn.Conv2dSame()
    x_2d    = makeConvLayer(x_2d,conv3,prop.conv3,process,num_out_channels)
    num_out_channels = prop.conv3.out_channels

    #####################################################
    # residual resnet
    process = kstd_m.messPaddingMessage("resnet layer",20)
    resnet2_1 = cnn.ResidualResNetStochasticDepth(pl(4,num_L))
    x_2d    = makeResNetLayer(x_2d,resnet2_1,prop.resnet2,process,num_out_channels,is_training)
    resnet2_2 = cnn.ResidualResNetStochasticDepth(pl(5,num_L))
    x_2d    = makeResNetLayer(x_2d,resnet2_2,prop.resnet2,process,num_out_channels,is_training)
    resnet2_3 = cnn.ResidualResNetStochasticDepth(pl(6,num_L))
    x_2d    = makeResNetLayer(x_2d,resnet2_3,prop.resnet2,process,num_out_channels,is_training)

    #####################################################
    # convolution layer
    process = kstd_m.messPaddingMessage("convolution layer",20)
    conv4   = cnn.Conv2dSame()
    x_2d    = makeConvLayer(x_2d,conv4,prop.conv4,process,num_out_channels)
    num_out_channels = prop.conv4.out_channels

    #####################################################
    # residual resnet
    process = kstd_m.messPaddingMessage("resnet layer",20)
    resnet3_1 = cnn.ResidualResNetStochasticDepth(pl(7,num_L))
    x_2d    = makeResNetLayer(x_2d,resnet3_1,prop.resnet3,process,num_out_channels,is_training)
    resnet3_2 = cnn.ResidualResNetStochasticDepth(pl(8,num_L))
    x_2d    = makeResNetLayer(x_2d,resnet3_2,prop.resnet3,process,num_out_channels,is_training)
    resnet3_3 = cnn.ResidualResNetStochasticDepth(pl(9,num_L))
    x_2d    = makeResNetLayer(x_2d,resnet3_3,prop.resnet3,process,num_out_channels,is_training)

    #####################################################
    # convolution layer
    process = kstd_m.messPaddingMessage("convolution layer",20)
    conv5   = cnn.Conv2dSame()
    x_2d    = makeConvLayer(x_2d,conv5,prop.conv5,process,num_out_channels)
    num_out_channels = prop.conv5.out_channels

    #####################################################
    # global average pooling layer
    process = kstd_m.messPaddingMessage("GAP layer",20)
    x_2d    = cnn.globalAveragePool(x_2d)

    y_cnn = x_2d

    #####################################################
    # full connected layer preparation

    fc_input_size = 512 #cnn.getX2dFeatureVolume(x_2d)
    y_cnn         = tf.reshape(x_2d, [-1, fc_input_size ])
 
    #####################################################
    # full connected layer
    process = "full connected layer"

    fc1 = cnn.FullConnected()
    fc1.setInputSize(fc_input_size)
    fc1.setOutputSize(prop.fc1_out)

    judgeParameterSettingError(fc1,process)

    W_f1     = fc1.createW0()
    y_cnn    = fc1.calculate(y_cnn,W_f1,prop.KEEP_PROP_ALL)

    kstd.echoIsAlready(process)
    fc_input_size = prop.fc1_out
    
    y_acc         = y_cnn

    return y_cnn,y_acc

def judgeParameterSettingError(model,process):
    if not model.isSetParameters():
        kstd.echoErrorOccured(process)
        kstd.exit()

def makeConvLayer(x_2d,conv,conv_para,process,num_preunit_out_channels):
    conv.setFilterW(conv_para.filter_w)
    conv.setFilterH(conv_para.filter_h)
    conv.setOutChannels(conv_para.out_channels)
    conv.setFilterInChannels(num_preunit_out_channels)
    conv.setStrideW(conv_para.stride_w)
    conv.setStrideH(conv_para.stride_h)
    judgeParameterSettingError(conv,process)

    W = conv.createW0()
    x_2d = conv.calculate(x_2d,W)

    kstd.echoIsAlready(process)

    return x_2d

def makePoolLayer(x_2d,pool,pool_para,process):
    pool.setFilterW(pool_para.stride_w)
    pool.setFilterH(pool_para.stride_h)
    pool.setStrideW(pool_para.stride_w)
    pool.setStrideH(pool_para.stride_h)
    judgeParameterSettingError(pool,process)

    x_2d = pool.calculate(x_2d)

    kstd.echoIsAlready(process)

    return x_2d

def makeResNetLayer(x_2d,resnet,resnet_para,process,num_preunit_out_channels,is_training):
    resnet.conv.setFilterW(resnet_para.filter_w)
    resnet.conv.setFilterH(resnet_para.filter_h)
    resnet.conv.setOutChannels(num_preunit_out_channels)
    resnet.conv.setFilterInChannels(num_preunit_out_channels)
    resnet.conv.setStrideW(resnet_para.stride_w)
    resnet.conv.setStrideH(resnet_para.stride_h)
    judgeParameterSettingError(resnet,process)

    W_r1,W_r2 = resnet.createW0()
    x_2d      = resnet.calculate(x_2d,W_r1,W_r2,is_training)

    kstd.echoIsAlready(process)

    return x_2d
#########################################################
# trainer
#########################################################
def getProgressMessage(ii,train_accuracy,train_entropy,timer):
    return ('step %5d/%d\taccuracy\t%0.2g\tentropy\t%0.2g\t(%ds/%dm)'
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
    y_cnn,y_acc                       = __cnnModel(x_2d,True)    
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
    y_cnn,y_acc                       = __cnnModel(x_2d,True)    
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
    y_cnn,y_acc                       = __cnnModel(x_2d,False)    
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
    y_cnn,y_acc = __cnnModel(x_2d,False)    

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




