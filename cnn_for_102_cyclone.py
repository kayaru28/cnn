#coding: UTF-8

import kayaru_standard_process as kstd
import kayaru_standard_messages as kstd_m
import cnn_std as cnn
import properties_for_102_cyclone as prop
import numpy as np
import file_interfase_for_102_cyclone as fi
import sys

MODE_SAVE = "save"

def echoStartSpecial(process=""):
    kstd.echoBlank()
    kstd.echoBar(70,"-")
    kstd.echoStart(process)
    kstd.echoBar(70,"-")
    kstd.echoBlank()

def echoIsAlreadySpecial(process=""):
    kstd.echoBlank()
    kstd.echoBar(15,"-- ")
    print(str(kstd.getTimeyyyymmddhhmmss()))
    kstd.echoIsAlready("    " + process)
    kstd.echoBar(15,"-- ")
    kstd.echoBlank()

def getDataSet(dto_data_set,file_path):

    ### image data reading
    dto_data_set.firstlizationImage(prop.image_wigth,prop.image_height)

    path = file_path.image_list
    csv_reader_for_image = kstd.CsvReader()
    csv_reader_for_image.openFile(path)
    csv_reader_for_image.readFile()

    image_list = np.array(csv_reader_for_image.getData())
    kstd.echoIsSetting("image      ","image_list")
    exit_code = dto_data_set.addFlatImageList(image_list)

    csv_reader_for_image.closeFile()

    kstd.judgeError(exit_code)

    ### label data reading
    dto_data_set.firstlizationLabel(prop.num_of_label_kind)

    path = file_path.label_list
    csv_reader_for_label = kstd.CsvReader()
    csv_reader_for_label.openFile(path)
    csv_reader_for_label.readFile()

    label_list = np.array(csv_reader_for_label.getData())
    kstd.echoIsSetting("label      ","label_list")
    exit_code = dto_data_set.addLabelList(label_list)

    csv_reader_for_label.closeFile()

    kstd.judgeError(exit_code)

def getDataSetForTest(dto_data_set,case):

    csv_reader_for_test_image = kstd.CsvReader()
    #csv_reader_for_test_image.openFile(prop.test_image_list_path)
    csv_reader_for_test_image.readFile()

    test_image_list = np.array(csv_reader_for_test_image.getData())
    kstd.echoIsSetting("test_image","image_list")
    exit_code = dto_data_set.addTestFlatImageList(test_image_list)

    csv_reader_for_test_image.closeFile()

    kstd.judgeError(exit_code)

def varCheckSpecial(dto,process_name):
    if(dto.varCheck() == kstd.NORMAL_CODE ):
        kstd.echoBlank()
        echoIsAlreadySpecial(process_name)
        kstd.echoBlank()
        kstd.echoBlank()
    else:
        kstd.echoErrorOccured(process_name + ":: varCheck")
        kstd.echoErrorCodeIs(dto.varCheck())
        kstd.exit()

#####################################################################################
#####################################################################################
if __name__ == "__main__":

    case = 0
    file_path = fi.filePath(case)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = ""  

    if mode == MODE_SAVE:
        echoStartSpecial("save mode")
        output_dir_path = kstd.getScriptDir() + "\\" + fi.getSaveDirName(case)
        exit_code = kstd.mkdir(output_dir_path)
        kstd.judgeError(exit_code)
        exit_code = kstd.cp(file_path.prop,output_dir_path)
        kstd.judgeError(exit_code)
        file_path.updateOutputDir(output_dir_path)

    process_name = "hyper parameter setting"
    echoStartSpecial(process_name)

    dto_hyper_param = cnn.DtoHyperParameterForTFCNN()
  
    #######################################################
    # hyper parameter settings
    #######################################################

    length = 20
    dto_hyper_param.setNumOfConvLayer(prop.num_of_conv_layer)
    kstd.echoIsSetting(kstd.getPaddingString("num_of_conv_layer",length),str(dto_hyper_param.num_of_conv_layer) )
    dto_hyper_param.setNumOfHiddenLayer(prop.num_of_hidden_layer)
    kstd.echoIsSetting(kstd.getPaddingString("num_of_hidden_layer",length),str(dto_hyper_param.num_of_hidden_layer) )
    dto_hyper_param.setDropRate(prop.drop_rate)
    kstd.echoIsSetting(kstd.getPaddingString("drop_rate",length),str(dto_hyper_param.drop_rate) )
    dto_hyper_param.setLearningRate(prop.learning_rate)
    kstd.echoIsSetting(kstd.getPaddingString("learning_rate",length),str(dto_hyper_param.learning_rate))
    dto_hyper_param.setLearningIteration(prop.learning_iteration)
    kstd.echoIsSetting(kstd.getPaddingString("learning_iter",length),str(dto_hyper_param.learning_iteration))
    dto_hyper_param.setBatchSize(prop.batch_size)
    kstd.echoIsSetting(kstd.getPaddingString("batch_size",length),str(dto_hyper_param.batch_size))
    
    # conv and pooling filter parameter
    for conv_layer_number in range(prop.num_of_conv_layer):
        dto_hyper_param.addFilterWigth(prop.filter_wigth[conv_layer_number])
        dto_hyper_param.addFilterHeight(prop.filter_height[conv_layer_number])
        dto_hyper_param.addNumOfOutCh(prop.num_of_out_ch[conv_layer_number])
        dto_hyper_param.addStrideConv(prop.stride_conv[conv_layer_number])
        dto_hyper_param.addStridePool(prop.stride_pool[conv_layer_number])
        dto_hyper_param.addShapePool(prop.shape_pool[conv_layer_number])

    varCheckSpecial(dto_hyper_param,process_name)

    #######################################################
    # data set setting
    #######################################################
    process_name = "data set setting"
    echoStartSpecial(process_name)

    dto_data_set = cnn.DtoDataSetForTFCNN()
    getDataSet(dto_data_set,file_path)

    varCheckSpecial(dto_data_set,process_name)
    
    dto_case_meta = cnn.DtoCaseMetaForTFCNN()

    path = file_path.learned_param
    dto_case_meta.setLearnedParameterFilePath(path)

    #######################################################
    # cnn executer
    #######################################################
    process_name = "convolution neural net"

    echoStartSpecial(process_name)
    cnn.cnnExecuter(dto_data_set,dto_hyper_param,dto_case_meta)



