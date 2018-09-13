#coding: UTF-8
import os

common_dir = "C:\\Users\\istor\\Desktop\\work\\102_DSL_cyclone_classification"
image_list_path = common_dir + "\\sample_list_0000.csv"
label_list_path = common_dir + "\\label_list_0000.csv"

################################################################################
#
# dataset parameters
#
################################################################################

num_of_label_kind = 2
image_wigth       = 64
image_height      = 64

################################################################################
#
# hyper parameters
#
################################################################################

num_of_hidden_layer = 10
num_of_conv_layer   = 2
drop_rate           = 0.8
learning_rate       = 0.7
learning_iteration  = 20
batch_size          = 10


########################################
### format
filter_height = []
filter_wigth  = []
num_of_out_ch = []
stride_conv   = []
stride_pool   = []
shape_pool    = []

########################################
### for 1st
filter_height.append( 4 )
filter_wigth .append( 4 )
num_of_out_ch.append( 32 )
stride_conv  .append( [1, 1, 1, 1] )
stride_pool  .append( [1, 2, 2, 1] )
shape_pool   .append( [1, 2, 2, 1] )

########################################
### for 2nd
filter_height.append( 4 )
filter_wigth .append( 4 )
num_of_out_ch.append( 40 )
stride_conv  .append( [1, 1, 1, 1] )
stride_pool  .append( [1, 2, 2, 1] )
shape_pool   .append( [1, 2, 2, 1] )






