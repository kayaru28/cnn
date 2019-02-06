#coding: UTF-8
import os
import numpy as np

PRINT_STEP = 5
PARA_SAVE  = 10

LEARNING_ITERATION = 70
BATCH_SIZE         = 50

LEARNING_RATE      = 0.0001
#LEARNING_RATE_MAX  = 0.0001
#LEARNING_RATE_MIN  = 0.000001

CYCLE_LOG_OUTPUT = 1
CYCLE_PARA_OUTPUT = 10

KEEP_PROP_ALL = 1.0

################################################################################
#
# model structure
#
################################################################################


conv1_filter_w     = 3
conv1_filter_h     = 3
conv1_out_channels = 16
conv1_stride_w     = 1
conv1_stride_h     = 1

pool1_filter_w = 3
pool1_filter_h = 3
pool1_stride_w = 1
pool1_stride_h = 1

fc1_out       = 10
fc1_keep_prop = 0.5







