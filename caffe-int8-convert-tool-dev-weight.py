# -*- coding: utf-8 -*-

# SenseNets is pleased to support the open source community by making caffe-int8-convert-tool available.
#
# Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


"""
Quantization module for generating the calibration tables will be used by 
quantized (INT8) models from FP32 models.with bucket split,[k, k, cin, cout]
cut into "cout" buckets.
This tool is based on Caffe Framework.
"""
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import math, copy
import matplotlib.pyplot as plt
import sys,os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import time
import datetime
from google.protobuf import text_format
from scipy import stats

np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description='find the pretrained caffe models int8 quantize scale value')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--mean', dest='mean',
                        help='value of mean', type=float, nargs=3)
    parser.add_argument('--norm', dest='norm',
                        help='value of normalize', type=float, nargs=1, default=1.0)                            
    parser.add_argument('--images', dest='images',
                        help='path to calibration images', type=str)
    parser.add_argument('--output', dest='output',
                        help='path to output calibration table file', type=str, default='calibration-dev.table')
    parser.add_argument('--group', dest='group',
                        help='enable the group scale', type=int, default=1)        
    parser.add_argument('--gpu', dest='gpu',
                        help='use gpu to forward', type=int, default=0)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()


# global params
QUANTIZE_NUM = 127
STATISTIC = 1
INTERVAL_NUM = 2048

# ugly global params
quantize_layer_lists = []


class QuantizeLayer:
    def __init__(self, name, blob_name, group_num):
        self.name = name
        self.blob_name = blob_name
        self.group_num = group_num
        self.weight_scale = np.zeros(group_num)
        self.blob_max = 0.0
        self.blob_distubution_interval = 0.0
        self.blob_distubution = np.zeros(INTERVAL_NUM)
        self.blob_scale = 1.0
        self.group_zero = np.zeros(group_num)

    def quantize_weight(self, weight_data):
        # spilt the weight data by cout num
        blob_group_data = np.array_split(weight_data, self.group_num)
        for i, group_data in enumerate(blob_group_data):
            max_val = np.max(group_data)
            min_val = np.min(group_data)
            threshold = max(abs(max_val), abs(min_val))
            if threshold < 0.0001:
                self.weight_scale[i] = 0
                self.group_zero[i] = 1
            else:
                self.weight_scale[i] = QUANTIZE_NUM / threshold
            print("%-20s group : %-5d max_val : %-10f scale_val : %-10f" % (self.name + "_param0", i, threshold, self.weight_scale[i]))

    def initial_blob_max(self, blob_data):
        # get the max value of blob
        max_val = np.max(blob_data)
        min_val = np.min(blob_data)
        self.blob_max = max(self.blob_max, max(abs(max_val), abs(min_val)))

    def initial_blob_distubution_interval(self):
        self.blob_distubution_interval = STATISTIC * self.blob_max / INTERVAL_NUM
        print("%-20s max_val : %-10.8f distribution_intervals : %-10.8f" % (self.name, self.blob_max, self.blob_distubution_interval))

    def initial_histograms(self, blob_data):
        # collect histogram of every group channel blob
        th = self.blob_max
        hist, hist_edge = np.histogram(blob_data, bins=INTERVAL_NUM, range=(0, th))
        self.blob_distubution += hist

    def quantize_blob(self):
        # calculate threshold  
        distribution = np.array(self.blob_distubution)
        # pick threshold which minimizes KL divergence
        threshold_bin = threshold_distribution(distribution) 
        threshold = (threshold_bin + 0.5) * self.blob_distubution_interval
        # get the activation calibration value
        self.blob_scale = QUANTIZE_NUM / threshold
        print("%-20s bin : %-8d threshold : %-10f interval : %-10f scale : %-10f" % (self.name, threshold_bin, threshold, self.blob_distubution_interval, self.blob_scale))

    
def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist
    
    
def threshold_distribution(distribution, target_bin=128):
    """
    Return the best threshold value. 
    Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    Args:
        distribution: list, activations has been processed by histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL 
    """   
    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[threshold-1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        # 
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin
        
        # merge hist into num_quantized_bins bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
        
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        # p = _smooth_distribution(p) # with some bugs, need to fix
        # q = _smooth_distribution(q)
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        
        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value



def net_forward(net, image_path, transformer):
    """
    network inference and statistics the cost time
    Args:
        net: the instance of Caffe inference
        image_path: a image need to be inference
        transformer:
    Returns:
        none
    """ 
    # load image
    image = caffe.io.load_image(image_path)
    # transformer.preprocess the image
    net.blobs['data'].data[...] = transformer.preprocess('data',image)
    # net forward
    output = net.forward()


def file_name(file_dir):
    """
    Find the all file path with the directory
    Args:
        file_dir: The source file directory
    Returns:
        files_path: all the file path into a list
    """
    files_path = []

    for root, dir, files in os.walk(file_dir):
        for name in files:
            file_path = root + "/" + name
            print(file_path)
            files_path.append(file_path)

    return files_path


def network_prepare(net, mean, norm):
    """
    instance the prepare process param of caffe network inference 
    Args:
        net: the instance of Caffe inference
        mean: the value of mean 
        norm: the value of normalize 
    Returns:
        none
    """
    print("Network initial")

    img_mean = np.array(mean)
    
    # initial transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # convert shape from RBG to BGR 
    transformer.set_transpose('data', (2,0,1))
    # load meanfile
    transformer.set_mean('data', img_mean)
    # resize image data from [0,1] to [0,255]
    transformer.set_raw_scale('data', 255)   
    # convert RGB -> BGR
    transformer.set_channel_swap('data', (2,1,0))   
    # normalize
    transformer.set_input_scale('data', norm)

    return transformer  


def weight_quantize(net, net_file, group_on):
    """
    CaffeModel convolution weight blob Int8 quantize
    Args:
        net: the instance of Caffe inference
        net_file: deploy caffe prototxt
    Returns:    
        none
    """
    print("\nQuantize the kernel weight:")

    # parse the net param from deploy prototxt
    params = caffe_pb2.NetParameter()
    with open(net_file) as f:
        text_format.Merge(f.read(), params)

    for i, layer in enumerate(params.layer):
        # find the convolution 3x3 and 1x1 layers to get out the weight_scale
        if(layer.type == "Convolution" or layer.type == "ConvolutionDepthwise"):
            kernel_size = layer.convolution_param.kernel_size[0]
            if(kernel_size == 3 or kernel_size == 1):
                weight_blob = net.params[layer.name][0].data
                # initial the instance of QuantizeLayer Class lists,you can use enable group quantize to generate int8 scale for each group layer.convolution_param.group
                if (group_on == 1):
                    quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], layer.convolution_param.num_output)
                else:
                    quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], 1)
                # quantize the weight value
                quanitze_layer.quantize_weight(weight_blob)
                # add the quantize_layer into the save list
                quantize_layer_lists.append(quanitze_layer)

    return None                


def activation_quantize(net, transformer, images_files):
    """
    Activation Int8 quantize, optimaize threshold selection with KL divergence,
    given a dataset, find the optimal threshold for quantizing it.
    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    Args:
        net: the instance of Caffe inference
        transformer: 
        images_files: calibration dataset
    Returns:
        none
    """
    print("\nQuantize the Activation:")
    # run float32 inference on calibration dataset to find the activations range
    for i , image in enumerate(images_files):
        # inference
        net_forward(net, image, transformer)
        # find max threshold
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten()
            layer.initial_blob_max(blob)
        if i % 100 == 0:
            print("loop stage 1 : %d/%d" % (i, len(images_files)))
    
    # calculate statistic blob scope and interval distribution
    for layer in quantize_layer_lists:
        layer.initial_blob_distubution_interval()   

    # for each layers
    # collect histograms of activations
    print("\nCollect histograms of activations:")
    for i, image in enumerate(images_files):
        net_forward(net, image, transformer)
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten()
            layer.initial_histograms(blob)
        if i % 100 == 0:
            print("loop stage 2 : %d/%d" % (i, len(images_files)))          

    # calculate threshold with KL divergence
    for layer in quantize_layer_lists:
        layer.quantize_blob()  

    return None


def save_calibration_file(calibration_path):
    calibration_file = open(calibration_path, 'w') 
    # save temp
    save_temp = []
    # save weight scale
    for layer in quantize_layer_lists:
        save_string = layer.name + "_param_0"
        for i in range(layer.group_num):
            save_string = save_string + " " + str(layer.weight_scale[i])
        save_temp.append(save_string)

    # save bottom blob scales
    for layer in quantize_layer_lists:
        save_string = layer.name + " " + str(layer.blob_scale)
        save_temp.append(save_string)

    # save into txt file
    for data in save_temp:
        calibration_file.write(data + "\n")

    calibration_file.close()

    # save calibration logs
    save_temp_log = []
    calibration_file_log = open(calibration_path + ".log", 'w')
    for layer in quantize_layer_lists:
        save_string = layer.name + ": value range 0 - " + str(layer.blob_max) \
                                 + ", interval " + str(layer.blob_distubution_interval) \
                                 + ", interval num " + str(INTERVAL_NUM) + "\n" \
                                 + str(layer.blob_distubution.astype(dtype=np.int64))
        save_temp_log.append(save_string)

    # save into txt file
    for data in save_temp_log:
        calibration_file_log.write(data + "\n")


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python caffe-int8-scale-tools-dev.py -h")


def main():
    """
    main function
    """

    # time start
    time_start = datetime.datetime.now()

    print(args)

    if args.proto == None or args.model == None or args.mean == None or args.images == None:
        usage_info()
        return None

    # deploy caffe prototxt path
    net_file = args.proto

    # trained caffemodel path
    caffe_model = args.model

    # mean value
    mean = args.mean

    # norm value
    norm = 1.0
    if args.norm != 1.0:
        norm = args.norm[0]

    # calibration dataset
    images_path = args.images

    # the output calibration file
    calibration_path = args.output

    # enable the group scale
    group_on = args.group

    # default use CPU to forwark
    if args.gpu != 0:
        caffe.set_device(0)
        caffe.set_mode_gpu()

    # initial caffe net and the forword model(GPU or CPU)
    net = caffe.Net(net_file,caffe_model,caffe.TEST)

    # prepare the cnn network
    transformer = network_prepare(net, mean, norm)

    # get the calibration datasets images files path
    images_files = file_name(images_path)

    # quanitze kernel weight of the caffemodel to find it's calibration table
    weight_quantize(net, net_file, group_on)

    # quantize activation value of the caffemodel to find it's calibration table
    activation_quantize(net, transformer, images_files)

    # save the calibration tables,best wish for your INT8 inference have low accuracy loss :)
    save_calibration_file(calibration_path)

    # time end
    time_end = datetime.datetime.now()

    print("\nCaffe Int8 Calibration table create success, it's cost %s, best wish for your INT8 inference has a low accuracy loss...\(^▽^)/...2333..." % (time_end - time_start))

if __name__ == "__main__":
    main()
