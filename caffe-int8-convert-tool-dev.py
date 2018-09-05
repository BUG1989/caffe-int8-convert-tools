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
quantized (INT8) models from FP32 models.
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
                        help='enable the group scale', type=int, default=0)        
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
        self.weight_scale = [0 for x in range(0, group_num)]
        self.blob_max = [0 for x in range(0, group_num)]
        self.blob_distubution_interval = [0 for x in range(0, group_num)]
        self.blob_distubution = [[0 for col in range(INTERVAL_NUM)] for row in range(group_num)]
        self.blob_scale = [1 for x in range(0, group_num)]
        self.group_zero = [0 for x in range(0, group_num)]

    def quantize_weight(self, weight_data):
        # spilt the weight data by group num
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
        # spilt the blob data by group num
        blob_group_data = np.array_split(blob_data, self.group_num)
        # interval for per bottom blob group channel
        for i, group_data in enumerate(blob_group_data):
            max_val = np.max(group_data)
            min_val = np.min(group_data)
            self.blob_max[i] = max(self.blob_max[i], max(abs(max_val), abs(min_val)))

    def initial_blob_distubution_interval(self):
        for i in range(0, self.group_num):
            if self.blob_max[i] < 0.000001:
                self.blob_scale[i] = 0
                self.group_zero[i] = 1
                self.blob_distubution_interval[i] = 0
            else:
                self.blob_distubution_interval[i] = STATISTIC * self.blob_max[i] / INTERVAL_NUM
            print("%-20s group : %-5d max_val : %-10.8f distribution_intervals : %-10.8f" % (self.name, i, self.blob_max[i], self.blob_distubution_interval[i]))

    def initial_histograms(self, blob_data):
        # spilt the blob data by group num
        blob_group_data = np.array_split(blob_data, self.group_num)
        # interval for per bottom blob group channel
        for i, group_data in enumerate(blob_group_data):    
            if self.blob_scale[i] == 0:
                continue
            else:    
                # collect histogram of every group channel blob
                add_to_distribution(group_data, self.blob_distubution[i], self.blob_distubution_interval[i])

    def quantize_blob(self):
        # calculate threshold
        for i in range(0, self.group_num):
            # sparse DepthwiseConvolution
            if self.blob_scale[i] == 0:
                print("%-20s group : %-5d bin : %-8d threshold : %-10f interval : %-10f scale : %-10f" % (self.name, i, 0, 0, self.blob_distubution_interval[i], self.blob_scale[i]))
            else:
                # normalize distributions
                normalize_distribution(self.blob_distubution[i])    
                distribution = np.array(self.blob_distubution[i])
                # pick threshold which minimizes KL divergence
                threshold_bin = threshold_distribution(distribution) 
                threshold = (threshold_bin + 0.5) * self.blob_distubution_interval[i]
                # get the activation calibration value
                self.blob_scale[i] = QUANTIZE_NUM / threshold
                print("%-20s group : %-5d bin : %-8d threshold : %-10f interval : %-10f scale : %-10f" % (self.name, i, threshold_bin, threshold, self.blob_distubution_interval[i], self.blob_scale[i]))

    def save_calibration(file_path):
        pass



def add_to_distribution(blob, distribution, interval):
    """
    add the distribution
    Args:
        blob: the output blob of caffe layer
        distribution: a list ,size is 2048
        interval: a float number
    Returns:
        none
    """     
    max_index = len(distribution) - 1

    indexes = np.minimum((np.abs(blob[blob!=0]) / interval).astype(np.int32), max_index)
    for index in indexes:
        distribution[index] = distribution[index] + 1


def normalize_distribution(distribution):
    """
    Normalize the input list
    Args:
        distribution: a list ,size is 2048
    Returns:
        none
    """     
    num_sum = sum(distribution)
    for i, data in enumerate(distribution):
        distribution[i] = data / float(num_sum)


def compute_kl_divergence(dist_a, dist_b):
    """
    Returen kl_divergence between 
    Args:
        dist_a: list original
        dist_b: list expand
    Returns:
        kl_divergence: float, kl_divergence 
    """ 
    nonzero_inds = dist_a != 0
    return np.sum(dist_a[nonzero_inds] * np.log(dist_a[nonzero_inds] / dist_b[nonzero_inds]))


def threshold_distribution(distribution, target_bin=128):
    """
    Returen the best cut off num of bin 
    Args:
        distribution: list, activations has been processed by histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL 
    """   
    target_threshold = target_bin
    min_kl_divergence = 1000
    length = distribution.size

    quantize_distribution = np.zeros(target_bin)

    threshold_sum = 0.0
    threshold_sum = sum(distribution[target_bin:])

    for threshold in range(target_bin, length):
        t_distribution = copy.deepcopy(distribution[:threshold])
        t_distribution[threshold-1] = t_distribution[threshold-1] + threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # ************************ threshold  ************************
        quantize_distribution = np.zeros(target_bin)
        num_per_bin = threshold / target_bin
        for i in range(0, target_bin):
            start = i * num_per_bin
            end = start + num_per_bin

            left_upper = (int)(math.ceil(start))
            if(left_upper > start):
                left_scale = left_upper - start
                quantize_distribution[i] += left_scale * distribution[left_upper - 1]

            right_lower = (int)(math.floor(end))
            if (right_lower < end):
                right_scale = end - right_lower
                quantize_distribution[i] += right_scale * distribution[right_lower]

            for j in range(left_upper, right_lower):
                quantize_distribution[i] += distribution[j]
        # ************************ threshold ************************

        # ************************ quantize ************************
        expand_distribution = np.zeros(threshold, dtype=np.float32)

        for i in range(0, target_bin):
            start = i * num_per_bin
            end = start + num_per_bin

            count = 0

            left_upper = (int)(math.ceil(start))
            left_scale = 0.0
            if (left_upper > start):
                left_scale = left_upper - start
                if (distribution[left_upper - 1] != 0):
                        count += left_scale

            right_lower = (int)(math.floor(end))
            right_scale = 0.0
            if (right_lower < end):
                right_scale = end - right_lower
                if (distribution[right_lower] != 0):
                    count += right_scale

            for j in range(left_upper, right_lower):
                if (distribution[j] != 0):
                    count = count + 1

            if count == 0:
                continue;
            expand_value = quantize_distribution[i] / count

            if (left_upper > start):
                if (distribution[left_upper - 1] != 0):
                    expand_distribution[left_upper - 1] += expand_value * left_scale
            if (right_lower < end):
                if (distribution[right_lower] != 0):
                    expand_distribution[right_lower] += expand_value * right_scale
            for j in range(left_upper, right_lower):
                if (distribution[j] != 0):
                    expand_distribution[j] += expand_value
        # ************************ quantize ************************

        kl_divergence = compute_kl_divergence(t_distribution, expand_distribution)

        if kl_divergence < min_kl_divergence:
            min_kl_divergence = kl_divergence
            target_threshold = threshold

    return target_threshold


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
    start = time.clock()
    output = net.forward()
    end = time.clock()
    print("%s forward time : %.3f s" % (image_path, end - start))


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
        if i == 0:
            if layer.type != "Input":
                raise ValueError("First layer should be input")

        # find the convolution 3x3 and 1x1 layers to get out the weight_scale
        if(layer.type == "Convolution" or layer.type == "ConvolutionDepthwise"):
            kernel_size = layer.convolution_param.kernel_size[0]
            if(kernel_size == 3 or kernel_size == 1):
                weight_blob = net.params[layer.name][0].data
                # initial the instance of QuantizeLayer Class lists,you can use enable group quantize to generate int8 scale for each group layer.convolution_param.group
                if (group_on == 1):
                    quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], layer.convolution_param.group)
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
        net_forward(net, image, transformer)
        print("loop stage 1 : %d" % (i))
        # find max threshold
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten()
            layer.initial_blob_max(blob)
    
    # calculate statistic blob scope and interval distribution
    for layer in quantize_layer_lists:
        layer.initial_blob_distubution_interval()   

    # for each layers
    # collect histograms of activations
    print("\nCollect histograms of activations:")
    for i, image in enumerate(images_files):
        net_forward(net, image, transformer)
        print("loop stage 2 : %d" % (i))    
        start = time.clock() 
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten()
            layer.initial_histograms(blob)
        end = time.clock()
        print("add cost %.3f s" % (end - start))

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
        save_string = layer.name
        for i in range(layer.group_num):
            save_string = save_string + " " + str(layer.blob_scale[i])
        save_temp.append(save_string)

    # save into txt file
    for data in save_temp:
        calibration_file.write(data + "\n")

    calibration_file.close()


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python caffe-int8-scale-tools.py -h")


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
    # weight_quantize(net, net_file)
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
