# -*- coding: utf-8 -*-

# SenseNets is pleased to support the open source community by making caffe-sparse-tool available.
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
Analyze module for generating the sparse-connection table
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
        description='find the pretrained caffe models sparse value')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--mean', dest='mean',
                        help='value of mean', type=float, nargs=3)
    parser.add_argument('--norm', dest='norm',
                        help='value of normalize', type=float, nargs=1, default=1.0)                            
    parser.add_argument('--images', dest='images',
                        help='path to sparse images', type=str)
    parser.add_argument('--output', dest='output',
                        help='path to output sparse file', type=str, default='sparse.table')    
    parser.add_argument('--gpu', dest='gpu',
                        help='use gpu to forward', type=int, default=0)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()

# ugly global params
sparse_layer_lists = []


class SparseLayer:
    def __init__(self, name, bottom_blob_name, top_blob_name, num_inch, num_outch):
        self.name = name
        self.bottom_blob_name = bottom_blob_name
        self.top_blob_name = top_blob_name
        self.num_inch = num_inch
        self.num_outch = num_outch
        self.top_blob_max = [0 for x in range(0, num_outch)]
        self.bottom_blob_max = [0 for x in range(0, num_inch)]
        self.weight_zero = [0 for x in range(0, num_outch)]
        self.inch_zero = [0 for x in range(0, num_inch)]
        self.outch_zero = [0 for x in range(0, num_outch)]

    def sparse_weight(self, weight_data):
        # spilt the weight data by outch num
        weight_outch_data = np.array_split(weight_data, self.num_outch)
        for i, data in enumerate(weight_outch_data):
            max_val = np.max(data)
            min_val = np.min(data)
            threshold = max(abs(max_val), abs(min_val))
            if threshold < 0.0001:
                self.weight_zero[i] = 1
            #print("%-20s group : %-5d max_val : %-10f scale_val : %-10f" % (self.name + "_param0", i, threshold, self.weight_scale[i]))

    def analyze_bottom_blob(self, blob_data):
        # spilt the blob data by inch num
        blob_inch_data = np.array_split(blob_data, self.num_inch)
        # interval for per bottom blob channel
        for i, data in enumerate(blob_inch_data):
            max_val = np.max(data)
            min_val = np.min(data)
            self.bottom_blob_max[i] = max(self.bottom_blob_max[i], max(abs(max_val), abs(min_val)))
            if max_val == min_val:
                self.inch_zero[i] = 1

    def analyze_top_blob(self, blob_data):
        # spilt the blob data by outch num
        blob_outch_data = np.array_split(blob_data, self.num_outch)
        # interval for per top blob channel
        for i, data in enumerate(blob_outch_data):
            max_val = np.max(data)
            min_val = np.min(data)
            self.top_blob_max[i] = max(self.top_blob_max[i], max(abs(max_val), abs(min_val)))
            if max_val == min_val:
                self.outch_zero[i] = 1

    def sparse_bottom_blob(self):
        for i in range(0, self.num_inch):
            if self.bottom_blob_max[i] < 0.0001:
                self.inch_zero[i] = 1

    def sparse_top_blob(self):
        for i in range(0, self.num_outch):
            if self.top_blob_max[i] < 0.0001:
                self.outch_zero[i] = 1
            #print("%-20s outch : %-5d max_val : %-10.8f " % (self.name, i, self.blob_max[i]))

    def display_sparse_info(self):
        count = 0
        for i in range(self.num_outch):
            if self.outch_zero[i] != 0 or self.weight_zero[i] !=0:
                count += 1
        print("%-20s outch : %-8d sparse : %-8d ratio : %-6.2f " % (self.name, self.num_outch, count, count / float(self.num_outch) * 100))

    def save_calibration(file_path):
        pass



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


def weight_sparse(net, net_file, transformer, images_files):
    """
    CaffeModel convolution weight blob sparse
    Args:
        net: the instance of Caffe inference
        net_file: deploy caffe prototxt
    Returns:    
        none
    """
    print("\nSparse the kernel weight:")
    
    # forward only once to find the bottom blob property
    net_forward(net, images_files[0], transformer)

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
                # find bottom blob channel num
                num_input = net.blobs[layer.bottom[0]].shape[1]
                # initial the instance of SparseLayer Class lists
                sparse_layer = SparseLayer(layer.name, layer.bottom[0], layer.top[0], num_input, layer.convolution_param.num_output)
                # sparse the weight value
                sparse_layer.sparse_weight(weight_blob)
                # add the sparse_layer into the save list
                sparse_layer_lists.append(sparse_layer)

    return None                


def activation_sparse(net, transformer, images_files):
    """
    Activation bottom/top blob sparse analyze
    Args:
        net: the instance of Caffe inference
        transformer: 
        images_files: sparse dataset
    Returns:  
        none
    """    
    print("\nAnalyze the sparse info of the Activation:")
    # run float32 inference on sparse dataset to analyze activations
    for i , image in enumerate(images_files):
        net_forward(net, image, transformer)
        # analyze bottom/top blob
        for layer in sparse_layer_lists:
            blob = net.blobs[layer.bottom_blob_name].data[0].flatten()
            layer.analyze_bottom_blob(blob)
            blob = net.blobs[layer.top_blob_name].data[0].flatten()
            layer.analyze_top_blob(blob)            
    
    # calculate top blob and flag the sparse channels in every layers
    for layer in sparse_layer_lists:
        layer.sparse_bottom_blob()
        layer.sparse_top_blob()

    return None


def save_sparse_file(sparse_path):
    sparse_file = open(sparse_path, 'w') 
    # save temp
    save_temp = []
    # save weight scale
    for layer in sparse_layer_lists:
        save_string = layer.name + "_weight"
        for i in range(layer.num_outch):
            if layer.weight_zero[i] != 0:
                save_string = save_string + " " + str(i)
        save_temp.append(save_string)

        # save bottom/top blob sparse channel
        save_string = layer.name + "_bottom"
        for i in range(layer.num_inch):
            if layer.inch_zero[i] != 0:
                save_string = save_string + " " + str(i)
        save_temp.append(save_string)

        save_string = layer.name + "_top   "
        for i in range(layer.num_outch):
            if layer.outch_zero[i] != 0:
                save_string = save_string + " " + str(i)
        save_temp.append(save_string)

    # save into txt file
    for data in save_temp:
        sparse_file.write(data + "\n")

    sparse_file.close()


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python caffe-sparse-tool.py -h")


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

    # the output sparse file
    sparse_path = args.output

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

    # analyze kernel weight of the caffemodel to find some channels whose weight value whole zero
    weight_sparse(net, net_file, transformer, images_files)

    # analyze activation value of the caffemodel to find some channels whose value whole zero or the same value(maybe the bisa value of latest conv layer)
    activation_sparse(net, transformer, images_files)

    # show sparse info
    for layer in sparse_layer_lists:
        layer.display_sparse_info()

    # save the sparse tables,best wish for your sparse have low accuracy loss :)
    save_sparse_file(sparse_path)

    # time end
    time_end = datetime.datetime.now()

    print("\nCaffe Sparse table create success, it's cost %s, best wish for your Sparse inference has a low accuracy loss...\(^▽^)/...2333..." % (time_end - time_start))

if __name__ == "__main__":
    main()
