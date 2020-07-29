# Caffe-Int8-Convert-Tools

This convert tools is base on TensorRT 2.0 Int8 calibration tools, which use the KL algorithm to find the suitable threshold to quantize the activions from Float32 to Int8(-127 - 127).

We provide the Classification(SqueezeNet_v1.1) and Detection(MobileNet_v1 SSD 300) demo based on [ncnn](https://github.com/Tencent/ncnn)(a high-performance neural network inference framework optimized for the mobile platform) and the community ready to support this implementation.

[The pull request in ncnn](https://github.com/Tencent/ncnn/pull/749)

## NCNN have a new convert tool to support Post-Training-Quantization 

Using this new [ncnn-quantization-tools](https://github.com/Tencent/ncnn/tree/master/tools/quantize), you can convert your ncnn model to ncnn int8 model directly. If you just want to deploy your model with ncnn,I suggest you use it.

## Reference

For details, please read the following PDF:

[8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf) 

MXNet quantization implementation:

[Quantization module for generating quantized (INT8) models from FP32 models](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py)

An introduction to the principles of a Chinese blog written by my friend([bruce.zhang](https://github.com/bigbigzxl)):

[The implement of Int8 quantize base on TensorRT](https://zhuanlan.zhihu.com/zhangxiaolongOptimization)

## HowTo

The purpose of this tool(caffe-int8-convert-tool-dev.py) is to test new features, such as mulit-channels quantization depend on group num.

This format is already supported in the [ncnn](https://github.com/Tencent/ncnn) latest version. I will do my best to transform some common network models into [classification-dev](https://github.com/BUG1989/caffe-int8-convert-tools/tree/master/classification-dev)

```
python caffe-int8-convert-tool-dev-weight.py -h
usage: caffe-int8-convert-tool-dev-weight.py [-h] [--proto PROTO] [--model MODEL]
                                  [--mean MEAN MEAN MEAN] [--norm NORM]
                                  [--images IMAGES] [--output OUTPUT]
                                  [--group GROUP] [--gpu GPU]

find the pretrained caffemodel int8 quantize scale value

optional arguments:
  -h, --help            show this help message and exit
  --proto PROTO         path to deploy prototxt.
  --model MODEL         path to pretrained caffemodel
  --mean MEAN           value of mean
  --norm NORM           value of normalize(scale value or std value)
  --images IMAGES       path to calibration images
  --output OUTPUT       path to output calibration table file
  --group GROUP         enable the group scale(0:disable,1:enable,default:1)
  --gpu GPU             use gpu to forward(0:disable,1:enable,default:0)
python caffe-int8-convert-tool-dev-weight.py --proto=test/models/mobilenet_v1.prototxt --model=test/models/mobilenet_v1.caffemodel --mean 103.94 116.78 123.68 --norm=0.017 --images=test/images/ output=mobilenet_v1.table --group=1 --gpu=1
```

### How to use the output file(calibration-dev.table)

For example in *MobileNet_v1_dev.table*

```
conv1_param_0 0.0 3779.48337933 482.140562772 1696.53814502
conv2_1/dw_param_0 0 72.129143 149.919382 // the convdw layer's weight scale every group is 0.0 72.129 149.919 ......
......
conv1 49.466518
conv2_1/dw 123.720796 // the convdw layer's bottom blobchannel scale is 123.720
......
```

Three steps to implement the *conv1* layer int8 convolution:

1. Quantize the bottom_blob and weight:

   ```
   bottom_blob_int8 = bottom_blob_float32 * data_scale(49.466518)
   weight_int8 = weight_float32 * weight_scale(156.639840)
   ```

2. Convolution_Int8:

   ```
   top_blob_int32 = bottom_blob_int8 * weight_int8
   ```

3. Dequantize the TopBlob_Int32 and add the bias:

   ```
   top_blob_float32 = top_blob_int32 / [data_scale(49.466518) * weight_scale(156.639840)] + bias_float32
   ```

## How to use with ncnn

[quantized int8 inference](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference#caffe-int8-convert-tools)

## Accuracy and Performance

#### We use ImageNet2012 Dataset to complete some classification test.

| Type                | Detail                                                |
| ------------------- | ----------------------------------------------------- |
| Calibration Dataset | ILSVRC2012_img_test   1k                              |
| Test Dataset        | ILSVRC2012_img_val    5k                              |
| Framework           | ncnn                                                  |
| Support Layer       | Convolution,ConvolutionDepthwise,ReLU                 |

The following table show the Top1 and Top5 different between Float32 and Int8 inference.

| Models          | FP32   |        | INT8   |        | Loss      |           |
| --------------- | ------ | ------ | ------ | ------ | --------- | --------- |
|                 | Top1   | Top5   | Top1   | Top5   | Diff Top1 | Diff Top5 |
| SqueezeNet v1.1 | 57.78% | 79.88% | 57.82% | 79.84% | +0.04%    | -0.04%    |
| MobileNet v1    | 67.26% | 87.92% | 66.74% | 87.43% | -0.52%    | -0.49%    |
| GoogleNet       | 68.50% | 88.84% | 68.62% | 88.68% | +0.12%    | -0.16%    |
| ResNet18        | 65.49% | 86.56% | 65.30% | 86.52% | -0.19%    | -0.04%    |
| ResNet50        | 71.80% | 89.90% | 71.76% | 90.06% | -0.04%    | +0.16%    |

#### We use VOC0712,MSCOCO Dataset to complete some detection test.

| Type         | Detail         |
| ------------ | -------------- |
| Test Dataset | VOC2007        |
| Unit         | mAP (Class 20) |

| Models           | FP32  | INT8  | Loss   |
| ---------------- | ----- | ----- | ------ |
| SqueezeNet SSD   | 61.80 | 61.27 | -0.53  |
| MobileNet_v1 SSD | 70.49 | 68.92 | -1.57  |

#### Speed up

The following table show the speedup between Float32 and Int8 inference. It should be noted that the winograd algorithm is enable in the Float32 and Int8 inference. The Hardware Platform is Hisi3519(Cortex-A17@880MHz)

| Uint(ms) | SqueezeNet v1.1 | MobileNet v1 | GoogleNet | ResNet18 | MobileNetv1 SSD | SqueezeNet SSD |
| -------- | --------------- | ------------ | --------- | -------- | --------------- | -------------- |
| Float32  | 282             | 490          | 1107      | 985      | 970             | 610            |
| Int8     | 192             | 369          | 696       | 531      | 605             | 498            |
| Ratio    | x1.46           | x1.33        | x1.59     | x1.85    | x1.60           | x1.22          |

#### Memory reduce

Runtime Memory : mbytes

| Models            | fp32-wino63 | int8-wino23 | int8-wino43 |
| ----------------- | ----------- | ----------- | ----------- |
| squeezenet_v1_1   | 50          | 30          | 32          |
| mobilenet_v1      | 61          | 35          | 35          |
| mobilenet_v1_ssd  | 90          | 45          | 45          |
| squeezenet_v1_ssd | 210         | 70          | 94          |
| resnet18          | 335         | 77          | 130         |
| googlenet_v1      | 154         | 72          | 89          |

Storage Memory : mbytes

| Models            | fp32 | int8 |
| ----------------- | ---- | ---- |
| squeezenet_v1_1   | 4.71 | 1.20 |
| mobilenet_v1      | 16.3 | 4.31 |
| mobilenet_v1_ssd  | 22.0 | 5.60 |
| squeezenet_v1_ssd | 21.1 | 5.37 |
| resnet18          | 44.6 | 11.2 |
| googlenet_v1      | 26.6 | 6.72 |

## Contributor

Thanks to NVIDIA for providing the principle of correlation entropy and ncnn's author [nihui](https://github.com/nihui) sharing his neural network inference framework.

Thanks to the help from the following friends:

Optimization Instructor : [Fugangping](https://github.com/fu1899), [bruce.zhang](https://github.com/bigbigzxl)

Algorithm : [xupengfeixupf](https://github.com/xupengfeixupf), [JansonZhu](https://github.com/JansonZhu), [wangxinwei](https://github.com/StarStyleSky), [lengmm](https://github.com/lengmm) 

Python : [daquexian](https://github.com/daquexian)

## License

BSD 3 Clause
