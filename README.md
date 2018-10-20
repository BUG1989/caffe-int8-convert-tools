# Caffe-Int8-Convert-Tools

This convert tools is base on TensorRT 2.0 Int8 calibration tools,which use the KL algorithm to find the suitable threshold to quantize the activions from Float32 to Int8(-128 - 127).

We provide the Classification(SqueezeNet_v1.1) and Detection(MobileNet_v1 SSD 300) demos based on [ncnn](https://github.com/Tencent/ncnn)(It is a high-performance neural network inference framework optimized for the mobile platform),and the community ready to support this implment.

[ncnn-int8](https://github.com/Tencent/ncnn/pull/487)

## Reference

For details, please read the following PDF:

[8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf) 

An introduction to the principles of a Chinese blog written by my friend:

[The implement of Int8 quantize base on TensorRT](https://note.youdao.com/share/?id=829ba6cabfde990e2832b048a4f492b3&type=note#/)

## HowTo

### New version

The purpose of this tool(caffe-int8-convert-tool-dev.py) is to test new features,such as mulit-channels quantization depend on group num.

This format is already supported in the [ncnn](https://github.com/Tencent/ncnn) latest version.I will do my best to transform some common network models into [classification-dev](https://github.com/BUG1989/caffe-int8-convert-tools/tree/master/classification-dev)

```
python caffe-int8-convert-tool-dev.py -h
usage: caffe-int8-convert-tool.py [-h] [--proto PROTO] [--model MODEL]
                                  [--mean MEAN MEAN MEAN] [--norm NORM]
                                  [--images IMAGES] [--output OUTPUT]
                                  [--group GROUP] [--gpu GPU]

find the pretrained caffe models int8 quantize scale value

optional arguments:
  -h, --help            show this help message and exit
  --proto PROTO         path to deploy prototxt.
  --model MODEL         path to pretrained weights
  --mean MEAN           value of mean
  --norm NORM           value of normalize(scale value)
  --images IMAGES       path to calibration images
  --output OUTPUT       path to output calibration table file
  --group GROUP         enable the group scale
  --gpu GPU             use gpu to forward
python caffe-int8-convert-tool-dev.py --proto=test/models/mobilenet_v1.prototxt --model=test/models/mobilenet_v1.caffemodel --mean 103.94 116.78 123.68 --norm=0.017 --images=test/images/ output=mobilenet_v1.table --gpu=1
```

Although it's done,but the speed of group quanization is very slow ......The difference from the old tool is that we try to get the int8_scale of bottom blob not the top blob. 

### How to use the output file(calibration-dev.table)

For example in *MobileNet_v1_dev.table*

```
conv1_param_0 156.639840
conv2_1/dw_param_0 0 72.129143 149.919382 // the convdw layer's weight scale every group is 0.0 72.129 149.919 ......
......
conv1 49.466518
conv2_1/dw 0 123.720796 48.705349 ...... // the convdw layer's bottom blob every group channel scale is 0.0 123.720 48.705 ......
......
```

### Old version

```
$ python caffe-int8-convert-tool.py --help
usage: caffe-int8-convert-tool.py [-h] [--proto PROTO] [--model MODEL]
                                  [--mean MEAN MEAN MEAN] [--norm NORM]
                                  [--images IMAGES] [--output OUTPUT]
                                  [--gpu GPU]

find the pretrained caffe models int8 quantize scale value

optional arguments:
  -h, --help            show this help message and exit
  --proto PROTO         path to deploy prototxt.
  --model MODEL         path to pretrained weights
  --mean MEAN           value of mean
  --norm NORM           value of normalize
  --images IMAGES       path to calibration images
  --output OUTPUT       path to output calibration table file
  --gpu GPU             use gpu to forward
  
$ python caffe-int8-convert-tool.py --proto=squeezenet_v1.1.prototxt --model=squeezenet.caffemodel --mean 104 117 123 --images=ILSVRC2012_1k --output=squeezenet_v1.1.table --gpu=1
```

Pay attention to the type of images,it is just the original image format,such as jpg or jpeg,do not us the type of caffe dataset(lmdb).
**It is recommended to provide representative calibration dataset for the given model use case,such as the validation or test dataset.**

### How to use the output file(calibration.table)

For example in *squeezenet_v1_1.table*

```
conv1_param_0 138.066410
fire2/squeeze1x1_param_0 92.028103 // the conv layer's weight scale is 92.028
......
data 0.841264
conv1 0.295743
pool1 0.161700  // the pool layer's top blob scale is 0.1617
fire2/squeeze1x1 0.089383 // the conv layer's top blob scale is 0.0893
......
```

Three steps to implement the *fire2/squeeze1x1* layer int8 convolution:

1. Quantize the bottom_blob and weight:

   ```
   bottom_blob_int8 = bottom_blob_float32 * data_scale(0.1617)
   weight_int8 = weight_float32 * weight_scale(92.028)
   ```

2. Convolution_Int8:

   ```
   top_blob_int32 = bottom_blob_int8 * weight_int8
   ```

3. Dequantize the TopBlob_Int32 and add the bias:

   ```
   top_blob_float32 = top_blob_int32 / [data_scale(0.1617) * weight_scale(92.028)] + bias_float32
   ```

## Accuracy and Performance

We used ImageNet2012 Dataset to complete some experiments.

| Type                | Detail                                                |
| ------------------- | ----------------------------------------------------- |
| Calibration Dataset | ILSVRC2012_img_test   1k                              |
| Test Dataset        | ILSVRC2012_img_val     5k                             |
| Framework           | ncnn                                                  |
| Support Layer       | Convolution3x3,Convolution1x1,ConvolutionDepthwise3x3 |

The following table show the Top1 and Top5 different between Float32 and Int8 inference.

|                 | FP32   |        | INT8   |        | Loss      |           |
| --------------- | ------ | ------ | ------ | ------ | --------- | --------- |
| NETWORK         | Top1   | Top5   | Top1   | Top5   | Diff Top1 | Diff Top5 |
| SqueezeNet v1.1 | 57.86% | 79.86% | 57.36% | 79.84% | -0.50%    | -0.02%    |
| MobileNet v1    | 67.78% | 87.62% | 64.92% | 85.22% | -2.86%    | -2.40%    |
| MobileNet v2    | 70.20% | 89.20% | 69.00% | 88.04% | -1.06%    | -1.16%    |
| GoogleNet v1    | 67.70% | 88.32% | 67.64% | 88.26% | -0.06%    | -0.06%    |
| ResNet-18       | 65.50% | 86.46% | 65.48% | 86.44% | -0.02%    | -0.02%    |
| ResNet-50       | 71.68% | 89.94% | 71.38% | 89.52% | -0.30%    | -0.32%    |

The following table show the speedup between Float32 and Int8 inference.It should be noted that the winograd algorithm is not used in the Float32 inference.The Hardware Platform is Hisi3519(Cortex-A17@1.2GHz)

| Uint(ms) | SqueezeNet v1.1 | MobileNet v1 | MobileNet v2 | GoogleNet | ResNet18 | MobileNetv1 SSD |
| -------- | --------------- | ------------ | ------------ | --------- | -------- | --------------- |
| Float32  | 382             | 568          | 392          | 1662      | 1869     | 1120            |
| Int8     | 242             | 369          | 311          | 1159      | 1159     | 701             |
| Ratio    | x1.30           | x1.41        | x1.28        | x1.43     | x1.61    | x1.47           |

## Sparse Connection Tool(*experimental* *stage*)

I tried to analyze the sparse connection of the CNN model.Using this tool,I've got some data([sparse-connection](https://github.com/BUG1989/caffe-int8-convert-tools/tree/master/sparse-connection)), and I hope you'll use it.

## Contributor

Thanks to our company [SenseNets](http://www.sensenets.com/home/) to support the open source project,and NVIDIA for providing the principle of correlation entropy,and ncnn's author [nihui](https://github.com/nihui) sharing his neural network inference framework.

Thanks to the help from the following friends:

Algorithm : [xupengfeixupf](https://github.com/xupengfeixupf), [JansonZhu](https://github.com/JansonZhu), wangxinwei, [lengmm](https://github.com/lengmm) 

Python : [daquexian](https://github.com/daquexian)

## License

BSD 3 Clause

