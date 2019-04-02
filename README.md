# face_recognation_with_Caffe_and_DPU
This is a face recognation application programme with CPU caffe implementation or FPGA accelerator via PCIE.

## Basic info
The basic algorithm is CNN, VGG-face specifically. And there are three versions of implementation as shown below.

## caffe with CPU
We put codes and necessary data in caffe_cpu folder. It is programmed with caffe python API, and you may install Caffe and specify the path of Caffe in the file test.py if you want to run.

## FPGA accelerator via PCIE
We implemented a neural network accelerator based on FPGA, and generated the weight file of the neural network and the instruction file needed by the accelerator through our tool chain. These data and the input/output of the network during the running are transmitted through the PCIE bus. Our code is based on the official driver of the PCIE of the FPGA IP provided by Xilinx.
The interface and control logic of this version are written in python.

## FPGA accelerator working with multithread CPU
The acceleration effect of the previous method is good, but the part of the CPU execution is too expensive, which becomes the bottleneck. So we have re-implemented the multi-threaded control program based on C++.
