# face_recognation_with_Caffe_and_DPU
This is a face recognation application program with CPU caffe implementation or FPGA accelerator via PCIE.

## Basic info
The basic algorithm is CNN, VGG-face specifically. And there are four versions of implementation as shown below. The basic process of the program is to read in the face picture pairs of the specified paths through the input text file, and then get two vectors after calculating by the neural network, and use the comparison of the inner product of the vectors and the threshold value to determine whether they belong to the same person or not.  

In order to run DPU-based versions, it is necessary to install Xilinx's official PICE driver(reference to the official website: AR# 65444 Xilinx PCI Express DMA Drivers and Software Guide) and test it according to its instructions, and then compile and run our program. If it doesn't work because of some driver mistake you may try to copy our program to under the official driver path and level with the original ***test/*** folder ,load and test driver in our folder then try again.

## Caffe with CPU
The code and necessary data are in the ***caffe_cpu/*** folder. 
This program is written with Caffe python API. 
Which means you need to install Caffe and specify the path of Caffe in the file test.py before running the program with the following command.  

The command to run the program:
```
python test.py input_list.txt output/results.out save show
```

## FPGA accelerator via PCIE with Python interface
We developed a neural network accelerator based on Xilinx FPGA, **KCU1500 board** specifically in this project, and generated the bitstream file of the accelerator, which can be used to configure the hardware. 
At the same time, using our tool chain and caffe model, we generated the weight file and instruction file of the neural network needed by the accelerator. 
These data files and the input and output of the neural network are transmitted between the host and the FPGA accelerate board through PCIE bus. 
Our program is based on Xilinx official PCIE driver and PCIE DMA IP of hardware.  

Bitstream file is stored in ***bit/*** folder and you can set your KCU1500 hardware with it.
Weight and instruction files are stored in ***weight/*** folder.
*We haven't offerd our tool chain here.*  

The interface and control logic of this version are written in python. And the command to run the program:
```
python test.py input_list.txt output/results.out save show
```

## FPGA accelerator with totaly C++ interface
The acceleration effect of the previous method is good, but the part of the CPU execution is too expensive, which becomes the bottleneck. So we have re-implemented the top control program with C++.  

There two sub c++ versions in ***tests_cpp/*** folder, one is running in iteration mode which means it will process one pair of picutres and show the result at one time, and the other is running in batch mode which means it will process all pairs of pictures and then calculate them and than output the results.   

The command to run the program:
```
./iteration_process input_list.txt output/results.out save show
```
Or:
```
./batch_process input_list.txt output/results.out save show
```

## FPGA accelerator with multithread
In order to further reduce the overhead of CPU processing and improve the average image processing throughput, a C++ multithreaded program based on Linux is implemented in ***tests_multithread/***.

In addition to the configuration and data input at the beginning of the program, “the image preprocessing”, “transmission & DPU calculation”， “post-processing of the output of the neural network” are encapsulated into three threads, which are executed by multi-threads using pthreads method.

The command to run the program:
```
./face_dpu_multithread input_list.txt output/results.out save show
```
