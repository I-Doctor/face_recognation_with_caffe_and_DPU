# coding=utf-8
from scipy import misc
import numpy as np
import argparse
import os,sys
import time
import copy
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn.metrics.pairwise as pw

# set caffe dir and workspace dir here
caffe_root = '/home/zk/workspace/installcaffe/caffe'
sys.path.insert(0,caffe_root)
import caffe
work_root = '/home/zk/workspace/15suo/Xilinx_Answer_65444_Linux_Files/tests_optimize/'



def pread(filename, nx, nz):

    pic = np.fromfile(filename, np.uint8)
    pic = pic[0:3*nx*nz]
    pic.shape = [nx, nz, 3]
    #print '   pic  shape: ', pic.shape
    #print '   pic  dtype: ', pic.dtype #, pic[100:110,100,0], pic[100:110,100,2]

    pic = pic * 2
    temp = np.zeros((224,224,1),dtype=np.uint8)
    temp[:,:,0] = pic [:,:,0]
    pic [:,:,0] = pic [:,:,2]
    pic [:,:,2] = temp[:,:,0]
    print '     pic  dtype: ', pic.dtype #, pic[100:110,100,0], pic[100:110,100,2]
    print("     read pic finished")

    return pic


def pgenerate(filename,outname):

    img = mpimg.imread(filename)
    #print '  image shape: ', img.shape
    img = misc.imresize(img, (224,224,3))
    #print '  image shape: ', img.shape
    #print '  image dtype: ', img.dtype #, img[100:110,100,0], img[100:110,100,2]

    temp = np.zeros((224,224,1),dtype=np.uint8)
    temp[:,:,0] = img [:,:,0]
    img [:,:,0] = img [:,:,2]
    img [:,:,2] = temp[:,:,0]
    img = img / 2
    print '      image dtype: ', img.dtype #, img[100:110,100,0], img[100:110,100,2]

    img.tofile(outname)


if __name__ == '__main__':

    all_start = time.time()

    # get args
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_list",
        help="List of input pairs of images and picutes .txt" +
            "two files in a line, based on work root such as: 1.jpg 2.jpg"
    )
    parser.add_argument(
        "output_file",
        help="Output results filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--threshold",
        default=0.711, # set threshold for judgement
        help="threshold for judgement, default is 0.794"
    )
    parser.add_argument(
        "--save_image",
        action='store_true', # set to show image during processing
        help="save the results with image."
    )
    parser.add_argument(
        "--show",
        action='store_true', # set to show image during processing
        help="show the results on screen after each pair."
    )
    args = parser.parse_args()
    # read input list of pairs from file
    print args.input_list
    print os.path.exists(args.input_list)
    input_list = open(args.input_list)
    pairs = input_list.readlines()

    print '==[0]== Finished reading parse at: ', time.time()-all_start

    # initialize dpu
    dpudriver = "./face_dpu_driver" 
    if os.path.exists(dpudriver):
        print "      initial dpu_driver"
        os.system(dpudriver + ' 1')

    # generate figure window for show model 
    count = 0
    out_file = open(args.output_file, 'w')
    print "      initial output file"
    if args.save_image:
        print "      initial image figure"
        plt.figure(figsize=(8,4),dpi=60)
        if args.show:
            print "      initial ion mode"
            plt.ion()

    print '==[1]== Initialized net on dpu at: ', time.time() - all_start, " Start loop:"

    
    for pair in pairs:
        inner_start = time.time()
        # load two images
        [img_1, img_2] = pair.split()
        print '--processing', count, 'pair: ', img_1, img_2
        print '    @@ Loaded images at: ', time.time() - inner_start

        # generate binary file needed by dpu from jpg pictures
        pgenerate(img_1,'/dev/shm/input_1.bin')
        pgenerate(img_2,'/dev/shm/input_2.bin')
        print '    @@ generated bins at: ', time.time() - inner_start

        # execute dpu driver to transport data and get results
        if os.path.exists(dpudriver):
            print "      execute dpu_driver"
            os.system(dpudriver)
        print '    @@ got output at: ', time.time() - inner_start
        
        # calculate distance for judgement
        result = 0
        if os.path.exists("/dev/shm/out_1.bin") and os.path.exists("/dev/shm/out_2.bin"):
            print "      execute judge"

            feature_1 = np.fromfile("/dev/shm/out_1.bin", np.int8)
            feature_2 = np.fromfile("/dev/shm/out_2.bin", np.int8)
            feature_1 = feature_1.astype(np.float)
            feature_2 = feature_2.astype(np.float)
            print '      feature shape: ', feature_1.shape, " & ", feature_2.shape
            feature_1 = feature_1.reshape((1,4096))
            feature_2 = feature_2.reshape((1,4096))
            mt = pw.pairwise_distances(feature_1, feature_2, metric='cosine')
            distance = mt[0][0]
            print "      Distance before normalization:", distance
            print "      Threshold: ", (float)(args.threshold)
            if distance < (float)(args.threshold) :
                result = 'Same'
            else:
                result = 'Different'

        # save
        print >> out_file, img_1, img_2, result
        print '    @@ saved result at: ', time.time() - inner_start
        count += 1

        # show and save image if necessary
        if args.save_image:
            print("      save picture")
            pic1 = pread("/dev/shm/input_1.bin", 224, 224)
            pic2 = pread("/dev/shm/input_2.bin", 224, 224)
            plt.suptitle(result,fontsize=14,color='b')
            plt.subplot(1,2,1)
            plt.imshow(pic1)
            plt.title(img_1,fontsize=12)
            plt.subplot(1,2,2)
            plt.imshow(pic2)
            plt.title(img_2,fontsize=12)
            plt.axis('on')
                        
            pattern = re.compile(r'\./.*/')
            outpicName = './output/'+re.sub(pattern,"",img_1)+'_'+re.sub(pattern,"",img_2)+'_'+result+'.png'
            plt.savefig(outpicName)
            print '    @@ save picture at: ', time.time() - inner_start
            if args.show:
                plt.pause(0.000000001)
                #plt.show()
                print '    @@ show at: ', time.time() - inner_start
        
    out_file.close()
    print '==[2]== Finished all', count, 'pairs in: ', time.time()-all_start
