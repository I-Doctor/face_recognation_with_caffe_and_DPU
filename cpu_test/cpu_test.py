# coding=utf-8
from scipy import misc
import numpy as np
import argparse
import sys,os
import time
import copy
import re
import matplotlib.pyplot as plt
import matplotlib.image as mping
import sklearn.metrics.pairwise as pw

# set caffe dir and workspace dir here
caffe_root = '/home/zk/workspace/installcaffe/caffe'
sys.path.insert(0,caffe_root)
import caffe
work_root = '/home/zk/workspace/15suo/cpu_test'



def main(argv):

    all_start = time.time()
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
        "--model_proto",
        default=os.path.join(work_root,
                "./vgg_face_caffe/VGG_FACE_deploy.prototxt"), # set deploy.prototxt
        help="Model definition file .prototxt"
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(work_root,
                "./vgg_face_caffe/VGG_FACE_finetuned.caffemodel"), # set deploy.caffemodel
        help="Trained model weights file .caffemodel"
    )
    parser.add_argument(
        "--threshold",
        default=0.794, # set threshold for judgement
        help="Trained model weights file .caffemodel"
    )
    parser.add_argument(
        "--save_image",
        action='store_true', # set to show image during processing
        help="save the results with image."
    )
    parser.add_argument(
        "--show",
        action='store_true', # set to show image during processing
        help="show the results on screen."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--images_dim",
        default='224,224', # set width and height
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(work_root,'./vgg_face_caffe/face_mean.binaryproto'), # set mean file
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    args = parser.parse_args()

    # read input list of pairs from file
    with open(args.input_list) as input_list:
        print "open input list..."
        pairs = input_list.readlines()
    print '==[0]== Finished reading parse at: ', time.time()-all_start

    # initialize net and input data transformer
    net = caffe.Net(args.model_proto, args.pretrained_model, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_mean('data', (np.load(mean_file).mean(1).mean(1)))
    mean = np.zeros(3)
    mean[0]=87
    mean[1]=97
    mean[2]=112
    transformer.set_mean('data', mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    
    # generate figure window for show model 
    count = 0
    out_file = open(args.output_file, 'w')
    if args.save_image:
        if args.show:
            plt.figure(figsize=(8,4),dpi=60)
            plt.ion()

    print '==[1]== Finished loading net at: ', time.time()-all_start, '  start loop:'

    
    # process pairs in file
    for pair in pairs:
        # load two images
        inner_start = time.time()
        [img_1, img_2] = pair.split()
        print '--processing', count, 'pair: ', img_1, img_2
        input_1 = caffe.io.load_image(os.path.join(work_root,img_1))
        input_2 = caffe.io.load_image(os.path.join(work_root,img_2))
        print '    @@ Loaded images at: ', time.time() - inner_start

        # calculate with caffe
        #st = time.time()
        temp = transformer.preprocess('data', input_1)
        print temp
        net.blobs['data'].data[...] = temp
        #print 'data_transformer:', time.time() - st
        out_1 = net.forward()
        feature_1 = copy.deepcopy(out_1['fc7'][0])
        for i in range(4070,4096):
            print feature_1[i]
        net.blobs['data'].data[...] = transformer.preprocess('data', input_2)
        out_2 = net.forward()
        feature_2 = copy.deepcopy(out_2['fc7'][0])
        print '    @@ got output at: ', time.time() - inner_start

        # calculate distance for judgement
        feature_1 = feature_1.reshape((1,4096))
        feature_2 = feature_2.reshape((1,4096))
        mt = pw.pairwise_distances(feature_1, feature_2, metric='cosine')
        distance = mt[0][0]
        print "      Distance before normalization:", distance
        if distance < args.threshold :
            result = 'Same'
        else:
            result = 'Different'
        print '    @@ calculated judgement at: ', time.time() - inner_start

        # save
        print >> out_file, img_1, img_2, result
        print '    @@ saved at: ', time.time() - inner_start
        count += 1

        # show and save image if necessary
        if args.save_image:
            print("      save picture")
            pic1 = mping.imread(img_1)
            pic2 = mping.imread(img_2)
            if not args.show:
                plt.figure(figsize=(8,4),dpi=60)
            plt.suptitle(result,fontsize=14,color='b')
            plt.subplot(1,2,1)
            plt.imshow(pic1)
            plt.title(img_1,fontsize=12)
            plt.subplot(1,2,2)
            plt.imshow(pic2)
            plt.title(img_2,fontsize=12)
            plt.axis('on')
            plt.pause(0.000000001)

            pattern = re.compile(r'\./.*/')
            outpicName = './output/'+re.sub(pattern,"",img_1)+'_'+re.sub(pattern,"",img_2)+'_'+result+'.png'
            plt.savefig(outpicName)
            print '    @@ save picture at: ', time.time() - inner_start
    
    out_file.close()

    print '==[2]== Finished all', count, 'pairs in: ', time.time()-all_start


if __name__ == '__main__':
    main(sys.argv)
