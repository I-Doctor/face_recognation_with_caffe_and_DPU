//=====================================================================
// (c) Copyright EFC of NICS; Tsinghua University. All rights reserved.
// Author   : Kai Zhong
// Email    : zhongk15@mails.tsinghua.edu.cn
// Create Date   : 2019.04.20
// File Name     : batch_process.cpp
// Description   : read pairs of picture name and process them with dpu
//                 then get the result of face recognation and write 
//                 into the result file or save in picture.
// Dependencies  : opencv 2.4.9
//                 g++
//=====================================================================

//---------------------------------------------------------------------
// include
//---------------------------------------------------------------------
#define _BSD_SOURCE
#define _XOPEN_SOURCE 500
#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <byteswap.h>
#include <errno.h>
#include <signal.h>
#include <ctype.h>
#include <termios.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


//---------------------------------------------------------------------
// namespace
//---------------------------------------------------------------------
using namespace std;
using namespace cv;


//---------------------------------------------------------------------
// define
//---------------------------------------------------------------------
/* base address of dpu which shouldn't be changed while hardware is not */
#define INST1_DDR_ADDR 0x6D000000
#define DATA1_DDR_ADDR 0x6A000000
#define WEIT1_DDR_ADDR 0x60000000
#define INST2_DDR_ADDR 0xED000000
#define DATA2_DDR_ADDR 0xEA000000
#define WEIT2_DDR_ADDR 0xE0000000
/* file name of PCIE DMA driver which shouldn't be changed */
#define DMA_H2C_DEVICE "/dev/xdma0_h2c_0"
#define DMA_C2H_DEVICE "/dev/xdma0_c2h_0"
#define DMA_REG_DEVICE "/dev/xdma0_user"

/* customized definations which can be changed according to different app */
#define THRESHOLD 0.29
#define WEIT_FILE_NAME "../weight/concat_svd_weight.bin"
#define INST_FILE_NAME "../weight/concat_svd_instr.bin"

/* define of some simple functions */
/* ltoh: little to host   htol: little to host */
#if __BYTE_ORDER == __LITTLE_ENDIAN
#  define ltohl(x)       (x)
#  define ltohs(x)       (x)
#  define htoll(x)       (x)
#  define htols(x)       (x)
#elif __BYTE_ORDER == __BIG_ENDIAN
#  define ltohl(x)     __bswap_32(x)
#  define ltohs(x)     __bswap_16(x)
#  define htoll(x)     __bswap_32(x)
#  define htols(x)     __bswap_16(x)
#endif

/* error output function */  
#define FATAL do { \
    fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); \
    exit(1); \
} while(0)

/* timing profiling function */  
#define TIMING(log_str)do { \
    gettimeofday(&end, NULL); \
    timeuse = 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec; \
    printf(log_str, pair, timeuse/1000000.0); \
} while(0)

#define MAP_SIZE (32*1024UL)
#define MAP_MASK (MAP_SIZE - 1)


//---------------------------------------------------------------------
// class and struct
//---------------------------------------------------------------------
class MyMat {
public:
    Mat _mat;
    // default constructor
    MyMat() {}
    // copy constructor overload (USED BY Push)
    MyMat(const MyMat& src) {
        src._mat.copyTo(_mat);
    }
    // Assignment (=) Operator overloading (USED BY Pop)
    MyMat& operator=(const MyMat& src) {
        src._mat.copyTo(_mat);
        return *this;
    }
};


//---------------------------------------------------------------------
// global varibale
//---------------------------------------------------------------------
static bool save_flag = 0;
static bool show_flag = 0;
vector<string> img_path_pairs; 
queue<MyMat> fifo_img_1;
queue<MyMat> fifo_img_2;
queue<String> fifo_img_name_1;
queue<String> fifo_img_name_2;
ofstream output_results;
long timeuse;
struct timeval start, end; 
stringstream ss;

//---------------------------------------------------------------------
// declaration of functions
//---------------------------------------------------------------------
static int test_dma_to_device(const char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, const char *filename);
static int test_dma_from_device(const char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, const char *filename);
static int reg_read(const char *devicename , uint32_t addr);
static int reg_write(const char *devicename , uint32_t addr,uint32_t writeval);
static void save_img(Mat &src1, Mat &src2, string info, string name, bool show_flag);
static double get_similarity(const Mat& first, const Mat& second);
static void preprocessing();
static void dpu_calculate();
static void result_output();



//---------------------------------------------------------------------
// main
//---------------------------------------------------------------------
int main(int argc, char* argv[]){

    // check arguements------------------------------------------------------
    if (argc == 3){
        printf("=====Start application WITHOUT saving or showing: ===========\n");
        printf("=====process %s and write results into %s \n",argv[1],argv[2]);
    }
    else if (argc==4 && 0==strcmp(argv[3],"save")){
        save_flag = 1;
        printf("=====Start application WITH saving image WITHOUT showing: ===\n");
        printf("=====process %s and write results into %s \n",argv[1],argv[2]);
    }
    else if (argc==5 && 0==strcmp(argv[4],"show")){
        save_flag = 1;
        show_flag = 1;
        printf("=====Start application WITH saving image WITH showing: ======\n");
        printf("=====process %s and write results into %s \n",argv[1],argv[2]);
    }
    else {
        printf("=====Please give the arguements as follow: \n");
        printf("=====basic       : ./dpu_face_app input_list.txt results.txt\n");
        printf("=====save image  : ./dpu_face_app input_list.txt results.txt save\n");
        printf("=====save & show : ./dpu_face_app input_list.txt results.txt save show\n");
        return 0;
    }


    // start init-----------------------------------------------------------
    gettimeofday(&start, NULL);  

    // config dpu
    printf("----@@Init the app ---------------------------------------------\n");
    printf("      Write init weights and insts into ddr.");
    test_dma_to_device(DMA_H2C_DEVICE, WEIT1_DDR_ADDR, 23859120,0,1, WEIT_FILE_NAME);
    test_dma_to_device(DMA_H2C_DEVICE, WEIT2_DDR_ADDR, 23859120,0,1, WEIT_FILE_NAME);
    printf("      weight_ok");
    test_dma_to_device(DMA_H2C_DEVICE, INST1_DDR_ADDR, 257644, 0,1, INST_FILE_NAME);
    test_dma_to_device(DMA_H2C_DEVICE, INST2_DDR_ADDR, 257644, 0,1, INST_FILE_NAME);
    printf("      instr_ok \n");

    // read input list into vector img_path_pairs
    printf("      Read input list into vector.\n");
    ifstream input_list(argv[1]); 
    if(!input_list) { 
        cerr << "Can't open the file.\n"; 
        FATAL; 
    }
    string line; 
    while(getline(input_list, line)) 
        img_path_pairs.push_back(line);
    input_list.close();

    // open output results file
    //ofstream output_results(argv[2]); 
    output_results.open(argv[2]);
    if(!output_results) { 
        cerr << "Can't open output file.\n"; 
        FATAL; 
    }
    
    // Init time
    int pair = 0;
    TIMING("    ##Finish%d [Init] time: %f\n");


    // calculating pairs---------------------------------------------------
    printf("------Start calculating ---------------------------------------\n");
    // preprocessing-------------------------------------------------------
    preprocessing();
    // dpu calculating-----------------------------------------------------
    dpu_calculate();
    // result & output-----------------------------------------------------
    result_output();

    // finish ---------------------------------------------------------------
    output_results.close();
    TIMING("======Finish%d [TOTAL] time: %f ============\n "); 

    return 0;
}


static int test_dma_to_device(const char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, const char *filename)
{
  struct timeval ss,ee;  
  gettimeofday(&ss, NULL );  
  long timeuse;

  int rc;
  char *buffer = NULL;
  char *allocated = NULL;

  posix_memalign((void **)&allocated, 4096/*alignment*/, size + 4096);
  assert(allocated);
  buffer = allocated + offset;

  int file_fd = -1;
  int fpga_fd = open(devicename, O_RDWR);
  assert(fpga_fd >= 0);

  if (filename) {
    file_fd = open(filename, O_RDONLY);
    assert(file_fd >= 0);
  }

  /* fill the buffer with data from file? */
    if (file_fd >= 0) {
      /* read data from file into memory buffer */
      rc = read(file_fd, buffer, size);
      if (rc != size) perror("read(file_fd)");
      assert(rc == size);
    }

  /* select AXI MM address */
  off_t off = lseek(fpga_fd, addr, SEEK_SET);
  while (count--) {
  
    /* write buffer to AXI MM address using SGDMA */
    rc = write(fpga_fd, buffer, size);
    assert(rc == size);
  }

  close(fpga_fd);
  if (file_fd >= 0) {
    close(file_fd);
  }
  free(allocated);
}


static int test_dma_from_device(const char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, const char *filename)
{

  struct timeval ss,ee;  
  gettimeofday(&ss, NULL );  
  long timeuse;

  int rc;
  char *buffer = NULL;
  char *allocated = NULL;

  posix_memalign((void **)&allocated, 4096/*alignment*/, size + 4096);
  assert(allocated);
  buffer = allocated + offset;

  int file_fd = -1;
  int fpga_fd = open(devicename, O_RDWR | O_NONBLOCK);
  assert(fpga_fd >= 0);

  /* create file to write data to */
  if (filename) {
    file_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC | O_SYNC, 0666);
    assert(file_fd >= 0);
  }

  while (count--) {
    /* select AXI MM address */
    off_t off = lseek(fpga_fd, addr, SEEK_SET);
    /* read data from AXI MM into buffer using SGDMA */
    rc = read(fpga_fd, buffer, size);
    if ((rc > 0) && (rc < size)) {
    }
    
    /* file argument given? */
    if ((file_fd >= 0)) {
      /* write buffer to file */
      rc = write(file_fd, buffer, size);
      assert(rc == size);
    }
  }

  close(fpga_fd);
  if (file_fd >=0) {
    close(file_fd);
  }
  free(allocated);
}



static int reg_read(const char *devicename , uint32_t addr) {
  int fd;
  void *map_base, *virt_addr; 
  uint32_t read_result, writeval;
  off_t target;
  /* access width */
  int access_width = 'w';
  char *device;

  device = strdup(devicename);
  target = addr;
  access_width = 'w';
  
  if ((fd = open(devicename, O_RDWR | O_SYNC)) == -1) FATAL;
  
  /* map one page */
  map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map_base == (void *) -1) FATAL;
  
  /* calculate the virtual address to be accessed */
  virt_addr = (char*)map_base + target;
  /* read only */
  
  read_result = *((uint32_t *) virt_addr);
  /* swap 32-bit endianess if host is not little-endian */
  read_result = ltohl(read_result);
  if (munmap(map_base, MAP_SIZE) == -1) FATAL;
  close(fd);
  return (int)read_result;
}


static int reg_write(const char *devicename , uint32_t addr,uint32_t writeval) {
  int fd;
  void *map_base, *virt_addr; 
  uint32_t read_result;
  off_t target;
  /* access width */
  int access_width = 'w';
  char *device;

  device = strdup(devicename);
  target = addr;

  if ((fd = open(devicename, O_RDWR | O_SYNC)) == -1) FATAL;
  //close(fd);
  //fd = open(devicename, O_RDWR | O_SYNC);
  //fflush(stdout);

  /* map one page */
  map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map_base == (void *) -1) FATAL;

  /* calculate the virtual address to be accessed */
  virt_addr = (char*)map_base + target; //cast to char* to calculate
  /* data value given, i.e. writing? */
  
  /* swap 32-bit endianess if host is not little-endian */
  writeval = htoll(writeval);  
  *((uint32_t *) virt_addr) = writeval;        
  if (munmap(map_base, MAP_SIZE) == -1) FATAL;
  close(fd);
  return 0;
}

static uint32_t getopt_integer(const char *optarg)
{
    int rc;
    uint32_t value;
    rc = sscanf(optarg, "0x%x", &value);
    if (rc <= 0)
    rc = sscanf(optarg, "%ul", &value);
    return value;
}


void save_img(Mat &src1, Mat &src2, string info, string name, bool show_flag)
{
    // calculate the rows and cols of new picture
    CV_Assert(src1.type()==src2.type());
    int rows=src1.rows>src2.rows?src1.rows+15:src2.rows+15;
    int cols=src1.cols+10+src2.cols;
    
    // copy src into dst and put text on
    Mat dst = Mat::zeros(rows, cols, src1.type());
    src1.copyTo(dst(Rect(0,           15, src1.cols, src1.rows)));
    src2.copyTo(dst(Rect(src1.cols+10,15, src2.cols, src2.rows)));
    putText(dst, info, Point(180,12), CV_FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255) );
    
    // save and show
    //cout<<name<<endl;
    imwrite(name, dst);
    if (show_flag) {
	printf("                    show image \n");
        namedWindow("result", CV_WINDOW_AUTOSIZE);
        imshow("result", dst);
	waitKey(1);
    }
}


double get_similarity(const Mat& first,const Mat& second)
{
    //cout<<first <<endl;
    double dotSum = first.dot(second);  // inner product
    //printf("dot:%f    ",dotSum);
    double normFirst  = norm(first);    // nomalization
    double normSecond = norm(second); 
    //printf("norm:%f , %f    ",normFirst,normSecond);
    if(normFirst!=0 && normSecond!=0){
        return dotSum/(normFirst*normSecond);
    }
}


void preprocessing()
{
    for(int pair=0; pair<img_path_pairs.size(); pair++){

        printf("--[%d]--preprocessing------\n", pair);

        // get images name
        string img_path_1, img_path_2;
        ss << img_path_pairs[pair];
        ss >> img_path_1;
        ss >> img_path_2;

        // read images
        Mat image_1, image_2;
        image_1 = imread(img_path_1, CV_LOAD_IMAGE_COLOR );
        image_2 = imread(img_path_2, CV_LOAD_IMAGE_COLOR );
        if (image_1.empty() || image_2.empty()) {
            cerr << "Image data error.\n";
            FATAL;
        }

        // process and push images and names into fifo
        int name_s, name_e;
        string img_name_1, img_name_2;
        name_s = img_path_1.find_last_of("/"); 
        name_e = img_path_1.find_last_of("."); 
        img_name_1 = img_path_1.substr(name_s+1,name_e-name_s-1);
        name_s = img_path_2.find_last_of("/"); 
        name_e = img_path_2.find_last_of("."); 
        img_name_2 = img_path_2.substr(name_s+1,name_e-name_s-1);
        fifo_img_name_1.push(img_name_1);
        fifo_img_name_2.push(img_name_2);

        MyMat img_1, img_2;
        resize(image_1, img_1._mat, Size(224,224),0,0, CV_INTER_AREA);
        resize(image_2, img_2._mat, Size(224,224),0,0, CV_INTER_AREA);
        fifo_img_1.push(img_1);
        fifo_img_2.push(img_2);

        // process and store images into bin file
        img_1._mat = img_1._mat/2;
        img_2._mat = img_2._mat/2;
        ofstream input_bin_1, input_bin_2;
        string input_bin_name_1, input_bin_name_2;
        ss.clear();
        ss << "/dev/shm/input_" << pair%10 << "_1.bin ";
        ss << "/dev/shm/input_" << pair%10 << "_2.bin ";
        ss >> input_bin_name_1;
        ss >> input_bin_name_2;
        input_bin_1.open(input_bin_name_1.c_str(), ios::out | ios::binary);
        input_bin_2.open(input_bin_name_2.c_str(), ios::out | ios::binary);
        if (!input_bin_1 || !input_bin_2) {
            cerr << "failed to creat input data file" << endl;
            FATAL;
        }
        for(int i=0;i<img_1._mat.rows;i++){
            for(int j=0;j<img_1._mat.cols;j++){
                input_bin_1.write((char*)&(img_1._mat.at<Vec3b>(i,j)), 3*sizeof(char));
                input_bin_2.write((char*)&(img_2._mat.at<Vec3b>(i,j)), 3*sizeof(char));
            }
        }
        input_bin_1.close();
        input_bin_2.close();

        TIMING("--#%d#--[Process] finish time: %f\n"); 
    }
}


void dpu_calculate()
{
    for(int pair=0; pair<img_path_pairs.size(); pair++){

        printf("        --[%d]--dpu_calculating------\n",pair);

        // calculate with dpu ------------------------------------------------
        int inited;
        int inited_1, inited_2;
        string input_bin_name_1, input_bin_name_2;
        ss.clear();
        ss << "/dev/shm/input_" << pair%10 << "_1.bin ";
        ss << "/dev/shm/input_" << pair%10 << "_2.bin ";
        ss >> input_bin_name_1;
        ss >> input_bin_name_2;

        // write data into ddr -----------------------------------------------
        printf("                Write data into ddr.\n");
        printf("                Write input_1 into ddr.\n");
        test_dma_to_device(DMA_H2C_DEVICE,DATA1_DDR_ADDR,0x24c00,0,1,input_bin_name_1.c_str());
        printf("                Write input_2 into ddr.\n");
        test_dma_to_device(DMA_H2C_DEVICE,DATA2_DDR_ADDR,0x24c00,0,1,input_bin_name_2.c_str());
        TIMING("        --#%d#--[Write] finish time: %f\n"); 

        // write config to run dpu  ------------------------------------------
        printf("                Write config into GPIO \n");
        reg_write(DMA_REG_DEVICE,0x0000,0x0);               // ideal state: no config
        reg_write(DMA_REG_DEVICE,0x1000,0x0);               // ideal state: no config
        inited_1 = reg_read(DMA_REG_DEVICE,0x0000) & 0x1;   // check inited
        if (! inited_1){	
            printf("                dpu 1 not inited\n");	// if not, init
            // write 0x1 to init
            reg_write(DMA_REG_DEVICE,0x0000,0x1);
            while ( ! (reg_read(DMA_REG_DEVICE,0x0000) & 0x1)){
                // wait until read 0x1 which refers to inited
                usleep(100);
            }
            reg_write(DMA_REG_DEVICE,0x1000,0x0);           // unconfig
        }
        printf("                dpu 1 inited \n");
        inited_2 = reg_read(DMA_REG_DEVICE,0x1000) & 0x1;   // check inited
        if (! inited_2){	
            printf("                dpu 2 not inited\n");	// if not, init
            // write 0x1 to init
            reg_write(DMA_REG_DEVICE,0x1000,0x1);
            while ( ! (reg_read(DMA_REG_DEVICE,0x1000) & 0x1)){
                // wait until read 0x1 which refers to inited
                usleep(100);
            }
            reg_write(DMA_REG_DEVICE,0x1000,0x0);           // unconfig
        }
        printf("                dpu 2 inited \n");
        usleep(100);
        // write config to run
        reg_write(DMA_REG_DEVICE,0x0000,0x2);       // config to run
        reg_write(DMA_REG_DEVICE,0x1000,0x2);       // config to run
        usleep(100);                                // avoid read before really start
        printf("                    both running... \n");
        while ( (reg_read(DMA_REG_DEVICE,0x0000) & 0x2) ){
            // wait until read 0x01 which refers to finished
            usleep(100);
        }
        printf("                dpu 1 finished \n");
        while ( (reg_read(DMA_REG_DEVICE,0x1000) & 0x2) ){
            // wait until read 0x01 which refers to finished
            usleep(100);
        }
        printf("                dpu 2 finished \n");
        reg_write(DMA_REG_DEVICE,0x0000,0x0);       // return to ideal
        reg_write(DMA_REG_DEVICE,0x1000,0x0);       // return to ideal
        TIMING("        --#%d#--[DPU] finish time: %f\n"); 

        // read results from DPU ------------------------------------------------
        printf("                Read results from ddr\n");
        string out_bin_name_1, out_bin_name_2;
        ss.clear();
        ss << "/dev/shm/out_" << pair%10 << "_1.bin ";
        ss << "/dev/shm/out_" << pair%10 << "_2.bin ";
        ss >> out_bin_name_1;
        ss >> out_bin_name_2;
        test_dma_from_device(DMA_C2H_DEVICE,DATA1_DDR_ADDR+4608,4096,0,1, out_bin_name_1.c_str());
        test_dma_from_device(DMA_C2H_DEVICE,DATA2_DDR_ADDR+4608,4096,0,1, out_bin_name_2.c_str());
        TIMING("        --#%d#--[Read] finish time: %f\n"); 

    }
}


void result_output()
{
    for(int pair=0; pair<img_path_pairs.size(); pair++){

        printf("                --[%d]--result_output------\n",pair);

        // calculate the result and save into file-------------------------------
        printf("                        Calculate the result\n");

        // read results from out files
        Mat result_1(1,4096, CV_8UC1);
        Mat result_2(1,4096, CV_8UC1);
        string out_bin_name_1, out_bin_name_2;
        ss.clear();
        ss << "/dev/shm/out_" << pair%10 << "_1.bin ";
        ss << "/dev/shm/out_" << pair%10 << "_2.bin ";
        ss >> out_bin_name_1;
        ss >> out_bin_name_2;
        ifstream out_bin_1(out_bin_name_1.c_str(), ios::in | ios::binary);
        ifstream out_bin_2(out_bin_name_2.c_str(), ios::in | ios::binary);
        if (!out_bin_1 || !out_bin_2) {
            cerr << "failed to open out result file" << endl;
            FATAL;
        }
        out_bin_1.read((char*)result_1.data, 4096*sizeof(char));        
        out_bin_2.read((char*)result_2.data, 4096*sizeof(char));        
        out_bin_1.close();
        out_bin_2.close();
        printf("                        Read results from output file.\n");

        // calculate the similarity
        double cos = get_similarity(result_1, result_2); 
        printf("                ##[cos]: %f \n", cos);
        string result = "different";
        if (cos > THRESHOLD){
            result = "same"; 
        }
        printf("                        Save final result into result file.\n");

        // save final result into result file
        string img_name_1_t, img_name_2_t; 
        MyMat img_1_t, img_2_t;
        img_name_1_t = fifo_img_name_1.front();
        img_name_2_t = fifo_img_name_2.front();
        fifo_img_name_1.pop();
        fifo_img_name_2.pop();
        img_1_t = fifo_img_1.front();
        img_2_t = fifo_img_2.front();
        fifo_img_1.pop();
        fifo_img_2.pop();
        output_results << img_name_1_t <<"  "<< img_name_2_t << "  " << result << endl;

        TIMING("                --#%d#--[Result] finish time: %f\n");

        // save result in image and show -----------------------------------------
        if (save_flag) {
            printf("                        Save the image\n");
            save_img(img_1_t._mat,img_2_t._mat, result, 
                        "output/"+img_name_1_t+img_name_2_t+".jpg", show_flag);

            TIMING("                --#%d#--[Image] finish time %f\n"); 
        }

    }

}
