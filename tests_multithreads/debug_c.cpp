//=====================================================================
// (c) Copyright EFC of NICS; Tsinghua University. All rights reserved.
// Author   : Kai Zhong
// Email    : zhongk15@mails.tsinghua.edu.cn
// Create Date   : 2019.03.20
// File Name     : dpu_face_application.cpp
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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
/* Usage address of dpu */
#define INST1_DDR_ADDR 0x6D000000
#define DATA1_DDR_ADDR 0x6A000000
#define WEIT1_DDR_ADDR 0x60000000
#define INST2_DDR_ADDR 0xED000000
#define DATA2_DDR_ADDR 0xEA000000
#define WEIT2_DDR_ADDR 0xE0000000
double THRESHOLD = 0.29;

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
  
#define FATAL do { \
    fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); \
    exit(1); \
    } while(0)
#define MAP_SIZE (32*1024UL)
#define MAP_MASK (MAP_SIZE - 1)


//---------------------------------------------------------------------
// declare
//---------------------------------------------------------------------
static bool save_flag = 0;
static bool show_flag = 0;
static int test_dma_to_device(const char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, const char *filename);
static int test_dma_from_device(const char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, const char *filename);
static int reg_read(const char *devicename , uint32_t addr);
static int reg_write(const char *devicename , uint32_t addr,uint32_t writeval);
static void timespec_sub(struct timespec *t1, const struct timespec *t2);
static void save_img(Mat &src1, Mat &src2, string info, string name, bool show_flag);
static double getSimilarity(const Mat& first, const Mat& second);
//float * array_reshape(const char * src, int length, int row, int col, int channel);
//float * file_reshape(int fid, int row, int col, int channel);
//static uint32_t getopt_integer(const char *optarg);
//int ceil_to(int x, int y);
/*
typedef Point3_<uint8_t> Pixel;
struct Operator_1 {
    void operator_1 ()(Pixel &pixel, const int * position) const
    {    
        if (position[0]==position[1]){
            cout<<pixel.x<<endl;
        }
    }
};*/
Mat   BGRToRGB(Mat img) { 
    Mat image(img.rows, img.cols, CV_8UC3); 
    for(int i=0; i<img.rows; ++i) { //获取第i行首像素指针 
        Vec3b *p1 = img.ptr<Vec3b>(i); 
        Vec3b *p2 = image.ptr<Vec3b>(i); 
        for(int j=0; j<img.cols; ++j) { //将img的bgr转为image的rgb 
            p2[j][2] = p1[j][0]; 
            p2[j][1] = p1[j][1]; 
            p2[j][0] = p1[j][2]; 
        } 
    } 
    return image; 
}


//---------------------------------------------------------------------
// main
//---------------------------------------------------------------------
int main(int argc, char* argv[]){

    // check arguements
    if (argc == 3){
        printf("=====Start application WITHOUT saving or showing: ======\n");
        printf("=====process %s and write results into %s \n",argv[1],argv[2]);
    }
    else if (argc==4 && 0==strcmp(argv[3],"save")){
        save_flag = 1;
        printf("=====Start application WITH saving image WITHOUT showing: ======\n");
        printf("=====process %s and write results into %s \n",argv[1],argv[2]);
    }
    else if (argc==5 && 0==strcmp(argv[4],"show")){
        save_flag = 1;
        show_flag = 1;
        printf("=====Start application WITH saving image WITH showing: =====\n");
        printf("=====process %s and write results into %s \n",argv[1],argv[2]);
    }
    else {
        printf("=====Please give the arguements as follow: \n");
        printf("=====basic       : ./dpu_face_app input_list.txt results.txt\n");
        printf("=====save image  : ./dpu_face_app input_list.txt results.txt save\n");
        printf("=====save & show : ./dpu_face_app input_list.txt results.txt save show\n");
        return 0;
    }

    struct timeval start,end;  
    gettimeofday(&start, NULL);  
    long timeuse;

    // start---------------------------------------------------------------
    // config dpu
    printf("  -@@Write init weights and insts into ddr.");
    test_dma_to_device("/dev/xdma0_h2c_0", WEIT1_DDR_ADDR, 23859120,0,1, "../../weight/concat_svd_weight.bin");
    test_dma_to_device("/dev/xdma0_h2c_0", WEIT2_DDR_ADDR, 23859120,0,1, "../../weight/concat_svd_weight.bin");
    printf("    weight_ok");
    test_dma_to_device("/dev/xdma0_h2c_0", INST1_DDR_ADDR, 257644, 0,1, "../../weight/concat_svd_instr.bin");
    test_dma_to_device("/dev/xdma0_h2c_0", INST2_DDR_ADDR, 257644, 0,1, "../../weight/concat_svd_instr.bin");
    printf("    instr_ok \n");

    // read input list into vector pic_pairs
    printf("    Read input list into vector.\n");
    ifstream input_list(argv[1]); 
    if(!input_list) { 
        cerr << "Can't open the file.\n"; 
        FATAL; 
    }
    string line; 
    vector<string> pic_pairs; 
    while(getline(input_list, line)) 
        pic_pairs.push_back(line);
    input_list.close();
    // open output results file
    ofstream output_results(argv[2]); 
    if(!output_results) { 
        cerr << "Can't open output file.\n"; 
        FATAL; 
    }

    gettimeofday(&end, NULL );  
    timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
            + end.tv_usec - start.tv_usec;  
    printf("   ##Finish. [Init] time: %f\n",timeuse /1000000.0); 

    // calculate---------------------------------------------------------------
    // calculating pairs
    for(int pair=0; pair<pic_pairs.size(); pair++){

        printf("  -@@Start calculating pair [%d]--------------------- .\n", pair);

        // preprocessing-------------------------------------------------------
        printf("     -@@Start preprocessing.\n");
        string picture_1, picture_2;
        stringstream input(pic_pairs[pair]);
        input>>picture_1;
        input>>picture_2;
        // read images
        Mat image_1, image_2;
        image_1 = imread(picture_1, CV_LOAD_IMAGE_COLOR );
        image_2 = imread(picture_2, CV_LOAD_IMAGE_COLOR );
        if (image_1.empty() || image_2.empty()) {
            cerr << "Image data error.\n";
            FATAL;
        }
        //cout<<(CV_8U==image_1.depth())<<endl;
        //cout<<(CV_8UC3==image_1.type())<<endl;
        //cout<<(3==image_1.elemSize())<<endl;
        // resize and store images
        Mat img_1, img_2;
        resize(image_1, img_1, Size(224,224),0,0,CV_INTER_AREA);
        resize(image_2, img_2, Size(224,224),0,0,CV_INTER_AREA);
        img_1 = img_1/2;
        img_2 = img_2/2;
        ofstream input_1, input_2;
        input_1.open("/dev/shm/input_0_1.bin", ios::out | ios::binary);
        input_2.open("/dev/shm/input_0_2.bin", ios::out | ios::binary);
        if (!input_1 || !input_2) {
            cerr << "failed to creat input data file" << endl;
            FATAL;
        }
        for(int i=0;i<img_1.rows;i++){
            for(int j=0;j<img_1.cols;j++){
                //cout<<i<<","<<j<<" "<<(int)(img_1.at<Vec3b>(i,j)[2])<<endl;
                //cout<<sizeof(img_1.at<Vec3b>(i,j)[2])<<endl;
                input_1.write((char*)&(img_1.at<Vec3b>(i,j)), 3*sizeof(char));
                input_2.write((char*)&(img_2.at<Vec3b>(i,j)), 3*sizeof(char));
            }
        }
        input_1.close();
        input_2.close();

        gettimeofday(&end, NULL );  
        timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
                + end.tv_usec - start.tv_usec;  
        printf("      ##Finish. [Process] time: %f\n",timeuse /1000000.0); 

        //image_1.forEach<Pixel>(Operator_1());
        //cvtColor(image1, image2, CV_RGB2GRAY);
        /*
        namedWindow("image_1", CV_WINDOW_AUTOSIZE);
        namedWindow("img_1", CV_WINDOW_AUTOSIZE);
        imshow("image_1", image_1);
        imshow("img_1", img_1);
        */

        // calculate with dpu ------------------------------------------------
        int inited;
        int inited_1, inited_2;
        // write data into ddr ------------------------------------------------
        printf("     -@@Write data into ddr.\n");
        printf("       Write input_1 into ddr.\n");
        test_dma_to_device("/dev/xdma0_h2c_0",DATA1_DDR_ADDR,0x24c00,0,1,"/dev/shm/input_0_1.bin");
        printf("       Write input_2 into ddr.\n");
        test_dma_to_device("/dev/xdma0_h2c_0",DATA2_DDR_ADDR,0x24c00,0,1,"/dev/shm/input_0_2.bin");

        gettimeofday(&end, NULL );  
        timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
                + end.tv_usec - start.tv_usec;  
        printf("      ##Write [data] time: %f\n",timeuse /1000000.0); 

        // write config to check inited -----------------------------------------------
        printf("     -@@Write config into GPIO \n");
        reg_write("/dev/xdma0_user",0x0000,0x0);  // ideal state: no config
        reg_write("/dev/xdma0_user",0x1000,0x0);  // ideal state: no config
        inited_1 = reg_read("/dev/xdma0_user",0x0000) & 0x1;// check inited
        if (! inited_1){	
            printf("       dpu 1 not inited\n");	    // if not, init
            // write 0x1 to init
            reg_write("/dev/xdma0_user",0x0000,0x1);
            while ( ! (reg_read("/dev/xdma0_user",0x0000) & 0x1)){
                // wait until read 0x1 which refers to inited
                usleep(100);
            }
            reg_write("/dev/xdma0_user",0x1000,0x0); // unconfig
        }
        printf("       dpu 1 inited \n");
        inited_2 = reg_read("/dev/xdma0_user",0x1000) & 0x1;// check inited
        if (! inited_2){	
            printf("       dpu 2 not inited\n");	    // if not, init
            // write 0x1 to init
            reg_write("/dev/xdma0_user",0x1000,0x1);
            while ( ! (reg_read("/dev/xdma0_user",0x1000) & 0x1)){
                // wait until read 0x1 which refers to inited
                usleep(100);
            }
            reg_write("/dev/xdma0_user",0x1000,0x0); // unconfig
        }
        printf("       dpu 2 inited \n");
        usleep(100);
        // write config to run
        reg_write("/dev/xdma0_user",0x0000,0x2);     // config to run
        reg_write("/dev/xdma0_user",0x1000,0x2);     // config to run
        usleep(100); // avoid read before really start running
        printf("       both running... \n");
        while ( (reg_read("/dev/xdma0_user",0x0000) & 0x2) ){
            // wait until read 0x01 which refers to finished
            usleep(100);
        }
        printf("       dpu 1 finished \n");
        while ( (reg_read("/dev/xdma0_user",0x1000) & 0x2) ){
            // wait until read 0x01 which refers to finished
            usleep(100);
        }
        printf("       dpu 2 finished \n");
        reg_write("/dev/xdma0_user",0x0000,0x0);     // return to ideal
        reg_write("/dev/xdma0_user",0x1000,0x0);     // return to ideal

        gettimeofday(&end, NULL );  
        timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
                + end.tv_usec - start.tv_usec;  
        printf("      ##Runing [DPU] time=%f\n",timeuse /1000000.0); 

        // read results from DPU ------------------------------------------------
        printf("     -@@Read results from ddr\n");
        test_dma_from_device("/dev/xdma0_c2h_0", DATA1_DDR_ADDR + 4608, 
        4096, 0, 1, "/dev/shm/out_0_1.bin");
        test_dma_from_device("/dev/xdma0_c2h_0", DATA2_DDR_ADDR + 4608, 
        4096, 0, 1, "/dev/shm/out_0_2.bin");

        gettimeofday(&end, NULL );  
        timeuse = 1000000 * ( end.tv_sec - start.tv_sec )
                + end.tv_usec - start.tv_usec;  
        printf("      ##Read [data] time: %f\n",timeuse /1000000.0); 

        // calculate the result and save into file-------------------------------
        printf("     -@@Calculate the result\n");
        // read results from out files
        Mat result_1(1,4096, CV_8UC1);
        Mat result_2(1,4096, CV_8UC1);
        ifstream input_result_1("/dev/shm/out_0_1.bin", ios::in | ios::binary);
        ifstream input_result_2("/dev/shm/out_0_2.bin", ios::out | ios::binary);
        if (!input_result_1 || !input_result_2) {
            cerr << "failed to open out result file" << endl;
            FATAL;
        }
        input_result_1.read((char*)result_1.data, 4096*sizeof(char));        
        input_result_2.read((char*)result_2.data, 4096*sizeof(char));        
        input_result_1.close();
        input_result_2.close();
        Mat result_1_f(1,4096, CV_32FC1);
        Mat result_2_f(1,4096, CV_32FC1);
        result_1.convertTo(result_1_f, CV_32FC1);
        result_2.convertTo(result_2_f, CV_32FC1);
        //cout<<result_1<<endl;
        printf("       Read results from output file.\n");
        double cos = getSimilarity(result_1, result_2); 
        printf("   ##[cos]: %f \n", cos);
        string result = "different";
        if (cos > THRESHOLD){
            result = "same"; 
        }
        printf("       Save final result into result file.\n");
        // save final result into result file
        int name_s, name_e;
        name_s = picture_1.find_last_of("/"); 
        name_e = picture_1.find_last_of("."); 
        picture_1 = picture_1.substr(name_s+1,name_e-name_s-1);
        name_s = picture_2.find_last_of("/"); 
        name_e = picture_2.find_last_of("."); 
        picture_2 = picture_2.substr(name_s+1,name_e-name_s-1);
        output_results << picture_1 <<"  "<< picture_2 << "  " << result << endl;

        gettimeofday(&end, NULL );  
        timeuse = 1000000 * ( end.tv_sec - start.tv_sec )
                + end.tv_usec - start.tv_usec; 
        printf("      ##Calculate [result] time: %f\n",timeuse /1000000.0);

        // save result in image and show -----------------------------------------
        if (save_flag) {
            printf("     -@@Save the image\n");
            save_img(img_1,img_2, result, "output/"+picture_1+picture_2+".jpg", show_flag);
            gettimeofday(&end, NULL ); 
            timeuse = 1000000 * ( end.tv_sec - start.tv_sec )
                    + end.tv_usec - start.tv_usec;  
            printf("      ##Save [image] time %f\n",timeuse /1000000.0); 
        }
    }

    // finish ---------------------------------------------------------------
    output_results.close();
    gettimeofday(&end, NULL );  
    timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
            + end.tv_usec - start.tv_usec;  
    printf("=====Finish. [TOTAL] time: %f\n",timeuse /1000000.0); 

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
  struct timespec ts_start, ts_end;

  posix_memalign((void **)&allocated, 4096/*alignment*/, size + 4096);
  assert(allocated);
  buffer = allocated + offset;
  //printf("host memory buffer = %p\n", buffer);

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
  
    //rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);
    /* write buffer to AXI MM address using SGDMA */
    rc = write(fpga_fd, buffer, size);
    assert(rc == size);
    //rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);
  }
  /* subtract the start time from the end time */
  //timespec_sub(&ts_end, &ts_start);
  /* display passed time, a bit less accurate but side-effects are accounted for */
  //printf("CLOCK_MONOTONIC reports %ld.%09ld seconds (total) for last transfer of %d bytes\n",
  //ts_end.tv_sec, ts_end.tv_nsec, size);
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
  struct timespec ts_start, ts_end;

  posix_memalign((void **)&allocated, 4096/*alignment*/, size + 4096);
  assert(allocated);
  buffer = allocated + offset;
  //printf("host memory buffer = %p\n", buffer);

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
    //rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);
    /* read data from AXI MM into buffer using SGDMA */
    rc = read(fpga_fd, buffer, size);
    if ((rc > 0) && (rc < size)) {
      //printf("Short read of %d bytes into a %d bytes buffer, could be a packet read?\n", rc, size);
    }
    
    //rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);
    /* file argument given? */
    if ((file_fd >= 0)) {
      /* write buffer to file */
      rc = write(file_fd, buffer, size);
      assert(rc == size);
    }
  }
  /* subtract the start time from the end time */
  //timespec_sub(&ts_end, &ts_start);
  /* display passed time, a bit less accurate but side-effects are accounted for */
  //printf("CLOCK_MONOTONIC reports %ld.%09ld seconds (total) for last transfer of %d bytes\n", ts_end.tv_sec, ts_end.tv_nsec, size);
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
//  printf("read device: %s\n", device);
  target = addr;
//  printf("  address: 0x%08x\n", (unsigned int)target);
  access_width = 'w';
  
  if ((fd = open(devicename, O_RDWR | O_SYNC)) == -1) FATAL;
//  printf("  character device %s opened and fd=%d .\n", devicename, fd);
  //fflush(stdout);
  
  /* map one page */
  map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map_base == (void *) -1) FATAL;
//  printf("  Memory mapped at address %p.\n", map_base); 
  //fflush(stdout);
  
  /* calculate the virtual address to be accessed */
  virt_addr = (char*)map_base + target;
  /* read only */
  
//  printf("Read 32-bit value at address 0x%08x (%p): 0x%08x\n", (unsigned int)target, virt_addr, *((uint32_t *) virt_addr));
  read_result = *((uint32_t *) virt_addr);
  /* swap 32-bit endianess if host is not little-endian */
  read_result = ltohl(read_result);
//  printf("actualy read value at address 0x%08x (%p): 0x%08x\n", (unsigned int)target, virt_addr, (unsigned int)read_result);   
  //fflush(stdout);
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
//  printf("write device: %s\n", device);
  target = addr;
//  printf("  address: 0x%08x\n", (unsigned int)target);

  if ((fd = open(devicename, O_RDWR | O_SYNC)) == -1) FATAL;
  //close(fd);
  //fd = open(devicename, O_RDWR | O_SYNC);
//  printf("  character device %s opened and fd=%d .\n", devicename, fd); 
  //fflush(stdout);

  /* map one page */
  map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map_base == (void *) -1) FATAL;
//    printf("  Memory mapped at address %p.\n", map_base); 
  //fflush(stdout);

  /* calculate the virtual address to be accessed */
  virt_addr = (char*)map_base + target; //cast to char* to calculate
  /* data value given, i.e. writing? */
  
//  printf("  Write 32-bits value 0x%08x to 0x%08x (%p)\n", (uint32_t)writeval, (unsigned int)target, virt_addr); 
  /* swap 32-bit endianess if host is not little-endian */
  writeval = htoll(writeval);  
  *((uint32_t *) virt_addr) = writeval;        
  //*((int *) virt_addr) = 100; printf("  write int 100 and it's actualy: %d \n", *((int *) virt_addr));
//	printf("  actualy write value 0x%08x to 0x%08x (%p)\n", *((uint32_t*)virt_addr), (unsigned int)target, virt_addr); 
  //fflush(stdout);
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
    //printf("sscanf() = %d, value = 0x%08x\n", rc, (unsigned int)value);
    return value;
}

/* Subtract timespec t2 from t1. Both t1 and t2 must already be normalized
 * i.e. 0 <= nsec < 1000000000 */
static void timespec_sub(struct timespec *t1, const struct timespec *t2)
{
    assert(t1->tv_nsec >= 0);
    assert(t1->tv_nsec < 1000000000);
    assert(t2->tv_nsec >= 0);
    assert(t2->tv_nsec < 1000000000);
    t1->tv_sec -= t2->tv_sec;
    t1->tv_nsec -= t2->tv_nsec;
    if (t1->tv_nsec >= 1000000000)
    {
        t1->tv_sec++;
        t1->tv_nsec -= 1000000000;
    }
    else if (t1->tv_nsec < 0)
    {
        t1->tv_sec--;
        t1->tv_nsec += 1000000000;
    }
}

int ceil_to(int x, int y)
{
    return x % y ? x + y - x % y : x;
}


void save_img(Mat &src1, Mat &src2, string info, string name, bool show_flag)
{
    // calculate the rows and cols of new picture
    CV_Assert(src1.type()==src2.type());
    int rows=src1.rows>src2.rows?src1.rows+15:src2.rows+15;
    int cols=src1.cols+10+src2.cols;
    src1 = src1 * 2;
    src2 = src2 * 2;
    
    // copy src into dst and put text on
    Mat dst = Mat::zeros(rows, cols, src1.type());
    src1.copyTo(dst(Rect(0,           15, src1.cols, src1.rows)));
    src2.copyTo(dst(Rect(src1.cols+10,15, src2.cols, src2.rows)));
    putText(dst, info, Point(180,12), CV_FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255) );
    
    // save and show
    cout<<name<<endl;
    imwrite(name, dst);
    if (show_flag) {
	printf("       show image \n");
        namedWindow("result", CV_WINDOW_AUTOSIZE);
        imshow("result", dst);
	waitKey(1);
    }
}


double getSimilarity(const Mat& first,const Mat& second)
{
    //cout<<first <<endl;
    double dotSum = first.dot(second);//内积
    //printf("dot:%f    ",dotSum);
    double normFirst  = norm(first);//取模
    double normSecond = norm(second); 
    //printf("norm:%f , %f    ",normFirst,normSecond);
    if(normFirst!=0 && normSecond!=0){
        return dotSum/(normFirst*normSecond);
    }
}

