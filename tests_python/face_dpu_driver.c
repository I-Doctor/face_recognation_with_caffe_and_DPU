// d52cbaca0ef8cf4fd3d6354deb5066970fb6511d02d18d15835e6014ed847fb0
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
#include <unistd.h>

//#include <cv.h>  
//#include <highgui.h> 

#include <unistd.h>
#include <math.h>

/* ltoh: little to host */
/* htol: little to host */
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
  
#define FATAL do { fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); exit(1); } while(0)
#define MAP_SIZE (32*1024UL)
#define MAP_MASK (MAP_SIZE - 1)
#define INST1_DDR_ADDR 0x6D000000
#define DATA1_DDR_ADDR 0x6A000000
#define WEIT1_DDR_ADDR 0x60000000
#define INST2_DDR_ADDR 0xED000000
#define DATA2_DDR_ADDR 0xEA000000
#define WEIT2_DDR_ADDR 0xE0000000

static int verbosity = 0;
static int read_back = 0;
static int allowed_accesses = 1;


char* result_before_fname = "/dev/shm/mem13.bin";
char* result_after_fname = "./123.bin";


static struct option const long_opts[] =
{
    {"device", required_argument, NULL, 'd'},
    {"address", required_argument, NULL, 'a'},
    {"size", required_argument, NULL, 's'},
    {"offset", required_argument, NULL, 'o'},
    {"count", required_argument, NULL, 'c'},
    {"file", required_argument, NULL, 'f'},
    {"verbose", no_argument, NULL, 'v'},
    {"help", no_argument, NULL, 'h'},
    {0, 0, 0, 0}
};
const int PRL=8;


int ceil_to(int x, int y)
{
    return x % y ? x + y - x % y : x;
}


static int test_dma_to_device(char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, char *filename);
static int test_dma_from_device(char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, char *filename);
static int reg_read(char *devicename , uint32_t addr);
static int reg_write(char *devicename , uint32_t addr,uint32_t writeval);
float * array_reshape(char * src, int length, int row, int col, int channel);
float * file_reshape(int fid, int row, int col, int channel);


static uint32_t getopt_integer(char *optarg)
{
    int rc;
    uint32_t value;
    rc = sscanf(optarg, "0x%x", &value);
    if (rc <= 0)
    rc = sscanf(optarg, "%ul", &value);
    //printf("sscanf() = %d, value = 0x%08x\n", rc, (unsigned int)value);
    return value;
}

/* Subtract timespec t2 from t1
 *
 * Both t1 and t2 must already be normalized
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

int main(int argc, char* argv[]){

    struct timeval start,end;  
    gettimeofday(&start, NULL );  
    long timeuse;

    // initial when argv[1]=='1'
    if(argc==2 && *(argv[1])=='1') {
        // init =======================================================
        printf("      Write init weights and insts into ddr.\n");
        test_dma_to_device("/dev/xdma0_h2c_0", WEIT1_DDR_ADDR, 23859120,0,1, "../weight/concat_svd_weight.bin");
        test_dma_to_device("/dev/xdma0_h2c_0", WEIT2_DDR_ADDR, 23859120,0,1, "../weight/concat_svd_weight.bin");
        printf("      weight");
        test_dma_to_device("/dev/xdma0_h2c_0", INST1_DDR_ADDR, 257644,
        0,1, "../weight/concat_svd_instr.bin");
        test_dma_to_device("/dev/xdma0_h2c_0", INST2_DDR_ADDR, 257644,
        0,1, "../weight/concat_svd_instr.bin");
        printf("      instr \n");
        printf("    ## Finished.\n");

        gettimeofday(&end, NULL );  
        timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
                + end.tv_usec - start.tv_usec;  
        printf("    ## init time: %f\n",timeuse /1000000.0); 
        return 0;
    }

    int inited;
    int inited_1, inited_2;
    // calculate ======================================================
    // write data into ddr
    printf("      Write input_1 into ddr.\n");
    test_dma_to_device("/dev/xdma0_h2c_0", DATA1_DDR_ADDR, 0x24c00, 
    0, 1, "/dev/shm/input_1.bin");
    printf("      Write input_2 into ddr.\n");
    test_dma_to_device("/dev/xdma0_h2c_0", DATA2_DDR_ADDR, 0x24c00, 
    0, 1, "/dev/shm/input_2.bin");

    gettimeofday(&end, NULL );  
    timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
            + end.tv_usec - start.tv_usec;  
    printf("    ## write data time: %f\n",timeuse /1000000.0); 

    // write config to check inited
    printf("      Write config into GPIO \n");
    reg_write("/dev/xdma0_user",0x0000,0x0);  // ideal state: no config
    reg_write("/dev/xdma0_user",0x1000,0x0);  // ideal state: no config
    inited_1 = reg_read("/dev/xdma0_user",0x0000) & 0x1;// check inited
    if (! inited_1){	
        printf("      dpu 1 not inited\n");	    // if not, init
        // write 0x1 to init
        reg_write("/dev/xdma0_user",0x0000,0x1);
        while ( ! (reg_read("/dev/xdma0_user",0x0000) & 0x1)){
            // wait until read 0x1 which refers to inited
            usleep(100);
        }
        reg_write("/dev/xdma0_user",0x1000,0x0); // unconfig
    }
    printf("      dpu 1 inited \n");
    inited_2 = reg_read("/dev/xdma0_user",0x1000) & 0x1;// check inited
    if (! inited_2){	
        printf("      dpu 2 not inited\n");	    // if not, init
        // write 0x1 to init
        reg_write("/dev/xdma0_user",0x1000,0x1);
        while ( ! (reg_read("/dev/xdma0_user",0x1000) & 0x1)){
            // wait until read 0x1 which refers to inited
            usleep(100);
        }
        reg_write("/dev/xdma0_user",0x1000,0x0); // unconfig
    }
    printf("      dpu 2 inited \n");
    usleep(100);
    // write config to run
    reg_write("/dev/xdma0_user",0x0000,0x2);     // config to run
    reg_write("/dev/xdma0_user",0x1000,0x2);     // config to run
    usleep(100); // avoid read before really start running
    printf("      both running... \n");
    while ( (reg_read("/dev/xdma0_user",0x0000) & 0x2) ){
        // wait until read 0x01 which refers to finished
        usleep(100);
    }
    printf("      dpu 1 finished \n");
    while ( (reg_read("/dev/xdma0_user",0x1000) & 0x2) ){
        // wait until read 0x01 which refers to finished
        usleep(100);
    }
    printf("      dpu 2 finished \n");
    reg_write("/dev/xdma0_user",0x0000,0x0);     // return to ideal
    reg_write("/dev/xdma0_user",0x1000,0x0);     // return to ideal

    gettimeofday(&end, NULL );  
    timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
            + end.tv_usec - start.tv_usec;  
    printf("    ## runing time=%f\n",timeuse /1000000.0); 

    printf("      Read results from ddr\n");
    test_dma_from_device("/dev/xdma0_c2h_0", DATA1_DDR_ADDR + 4608, 
    4096, 0, 1, "/dev/shm/out_1.bin");
    test_dma_from_device("/dev/xdma0_c2h_0", DATA2_DDR_ADDR + 4608, 
    4096, 0, 1, "/dev/shm/out_2.bin");

    gettimeofday(&end, NULL );  
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    printf("    ## read time: %f\n",timeuse /1000000.0); 

    return 0;
}


static int test_dma_to_device(char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, char *filename)
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


static int test_dma_from_device(char *devicename, uint32_t addr, uint32_t size, uint32_t offset, uint32_t count, char *filename)
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



static int reg_read(char *devicename , uint32_t addr) {
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
  virt_addr = map_base + target;
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


static int reg_write(char *devicename , uint32_t addr,uint32_t writeval) {
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
  virt_addr = map_base + target;
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

/* one core
    // calculate input_1===============================================
    printf("      Write input_1 into ddr.\n");
    test_dma_to_device("/dev/xdma0_h2c_0", DATA1_DDR_ADDR+, 0x24c00 ,0,1,"/dev/shm/input_1.bin");

    gettimeofday(&end, NULL );  
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    printf("    ## write 1 time: %f\n",timeuse /1000000.0); 

    printf("      Write config into GPIO\n");
    reg_write("/dev/xdma0_user",0x0,0x0);						        // ideal state: no config
    inited = reg_read("/dev/xdma0_user",0x0) & 0x00000001;	// check if inited
    if (! inited){	
        printf("      not inited\n");								            // if not, init
        // write 0x1 to init
        reg_write("/dev/xdma0_user",0x0,0x1);
        while ( ! (reg_read("/dev/xdma0_user",0x0) & 0x1)){		// wait until read 0x01 which refers to inited
            usleep(100);
        }
        reg_write("/dev/xdma0_user",0x0,0x0);					        // unconfig
    }
    printf("      inited\n");
    usleep(100);
    reg_write("/dev/xdma0_user",0x0,0x2);						        // config to run
    usleep(100);												                    // avoid read before really start running
    printf("      running 1 \n");
    while ( (reg_read("/dev/xdma0_user",0x0) & 0x2) ){			// wait until read 0x01 which refers to finished
        usleep(100);
    }
    reg_write("/dev/xdma0_user",0x0,0x0);						        // return to ideal: no config

    gettimeofday(&end, NULL );  
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    printf("    ## run 1 time=%f\n",timeuse /1000000.0); 

    printf("      Read out_1 from ddr\n");
    test_dma_from_device("/dev/xdma0_c2h_0", DATA_DDR_ADDR+(0x6A000000-0x60000000) +4608, 4096 ,0,1,"/dev/shm/out_1.bin");

    gettimeofday(&end, NULL );  
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    printf("    ## read 1 time: %f\n",timeuse /1000000.0); 


    // calculate input_2 ========================================================================================
    printf("      Write input_2 into ddr.\n");
    test_dma_to_device("/dev/xdma0_h2c_0", DATA_DDR_ADDR+(0x6A000000-0x60000000)+0, 0x24c00 ,0,1,"/dev/shm/input_2.bin");

    gettimeofday(&end, NULL );  
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    printf("    ## write 2 time: %f\n",timeuse /1000000.0); 

    printf("     Write config into GPIO\n");
    reg_write("/dev/xdma0_user",0x0,0x0);						        // ideal state: no config
    inited = reg_read("/dev/xdma0_user",0x0) & 0x00000001;	// check if inited
    if (! inited){	
        printf("      not inited\n");								            // if not, init
        // write 0x1 to init
        reg_write("/dev/xdma0_user",0x0,0x1);
        while ( ! (reg_read("/dev/xdma0_user",0x0) & 0x1)){		// wait until read 0x1 which refers to inited
            usleep(100);
        }
        reg_write("/dev/xdma0_user",0x0,0x0);					        // unconfig
    }
    printf("      inited\n");
    usleep(100);
    reg_write("/dev/xdma0_user",0x0,0x2);						        // config to run
    usleep(100);												                    // avoid read before really start running
    printf("      running\n");
    while ( (reg_read("/dev/xdma0_user",0x0) & 0x2) ){			// wait until read 0x01 which refers to finished
        usleep(100);
    }
    reg_write("/dev/xdma0_user",0x0,0x0);					        	// return to ideal: no config

    gettimeofday(&end, NULL );  
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    printf("    ## run 2 time: %f\n",timeuse /1000000.0); 

    printf("      Read out_2 from ddr\n");
    test_dma_from_device("/dev/xdma0_c2h_0", DATA_DDR_ADDR+(0x6A000000-0x60000000) + 4608, 4096 ,0,1,"/dev/shm/out_2.bin");

    gettimeofday(&end, NULL );  
    timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;  
    printf("    ## read 2 time=%f\n",timeuse /1000000.0); 
*/
