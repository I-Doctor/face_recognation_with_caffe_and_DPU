CP  = g++

all: face_dpu_multithread

face_dpu_multithread: face_dpu_multithread.o
	$(CP) -o $@ $^ -lopencv_core -lopencv_imgproc -lopencv_highgui -lpthread 

%.o: %.c
	$(CC) -c -std=c99 -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE -g 

%.o: %.cpp
	$(CP) -o $@ -c $< 

clean:
	rm -rf *.o *.bin face_dpu_multithread

clear:
	rm -rf *.o 
