CC = g++
TARGET = batch_process iteration_process
SRCS = batch_process.cpp iteration_process.cpp
OBJS = $(SRCS:.cpp=.o)
DLIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui

all: $(TARGET)

$(TARGET): %: %.o
	$(CC) -o $@ $^ $(DLIBS)  

$(OBJS): %.o: %.cpp
	$(CC) -o $@ -c $<

clean:
	rm -rf $(TARGET) $(OBJS)
