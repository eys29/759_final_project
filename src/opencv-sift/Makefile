CXX = g++
CXXFLAGS = -I/disk/zli2793/tools/miniforge3/envs/ece759/include/opencv4/
LDFLAGS = -L/disk/zli2793/tools/miniforge3/envs/ece759/lib 
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_xfeatures2d

TARGET = bin/main
SRC = main.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): mkdir_bin $(SRC) 
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

mkdir_bin:
	mkdir -p bin

clean:
	rm -f $(TARGET)
