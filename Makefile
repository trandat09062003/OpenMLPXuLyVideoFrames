CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv 2>/dev/null || echo -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui)

TARGET = video_omp
SRC = video_openmp.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET) *.o test_out.mp4 output_processed.mp4 benchmark_result.png
