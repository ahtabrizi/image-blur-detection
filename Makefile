# Define the compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall

# Linker flags
LDFLAGS = `pkg-config --cflags --libs opencv4`

# Define the source file
SRC = image_blurr_detection.cpp

# Define the executable output
TARGET = image_blurr_detection

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)