# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I. # -I. tells compiler to look for headers in current dir

# Linker flags (if any)
LDFLAGS =

# Target executable name
TARGET = au

# Source files
SOURCES = main.cpp tensor.cpp Activations.cpp

# Object files (derived from source files)
OBJECTS = $(SOURCES:.cpp=.o)

# Default target: build the executable
all: $(TARGET)

# Rule to link the executable from object files
$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ -o $@

# Rule to compile .cpp files into .o files
# %.o depends on its .cpp file and the tensor.h header
%.o: %.cpp tensor.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Target to clean up build files
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean