# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Werror -I.. -I../BaseTensor -I../AutoGrad $(INCLUDE)

# OpenBLAS include and libs
INCLUDE = -I/opt/homebrew/Cellar/openblas/0.3.30/include
LIBS = -L/opt/homebrew/Cellar/openblas/0.3.30/lib -lopenblas

# Target executable
TARGET = love

# Source files
SOURCES = main.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Main rule
all: $(TARGET)

# Linking rule
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Compilation rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)
