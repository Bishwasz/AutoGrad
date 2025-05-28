CXX = g++
CXXFLAGS = -std=c++17 -Wall -Werror -I.. -I../BaseTensor -I../AutoGrad

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
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)
