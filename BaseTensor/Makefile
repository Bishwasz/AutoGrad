CXX = g++ -std=c++17 -O3 -mavx -mfma
TARGET = main
SRC = main.cpp

# OpenBLAS paths - direct to the Cellar location where cblas.h exists
INCLUDE = -I/opt/homebrew/Cellar/openblas/0.3.30/include
LIBS = -L/opt/homebrew/Cellar/openblas/0.3.30/lib -lopenblas

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) $(INCLUDE) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)