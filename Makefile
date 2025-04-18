# Simple Makefile for Tensor class

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

all: autog_test

autog_test: autog.cpp 
	$(CXX) $(CXXFLAGS) autog.cpp -o autog_test

clean:
	rm -f autog_test
