CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I/opt/homebrew/Cellar/openblas/0.3.30/include
LDFLAGS = -L/opt/homebrew/Cellar/openblas/0.3.30/lib -lopenblas

main: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main $(LDFLAGS)
