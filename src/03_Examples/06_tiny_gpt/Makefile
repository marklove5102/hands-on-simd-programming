CXX=g++
CXXFLAGS=-mavx2 -mfma -masm=att -std=c++17
LDFLAGS=-lstdc++fs
TARGET=simd_program
ASMFILE=main.s
SRCFILE=main.cpp

all: $(TARGET)

$(TARGET): $(SRCFILE)
	$(CXX) $(CXXFLAGS) $(SRCFILE) -o $(TARGET) $(LDFLAGS)

asm: $(SRCFILE)
	$(CXX) $(CXXFLAGS) -S $(SRCFILE) -o $(ASMFILE)

clean:
	rm -f $(TARGET) $(ASMFILE)
