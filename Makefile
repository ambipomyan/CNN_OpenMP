default:
	/opt/llvm/llvm-14.x-install/bin/clang++ -I/usr/include/opencv4 -DOPENCV -std=c++11 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -fopenmp-targets=nvptx64 -g -O0 -Ofast -lm -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -L/opt/llvm/llvm-14.x-install/lib main.cpp conv-omp-gpu.cpp connect-omp-gpu.cpp util.cpp -o main

run:
	./main 2000 50 20 50 1

clean:
	rm -rf main
