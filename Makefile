default:
	/home/kyan2/llvm-15.x-install/bin/clang++ -I/usr/include/opencv4 -DOPENCV -std=c++11 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -fopenmp-targets=nvptx64 -g -O0 -Ofast -lm -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -L/home/kyan2/llvm-15.x-install/lib main-multi-gpu.cpp helpers_main.cpp conv-omp-gpu.cpp connect-omp-gpu.cpp util.cpp -o main

omp-cpu:
	/home/kyan2/llvm-15.x-install/bin/clang++ -I/usr/include/opencv4 -DOPENCV -std=c++11 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -g -O0 -Ofast -lm -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -L/home/kyan2/llvm-15.x-install/lib main-multi-gpu.cpp helpers_main_evals.cpp conv-omp-cpu.cpp connect-omp-cpu.cpp util.cpp -o main-omp-cpu

cudnn:
	/opt/llvm/llvm-14.x-install/bin/clang++ -I/usr/include/opencv4 -I/usr/local/cuda/include -DOPENCV -std=c++11 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -g -O0 -Ofast -lm -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lcudart -lcublas -lcudnn -L/opt/llvm/llvm-14.x-install/lib -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 main-multi-gpu.cpp helpers_main_evals.cpp conv-cudnn.cpp connect-cudnn.cpp util.cpp -o main-cudnn

run:
	./main 2000 500 50 10 1

clean:
	rm -rf main main-omp-cpu main-cudnn weights
