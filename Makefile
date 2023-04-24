default:
	clang++ -std=c++11 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -fopenmp-targets=nvptx64 -g -O0 -Ofast -lm `pkg-config --cflags --libs opencv4` main-multi-gpu.cpp helpers_main.cpp conv-omp-gpu.cpp connect-omp-gpu.cpp util.cpp -o main

omp-cpu:
	clang++ -std=c++11 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -g -O0 -Ofast -lm `pkg-config --cflags --libs opencv4` main-multi-gpu.cpp helpers_main_evals.cpp conv-omp-cpu.cpp connect-omp-cpu.cpp util.cpp -o main-omp-cpu

cudnn:
	clang++ -std=c++11 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -g -O0 -Ofast -I/usr/local/cuda/include -lm `pkg-config --cflags --libs opencv4` -L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn main-multi-gpu.cpp helpers_main_evals.cpp conv-cudnn.cpp connect-cudnn.cpp util.cpp -o main-cudnn

api:
	clang++ -std=c++11 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -fopenmp-targets=nvptx64 -g -O0 -Ofast -lm `pkg-config --cflags --libs opencv4` main-api.cpp helpers_main.cpp conv-omp-gpu.cpp connect-omp-gpu.cpp util.cpp -o main-api


run:
	./main-api 2000 100 10 10 1

clean:
	rm -rf main main-omp-cpu main-cudnn main-api weights
