all: HR_euler
NVCCFLAGS = -use_fast_math

HR_euler: kernel.o HR_euler.cpp
	LD_RUN_PATH=/usr/local/cuda-6.0/lib64 g++-4.4 -I/usr/local/cuda-6.0/include HR_euler.cpp -o HR_euler init_variable.o kernel.o -L/usr/local/cuda-6.0/lib64 -L/usr/lib/nvidia-current -lcuda -lcudart -lm

kernel.o: kernel.cu init_variable.o kernel.h
	LD_RUN_PATH=/usr/local/cuda-6.0/lib64 nvcc -I/usr/local/cuda-6.0/include -arch=sm_20 -c kernel.cu -L/usr/local/cuda-6.0/lib64 -L/usr/lib/nvidia-current -lcuda -lcudart -lm

init_variable.o: init_variable.cpp
	LD_RUN_PATH=/usr/local/cuda-6.0/lib64 g++-4.4 -I/usr/local/cuda-6.0/include -c init_variable.cpp -L/usr/local/cuda-6.0/lib64 -L/usr/lib/nvidia-current -lcuda -lcudart -lm
 
