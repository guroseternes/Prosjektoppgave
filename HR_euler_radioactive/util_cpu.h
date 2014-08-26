#include "cuda.h"
#include "gpu_ptr.h"
#include "kernel_arg_structs.h"
#include <sys/time.h>
#include <math.h>

// Print GPU properties
void print_properties(){
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
         printf("Device count: %d\n", deviceCount);

        cudaDeviceProp p;
        cudaSetDevice(0);
        cudaGetDeviceProperties (&p, 0);
        printf("Compute capability: %d.%d\n", p.major, p.minor);
        printf("Name: %s\n" , p.name);
	printf("Compute concurrency %i\n", p.concurrentKernels);
        printf("\n\n");
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

float Linf(cpu_ptr_2D exactSol, cpu_ptr_2D numSol){
	int nx = exactSol.get_nx();
	int ny = exactSol.get_ny();
	float max = 0;
	float error = 0;
	for (int i=0; i<nx; i++){
//		printf("Error %.8f\n", error);
		for (int j=0; j<ny; j++){ 
			error = fabs(exactSol(i,j)-numSol(i,j));
			if (error > max){
			//	printf("max %.8f\n", max);
				max = error;
			}
		}
	}
	return max;
}
		
float L1(cpu_ptr_2D exactSol, cpu_ptr_2D numSol){
	int nx = exactSol.get_nx();
	int ny = exactSol.get_ny();
	int step = nx/50;
	float max = 0;
	float error;
	int counter = 0;
	for (int i=0; i<nx; i++){
		counter = 0;
		error = 0;
		for (int j=0; j<ny; j+=step){ 
			error += fabs(exactSol(i,j)-numSol(i,j));
			counter++;
		}
		if (error > max)
			max = error;
	}
	printf("Counter %i\n", counter);	
	return max;
}

float L1test(cpu_ptr_2D exactSol, cpu_ptr_2D numSol){
	int nx = exactSol.get_nx();
	int ny = exactSol.get_ny();
	float max = 0;
	float error;
	for (int i=0; i<nx; i++){
		error = 0;
		for (int j=0; j<ny; j++){ 
			error += fabs(exactSol(i,j)-numSol(i,j));
		}
		if (error > max)
			max = error;
	}
	float scale = nx/50;
	return max/scale;
}

inline void set_bc_args(collBCKernelArgs* args, gpu_raw_ptr U0, gpu_raw_ptr U1, gpu_raw_ptr U2, gpu_raw_ptr U3, gpu_raw_ptr U4, gpu_raw_ptr U5, unsigned int NX, unsigned int NY,int border){

	args->U0 = U0;
        args->U1 = U1;
        args->U2 = U2;
        args->U3 = U3;
	args->U4 = U4;
	args->U5 = U5;
	
	args->NX = NX;
	args->NY = NY;
	args->global_border = border;
}

inline void set_rk_args(RKKernelArgs* args, float* dt, gpu_raw_ptr U0, gpu_raw_ptr U1, gpu_raw_ptr U2, gpu_raw_ptr U3, gpu_raw_ptr U4, gpu_raw_ptr U5, gpu_raw_ptr R0, gpu_raw_ptr R1, gpu_raw_ptr R2, gpu_raw_ptr R3, gpu_raw_ptr R4, gpu_raw_ptr R5, gpu_raw_ptr Q0, gpu_raw_ptr Q1, gpu_raw_ptr Q2, gpu_raw_ptr Q3, gpu_raw_ptr Q4, gpu_raw_ptr Q5, unsigned int nx,unsigned int ny, int border, float lambda){

	args->dt = dt;

	args->U0 = U0;
	args->U1 = U1;
	args->U2 = U2;
	args->U3 = U3;
	args->U4 = U4;
	args->U5 = U5;

	args->R0 = R0;
	args->R1 = R1;
	args->R2 = R2;
	args->R3 = R3;
	args->R4 = R4;
	args->R5 = R5;

	args->Q0 = Q0;
	args->Q1 = Q1;
	args->Q2 = Q2;
	args->Q3 = Q3;
	args->Q5 = Q5;
	args->Q4 = Q4;

 	args->nx = nx;
	args->ny = ny;
	args->global_border = border;
	args->lambda = lambda;
}


inline void set_dt_args(DtKernelArgs* args, float* L, float* dt, unsigned int nElements, float dx, float dy, float scale){
	args->L = L;
	args->dt = dt;
	args->nElements = nElements;
	args->dx = dx;
	args->dy = dy;
	args->scale = scale;
}


inline void set_flux_args(FluxKernelArgs* args, float* L, gpu_raw_ptr U0, gpu_raw_ptr U1, gpu_raw_ptr U2, gpu_raw_ptr U3, gpu_raw_ptr U4, gpu_raw_ptr U5, gpu_raw_ptr R0, gpu_raw_ptr R1, gpu_raw_ptr R2, gpu_raw_ptr R3, gpu_raw_ptr R4, gpu_raw_ptr R5, unsigned int nx, unsigned int ny, int border, float dx, float dy, float theta, float gamma, float alphaA, float alphaB, float betaA, float betaB, int innerDimX, int innerDimY){
	args->L = L; 	

	args->U0 = U0;
	args->U1 = U1;
	args->U2 = U2;
	args->U3 = U3;
	args->U4 = U4;
	args->U5 = U5;

	args->R0 = R0;
	args->R1 = R1;
	args->R2 = R2;
	args->R3 = R3;
	args->R4 = R4; 	
	args->R5 = R5;

	args->nx = nx;
	args->ny = ny;
	args->global_border = border;
	args->dx = dx;
	args->dy = dy;

	args->gamma = gamma;
	args->theta = theta;
	args->alphaA = alphaA;
	args->alphaB = alphaB;
	args->betaA = betaA;
	args->betaB = betaB;

	args->innerDimX = innerDimX;
	args->innerDimY = innerDimY;

}

void setLandDt(int nElements, float* L_host, float* L_device, float* dt_device){

	L_host = new float[nElements];
	for (int i = 0; i < nElements; i++)
		L_host[i] = FLT_MAX;

	cudaMalloc((void**)&L_device, sizeof(float)*(nElements));
	cudaMemcpy(L_device,L_host, sizeof(float)*(nElements), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dt_device, sizeof(float));
}


void computeGridBlock(dim3& gridBlock, dim3& threadBlock, int NX, int NY, int tiledimX, int tiledimY, int blockdimX, int blockdimY){

        int gridDimx =  (NX + tiledimX - 1)/tiledimX;
        int gridDimy =  (NY + tiledimY - 1)/tiledimY;
                                                                                                                       
        threadBlock.x = blockdimX;
        threadBlock.y = blockdimY;
        gridBlock.x = gridDimx;
        gridBlock.y = gridDimy;
}
