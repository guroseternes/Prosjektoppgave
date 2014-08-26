#include "cpu_ptr.h"
#include "kernel.h"
#include "ICsquare.h" 
#include "util_cpu.h"
#include "configurations.h"
#include "cudaProfiler.h"
#include "cuda_profiler_api.h"

// Global variables, will be moved to h-file when ready
// Grid parameters
unsigned int nx = 1600;
unsigned int ny = 1600;
int border = 2;

//Time parameters
float timeLength = 2.0;
float currentTime = 0;
float dt;
float cfl_number = 0.475f;
float theta = 2.0f;
float gasGam = 1.4f;
int step = 0;
int maxStep = 6000;

int main(int argc,char **argv){

// Print GPU properties
//print_properties();

// Files to print the result after the last time step
FILE *rho_file;
FILE *E_file;
rho_file = fopen("rho_final.txt", "w");
E_file = fopen("E_final.txt", "w");

// Construct initial condition for problem
ICsinus Config(-1.0, 1.0, -1.0, 1.0); 
//ICsquare Config(0.5,0.5,gasGam);

// Set initial values for Configuration 1

//Config.set_rho(rhoConfig19);
//Config.set_pressure(pressureConfig19);
//Config.set_u(uConfig19);
//Config.set_v(vConfig19);

// Determining global border based on left over tiles (a little hack)
int globalPadding;
globalPadding = (nx+2*border+16)/16;
globalPadding = 16*globalPadding - (nx+2*border);
//printf("Globalpad: %i\n", globalPadding);

// Change border to add padding
//border = border + globalPadding/2;

// Initiate the matrices for the unknowns in the Euler equations
cpu_ptr_2D rho(nx, ny, border,1);
cpu_ptr_2D E(nx, ny, border,1);
cpu_ptr_2D rho_u(nx, ny, border,1);
cpu_ptr_2D rho_v(nx, ny, border,1);

// Set initial condition
Config.setIC(rho, rho_u, rho_v, E);

double timeStart = get_wall_time();

// Test 
cpu_ptr_2D rho_dummy(nx, ny, border);
rho_dummy.xmin = -1.0;
rho_dummy.ymin = -1.0;

cpu_ptr_2D E_dummy(nx, ny, border);
E_dummy.xmin = -1.0;
E_dummy.ymin = -1.0;

// Set block and grid sizes
dim3 gridBC = dim3(1, 1, 1);
dim3 blockBC = dim3(BLOCKDIM_BC,1,1);

dim3 gridBlockFlux;
dim3 threadBlockFlux;

dim3 gridBlockRK;
dim3 threadBlockRK;

computeGridBlock(gridBlockFlux, threadBlockFlux, nx + 2*border, ny + 2*border, INNERTILEDIM_X, INNERTILEDIM_Y, BLOCKDIM_X, BLOCKDIM_Y);

computeGridBlock(gridBlockRK, threadBlockRK, nx + 2*border, ny + 2*border, BLOCKDIM_X_RK, BLOCKDIM_Y_RK, BLOCKDIM_X_RK, BLOCKDIM_Y_RK);

int nElements = gridBlockFlux.x*gridBlockFlux.y;

// Allocate memory for the GPU pointers
gpu_ptr_1D L_device(nElements);
gpu_ptr_1D dt_device(1);

gpu_ptr_2D rho_device(nx, ny, border);
gpu_ptr_2D E_device(nx, ny, border);
gpu_ptr_2D rho_u_device(nx, ny, border);
gpu_ptr_2D rho_v_device(nx, ny, border); 

gpu_ptr_2D R0(nx, ny, border);
gpu_ptr_2D R1(nx, ny, border);
gpu_ptr_2D R2(nx, ny, border);
gpu_ptr_2D R3(nx, ny, border);

gpu_ptr_2D Q0(nx, ny, border);
gpu_ptr_2D Q1(nx, ny, border);
gpu_ptr_2D Q2(nx, ny, border);
gpu_ptr_2D Q3(nx, ny, border);

// Allocate pinned memory on host
init_allocate();

// Set BC arguments
set_bc_args(BCArgs[0], rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), nx+2*border, ny+2*border, border);
set_bc_args(BCArgs[1], Q0.getRawPtr(), Q1.getRawPtr(), Q2.getRawPtr(), Q3.getRawPtr(), nx+2*border, ny+2*border, border);
set_bc_args(BCArgs[2], rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), nx+2*border, ny+2*border, border);

// Set FLUX arguments
set_flux_args(fluxArgs[0], L_device.getRawPtr(), rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), R0.getRawPtr(),R1.getRawPtr(), R2.getRawPtr(), R3.getRawPtr(), nx, ny, border, rho.get_dx(), rho.get_dy(), theta, gasGam, INNERTILEDIM_X, INNERTILEDIM_Y);
set_flux_args(fluxArgs[1], L_device.getRawPtr(), Q0.getRawPtr(), Q1.getRawPtr(), Q2.getRawPtr(), Q3.getRawPtr(), R0.getRawPtr(),R1.getRawPtr(), R2.getRawPtr(), R3.getRawPtr(), nx, ny, border, rho.get_dx(), rho.get_dy(), theta, gasGam, INNERTILEDIM_X, INNERTILEDIM_Y);

// Set TIME argument
set_dt_args(dtArgs, L_device.getRawPtr(), dt_device.getRawPtr(), nElements, rho.get_dx(), rho.get_dy(), cfl_number);

// Set Rk arguments
set_rk_args(RKArgs[0], dt_device.getRawPtr(), rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), R0.getRawPtr(), R1.getRawPtr(), R2.getRawPtr(), R3.getRawPtr(), Q0.getRawPtr(), Q1.getRawPtr(), Q2.getRawPtr(), Q3.getRawPtr(), nx, ny, border); 
set_rk_args(RKArgs[1], dt_device.getRawPtr(), Q0.getRawPtr(), Q1.getRawPtr(), Q2.getRawPtr(), Q3.getRawPtr(), R0.getRawPtr(), R1.getRawPtr(), R2.getRawPtr(), R3.getRawPtr(), rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), nx, ny, border); 


L_device.set(FLT_MAX);

R0.set(0,0,0,nx,ny,border); 
R1.set(0,0,0,nx,ny,border); 
R2.set(0,0,0,nx,ny,border); 
R3.set(0,0,0,nx,ny,border); 

Q0.set(0,0,0,nx,ny,border); 
Q1.set(0,0,0,nx,ny,border); 
Q2.set(0,0,0,nx,ny,border); 
Q3.set(0,0,0,nx,ny,border); 

rho_device.upload(rho.get_ptr());
rho_u_device.upload(rho_u.get_ptr());
rho_v_device.upload(rho_v.get_ptr());
E_device.upload(E.get_ptr());

// Update boudries
callCollectiveSetBCPeriodic(gridBC, blockBC, BCArgs[0]);

//Create cuda stream
cudaStream_t stream1;
cudaStreamCreate(&stream1);
cudaEvent_t dt_complete;
cudaEventCreate(&dt_complete);


while (currentTime < timeLength && step < maxStep){	
	
	//RK1	
	//Compute flux
	callFluxKernel(gridBlockFlux, threadBlockFlux, 0, fluxArgs[0]);	
	
	// Compute timestep (based on CFL condition)
	callDtKernel(TIMETHREADS, dtArgs);
	
	cudaMemcpyAsync(dt_host, dt_device.getRawPtr(), sizeof(float), cudaMemcpyDeviceToHost, stream1);
	cudaEventRecord(dt_complete, stream1);

	// Perform RK1 step
	callRKKernel(gridBlockRK, threadBlockRK, 0, RKArgs[0]);
	
	//Update boudries
	callCollectiveSetBCPeriodic(gridBC, blockBC, BCArgs[1]);		

	//RK2
	// Compute flux
	callFluxKernel(gridBlockFlux, threadBlockFlux, 1, fluxArgs[1]);

	//Perform RK2 step
	callRKKernel(gridBlockRK, threadBlockRK, 1, RKArgs[1]);	

	//cudaEventRecord(srteam_sync, srteam1);

	callCollectiveSetBCPeriodic(gridBC, blockBC, BCArgs[2]);

	cudaEventSynchronize(dt_complete);

	step++;	
	currentTime += *dt_host;	
//	printf("Step: %i, current time: %.6f dt:%.6f\n" , step,currentTime, dt_host[0]);

}


//cuProfilerStop();
//cudaProfilerStop();

printf("Elapsed time %.5f", get_wall_time() - timeStart);

rho_device.download(rho_dummy.get_ptr());
rho_dummy.printToFile(rho_file, true, false);


Config.exactSolution(E_dummy, currentTime);
E_dummy.printToFile(E_file, true, false);

float LinfError = Linf(E_dummy, rho_dummy);
float L1Error = L1(E_dummy, rho_dummy); 
float L1Error2 = L1test(E_dummy, rho_dummy);

printf("nx: %i step: %i, current time: %.6f dt:%.6f\n" , nx, step,currentTime, dt_host[0]); 
// Print test

printf("nx: %i\t Linf error %.9f\t L1 error %.7f L1test erro %.7f", nx, LinfError, L1Error, L1Error2);

/*
cudaMemcpy(L_host, L_device, sizeof(float)*(nElements), cudaMemcpyDeviceToHost);
for (int i =0; i < nElements; i++)
	printf(" %.7f ", L_host[i]); 
*/


printf("%s\n", cudaGetErrorString(cudaGetLastError()));

return(0);
}


