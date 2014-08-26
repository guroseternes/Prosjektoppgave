#include "util.h"
#include "kernel.h"
#include "cuda.h"
#include "math.h"

__constant__ FluxKernelArgs flux_ctx;
__constant__ DtKernelArgs dt_ctx;
__constant__ RKKernelArgs rk_ctx;

void init_allocate(){
	for (int i=0; i<3; i++){
		cudaHostAlloc(&BCArgs[i], sizeof(collBCKernelArgs), cudaHostAllocWriteCombined);
		//cudaMallocHost(&BCArgs[i], sizeof(collBCKernelArgs));

		cudaHostAlloc(&fluxArgs[i], sizeof(FluxKernelArgs), cudaHostAllocWriteCombined);

		cudaHostAlloc(&RKArgs[i], sizeof(RKKernelArgs), cudaHostAllocWriteCombined);

		cudaHostAlloc(&dtArgs, sizeof(DtKernelArgs), cudaHostAllocWriteCombined);

		cudaHostAlloc(&dt_host, sizeof(float), cudaHostAllocWriteCombined);
		//cudaMallocHost(&fluxArgs[i], sizeof(FluxKernelArgs));
	}
}

__global__ void RKKernel(int step){ 

	float dt = rk_ctx.dt[0];
	float u0,u1,u2,u3,r0,r1,r2,r3,q0,q1,q2,q3;
	int global_border = rk_ctx.global_border;

	// Global indexes     
        int xid = blockIdx.x*blockDim.x + threadIdx.x - global_border;
        int yid = blockIdx.y*blockDim.y + threadIdx.y - global_border;

        if ( xid < 0 || xid >= rk_ctx.nx || yid < 0 || yid >= rk_ctx.ny ) return; 

	u0 = global_index(rk_ctx.U0.ptr, rk_ctx.U0.pitch, xid, yid, global_border)[0];
	u1 = global_index(rk_ctx.U1.ptr, rk_ctx.U1.pitch, xid, yid, global_border)[0];
	u2 = global_index(rk_ctx.U2.ptr, rk_ctx.U2.pitch, xid, yid, global_border)[0];
	u3 = global_index(rk_ctx.U3.ptr, rk_ctx.U3.pitch, xid, yid, global_border)[0];		

	r0 = global_index(rk_ctx.R0.ptr, rk_ctx.R0.pitch, xid, yid, global_border)[0];
        r1 = global_index(rk_ctx.R1.ptr, rk_ctx.R1.pitch, xid, yid, global_border)[0];
        r2 = global_index(rk_ctx.R2.ptr, rk_ctx.R2.pitch, xid, yid, global_border)[0];
        r3 = global_index(rk_ctx.R3.ptr, rk_ctx.R3.pitch, xid, yid, global_border)[0];

	if (step == 0) {
		q0 =  u0 + dt*r0;
		q1 =  u1 + dt*r1;
		q2 =  u2 + dt*r2;
		q3 =  u3 + dt*r3;
	}
	else {
		q0 = global_index(rk_ctx.Q0.ptr, rk_ctx.Q0.pitch, xid, yid, global_border)[0];
        	q1 = global_index(rk_ctx.Q1.ptr, rk_ctx.Q1.pitch, xid, yid, global_border)[0];
        	q2 = global_index(rk_ctx.Q2.ptr, rk_ctx.Q2.pitch, xid, yid, global_border)[0];
        	q3 = global_index(rk_ctx.Q3.ptr, rk_ctx.Q3.pitch, xid, yid, global_border)[0];

		q0 = 0.5f*(q0 + (u0 + dt*r0));
		q1 = 0.5f*(q1 + (u1 + dt*r1));
		q2 = 0.5f*(q2 + (u2 + dt*r2));
		q3 = 0.5f*(q3 + (u3 + dt*r3));
	}

	global_index(rk_ctx.Q0.ptr, rk_ctx.Q0.pitch, xid, yid, global_border)[0] = q0;
        global_index(rk_ctx.Q1.ptr, rk_ctx.Q1.pitch, xid, yid, global_border)[0] = q1;
        global_index(rk_ctx.Q2.ptr, rk_ctx.Q2.pitch, xid, yid, global_border)[0] = q2;
        global_index(rk_ctx.Q3.ptr, rk_ctx.Q3.pitch, xid, yid, global_border)[0] = q3;
}

void callRKKernel(dim3 grid, dim3 block, int step, RKKernelArgs* h_ctx){	
	cudaMemcpyToSymbolAsync(rk_ctx, h_ctx, sizeof(RKKernelArgs), 0, cudaMemcpyHostToDevice);
	RKKernel<<<grid, block>>>(step);
}

__global__ void DtKernel(int nThreads){

	extern __shared__ float sdata[];
	volatile float* sdata_volatile = sdata;
	unsigned int tid = threadIdx.x;
	int threads = nThreads;
	float dt;
	//printf("THREADID %i",tid);

	sdata[tid] = FLT_MAX;

	for (unsigned int i=tid; i<dt_ctx.nElements; i += threads)
		sdata[tid] = min(sdata[tid], dt_ctx.L[i]);
		__syncthreads();
	//	if (tid == 0){
	//		printf("START\n");
	//		for (int k=0; k<nThreads; k++)
	//			printf(" %.5f\t",sdata[k]);  
	//	}	
	//Now, reduce all elements into a single element
	if (threads >= 512) {
		if (tid < 256) sdata[tid] = min(sdata[tid], sdata[tid + 256]);
		__syncthreads();
	}
	if (threads >= 256) {
		if (tid < 128) sdata[tid] = min(sdata[tid], sdata[tid + 128]);
		__syncthreads();
	}
	if (threads >= 128) {
		if (tid < 64) sdata[tid] = min(sdata[tid], sdata[tid + 64]);
		__syncthreads();
	}
	if (tid < 32) {
		if (threads >= 64) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid + 32]);
		if (tid < 16) {
			if (threads >= 32) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid + 16]);
			if (threads >= 16) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  8]);
			if (threads >=  8) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  4]);
			if (threads >=  4) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  2]);
			if (threads >=  2) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  1]);
		}

		if (tid == 0) {
			dt = sdata_volatile[tid];
			if (dt == FLT_MAX) {
				//If no water at all, and no sources, 
				//we really do not need to simulate, 
				//but using FLT_MAX will make things crash...
				dt = 1.0e-7f;
			}
			dt_ctx.dt[tid] = dt*dt_ctx.scale;
		//	printf("TID %i",tid); 
		}
	}
}


void callDtKernel(int nThreads, DtKernelArgs* h_ctx){

	cudaMemcpyToSymbolAsync(dt_ctx, h_ctx, sizeof(DtKernelArgs), 0, cudaMemcpyHostToDevice);	
	DtKernel<<<1,nThreads,sizeof(float)*nThreads>>>(nThreads);
}

inline __device__ void fluxAndLambdaFuncF(float& rho, float& U1, float& U2, float& U3,
		const float& gamma,
		float& F0, float& F1, float& F2, float& F3,
		float& u, float& v,float& c){

	float pressure, E;

	// Vaues needed to compute the eigenvalues
	u = U1/rho;
	v = U2/rho;
	E = U3;
        pressure = (gamma - 1.0f)*(E-0.5f*rho*(u*u + v*v));
        c = sqrtf(gamma*pressure/rho);	      

	// Flux computation
	F0 = U1;
	F1 = U1*u +  pressure;
	F2 = U1*v;
	F3 = u*(E+pressure);
}

inline __device__ void fluxAndLambdaFuncG(float& rho, float& U1, float& U2, float& U3,
                const float& gamma,
                float& G0, float& G1, float& G2, float& G3,
                float& u, float& v,float& c){
        
        float pressure, E;

	// Vaues needed to compute the eigenvalues
	u = U1/rho;
	v = U2/rho;
	E = U3;
        pressure = (gamma - 1.0f)*(E-0.5f*rho*(u*u + v*v));
	c = sqrtf(gamma*pressure/rho);
        //if (pressure < 0)
                //printf("ZERO alert compute G and Lambda gamma:%.3f pressure: %.3f rho:%.3f  rho_u:%.3f rho_v%.3f E%.3f\n", gamma,pressure,rho,U1,U2,E);
        // Flux computation
        G0 = U2;
        G1 = U2*u;
        G2 = U2*v + pressure;
        G3 = v*(E+pressure);
}

inline __device__ float minEigenVal(float a, float b) {
	return fminf(fminf(a, b), 0.0f);
}

inline __device__ float maxEigenVal(float a, float b) {
	return fmaxf(fmaxf(a, b), 0.0f);
}

inline __device__ float sign(float& a) {
	/**
	  * The following works by bit hacks. In non-obfuscated code, something like
	  *  float r = ((int&)a & 0x7FFFFFFF)!=0; //set r to one or zero
	  *  (int&)r |= ((int&)a & 0x80000000);   //Copy sign bit of a
	  *  return r;
	  */
#ifndef NEW_SIGN
	return (signed((int&)a & 0x80000000) >> 31 ) | ((int&)a & 0x7FFFFFFF)!=0;
#else
	float r = ((int&)a & 0x7FFFFFFF)!=0;
	return copysignf(r, a);
#endif
}


inline __device__ float minmod(float a, float b, float c){
	return 0.25f
	*sign(a)
	*(sign(a) + sign(b))
	*(sign(b) + sign(c))
	*fminf( fminf(fabsf(a), fabsf(b)), fabsf(c) );


/*	if ( a > 0 && b > 0 && c > 0)
		return fminf(c,fminf(a,b));
	else if ( a < 0 && b < 0 && c < 0)
		return fmaxf(c,fmaxf(a,b));
	else
		return 0.0;
*/
}

inline __device__ float limiter(float u_plus, float u_center, float u_minus){
	return minmod(flux_ctx.theta*(u_plus-u_center),(u_plus-u_minus)*0.5f, flux_ctx.theta*(u_center-u_minus));
}

inline __device__ void reconstructPointVal(float (&U)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], float (&Ux)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], float (&Uy)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], unsigned int i, unsigned int j){
	float u_center,u_south,u_north,u_east,u_west;

	float ux_out, uy_out;

	for (int l=0; l<4; l++){
			u_center = U[l][i][j];
                        u_south = U[l][i][j-1];
                        u_north = U[l][i][j+1];
                        u_west = U[l][i-1][j];
                        u_east = U[l][i+1][j];

			// Compute interface values, each cell computes 
                        ux_out = 0.5f*limiter(u_east, u_center, u_west);
                        uy_out = 0.5f*limiter(u_north, u_center, u_south);

			Ux[l][i][j] = ux_out;
			Uy[l][i][j] = uy_out;			
	}

}

inline __device__ float computeFluxWest(float (&U)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], float (&Ux)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], float (&Uy)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], float (&Uout)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], unsigned int i, unsigned int j){
	
	float U0ma, U1ma, U2ma, U3ma;
	float U0mb, U1mb, U2mb, U3mb;
	float U0pa, U1pa, U2pa, U3pa;
	float U0pb, U1pb, U2pb, U3pb;
	float FG0pa, FG1pa, FG2pa, FG3pa;
	float FG0pb, FG1pb, FG2pb, FG3pb;
	float FG0ma, FG1ma, FG2ma, FG3ma;
	float FG0mb, FG1mb, FG2mb, FG3mb;
	float upa,vpa,cpa,uma,vma,cma;
	float upb,vpb,cpb,umb,vmb,cmb;
	float ama, apa;
	float amb, apb;
	float alpha = 1.0/sqrtf(3);
	
	// The eastern reconstruction point of u(i-1,j)
	U0ma = U[0][i-1][j] + Ux[0][i-1][j] + 2*alpha*Uy[0][i-1][j];
	U1ma = U[1][i-1][j] + Ux[1][i-1][j] + 2*alpha*Uy[1][i-1][j];
	U2ma = U[2][i-1][j] + Ux[2][i-1][j] + 2*alpha*Uy[2][i-1][j];
	U3ma = U[3][i-1][j] + Ux[3][i-1][j] + 2*alpha*Uy[3][i-1][j];
	
	// The eastern reconstruction point of u(i-1,j)
	U0mb = U[0][i-1][j] + Ux[0][i-1][j] - 2*alpha*Uy[0][i-1][j];
	U1mb = U[1][i-1][j] + Ux[1][i-1][j] - 2*alpha*Uy[1][i-1][j];
	U2mb = U[2][i-1][j] + Ux[2][i-1][j] - 2*alpha*Uy[2][i-1][j];
	U3mb = U[3][i-1][j] + Ux[3][i-1][j] - 2*alpha*Uy[3][i-1][j];
	
	// The western reconstruction point of u(i,j)
	U0pa = U[0][i][j] - Ux[0][i][j] + 2*alpha*Uy[0][i][j];
        U1pa = U[1][i][j] - Ux[1][i][j] + 2*alpha*Uy[1][i][j];
        U2pa = U[2][i][j] - Ux[2][i][j] + 2*alpha*Uy[2][i][j];
        U3pa = U[3][i][j] - Ux[3][i][j] + 2*alpha*Uy[3][i][j];
		
	// The western reconstruction point of u(i,j)
	U0pb = U[0][i][j] - Ux[0][i][j] - 2*alpha*Uy[0][i][j];
        U1pb = U[1][i][j] - Ux[1][i][j] - 2*alpha*Uy[1][i][j];
        U2pb = U[2][i][j] - Ux[2][i][j] - 2*alpha*Uy[2][i][j];
        U3pb = U[3][i][j] - Ux[3][i][j] - 2*alpha*Uy[3][i][j];

	fluxAndLambdaFuncF(U0pa, U1pa, U2pa, U3pa, flux_ctx.gamma, FG0pa, FG1pa, FG2pa, FG3pa, upa, vpa, cpa);
	fluxAndLambdaFuncF(U0ma, U1ma, U2ma, U3ma, flux_ctx.gamma, FG0ma, FG1ma, FG2ma, FG3ma, uma, vma, cma);

	fluxAndLambdaFuncF(U0pb, U1pb, U2pb, U3pb, flux_ctx.gamma, FG0pb, FG1pb, FG2pb, FG3pb, upb, vpb, cpb);
	fluxAndLambdaFuncF(U0mb, U1mb, U2mb, U3mb, flux_ctx.gamma, FG0mb, FG1mb, FG2mb, FG3mb, umb, vmb, cmb);

	ama = minEigenVal(uma-cma, upa-cpa);
	apa = maxEigenVal(uma+cma, upa+cpa);

	amb = minEigenVal(umb-cmb, upb-cpb);
	apb = maxEigenVal(umb+cmb, upb+cpb);

	__syncthreads();
	
	float flux0a = ((apa*FG0ma - ama*FG0pa) + apa*ama*(U0pa-U0ma))/(apa-ama);
	float flux1a = ((apa*FG1ma -ama*FG1pa) + apa*ama*(U1pa-U1ma))/(apa-ama);
	float flux2a = ((apa*FG2ma -ama*FG2pa) + apa*ama*(U2pa-U2ma))/(apa-ama);
	float flux3a = ((apa*FG3ma -ama*FG3pa) + apa*ama*(U3pa-U3ma))/(apa-ama);

	float flux0b = ((apb*FG0mb -amb*FG0pb) + apb*amb*(U0pb-U0mb))/(apb-amb);
	float flux1b = ((apb*FG1mb -amb*FG1pb) + apb*amb*(U1pb-U1mb))/(apb-amb);
	float flux2b = ((apb*FG2mb -amb*FG2pb) + apb*amb*(U2pb-U2mb))/(apb-amb);
	float flux3b = ((apb*FG3mb -amb*FG3pb) + apb*amb*(U3pb-U3mb))/(apb-amb);
	
	Uout[0][i][j] = 0.5f*(flux0a + flux0b);
	Uout[1][i][j] = 0.5f*(flux1a + flux1b);
	Uout[2][i][j] = 0.5f*(flux2a + flux2b);
	Uout[3][i][j] = 0.5f*(flux3a + flux3b);	

	apa = 0.5*(apa+apb);
	ama = 0.5*(ama+amb);	

	return flux_ctx.dx/fmaxf(apa, -ama);
}
 	
inline __device__ float computeFluxSouth(float (&U)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], float (&Ux)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], float (&Uy)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], float (&Uout)[4][BLOCKDIM_X][SM_BLOCKDIM_Y], unsigned int i, unsigned int j){

	float U0ma, U1ma, U2ma, U3ma;
	float U0mb, U1mb, U2mb, U3mb;
	float U0pa, U1pa, U2pa, U3pa;
	float U0pb, U1pb, U2pb, U3pb;
	float FG0pa, FG1pa, FG2pa, FG3pa;
	float FG0pb, FG1pb, FG2pb, FG3pb;
	float FG0ma, FG1ma, FG2ma, FG3ma;
	float FG0mb, FG1mb, FG2mb, FG3mb;
	float upa,vpa,cpa,uma,vma,cma;
	float upb,vpb,cpb,umb,vmb,cmb;
	float ama, apa;
	float amb, apb;
	float alpha = 1.0/sqrtf(3);

        // The eastern reconstruction point of u(i-1,j)
        U0ma = U[0][i][j-1] + Uy[0][i][j-1] + 2*alpha*Ux[0][i][j-1];
        U1ma = U[1][i][j-1] + Uy[1][i][j-1] + 2*alpha*Ux[1][i][j-1];
        U2ma = U[2][i][j-1] + Uy[2][i][j-1] + 2*alpha*Ux[2][i][j-1];
        U3ma = U[3][i][j-1] + Uy[3][i][j-1] + 2*alpha*Ux[3][i][j-1];
 
       // The eastern reconstruction point of u(i-1,j)
        U0mb = U[0][i][j-1] + Uy[0][i][j-1] - 2*alpha*Ux[0][i][j-1];
        U1mb = U[1][i][j-1] + Uy[1][i][j-1] - 2*alpha*Ux[1][i][j-1];
        U2mb = U[2][i][j-1] + Uy[2][i][j-1] - 2*alpha*Ux[2][i][j-1];
        U3mb = U[3][i][j-1] + Uy[3][i][j-1] - 2*alpha*Ux[3][i][j-1];

        // The western reconstruction point of u(i,j)
        U0pa = U[0][i][j] - Uy[0][i][j] + 2*alpha*Ux[0][i][j];
        U1pa = U[1][i][j] - Uy[1][i][j] + 2*alpha*Ux[1][i][j];
        U2pa = U[2][i][j] - Uy[2][i][j] + 2*alpha*Ux[2][i][j];
        U3pa = U[3][i][j] - Uy[3][i][j] + 2*alpha*Ux[3][i][j];

        // The western reconstruction point of u(i,j)
        U0pb = U[0][i][j] - Uy[0][i][j] - 2*alpha*Ux[0][i][j];
        U1pb = U[1][i][j] - Uy[1][i][j] - 2*alpha*Ux[1][i][j];
        U2pb = U[2][i][j] - Uy[2][i][j] - 2*alpha*Ux[2][i][j];
        U3pb = U[3][i][j] - Uy[3][i][j] - 2*alpha*Ux[3][i][j];

	fluxAndLambdaFuncG(U0pa, U1pa, U2pa, U3pa, flux_ctx.gamma, FG0pa, FG1pa, FG2pa, FG3pa, upa, vpa, cpa);
	fluxAndLambdaFuncG(U0ma, U1ma, U2ma, U3ma, flux_ctx.gamma, FG0ma, FG1ma, FG2ma, FG3ma, uma, vma, cma);

	fluxAndLambdaFuncG(U0pb, U1pb, U2pb, U3pb, flux_ctx.gamma, FG0pb, FG1pb, FG2pb, FG3pb, upb, vpb, cpb);
	fluxAndLambdaFuncG(U0mb, U1mb, U2mb, U3mb, flux_ctx.gamma, FG0mb, FG1mb, FG2mb, FG3mb, umb, vmb, cmb);

	ama = minEigenVal(vma-cma, vpa-cpa);
	apa = maxEigenVal(vma+cma, vpa+cpa);

	amb = minEigenVal(vmb-cmb, vpb-cpb);
	apb = maxEigenVal(vmb+cmb, vpb+cpb);

	__syncthreads();
	
	float flux0a = ((apa*FG0ma -ama*FG0pa) + apa*ama*(U0pa-U0ma))/(apa-ama);
	float flux1a = ((apa*FG1ma -ama*FG1pa) + apa*ama*(U1pa-U1ma))/(apa-ama);
	float flux2a = ((apa*FG2ma -ama*FG2pa) + apa*ama*(U2pa-U2ma))/(apa-ama);
	float flux3a = ((apa*FG3ma -ama*FG3pa) + apa*ama*(U3pa-U3ma))/(apa-ama);

	float flux0b = ((apb*FG0mb -amb*FG0pb) + apb*amb*(U0pb-U0mb))/(apb-amb);
	float flux1b = ((apb*FG1mb -amb*FG1pb) + apb*amb*(U1pb-U1mb))/(apb-amb);
	float flux2b = ((apb*FG2mb -amb*FG2pb) + apb*amb*(U2pb-U2mb))/(apb-amb);
	float flux3b = ((apb*FG3mb -amb*FG3pb) + apb*amb*(U3pb-U3mb))/(apb-amb);
	
	Uout[0][i][j] = 0.5f*(flux0a + flux0b);
	Uout[1][i][j] = 0.5f*(flux1a + flux1b);
	Uout[2][i][j] = 0.5f*(flux2a + flux2b);
	Uout[3][i][j] = 0.5f*(flux3a + flux3b);	

	apa = 0.5*(apa+apb);
	ama = 0.5*(ama+amb);	

	return flux_ctx.dx/fmaxf(apa, -ama);
}


__global__ void fluxKernel(int step){

	int global_border = flux_ctx.global_border;
	float dx = flux_ctx.dx;
	float dy = flux_ctx.dy;
 
	// Global indexes, multiply by tiledim because each block has a halo/border	
	int xid = blockIdx.x*flux_ctx.innerDimX + threadIdx.x - global_border;
	int yid = blockIdx.y*flux_ctx.innerDimY + threadIdx.y - global_border;

	xid = fminf(xid, flux_ctx.nx+global_border-1);
	yid = fminf(yid, flux_ctx.ny+global_border-1);

	// Local id
	int i = threadIdx.x;
	int j = threadIdx.y;

	float r = FLT_MAX;
	float r0, r1, r2, r3;

	const int nthreads = BLOCKDIM_X*BLOCKDIM_Y;

	__shared__ float timeStep[BLOCKDIM_X][BLOCKDIM_Y];
	timeStep[i][j] = FLT_MAX;

	__shared__ float local_U[4][BLOCKDIM_X][SM_BLOCKDIM_Y];
	__shared__ float local_Ux[4][BLOCKDIM_X][SM_BLOCKDIM_Y];
	__shared__ float local_Uy[4][BLOCKDIM_X][SM_BLOCKDIM_Y];
	__shared__ float local_Uoutx[4][BLOCKDIM_X][SM_BLOCKDIM_Y];
	__shared__ float local_Uouty[4][BLOCKDIM_X][SM_BLOCKDIM_Y];

	local_U[0][i][j] = global_index(flux_ctx.U0.ptr, flux_ctx.U0.pitch, xid, yid, global_border)[0];
	local_U[1][i][j] = global_index(flux_ctx.U1.ptr, flux_ctx.U1.pitch, xid, yid, global_border)[0];
	local_U[2][i][j] = global_index(flux_ctx.U2.ptr, flux_ctx.U2.pitch, xid, yid, global_border)[0];
	local_U[3][i][j] = global_index(flux_ctx.U3.ptr, flux_ctx.U3.pitch, xid, yid, global_border)[0];	

	__syncthreads();

	if ( i > 0 && i < BLOCKDIM_X - 1 && j > 0 && j < BLOCKDIM_Y - 1){
		reconstructPointVal(local_U, local_Ux, local_Uy, i, j);
	}

	__syncthreads();


	if ( i > 1 && i < TILEDIM_X + 1 && j > 1 && j < TILEDIM_Y)
		r = min(r, computeFluxWest(local_U, local_Ux, local_Uy, local_Uoutx,i, j));
	if ( i > 1 && i < TILEDIM_X  && j > 1 && j < TILEDIM_Y + 1)
		r = computeFluxSouth(local_U, local_Ux, local_Uy, local_Uouty, i, j);

	int p = threadIdx.y*blockDim.x+threadIdx.x;

	__syncthreads();

	if (xid > -1 && xid < flux_ctx.nx && yid > -1 && yid < flux_ctx.ny){
		if ( i > 1 && i < TILEDIM_X  && j > 1 && j < TILEDIM_Y){

			r0 = (local_Uoutx[0][i][j] - local_Uoutx[0][i+1][j])/dx + (local_Uouty[0][i][j] - local_Uouty[0][i][j+1])/dy;	
			r1 = (local_Uoutx[1][i][j] - local_Uoutx[1][i+1][j])/dx + (local_Uouty[1][i][j] - local_Uouty[1][i][j+1])/dy;   
			r2 = (local_Uoutx[2][i][j] - local_Uoutx[2][i+1][j])/dx + (local_Uouty[2][i][j] - local_Uouty[2][i][j+1])/dy;   
			r3 = (local_Uoutx[3][i][j] - local_Uoutx[3][i+1][j])/dx + (local_Uouty[3][i][j] - local_Uouty[3][i][j+1])/dy;   

			global_index(flux_ctx.R0.ptr, flux_ctx.R0.pitch, xid, yid, global_border)[0] = r0;//local_Ux[0][i][j]; 		
			global_index(flux_ctx.R1.ptr, flux_ctx.R1.pitch, xid, yid, global_border)[0] = r1;
			global_index(flux_ctx.R2.ptr, flux_ctx.R2.pitch, xid, yid, global_border)[0] = r2;
			global_index(flux_ctx.R3.ptr, flux_ctx.R3.pitch, xid, yid, global_border)[0] = r3;//local_Uy[0][i][j];

			timeStep[0][p] = r;

		}
	}

//Now, find and write out the maximal eigenvalue in this block
	if (step==0) {
	//	__syncthreads();
		volatile float* B_volatile = timeStep[0];
		//int p = threadIdx.y*blockDim.x+threadIdx.x; //reuse p for indexing
		//printf(" %i ", p);
		//Write the maximum eigenvalues computed by this thread into shared memory
		//Only consider eigenvalues within the internal domain
	/*	if (xid < flux_ctx.nx && yid < flux_ctx.ny && xid >= 0 && yid >=0){
			timeStep[0][p] = r; 
		}	
	*/
		__syncthreads();		

		//First use all threads to reduce min(1024, nthreads) values into 64 values
		//This first outer test is a compile-time test simply to remove statements if nthreads is less than 512.
		if (nthreads >= 512) {
			//This inner test (p < 512) first checks that the current thread should
			//be active in the reduction from min(1024, nthreads) elements to 512. Makes little sense here, but
			//a lot of sense for the last test where there should only be 64 active threads.
			//The second part of this test ((p+512) < nthreads) removes the threads that would generate an
			//out-of-bounds access to shared memory
			if (p < 512 && (p+512) < nthreads) timeStep[0][p] = fminf(timeStep[0][p], timeStep[0][p + 512]); //min(1024, nthreads)=>512
			__syncthreads();
		}

		if (nthreads >= 256) { 
			if (p < 256 && (p+256) < nthreads) timeStep[0][p] = fminf(timeStep[0][p], timeStep[0][p + 256]); //min(512, nthreads)=>256
			__syncthreads();
		}
		if (nthreads >= 128) {
			if (p < 128 && (p+128) < nthreads) timeStep[0][p] = fminf(timeStep[0][p], timeStep[0][p + 128]); //min(256, nthreads)=>128
			__syncthreads();
		}
		if (nthreads >= 64) {
			if (p < 64 && (p+64) < nthreads) timeStep[0][p] = fminf(timeStep[0][p], timeStep[0][p + 64]); //min(128, nthreads)=>64
			__syncthreads();
		}

		//Let the last warp reduce 64 values into a single value
		//Will generate out-of-bounds errors for nthreads < 64
		if (p < 32) {
			if (nthreads >= 64) B_volatile[p] = fminf(B_volatile[p], B_volatile[p + 32]); //64=>32
			if (nthreads >= 32) B_volatile[p] = fminf(B_volatile[p], B_volatile[p + 16]); //32=>16
			if (nthreads >= 16) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  8]); //16=>8
			if (nthreads >=  8) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  4]); //8=>4
			if (nthreads >=  4) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  2]); //4=>2
			if (nthreads >=  2) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  1]); //2=>1
		}

		if (threadIdx.y + threadIdx.x == 0) flux_ctx.L[blockIdx.x*gridDim.y + blockIdx.y] = B_volatile[0];


	}

}

void callFluxKernel(dim3 grid, dim3 block, int step, FluxKernelArgs* h_ctx){

	cudaMemcpyToSymbolAsync(flux_ctx, h_ctx, sizeof(FluxKernelArgs), 0, cudaMemcpyHostToDevice);
	fluxKernel<<<grid, block>>>(step);
}



// Set wall boundry condition
__global__ void setBCPeriodic(gpu_raw_ptr U, unsigned int NX, unsigned int NY, int border){

	int threads = blockDim.x*blockDim.y;	

	float* B_in;
	float* B_out;

	int nx = NX-2*border;
	int ny = NY-2*border;

	int tid = threadIdx.y*blockDim.x+threadIdx.x;

	int kin;
	int kk;

	// SOUTH
	for (int b = 0; b < border; b++){
		B_out = global_index(U.ptr, U.pitch, 0, -1 - b, border);   
		B_in = global_index(U.ptr, U.pitch, 0, ny -1 - b, border);
		for (int k = tid; k < nx+border*2; k+=threads){
			kk = k-border;
			kin = min(kk,nx-1);
			kin = max(kin,0);			
			B_out[kk] = B_in[kin];
		}
	}

	// NORTH
	for (int b = 0; b < border; b++){
                B_out = global_index(U.ptr, U.pitch, 0, ny + b, border);   
                B_in = global_index(U.ptr, U.pitch, 0, 0 + b, border);
		for (int k = tid; k < nx+border*2; k+=threads){
			kk = k-border;
			kin = min(kk,nx-1);
			kin = max(kin,0);			
			B_out[kk] = B_in[kin];
		}

        }

	// WEST
	for (int k = tid; k < ny+border*2; k+= threads){
		kk = k-border;
        	B_out = global_index(U.ptr, U.pitch, 0, kk, border); 	
		kin = min(kk,ny-1);
		kin = max(kin,0);			
		for (int b = 0; b < border; b++)
                	B_out[-1-b] = global_index(U.ptr, U.pitch, nx -1 - b, kin, border)[0];                      
        }

	// EAST
        for (int k = tid; k < ny+border*2; k+= threads){
		kk = k-border;
                B_out = global_index(U.ptr, U.pitch, nx, kk, border);     
		kin = min(kk,ny-1);
		kin = max(kin,0);			
                for (int b = 0; b < border; b++)
                        B_out[b] = global_index(U.ptr, U.pitch, 0 + b, kin,border)[0];
        }
}


void callSetBCPeriodic(dim3 grid, dim3 block, gpu_raw_ptr U, unsigned int NX, unsigned int NY, int border){
	setBCPeriodic<<<grid, block>>>(U, NX, NY, border);
}	 

void callCollectiveSetBCPeriodic(dim3 grid, dim3 block, const collBCKernelArgs* arg){

	callSetBCPeriodic(grid, block, arg->U0, arg->NX, arg->NY, arg->global_border); 
	callSetBCPeriodic(grid, block, arg->U1, arg->NX, arg->NY, arg->global_border);
        callSetBCPeriodic(grid, block, arg->U2, arg->NX, arg->NY, arg->global_border);
        callSetBCPeriodic(grid, block, arg->U3, arg->NX, arg->NY, arg->global_border);
}


// Set wall boundry condition
__global__ void setBCOpen(gpu_raw_ptr U, unsigned int NX, unsigned int NY, int border){

	int threads = blockDim.x*blockDim.y;	

	float* B_in;
	float* B_out;

	int nx = NX-2*border;
	int ny = NY-2*border;

	int tid = threadIdx.y*blockDim.x+threadIdx.x;

	int kin;
	int kk;

	// SOUTH
	for (int b = 0; b < border; b++){
		B_out = global_index(U.ptr, U.pitch, 0, -1 - b, border);   
		B_in = global_index(U.ptr, U.pitch, 0, 0, border);
		for (int k = tid; k < nx+border*2; k+=threads){
			kk = k-border;
			kin = min(kk,nx-1);
			kin = max(kin,0);			
			B_out[kk] = B_in[kin];
		}
	}
	// NORTH
	for (int b = 0; b < border; b++){
                B_out = global_index(U.ptr, U.pitch, 0, ny + b, border);   
                B_in = global_index(U.ptr, U.pitch, 0, ny - 1, border);
		for (int k = tid; k < nx+border*2; k+=threads){
			kk = k-border;
			kin = min(kk,nx-1);
			kin = max(kin,0);			
			B_out[kk] = B_in[kin];
		}

        }

	// WEST
	for (int k = tid; k < ny+border*2; k+= threads){
		kk = k-border;
        	B_out = global_index(U.ptr, U.pitch, 0, kk, border); 	
		kin = min(kk,nx-1);
		kin = max(kin,0);			
		for (int b = 0; b < border; b++)
                	B_out[-1-b] = global_index(U.ptr, U.pitch, 0, kin, border)[0];                      
        }

	// EAST
        for (int k = tid; k < ny+border*2; k+= threads){
		kk = k-border;
                B_out = global_index(U.ptr, U.pitch, nx, kk, border);     
		kin = min(kk,nx-1);
		kin = max(kin,0);			
                for (int b = 0; b < border; b++)
                        B_out[b] = global_index(U.ptr, U.pitch, nx - 1, kin,border)[0];
        }
}

void callSetBCOpen(dim3 grid, dim3 block, gpu_raw_ptr U, unsigned int NX, unsigned int NY, int border){
	setBCOpen<<<grid, block>>>(U, NX, NY, border);
}	 


void callCollectiveSetBCOpen(dim3 grid, dim3 block, const collBCKernelArgs* arg){

	//cudaMemcpyToSymbolAsync(bc_ctx, arg->, sizeof(collBCKernelArgs), 0, cudaMemcpyHostToDevice);	
	callSetBCOpen(grid, block, arg->U0, arg->NX, arg->NY, arg->global_border); 
	callSetBCOpen(grid, block, arg->U1, arg->NX, arg->NY, arg->global_border);
        callSetBCOpen(grid, block, arg->U2, arg->NX, arg->NY, arg->global_border);
        callSetBCOpen(grid, block, arg->U3, arg->NX, arg->NY, arg->global_border);
}


// Set wall boundry condition
__global__ void setBCWall(gpu_raw_ptr U, unsigned int NX, unsigned int NY, int border){

	int threads = blockDim.x*blockDim.y;	

	float* B_in;
	float* B_out;

	int nx = NX-2*border;
	int ny = NY-2*border;

	int tid = threadIdx.y*blockDim.x+threadIdx.x;

	int kin;

	// SOUTH
	for (int b = 0; b < border; b++){
		B_out = global_index(U.ptr, U.pitch, 0, -1 - b, border);   
		B_in = global_index(U.ptr, U.pitch, 0, 0 + b, border);
		for (int k = tid-2; k < nx+border; k+=threads){
			kin = min(k,nx-1);
			kin = max(kin,0);			
			B_out[k] = B_in[kin];
		}
	}
	// NORTH
	for (int b = 0; b < border; b++){
                B_out = global_index(U.ptr, U.pitch, 0, ny + b, border);   
                B_in = global_index(U.ptr, U.pitch, 0, ny - 1 - b, border);
		for (int k = tid-2; k < nx+border; k+=threads){
			kin = min(k,nx-1);
			kin = max(kin,0);			
			B_out[k] = B_in[kin];
		}

        }

	// WEST
	for (int k = tid-2; k < ny; k+= threads){
		printf("k: %i", k);
        	B_out = global_index(U.ptr, U.pitch, 0, k, border); 	
		kin = min(k,nx-1);
		kin = max(kin,0);			
		for (int b = 0; b < border; b++)
                	B_out[-1-b] = global_index(U.ptr, U.pitch, 0 + b, kin, border)[0];                      
        }

	// EAST
        for (unsigned int k = tid; k < ny; k+= threads){
                B_out = global_index(U.ptr, U.pitch, nx, k, border);     
		kin = min(k,nx-1);
		kin = max(kin,0);			
                for (int b = 0; b < border; b++)
                        B_out[b] = global_index(U.ptr, U.pitch, nx - 1 - b, kin,border)[0];
        }

}

void callSetBCWall(dim3 grid, dim3 block, gpu_raw_ptr U, unsigned int NX, unsigned int NY, int border){
	setBCWall<<<grid, block>>>(U, NX, NY, border);
}	 


void callCollectiveSetBCWall(dim3 grid, dim3 block, const collBCKernelArgs* arg){

	callSetBCWall(grid, block, arg->U0, arg->NX, arg->NY, arg->global_border); 
	callSetBCWall(grid, block, arg->U1, arg->NX, arg->NY, arg->global_border);
        callSetBCWall(grid, block, arg->U2, arg->NX, arg->NY, arg->global_border);
        callSetBCWall(grid, block, arg->U3, arg->NX, arg->NY, arg->global_border);
}
