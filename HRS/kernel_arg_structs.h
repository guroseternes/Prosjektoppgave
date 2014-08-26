#ifndef KARGS_H_
#define KARGS_H_

#include  "gpu_ptr.h" 


struct collBCKernelArgs {
        gpu_raw_ptr U0, U1, U2, U3; //!< input/output
        //float north_arg, south_arg, east_arg, west_arg;
        unsigned int NX, NY;
        int global_border;
};

struct BCKernelArgs {
        gpu_raw_ptr U; //!< input/output
        //float north_arg, south_arg, east_arg, west_arg;
        unsigned int NX, NY;
	int global_border;
};


struct DtKernelArgs {
        float* L;               //!< Maximal eigenvalues for each block
        float* dt;              //!< Output delta t
        unsigned int nElements;  //!< Elements in eigenvalue buffer
        float dx;                       //!< Spatial distance between cells in x direction
        float dy;                       //!< Spatial distance between cells in y direction
        float scale;            //!< Scaling of dt to guarantee to maintain stable solution
};


struct RKKernelArgs {
        gpu_raw_ptr  Q0, Q1, Q2, Q3; //!< Newly calculated Q-vector (and input Q-vector for second step of RK)
        gpu_raw_ptr  U0, U1, U2, U3; //!< U-vector at current timestep
        gpu_raw_ptr  R0, R1, R2, R3; //!< Net flux in and out of cells
        float* dt;         //!< Timestep
        float gamma;           //!< Gravitational constant
        unsigned int nx;   //!< Computational domain widht
        unsigned int ny;   //!< Computational domain height
	int global_border;
};

/**
 * Parameters used by the flux kernel
 */
struct  FluxKernelArgs {
        gpu_raw_ptr U0, U1, U2, U3;        //!< U vector given at cell midpoints
        gpu_raw_ptr R0, R1, R2, R3;        //!< Source term and net flux in and out of each cell.
       // float *active_compact_x, *active_compact_y;      //!< compacted map of active blocks for current timestep
        float* L;                      //!< Maximal eigenvalues for each block
        float dx;                      //!< Spatial distance between cells in x direction
        float dy;                      //!< Spatial distance between cells in y direction
        unsigned int nx;               //!< Domain size without ghost cells
        unsigned int ny;               //!< Domain size without ghost cells
	int global_border;		//!< Ghost cell border
       	
	float gamma; 
	float theta; 

	int innerDimX;
	int innerDimY;                     //!< For min-mod function
};

#endif /* KARGS_H_ */
