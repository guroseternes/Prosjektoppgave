#ifndef ICsquare_H_
#define ICsquare_H_

//Class for storing a square initial condition Riemann problem, where each quadrant has its own initial values 
#include <math.h>
#include <iostream>
#include <cassert>
#include <vector>
#define PI 3.1415926535897932384626433832795 

class ICsinus{
public:
	ICsinus(float xmin = 0, float xmax = 1, float ymin = 0, float ymax = 1);
	
	float setIC(cpu_ptr_2D &rho, cpu_ptr_2D &rho_u, cpu_ptr_2D &rho_v, cpu_ptr_2D &E, cpu_ptr_2D &radioactive);

	void exactSolution(cpu_ptr_2D &rho, float time);

private:
	float xmin, ymin, xmax, ymax;
	float u,v,p,gamNew;
};


ICsinus::ICsinus(float xmin, float xmax, float ymin, float ymax):xmin(xmin),ymin(ymin), xmax(xmax), ymax(ymax), u(1.0), v(-0.5f), p(1.0f), gamNew(1.4f){}


float ICsinus::setIC(cpu_ptr_2D &rho, cpu_ptr_2D &rho_u, cpu_ptr_2D &rho_v, cpu_ptr_2D &E, cpu_ptr_2D &radioactive){	
	float polutionCount = 0;
	int nx = rho.get_nx();
	int ny = rho.get_ny();
	float dx = (xmax-xmin)/(float) nx;
	float dy = (ymax-ymin)/(float) ny;
	float x, y;

	rho.xmin = xmin;
	rho.xmax = xmax;	
	rho.ymin = ymin;
	rho.ymax = ymax;
	
	rho_u.xmin = xmin;
	rho_u.xmin = xmin;
	rho_u.xmax = xmax;	
	rho_u.ymin = ymin;
	
	rho_v.ymax = ymax;
	rho_v.xmax = xmax;	
	rho_v.ymin = ymin;
	rho_v.ymax = ymax;
	
	E.xmin = xmin;
	E.xmax = xmax;	
	E.ymin = ymin;
	E.ymax = ymax;
	
	float counter = 0;

	for (int i = 0; i < nx; i++){
		x = dx*i+dx*0.5f+xmin ;
		for (int j=0; j < ny; j++){
			y = dy*j+dy*0.5f+ymin;
			rho(i,j) = 1.0f + 0.2f*sinf(PI*(x+y));
			rho_u(i,j) = u*rho(i,j);
			rho_v(i,j) = v*rho(i,j);
			E(i,j) = p/(gamNew -1.0f) + 0.5f*rho(i,j)*(u*u + v*v);

			
			if ( (x)*(x) + (y)*(y)  <= 0.05){
				radioactive(i,j) = 1*rho(i,j);
				polutionCount += radioactive(i,j);
			}

		}
	}
/*
	for (int i = 0; i < nx; i++){
		x = dx*i+dx*0.5f+xmin ;
		for (int j=0; j < ny; j++){
			y = dy*j+dy*0.5f+ymin;
			if ( (x)*(x) + (y)*(y)  <= 0.05){
				radioactive(i,j) = radioactive(i,j)/counter;
				polutionCount += radioactive(i,j);
				counter++;
			}

		}
	}
*/
	return polutionCount;
}

void ICsinus::exactSolution(cpu_ptr_2D &rho, float time){

	int nx = rho.get_nx();
	int ny = rho.get_ny();
	float dx = (xmax-xmin)/(float) nx;
	float dy = (ymax-ymin)/(float) ny;
	float x, y;
	
	for (int i = 0; i < nx; i++){
		x = dx*i + dx*0.5;
		for (int j=0; j < ny; j++){
			y = dy*j+dy*0.5;
			rho(i,j) = 1.0f + 0.2f*sinf(PI*(x+y-time*(u+v)));
		}
	}
}	
 


class ICsquare{
public:
	// Constructor, assumes we are dealing with the positive unit sqaure, but this is optional
	ICsquare(float x_intersect, float y_intersect,float gam, float xmin = 0,float xmax = 1,float  ymin = 0,float ymax = 1);
	
	void set_rho(float* rho);
	void set_pressure(float* pressure);
	void set_u(float* u);
	void set_v(float* v);
	void set_gam(float gam);
	
	float setIC(cpu_ptr_2D &rho, cpu_ptr_2D &rho_u, cpu_ptr_2D &rho_v, cpu_ptr_2D &E, cpu_ptr_2D &rad);


private:
	// Where quadrant division lines intersect
	float x_intersect, y_intersect;
	
	// Initial pressure and density
	float pressure_array[4];
	float rho_array[4];

	// Initial speeds in x and y-directions
	float u_array[4];
	float v_array[4];

	//x and y limits for the sqaure
	float xmin, ymin, xmax, ymax, gam;
};

ICsquare::ICsquare(float x_intersect, float y_intersect, float gam, float xmin, float xmax, float ymin, float ymax):x_intersect(x_intersect), y_intersect(y_intersect),gam(gam),\
xmin(xmin),ymin(ymin), xmax(xmax), ymax(ymax){
}

void ICsquare::set_gam(float gam){
	gam = gam;
}

void ICsquare::set_rho(float* rho){
	for (int i=0; i<4; i++)
		rho_array[i] = rho[i];
}

void ICsquare::set_u(float* u){
	for (int i=0; i<4; i++)
                u_array[i] = u[i];
}

void ICsquare::set_v(float* v){
	for (int i=0; i<4; i++)
                v_array[i] = v[i];
}

void ICsquare::set_pressure(float* pressure){
	for (int i=0; i<4; i++)
                pressure_array[i] = pressure[i];
}

float ICsquare::setIC(cpu_ptr_2D &rho, cpu_ptr_2D &rho_u, cpu_ptr_2D &rho_v, cpu_ptr_2D &E, cpu_ptr_2D &radioactive){
	int nx = rho.get_nx();
	int ny = rho.get_ny();
	float fluid = 1000;
	float dx = (xmax-xmin)/(float) nx;
	float dy = (ymax-ymin)/(float) ny;
	float x, y;
	float polutionCount = 0;
	int quad;
	for (int i = 0; i < nx; i++){
		x = dx*i;
		for (int j=0; j < ny; j++){
			y = dy*j;
			// Quadrant 1
			if (x >= x_intersect && y >= y_intersect)
				quad = 0;
			// Quadrant 2
			else if (x < x_intersect && y >= y_intersect)
				quad = 1;
			// Quadrant 3
			else if ( x < x_intersect && y < y_intersect)
				quad = 2;
			// Quadrant 4
			else
				quad = 3;
			// Set initial values
			rho(i,j) = rho_array[quad];
			//printf("%.3f ", rho(i,j));
			rho_u(i,j) = rho_array[quad]*u_array[quad];
			rho_v(i,j) = rho_array[quad]*v_array[quad];
			E(i,j) = pressure_array[quad]/(gam -1.0f) + 0.5f*rho_array[quad]*(u_array[quad]*u_array[quad] + v_array[quad]*v_array[quad]);

				
			if ( (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)  <= 0.05){
				radioactive(i,j) = 1*rho(i,j);
				polutionCount += radioactive(i,j);
			}


/*			if ( x==0.5 && y ==0.5){
				radioactive(i,j) = 1;
				polutionCount += 1;
			}


			if ( (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)  <= 0.02){
				radioactive(i,j) = 1;
				polutionCount += 1;
			}
*/				
		}
	}
	return polutionCount;

}
							
  

	




#endif
