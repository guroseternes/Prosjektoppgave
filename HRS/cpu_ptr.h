#ifndef CPU_PTR_H_
#define CPU_PTR_H_

#include <stdio.h>
#include <string>
#include <stdlib.h>
//#include <iostream>
#include <cassert>
#include <vector>

class cpu_ptr_2D{
public:
	// Regular constructor
	cpu_ptr_2D(unsigned int nx, unsigned int ny, unsigned int border, bool setToZero = false);

	// Copy constructor
	cpu_ptr_2D(const cpu_ptr_2D& other);

	// Deconstructor
	~cpu_ptr_2D();
	
	float xmin, xmax, ymin, ymax;

	int get_nx();
	int get_ny();

	float get_dx();
	float get_dy();

	float* get_ptr(){return data;};

	void set_time(float time);

	cpu_ptr_2D& operator=(const cpu_ptr_2D& rhs);

	// Access elements
	float &operator()(unsigned int i, unsigned int j);		

	void printToFile(FILE* filePtr, bool withHeader = false, bool withBorder = false);

private:
	unsigned int nx, ny, border, NX, NY;
	float time;
	float *data;
 	void allocateMemory();
};


float &cpu_ptr_2D::operator() (unsigned int i, unsigned int j){
	return data[((j) + border)*(nx + 2*border) + i + border];
}

cpu_ptr_2D::cpu_ptr_2D(unsigned int nx, unsigned int ny, unsigned int border, bool setToZero): nx(nx), ny(ny), border(border), NX(nx+2*border), NY(ny+2*border), xmin(0), ymin(0), xmax(1.0), ymax(1.0),time(0){
	allocateMemory();
	if (setToZero){
		for (int j = 0; j < NY; j++){
			for (int i = 0; i < NX; i++){ 
				this->operator()(i-border,j-border) = 0.0;
			}	
		}
	}
}

cpu_ptr_2D::cpu_ptr_2D(const cpu_ptr_2D& other):nx(other.nx),ny(other.ny),border(other.border),NX(other.NX),NY(other.NY),xmin(other.xmin), ymin(other.ymin),xmax(other.xmax),ymax(other.ymax),time(other.time){
	allocateMemory();
	for (int i = 0; i < NX*NY; i++){
		data[i] = other.data[i];
	}
}

cpu_ptr_2D::~cpu_ptr_2D(){
	delete [] data;
}

void cpu_ptr_2D::allocateMemory(){
	data = new float[NX*NY];
}

int cpu_ptr_2D::get_nx(){
	return nx;
}

int cpu_ptr_2D::get_ny(){
	return ny;
}

float cpu_ptr_2D::get_dx(){
	return ((xmax - xmin)/(float)nx);
}

float cpu_ptr_2D::get_dy(){
        return ((ymax - ymin)/(float)ny);
}

void cpu_ptr_2D::set_time(float time){
	time = time;
}


cpu_ptr_2D &cpu_ptr_2D::operator = (const cpu_ptr_2D &rhs){
	if(this == &rhs){
		return *this;
	}else{
		if (NX != rhs.NX || NY != rhs.NY){
			this->~cpu_ptr_2D();
			nx = rhs.nx; ny = rhs.ny; border=rhs.border;
			NX = rhs.NX; NY = rhs.NY;
			allocateMemory();
		}
		for (int i = 0; i < NX*NY; i++){
			data[i]=rhs.data[i];
		}	
	}
}

void cpu_ptr_2D::printToFile(FILE* filePtr, bool withHeader, bool withBorder){

	float dx = (xmax - xmin)/(float)nx;
	float dy = (ymax -ymin)/(float)ny;
	if (not withBorder){	
		if (withHeader){
			fprintf(filePtr, "nx: %i ny: %i\nborder: %i\t time: %f\n", nx, ny, border,time);
			fprintf(filePtr, "xmin: %.1f xmax: %.1f\nymin: %.1f ymax: %.1f\n", xmin, xmax, ymin, ymax);
		}
		for (int j=0; j<nx; j++){
			for (int i=0; i<nx; i++){
				fprintf(filePtr, "%.3f\t%.3f\t%.3f\t%.3f\n", time, xmin + dx*i, ymin + dy*j, this->operator()(i,j));
			}
		}
	}else{
		if (withHeader){
			fprintf(filePtr, "nx: %i ny: %i\nborder: %i\t time: %f\n", NX, NY, border,time);
			fprintf(filePtr, "xmin: %.1f xmax: %.1f\nymin: %.1f ymax: %.1f\n", xmin, xmax, ymin, ymax);
		}
		for (int j=0; j<NY; j++){
			for (int i=0; i<NX; i++){
				fprintf(filePtr, "%.3f\t%.3f\t%.3f\t%.3f\n", time, xmin + dx*i-dx*border, ymin + dy*j-dy*border, this->operator()(i-border,j-border));
			}
		}

	}

}




#endif
