#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include <ap_int.h>

typedef float DataType;

#define SIDE_LEN 180								// The length of a side of the systolic array
//#define ARRAY_SIZE		(SIDE_LEN*SIDE_LEN)

extern "C"{
void systolic_array(int row, int col, int col1, DataType din_a[], DataType din_b[], DataType bias[], DataType out[]);
}

#endif



