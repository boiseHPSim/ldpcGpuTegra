#ifndef __TEST_H__
#define __TEST_H__

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>


#include <cuda.h>
#include <cuda_runtime.h>

class test
{
public:
	test();
	~test();
	void runMytest();
};


#endif // __TEST_H__
