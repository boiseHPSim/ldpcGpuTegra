
#include "test.h"

#define checkCudaErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void testKernel(int val)
{
    printf("hi\n");
}

test::test()
{

}

test::~test()
{

}

void test::runMytest()
{
//    int devID;
//    cudaDeviceProp props;
//
//    // This will pick the best possible CUDA capable device
//    devID = 0;

    //Get GPU information
//    checkCudaErrors(cudaGetDevice(&devID));
//    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
//    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
//           devID, props.name, props.major, props.minor);
//
//    printf("printf() is called. Output:\n\n");

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    testKernel<<<dimGrid, dimBlock>>>(10);
    cudaDeviceSynchronize();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
//    cudaDeviceReset();
}
