/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void VecAdd(float* A, float* B, float* C, int numElements) {
    int i = blockDim.x *blockIdx.x + threadIdx.x;

    if (i < numElements)
        C[i] = A[i] + B[i];
}


/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 20;

    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float* h_A = (float*) malloc(size);
    float* h_B = (float*) malloc(size);
    float* h_C = (float*) malloc(size);
    if(h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Error allocating the host buffer!\n");
        return EXIT_FAILURE;
    }


    // initialize host vectors
    for (int i=0; i < numElements; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        printf("init: %f | %f\n", h_A[i], h_B[i]);
    }


    float* d_A, *d_B, *d_C;
    err = cudaMalloc((void**) &d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating %d bytes on the device!\nError: %s\n", size, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**) &d_B, size);
    if (err != cudaSuccess)  {
        fprintf(stderr, "Error allocating %d bytes on the device!\nError: %s\n", size, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**) &d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating %d bytes on the device!\nError: %s\n", size, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("copy vectors to the device!\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying buffer A to the device: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying buffer B to the device: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // call kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error invoking kernel function: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }


    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)  {
        fprintf(stderr, "Error copying result buffer C back to the host: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    for (int i=0; i < numElements; i++) {
        printf("%d: %f + %f = %f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);






    printf("Done\n");
    return 0;
}

