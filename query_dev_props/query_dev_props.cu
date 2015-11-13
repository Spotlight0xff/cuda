#include <stdio.h>

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i=0; i < nDevices; i++) {
        cudaDeviceProp prop;
        // query the device properties of the i-th device
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("\tDevice Name: %s\n", prop.name);
        printf("\tMajor compute capability: %d.%d\n", prop.major, prop.minor);
        printf("\tDevice Global Memory: %f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("\tShared Memory per Block: %d bytes\n", prop.sharedMemPerBlock);
        printf("\tMap Host Memory available (pinned Memory): %s\n", prop.canMapHostMemory ? "true": "false");
        printf("\tMemory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("\tMemory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("\tPeak Memory Bandwidth: %f GB/s\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8)/1.0e6);
        printf("\tNumber of asynchronous engines: %d\n", prop.asyncEngineCount);
        printf("\tL2 Cache bytes: %d\n", prop.l2CacheSize);
        printf("\tConcurrent Kernels: %d\n", prop.concurrentKernels);
    }
}
