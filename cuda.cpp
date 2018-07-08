#include<cstdlib>
#include<iostream>
#include <cassert>
#include<string>
#include<map>
#include <chrono>
#include "cuda.h"
#include "nvrtc.h"

static const char* source = "\textern \"C\" __global__ void vecAdd(int *a, int *b, int *c, int n)\n"
                            "\t{\n"
                            "\tint id =  threadIdx.x;\t"
                            "\tif (id < n)\n"
                            "\tc[id] = a[id] + b[id];\n"
                            "}\t";


void checkCudaErrors(CUresult err) {
    if (err != CUDA_SUCCESS) {
        std::cout << err << std::endl;
    }
    assert(err == CUDA_SUCCESS);
}

const int N = 1024 * 1024 * 512;

int main(int argc,char *argv[])
{
    int size = N;
    int i;
    nvrtcProgram program;
    nvrtcResult result = nvrtcCreateProgram ( &program, source, nullptr, 0, nullptr, nullptr);
    printf("%s\n", nvrtcGetErrorString(result));
    nvrtcResult compileResult =  nvrtcCompileProgram ( program, 0, nullptr);
    printf("%s\n", nvrtcGetErrorString(compileResult));

    size_t logSize;
    nvrtcGetProgramLogSize(program, &logSize);
    auto *log = new char[logSize];
    nvrtcGetProgramLog(program, log);

    size_t ptxSize;
    nvrtcGetPTXSize(program, &ptxSize);
    auto *ptx = new char[ptxSize];
    nvrtcGetPTX(program, ptx);

    printf("%s", ptx);

    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction function;

    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
    checkCudaErrors(cuModuleLoadDataEx( &cuModule , ptx , 0, nullptr, nullptr));
    checkCudaErrors(cuModuleGetFunction(&function, cuModule, "vecAdd"));

    auto *Host_a = (int*)malloc(sizeof(int)*N);
    auto *Host_b = (int*)malloc(sizeof(int)*N);
    auto *Host_c = (int*)malloc(sizeof(int)*N);
    for(i = 0; i < N; i++) {
        Host_a[i] = i;
        Host_b[i] = N - i;
    }


    CUdeviceptr devBufferA;
    checkCudaErrors(cuMemAlloc(&devBufferA, sizeof(int) * N));
    checkCudaErrors(cuMemcpyHtoD(devBufferA, Host_a, sizeof(int) * N));

    CUdeviceptr devBufferB;
    checkCudaErrors(cuMemAlloc(&devBufferB, sizeof(int) * N));
    checkCudaErrors(cuMemcpyHtoD(devBufferB, Host_b, sizeof(int) * N));

    CUdeviceptr devBufferC;
    checkCudaErrors(cuMemAlloc(&devBufferC, sizeof(int) * N));

    void* KernelParams[] = {&devBufferA, &devBufferB, &devBufferC, &size};

    unsigned  int blockSize, gridSize;

    blockSize = 1024;

    gridSize = (unsigned int)ceil((float)N/blockSize);

    auto start = std::chrono::high_resolution_clock::now();

    checkCudaErrors(cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0,
                                   nullptr, KernelParams, nullptr));

    auto done = std::chrono::high_resolution_clock::now();

    checkCudaErrors(cuMemcpyDtoH(Host_c, devBufferC, N*sizeof(int)));

    long resultTime = std::chrono::duration_cast<std::chrono::nanoseconds>(done - start).count();

    printf("Time spent: %ld\n", resultTime);

    checkCudaErrors(cuMemFree(devBufferA));
    checkCudaErrors(cuMemFree(devBufferB));
    checkCudaErrors(cuMemFree(devBufferC));
    checkCudaErrors(cuModuleUnload(cuModule));
    checkCudaErrors(cuCtxDestroy(cuContext));

}
