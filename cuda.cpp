#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<map>
#include "cuda.h"

static const char source[] = "";

int main(int argc,char *argv[])
{
    CUdevice cuDevice;
    CUcontext cuContext;
    CUfunction func;
    CUresult ret;
    CUmodule cuModule;
    cuInit(0);

    ret = cuDeviceGet(&cuDevice, 0);
    if (ret != CUDA_SUCCESS) { exit(1);}

    ret = cuCtxCreate(&cuContext, 0, cuDevice);
    if (ret != CUDA_SUCCESS) { exit(1);}

    const unsigned int jitNumOptions = 3;
    auto *jitOptions = new CUjit_option[jitNumOptions];
    auto **jitOptVals = new void*[jitNumOptions];

    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    int jitLogBufferSize = 1024*1024;
    jitOptVals[0] = (void *)jitLogBufferSize;

    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
    auto *jitLogBuffer = new char[jitLogBufferSize];
    jitOptVals[1] = jitLogBuffer;
    jitOptions[2] = CU_JIT_WALL_TIME;

    ret = cuModuleLoadDataEx( &cuModule , source , jitNumOptions, jitOptions, (void **)jitOptVals );
    if (ret != CUDA_SUCCESS) { exit(1);}
}
