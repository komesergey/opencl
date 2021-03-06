#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static const char source[] =
"__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {\n"
    "int i = get_global_id(0);\n"
    "C[i] = A[i] + B[i];\n"
"}\n";

long testSpeed(cl_device_id current_id){
    int i;
    cl_int ret;
    const int LIST_SIZE = 1024 * 1024 * 512;
    auto *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    auto *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    auto start = std::chrono::high_resolution_clock::now();

    const char *source_str = &source[0];
    size_t source_size;
    source_size = sizeof(source);
    cl_context context = clCreateContext(nullptr, 1, &current_id, nullptr, nullptr, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, current_id, 0, &ret);
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), nullptr, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), nullptr, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int), nullptr, &ret);
    clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), B, 0, nullptr, nullptr);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    clBuildProgram(program, 1, &current_id, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    size_t global_item_size = LIST_SIZE;
    size_t local_item_size = 64;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_item_size, &local_item_size, 0, nullptr, nullptr);
    auto *C = (int*)malloc(sizeof(int)*LIST_SIZE);
    clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), C, 0, nullptr, nullptr);

    auto done = std::chrono::high_resolution_clock::now();

    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(a_mem_obj);
    clReleaseMemObject(b_mem_obj);
    clReleaseMemObject(c_mem_obj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return std::chrono::duration_cast<std::chrono::milliseconds>(done - start).count();
}

int main() {
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;

    clGetPlatformIDs(0, nullptr, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, nullptr);

    for (i = 0; i < platformCount; i++) {
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        devices = (cl_device_id *) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, nullptr);
        for (j = 0; j < deviceCount; j++) {
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, nullptr, &valueSize);
            value = (char *) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, nullptr);
            printf("%d. Device: %s\n", j + 1, value);
            long time = testSpeed(devices[j]);
            printf("Time: %ld\n", time);
            free(value);
        }
        free(devices);
    }

    free(platforms);
    return 0;
}