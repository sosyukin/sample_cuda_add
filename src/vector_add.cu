#include <cuda_runtime.h>
#include <helper_cuda.h>
#define assert(x) \
    if (!x) exit(1)
__global__ void vector_add(const float *a, const float *b, float * c, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    cudaError_t err = cudaSuccess;
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    if (h_a == nullptr || h_b == nullptr || h_c == nullptr)
        return 1;
    for (int i = 0; i < numElements; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }
    float *d_a = nullptr;
    err = cudaMalloc((void**)&d_a, size);
    assert(err == cudaSuccess); 
    float *d_b = nullptr;
    err = cudaMalloc((void**)&d_b, size);
    assert(err == cudaSuccess);
    float *d_c = nullptr;
    err = cudaMalloc((void**)&d_c, size);
    assert(err == cudaSuccess);
    
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
    err = cudaGetLastError();
    assert(err == cudaSuccess);
    
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    for (int i = 0; i < numElements; ++i) {
        assert(fabs(h_a[i] + h_b[i] - h_c[i]) < 1e-5);
    }
    err = cudaFree(d_a);
    err = cudaFree(d_b);
    err = cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
