#include <cassert>
#include <chrono>
#include <iostream>
#include <random>


// ============================================================================
// SETTINGS
const int DIMENSION[]   = {2, 3, 4};
const int PERMUTATION[] = {1, 2, 0};
const int TILE          = 32;


// ============================================================================
// KERNELS
__global__ void transposeKernel(const float* d_input, float* d_output,
                                const int m, const int n) {
    __shared__ float buffer[TILE][TILE + 1];

    // read matrix in linear order
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if ((col < n) && (row < m)) {
        buffer[threadIdx.y][threadIdx.x] = d_input[row * n + col];
    }
    __syncthreads();

    // write transposed matrix in linear order
    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    if ((col < m) && (row < n)) {
        // transpose is done with buffer
        d_output[row * m + col] = buffer[threadIdx.x][threadIdx.y];
    }
}


// ============================================================================
// UTILS FUNCTIONS
inline cudaError_t CHECK_CUDA(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void array_rand_init(float* array, int size) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine            generator(seed);
    std::uniform_real_distribution<float> distribution(0, 1);
    for (int i = 0; i < size; i++)
        array[i] = distribution(generator);
}

void array_print(const float* tensor, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << tensor[i] << ", ";
    }
    std::cout << std::endl;
}

void transpose_cpu(const float* input, float* output, const int* iDim,
                   const int* p) {
    int oDim[] = {iDim[p[0]], iDim[p[1]], iDim[p[2]]};
    for (int i = 0; i < iDim[0]; ++i) {
        for (int j = 0; j < iDim[1]; ++j) {
            for (int k = 0; k < iDim[2]; ++k) {
                int idx[]  = {i, j, k};
                int odx[]  = {idx[p[0]], idx[p[1]], idx[p[2]]};
                int iIndex = (idx[0] * iDim[1] * iDim[2]) + (idx[1] * iDim[2]) +
                             (idx[2]);
                int oIndex = (odx[0] * oDim[1] * oDim[2]) + (odx[1] * oDim[2]) +
                             (odx[2]);
                output[oIndex] = input[iIndex];
            }
        }
    }
}

void process_test(const float* gold, const float* result, int size,
                  float host_ms, float device_ms) {
    // Check Correctness
    for (int i = 0; i < size; ++i) {
        if (result[i] != gold[i]) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "       i  = " << i << std::endl;
            std::cout << "  gold[i] = " << gold[i] << std::endl;
            std::cout << "result[i] = " << result[i] << std::endl;
            return;
        }
    }
    // Note: GB/sec = (Byte*1e-9)/(msec*1e-3) = (Byte*1e6)/msec
    double bandwitdh = 2 * size * sizeof(float) * 1e-6 * device_ms;
    std::cout << "Bandwidth (GB/s): " << bandwitdh << std::endl;
    std::cout << "     time (msec): " << device_ms << std::endl;
    std::cout << "     Time Speedup: " << host_ms / device_ms << "x"
              << std::endl;
    std::cout << std::endl;
}


// ============================================================================
// MAIN
int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------------
    // GET INFO
    int SIZE = 1;
    for (auto&& e : DIMENSION)
        SIZE *= e;
    const int BYTES = SIZE * sizeof(float);

    // ------------------------------------------------------------------------
    // PRINT INFO
    std::cout << "TILE:    " << TILE << std::endl;
    std::cout << "SIZE:    " << SIZE << std::endl;
    std::cout << "DIMENSION:   (";
    for (auto&& e : DIMENSION)
        std::cout << e << ", ";
    std::cout << ")" << std::endl;
    std::cout << "PERMUTATION: (";
    for (auto&& e : PERMUTATION)
        std::cout << e << ", ";
    std::cout << ")" << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // SETUP TIMERS
    float       host_ms, device_ms;
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    // HOST MEMORY ALLOCATION
    float* h_input  = new float[SIZE]{};
    float* h_output = new float[SIZE]{};
    float* h_gold   = new float[SIZE]{};

    // HOST INITIALIZATION
    for (int i = 0; i < SIZE; ++i)
        h_input[i] = i * 1.0f;
    // array_rand_init(h_input, SIZE);


    // HOST EXECUTION
    start = std::chrono::steady_clock::now();
    transpose_cpu(h_input, h_gold, DIMENSION, PERMUTATION);
    end     = std::chrono::steady_clock::now();
    host_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() *
              1e-6;
    std::cout << "Host Time (msec): " << host_ms << std::endl;


    // TEST
    array_print(h_input, SIZE);
    array_print(h_gold, SIZE);


    // //
    // ------------------------------------------------------------------------
    // // DEVICE MEMORY ALLOCATION
    // float *d_input, *d_output;
    // CHECK_CUDA(cudaMalloc(&d_input, BYTES));
    // CHECK_CUDA(cudaMalloc(&d_output, BYTES));

    // // COPY DATA FROM HOST TO DEVICE
    // CHECK_CUDA(cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice));

    // // KERNEL EXECUTE
    // CHECK_CUDA(cudaEventRecord(startEvent, 0));
    // transposeKernel<<<DimGrid, DimBlock>>>(d_input, d_output, M, N);
    // CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    // CHECK_CUDA(cudaEventSynchronize(stopEvent));
    // CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));

    // // COPY DATA FROM DEVICE TO HOST
    // CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES,
    // cudaMemcpyDeviceToHost)); process_test(h_gold, h_output, SIZE, host_ms,
    // device_ms);


    // //
    // ------------------------------------------------------------------------
    // // DEVICE MEMORY DEALLOCATION
    // CHECK_CUDA(cudaEventDestroy(startEvent));
    // CHECK_CUDA(cudaEventDestroy(stopEvent));

    // CHECK_CUDA(cudaFree(d_input));
    // CHECK_CUDA(cudaFree(d_output));

    // cudaDeviceReset();

    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
    delete[] h_gold;

    return 0;
}
