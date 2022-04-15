#include "CheckError.cuh"
// #include "Timer.cuh"
// using namespace timer;

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>


// ============================================================================
// GlOBAL VARS
const bool print      = 0;
const int  NUM_TESTS  = 100;
const int  M          = 1024;
const int  N          = 1024;
const int  TILE       = 32;
const int  TILE_SPLIT = 4;
const dim3 DimBlock1(TILE, TILE, 1);
const dim3 DimGrid1(std::ceil((float)N / TILE), std::ceil((float)M / TILE), 1);
const dim3 DimBlock2(TILE, TILE / TILE_SPLIT, 1);
const dim3 DimGrid2(std::ceil((float)N / TILE), std::ceil((float)M / TILE), 1);

const int SIZE  = M * N;
const int BYTES = SIZE * sizeof(float);


// ============================================================================
// KERNELS
__global__ void kCopy(const float* d_input, float* d_output, const int m,
                      const int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < m) {
        d_output[row * n + col] = d_input[row * n + col];
    }
}

__global__ void kTranspose_naive(const float* d_input, float* d_output,
                                 const int m, const int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < m) {
        d_output[col * m + row] = d_input[row * n + col];
    }
}

__global__ void kTranspose_shm(const float* d_input, float* d_output,
                               const int m, const int n) {
    __shared__ float buffer[TILE][TILE];

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

__global__ void kTranspose_shm_bank(const float* d_input, float* d_output,
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

__global__ void kTranspose_naive_index(const float* d_input, float* d_output,
                                       const int m, const int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y * TILE_SPLIT + threadIdx.y;

    if (col < n && row < m) {
        for (int j = 0; j < TILE; j += TILE / TILE_SPLIT)
            d_output[col * m + (row + j)] = d_input[(row + j) * n + col];
    }
}

__global__ void kTranspose_shm_index(const float* d_input, float* d_output,
                                     const int m, const int n) {
    __shared__ float buffer[TILE][TILE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y * TILE_SPLIT + threadIdx.y;

    if (col < n && row < m) {
        for (int j = 0; j < TILE; j += TILE / TILE_SPLIT)
            buffer[threadIdx.y + j][threadIdx.x] = d_input[(row + j) * n + col];
    }
    __syncthreads();

    col = blockIdx.y * blockDim.x + threadIdx.x;
    row = blockIdx.x * blockDim.y * TILE_SPLIT + threadIdx.y;
    if (col < n && row < m) {
        for (int j = 0; j < TILE; j += TILE / TILE_SPLIT)
            d_output[(row + j) * m + col] =
                buffer[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void kTranspose_shm_bank_index(const float* d_input, float* d_output,
                                          const int m, const int n) {
    __shared__ float buffer[TILE][TILE + 1];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y * TILE_SPLIT + threadIdx.y;

    if (col < n && row < m) {
        for (int j = 0; j < TILE; j += TILE / TILE_SPLIT)
            buffer[threadIdx.y + j][threadIdx.x] = d_input[(row + j) * n + col];
    }
    __syncthreads();

    col = blockIdx.y * blockDim.x + threadIdx.x;
    row = blockIdx.x * blockDim.y * TILE_SPLIT + threadIdx.y;
    if (col < n && row < m) {
        for (int j = 0; j < TILE; j += TILE / TILE_SPLIT)
            d_output[(row + j) * m + col] =
                buffer[threadIdx.x][threadIdx.y + j];
    }
}


// ============================================================================
// UTILS FUNCTIONS

void printMatrix(const float* matrix, int m, int n, std::string name) {
    if (print) {
        std::cout << name << std::endl;
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                std::cout << matrix[row * n + col] << " \n"[col == n - 1];
            }
        }
        std::cout << std::endl;
    }
}

void transpose_cpp(const float* matrix, float* result, int m, int n) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            result[col * M + row] = matrix[row * N + col];
        }
    }
}

bool equals_array(const float* gold, const float* result, int size) {
    bool correct = true;
    for (int i = 0; i < size; ++i)
        if (result[i] != gold[i]) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "       i  = " << i << std::endl;
            std::cout << "  gold[i] = " << gold[i] << std::endl;
            std::cout << "result[i] = " << result[i] << std::endl;
            correct = false;
            break;
        }
    return correct;
}

void process_test(const float* gold, const float* result, int size,
                  float copy_bandwidth, float host_ms, float ms) {
    printMatrix(result, N, M, "output");
    bool correct = equals_array(gold, result, size);
    if (correct) {
        double bandwitdh = 2 * size * sizeof(float) * 1e-6 * NUM_TESTS / ms;
        std::cout << "Bandwidth (GB/s): " << bandwitdh << std::endl;
        double time = ms / NUM_TESTS;
        std::cout << "     time (msec): " << time << std::endl;
        double speedup_bandwitdh = bandwitdh / copy_bandwidth;
        double speedup_time      = host_ms / time;
        std::cout << "Bandwidth Speedup: " << speedup_bandwitdh << "x"
                  << std::endl;
        std::cout << "     Time Speedup: " << speedup_time << "x" << std::endl;
    }
    std::cout << std::endl;
}


// ============================================================================
// MAIN
int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------------
    // PRINT INFO
    std::cout << "M:       " << M << std::endl;
    std::cout << "N:       " << N << std::endl;
    std::cout << "TILE:    " << TILE << std::endl;
    std::cout << "DimBlock1: " << DimBlock1.x << ", " << DimBlock1.y << ", "
              << DimBlock1.z << std::endl;
    std::cout << "DimGrid1: " << DimGrid1.x << ", " << DimGrid1.y << ", "
              << DimGrid1.z << std::endl;
    std::cout << "DimBlock2: " << DimBlock2.x << ", " << DimBlock2.y << ", "
              << DimBlock2.z << std::endl;
    std::cout << "DimGrid2: " << DimGrid2.x << ", " << DimGrid2.y << ", "
              << DimGrid2.z << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // SETUP TIMERS
    cudaEvent_t startEvent, stopEvent;
    SAFE_CALL(cudaEventCreate(&startEvent));
    SAFE_CALL(cudaEventCreate(&stopEvent));
    float host_ms, device_ms;


    // HOST MEMORY ALLOCATION
    float* h_input  = new float[M * N]{};
    float* h_output = new float[M * N]{};
    float* h_gold   = new float[M * N]{};

    // HOST INITIALIZATION
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = i * 1.0f;
    }
    printMatrix(h_input, M, N, "input");

    // HOST EXECUTION
    SAFE_CALL(cudaEventRecord(startEvent, 0));
    transpose_cpp(h_input, h_gold, M, N);
    SAFE_CALL(cudaEventRecord(stopEvent, 0));
    SAFE_CALL(cudaEventSynchronize(stopEvent));
    SAFE_CALL(cudaEventElapsedTime(&host_ms, startEvent, stopEvent));
    printMatrix(h_gold, N, M, "gold");
    std::cout << "Host Time (msec): " << host_ms << std::endl;


    // ------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    float *d_input, *d_output;
    SAFE_CALL(cudaMalloc(&d_input, BYTES))
    SAFE_CALL(cudaMalloc(&d_output, BYTES))

    // COPY DATA FROM HOST TO DEVICE
    SAFE_CALL(cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice))
    // SAFE_CALL(cudaMemcpy(d_output, h_output, BYTES, cudaMemcpyHostToDevice))


    // ------------------------------------------------------------------------
    // GENERARE COPY BANDWITH
    SAFE_CALL(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kCopy<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);  // warmup
    SAFE_CALL(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_TESTS; ++i)
        kCopy<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    SAFE_CALL(cudaEventRecord(stopEvent, 0));
    SAFE_CALL(cudaEventSynchronize(stopEvent));
    SAFE_CALL(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA_ERROR
    SAFE_CALL(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost))
    bool   correct = equals_array(h_input, h_output, SIZE);
    double copy_bandwidth =
        2 * SIZE * sizeof(float) * 1e-6 * NUM_TESTS / device_ms;
    std::cout << "Copy Bandwidth (GB/s): " << copy_bandwidth << std::endl
              << std::endl;


    // ------------------------------------------------------------------------
    std::cout << "TEST: transpose naive" << std::endl;
    SAFE_CALL(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_naive<<<DimGrid1, DimBlock1>>>(d_input, d_output, M,
                                              N);  // warmup
    SAFE_CALL(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_TESTS; ++i)
        kTranspose_naive<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    SAFE_CALL(cudaEventRecord(stopEvent, 0));
    SAFE_CALL(cudaEventSynchronize(stopEvent));
    SAFE_CALL(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA_ERROR
    SAFE_CALL(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost))
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout << "TEST: transpose with shared memory" << std::endl;
    SAFE_CALL(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm<<<DimGrid1, DimBlock1>>>(d_input, d_output, M,
                                            N);  // warmup
    SAFE_CALL(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_TESTS; ++i)
        kTranspose_shm<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    SAFE_CALL(cudaEventRecord(stopEvent, 0));
    SAFE_CALL(cudaEventSynchronize(stopEvent));
    SAFE_CALL(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA_ERROR
    SAFE_CALL(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost))
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout
        << "TEST: transpose with shared memory and bank conflict avoidance"
        << std::endl;
    SAFE_CALL(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm_bank<<<DimGrid1, DimBlock1>>>(d_input, d_output, M,
                                                 N);  // warmup
    SAFE_CALL(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_TESTS; ++i)
        kTranspose_shm_bank<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    SAFE_CALL(cudaEventRecord(stopEvent, 0));
    SAFE_CALL(cudaEventSynchronize(stopEvent));
    SAFE_CALL(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA_ERROR
    SAFE_CALL(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost))
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout << "TEST: transpose naive with reduced index calculation"
              << std::endl;
    SAFE_CALL(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_naive_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                    N);  // warmup
    SAFE_CALL(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_TESTS; ++i)
        kTranspose_naive_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                        N);
    SAFE_CALL(cudaEventRecord(stopEvent, 0));
    SAFE_CALL(cudaEventSynchronize(stopEvent));
    SAFE_CALL(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA_ERROR
    SAFE_CALL(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost))
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout
        << "TEST: transpose with shared memory and reduced index calculation"
        << std::endl;
    SAFE_CALL(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                  N);  // warmup
    SAFE_CALL(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_TESTS; ++i)
        kTranspose_shm_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M, N);
    SAFE_CALL(cudaEventRecord(stopEvent, 0));
    SAFE_CALL(cudaEventSynchronize(stopEvent));
    SAFE_CALL(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA_ERROR
    SAFE_CALL(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost))
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout << "TEST: transpose with shared memory, bank conflict "
                 "avoidance and reduced index calculation"
              << std::endl;
    SAFE_CALL(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm_bank_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                       N);  // warmup
    SAFE_CALL(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_TESTS; ++i)
        kTranspose_shm_bank_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                           N);
    SAFE_CALL(cudaEventRecord(stopEvent, 0));
    SAFE_CALL(cudaEventSynchronize(stopEvent));
    SAFE_CALL(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA_ERROR
    SAFE_CALL(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost))
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL(cudaFree(d_input))
    SAFE_CALL(cudaFree(d_output))

    cudaDeviceReset();

    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
    delete[] h_gold;

    return 0;
}