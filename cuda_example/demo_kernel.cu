#include <cassert>
#include <chrono>
#include <iostream>


// ============================================================================
// GlOBAL VARS
const bool PRINT      = 0;
const int  NUM_REPS   = 100;
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

__global__ void kCopy_shm(const float* d_input, float* d_output, const int m,
                          const int n) {
    __shared__ float buffer[TILE][TILE];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if ((col < n) && (row < m)) {
        buffer[threadIdx.y][threadIdx.x] = d_input[row * n + col];
    }
    __syncthreads();

    if ((col < m) && (row < n)) {
        d_output[row * m + col] = buffer[threadIdx.y][threadIdx.x];
    }
}

__global__ void kCopy_shm_bank(const float* d_input, float* d_output,
                               const int m, const int n) {
    __shared__ float buffer[TILE][TILE + 1];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if ((col < n) && (row < m)) {
        buffer[threadIdx.y][threadIdx.x] = d_input[row * n + col];
    }
    __syncthreads();

    if ((col < m) && (row < n)) {
        d_output[row * m + col] = buffer[threadIdx.y][threadIdx.x];
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
inline cudaError_t CHECK_CUDA(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void printMatrix(const float* matrix, int m, int n, std::string name) {
    if (PRINT) {
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

bool check_array(const float* gold, const float* result, int size) {
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
    bool correct = check_array(gold, result, size);
    if (correct) {
        // Note: GB/sec = (Byte*1e-9)/(msec*1e-3) = (Byte*1e6)/msec
        double bandwitdh = 2 * size * sizeof(float) * 1e-6 * NUM_REPS / ms;
        std::cout << "Bandwidth (GB/s): " << bandwitdh << std::endl;
        double time = ms / NUM_REPS;
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
    float       host_ms, device_ms;
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

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
    start = std::chrono::steady_clock::now();
    transpose_cpp(h_input, h_gold, M, N);
    end     = std::chrono::steady_clock::now();
    host_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() *
              1e-6;
    printMatrix(h_gold, N, M, "gold");
    std::cout << "Host Time (msec): " << host_ms << std::endl;


    // ------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, BYTES));
    CHECK_CUDA(cudaMalloc(&d_output, BYTES));

    // COPY DATA FROM HOST TO DEVICE
    CHECK_CUDA(cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(d_output, h_output, BYTES,
    // cudaMemcpyHostToDevice));


    // ------------------------------------------------------------------------
    // GENERARE COPY BANDWITH and TEST shm, bank
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kCopy<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kCopy<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    check_array(h_input, h_output, SIZE);
    double copy_bandwidth =
        2 * SIZE * sizeof(float) * 1e-6 * NUM_REPS / device_ms;
    std::cout << "Copy Bandwidth (GB/s):            " << copy_bandwidth
              << std::endl;

    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kCopy_shm<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kCopy_shm<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    check_array(h_input, h_output, SIZE);
    std::cout << "Copy+SHMEM Bandwidth (GB/s):      "
              << (2 * SIZE * sizeof(float) * 1e-6 * NUM_REPS / device_ms)
              << std::endl;

    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kCopy_shm_bank<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kCopy_shm_bank<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    check_array(h_input, h_output, SIZE);
    std::cout << "Copy+SHMEM+BANK Bandwidth (GB/s): "
              << (2 * SIZE * sizeof(float) * 1e-6 * NUM_REPS / device_ms)
              << std::endl
              << std::endl;


    // ------------------------------------------------------------------------
    std::cout << "TEST: transpose naive" << std::endl;
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_naive<<<DimGrid1, DimBlock1>>>(d_input, d_output, M,
                                              N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kTranspose_naive<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout << "TEST: transpose with shared memory" << std::endl;
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm<<<DimGrid1, DimBlock1>>>(d_input, d_output, M,
                                            N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kTranspose_shm<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout
        << "TEST: transpose with shared memory and bank conflict avoidance"
        << std::endl;
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm_bank<<<DimGrid1, DimBlock1>>>(d_input, d_output, M,
                                                 N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kTranspose_shm_bank<<<DimGrid1, DimBlock1>>>(d_input, d_output, M, N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout << "TEST: transpose naive with reduced index calculation"
              << std::endl;
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_naive_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                    N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kTranspose_naive_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                        N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout
        << "TEST: transpose with shared memory and reduced index calculation"
        << std::endl;
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                  N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kTranspose_shm_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M, N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    std::cout << "TEST: transpose with shared memory, bank conflict "
                 "avoidance and reduced index calculation"
              << std::endl;
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm_bank_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                       N);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kTranspose_shm_bank_index<<<DimGrid2, DimBlock2>>>(d_input, d_output, M,
                                                           N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    process_test(h_gold, h_output, SIZE, copy_bandwidth, host_ms, device_ms);


    // ------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    cudaDeviceReset();

    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
    delete[] h_gold;

    return 0;
}
