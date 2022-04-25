#include <cassert>
#include <chrono>
#include <iostream>


// ============================================================================
// GlOBAL VARS
const int  NUM_REPS = 100;
const int  M        = 10240;
const int  N        = 10240;
const int  TILE     = 32;
const dim3 DimBlock(TILE, TILE, 1);
const dim3 DimGrid(std::ceil((float)N / TILE), std::ceil((float)M / TILE), 1);

const int SIZE  = M * N;
const int BYTES = SIZE * sizeof(float);


// ============================================================================
// KERNELS
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


// ============================================================================
// UTILS FUNCTIONS
inline cudaError_t CHECK_CUDA(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
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
                  float host_ms, float ms) {
    bool correct = check_array(gold, result, size);
    if (correct) {
        // Note: GB/sec = (Byte*1e-9)/(msec*1e-3) = (Byte*1e6)/msec
        double bandwitdh = 2 * size * sizeof(float) * 1e-6 * NUM_REPS / ms;
        std::cout << "Bandwidth (GB/s): " << bandwitdh << std::endl;
        double time = ms / NUM_REPS;
        std::cout << "     time (msec): " << time << std::endl;
        std::cout << "     Time Speedup: " << host_ms / time << "x"
                  << std::endl;
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
    std::cout << "DimBlock: " << DimBlock.x << ", " << DimBlock.y << ", "
              << DimBlock.z << std::endl;
    std::cout << "DimGrid: " << DimGrid.x << ", " << DimGrid.y << ", "
              << DimGrid.z << std::endl;
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

    // HOST EXECUTION
    start = std::chrono::steady_clock::now();
    transpose_cpp(h_input, h_gold, M, N);
    end     = std::chrono::steady_clock::now();
    host_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() *
              1e-6;
    std::cout << "Host Time (msec): " << host_ms << std::endl;

    // ------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, BYTES));
    CHECK_CUDA(cudaMalloc(&d_output, BYTES));

    // COPY DATA FROM HOST TO DEVICE
    CHECK_CUDA(cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice));

    // KERNEL EXECUTE
    CHECK_CUDA(cudaMemset(d_output, 0, BYTES));  // Initialize output
    kTranspose_shm_bank<<<DimGrid, DimBlock>>>(d_input, d_output, M,
                                               N);  // warm-up
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        kTranspose_shm_bank<<<DimGrid, DimBlock>>>(d_input, d_output, M, N);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));

    // COPY DATA FROM DEVICE TO HOST
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    process_test(h_gold, h_output, SIZE, host_ms, device_ms);


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
