#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>


// ============================================================================
// SETTINGS
const int NUM_REPS = 100;
const int TILE     = 2;
const int BLOCK_X  = TILE;  // dim2
const int BLOCK_Y  = TILE;  // dim1
const int BLOCK_Z  = TILE;  // dim0

const int DIMENSION[]   = {2, 3, 4};
const int PERMUTATION[] = {1, 2, 0};
// const int PERMUTATION[] = {1, 2, 0};


// ============================================================================
// CUDA SECTION
inline cudaError_t CHECK_CUDA(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void matrix_transpose_kernel(const float* d_input, float* d_output,
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

__global__ void copy_kernel(const float* d_input, float* d_output, int dim0,
                            int dim1, int dim2, int p0, int p1, int p2) {
    int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx1 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx0 < dim0 && idx1 < dim1 && idx2 < dim2) {
        int iIndex       = (idx0 * dim1 * dim2) + (idx1 * dim2) + idx2;
        d_output[iIndex] = d_input[iIndex];
    }
}

__global__ void transpose_naive_kernel(const float* d_input, float* d_output,
                                       int dim0, int dim1, int dim2, int p0,
                                       int p1, int p2) {
    int idx[3] = {
        blockIdx.z * blockDim.z + threadIdx.z,  // i
        blockIdx.y * blockDim.y + threadIdx.y,  // j
        blockIdx.x * blockDim.x + threadIdx.x,  // k
    };
    if (idx[0] < dim0 && idx[1] < dim1 && idx[2] < dim2) {
        int iDim[3] = {dim0, dim1, dim2};
        int oDim[3] = {iDim[p0], iDim[p1], iDim[p2]};
        int odx[3]  = {idx[p0], idx[p1], idx[p2]};
        int iIndex = (idx[0] * iDim[1] * iDim[2]) + (idx[1] * iDim[2]) + idx[2];
        int oIndex = (odx[0] * oDim[1] * oDim[2]) + (odx[1] * oDim[2]) + odx[2];
        d_output[oIndex] = d_input[iIndex];
    }
}

__global__ void transpose_shmem_kernel3(const float* d_input, float* d_output,
                                        int dim0, int dim1, int dim2, int p0,
                                        int p1, int p2) {
    __shared__ float buffer[TILE][TILE][TILE];
    int              idx[3] = {
        blockIdx.z * blockDim.z + threadIdx.z,  // i
        blockIdx.y * blockDim.y + threadIdx.y,  // j
        blockIdx.x * blockDim.x + threadIdx.x,  // k
    };

    int iDim[3] = {dim0, dim1, dim2};
    if (idx[0] < iDim[0] && idx[1] < iDim[1] && idx[2] < iDim[2]) {
        int iIndex = (idx[0] * iDim[1] * iDim[2]) + (idx[1] * iDim[2]) + idx[2];
        buffer[threadIdx.z][threadIdx.y][threadIdx.x] = d_input[iIndex];
    }

    __syncthreads();

    int oDim[3] = {iDim[p0], iDim[p1], iDim[p2]};
    int odx[3]  = {idx[p0], idx[p1], idx[p2]};
    if (odx[0] < oDim[0] && odx[1] < oDim[1] && odx[2] < oDim[2]) {
        int threads[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
        int oIndex = (idx[0] * iDim[1] * iDim[2]) + (idx[1] * iDim[2]) + idx[2];
        d_output[oIndex] = buffer[threads[p0]][threads[p1]][threads[p2]];
    }
}

__global__ void transpose_shmem_kernel2(const float* d_input, float* d_output,
                                        int dim0, int dim1, int dim2, int p0,
                                        int p1, int p2) {
    // Permutation = (1, 2, 0)
    __shared__ float buffer[TILE][TILE][TILE];

    int iDim[3] = {dim0, dim1, dim2};
    int i       = blockIdx.z * TILE + threadIdx.z;
    int j       = blockIdx.y * TILE + threadIdx.y;
    int k       = blockIdx.x * TILE + threadIdx.x;
    if (i < iDim[0] && j < iDim[1] && k < iDim[2]) {
        int iIndex = (i * iDim[1] * iDim[2]) + (j * iDim[2]) + k;
        buffer[threadIdx.z][threadIdx.y][threadIdx.x] = d_input[iIndex];
    }
    __syncthreads();

    int oDim[3] = {dim1, dim2, dim0};
    i           = blockIdx.y * TILE + threadIdx.z;
    j           = blockIdx.x * TILE + threadIdx.y;
    k           = blockIdx.z * TILE + threadIdx.x;
    if (i < oDim[0] && j < oDim[1] && k < oDim[2]) {
        int threads[3]   = {threadIdx.z, threadIdx.y, threadIdx.x};
        int oIndex       = (i * oDim[1] * oDim[2]) + (j * oDim[2]) + k;
        d_output[oIndex] = buffer[threadIdx.x][threadIdx.z][threadIdx.y];
    }
}

__global__ void transpose_shmem_kernel(const float* d_input, float* d_output,
                                       int dim0, int dim1, int dim2, int p0,
                                       int p1, int p2) {
    __shared__ float buffer[TILE][TILE][TILE];

    int iDim[3] = {dim0, dim1, dim2};
    int i       = blockIdx.z * TILE + threadIdx.z;
    int j       = blockIdx.y * TILE + threadIdx.y;
    int k       = blockIdx.x * TILE + threadIdx.x;
    if (i < iDim[0] && j < iDim[1] && k < iDim[2]) {
        int iIndex = (i * iDim[1] * iDim[2]) + (j * iDim[2]) + k;
        buffer[threadIdx.z][threadIdx.y][threadIdx.x] = d_input[iIndex];
    }
    __syncthreads();

    int oDim[3]   = {iDim[p0], iDim[p1], iDim[p2]};
    int blocks[3] = {blockIdx.z, blockIdx.y, blockIdx.x};
    i             = blocks[p0] * TILE + threadIdx.z;
    j             = blocks[p1] * TILE + threadIdx.y;
    k             = blocks[p2] * TILE + threadIdx.x;
    if (i < oDim[0] && j < oDim[1] && k < oDim[2]) {
        int threads[3]   = {threadIdx.z, threadIdx.y, threadIdx.x};
        int targets[3]   = {threads[p0], threads[p1], threads[p2]};
        int oIndex       = (i * oDim[1] * oDim[2]) + (j * oDim[2]) + k;
        d_output[oIndex] = buffer[targets[p0]][targets[p1]][targets[p2]];
    }
}

__global__ void kTranspose_shm(const float* d_input, float* d_output,
                               const int dim0, const int dim1) {
    __shared__ float buffer[TILE][TILE];

    // read matrix in linear order
    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;
    if ((j < dim1) && (i < dim0)) {
        buffer[threadIdx.y][threadIdx.x] = d_input[i * dim1 + j];
    }
    __syncthreads();

    // write transposed matrix in linear order
    i = blockIdx.x * TILE + threadIdx.y;
    j = blockIdx.y * TILE + threadIdx.x;
    if ((j < dim0) && (i < dim1)) {
        // transpose is done with buffer
        d_output[i * dim0 + j] = buffer[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// C++ SECTION
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
                int idx[] = {i, j, k};
                int odx[] = {idx[p[0]], idx[p[1]], idx[p[2]]};
                int iIndex =
                    (idx[0] * iDim[1] * iDim[2]) + (idx[1] * iDim[2]) + idx[2];
                int oIndex =
                    (odx[0] * oDim[1] * oDim[2]) + (odx[1] * oDim[2]) + odx[2];
                output[oIndex] = input[iIndex];
            }
        }
    }
}

void array_check(const float* gold, const float* result, int size) {
    // Check Correctness
    for (int i = 0; i < size; ++i) {
        if (result[i] != gold[i]) {
            std::cout << "!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!" << std::endl;
            std::cout << "       i  = " << i << std::endl;
            std::cout << "  gold[i] = " << gold[i] << std::endl;
            std::cout << "result[i] = " << result[i] << std::endl;
            return;
        }
    }
}


// ============================================================================
// MAIN
int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------------
    // GET INFO
    int        size  = DIMENSION[0] * DIMENSION[1] * DIMENSION[2];
    const int  bytes = size * sizeof(float);
    const dim3 DimBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 DimGrid(std::ceil((float)DIMENSION[2] / BLOCK_X),
                       std::ceil((float)DIMENSION[1] / BLOCK_Y),
                       std::ceil((float)DIMENSION[0] / BLOCK_Z));
    // ------------------------------------------------------------------------
    // PRINT INFO
    std::cout << "TILE:    " << TILE << std::endl;
    std::cout << "size:    " << size << std::endl;
    std::cout << "DIMENSION:   (" << DIMENSION[0] << ", " << DIMENSION[1]
              << ", " << DIMENSION[2] << ")" << std::endl;
    std::cout << "PERMUTATION: (" << PERMUTATION[0] << ", " << PERMUTATION[1]
              << ", " << PERMUTATION[2] << ")" << std::endl;
    std::cout << "DimBlock:    (" << DimBlock.x << ", " << DimBlock.y << ", "
              << DimBlock.z << ")" << std::endl;
    std::cout << "DimGrid:     (" << DimGrid.x << ", " << DimGrid.y << ", "
              << DimGrid.z << ")" << std::endl;
    std::cout << std::endl;


    // ------------------------------------------------------------------------
    // SETUP TIMERS
    float       host_ms, kernel_ms;
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;


    // ------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION AND INITIALIZATION
    float* h_input  = new float[size]{};
    float* h_output = new float[size]{};
    float* h_gold   = new float[size]{};

    for (int i = 0; i < size; ++i)
        h_input[i] = i * 1.0f;
    // array_rand_init(h_input, size);


    // ------------------------------------------------------------------------
    // HOST EXECUTION
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_REPS; ++i)
        transpose_cpu(h_input, h_gold, DIMENSION, PERMUTATION);
    end     = std::chrono::steady_clock::now();
    host_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() *
              1e-6 / NUM_REPS;
    std::cout << "       Host Time (ms): " << host_ms << std::endl;


    // ------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION AND INITIALIZATION
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));


    // ------------------------------------------------------------------------
    // GENERARE COPY BANDWITH
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    copy_kernel<<<DimGrid, DimBlock>>>(
        d_input, d_output, DIMENSION[0], DIMENSION[1], DIMENSION[2],
        PERMUTATION[0], PERMUTATION[1], PERMUTATION[2]);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        copy_kernel<<<DimGrid, DimBlock>>>(
            d_input, d_output, DIMENSION[0], DIMENSION[1], DIMENSION[2],
            PERMUTATION[0], PERMUTATION[1], PERMUTATION[2]);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    array_check(h_input, h_output, size);
    kernel_ms /= NUM_REPS;
    float copy_bandwidth = 2 * bytes * 1e-6 / kernel_ms;
    std::cout << "Copy Bandwidth (GB/s):  " << copy_bandwidth << std::endl;


    // ------------------------------------------------------------------------
    // TRANSPOSE NAIVE
    std::cout << "\nTranspose Naive" << std::endl;
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose_naive_kernel<<<DimGrid, DimBlock>>>(
        d_input, d_output, DIMENSION[0], DIMENSION[1], DIMENSION[2],
        PERMUTATION[0], PERMUTATION[1], PERMUTATION[2]);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose_naive_kernel<<<DimGrid, DimBlock>>>(
            d_input, d_output, DIMENSION[0], DIMENSION[1], DIMENSION[2],
            PERMUTATION[0], PERMUTATION[1], PERMUTATION[2]);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    array_check(h_gold, h_output, size);
    kernel_ms /= NUM_REPS;
    std::cout << "       Time (ms): " << kernel_ms << std::endl;
    std::cout << " Bandwidth(GB/s): " << 2 * bytes * 1e-6 / kernel_ms
              << std::endl;
    std::cout << "    Speedup (ms): " << host_ms / kernel_ms << "x"
              << std::endl;


    // ------------------------------------------------------------------------
    // TRANSPOSE WITH SHARED MEMORY
    std::cout << "\nTranspose with shared Memory" << std::endl;
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose_shmem_kernel<<<DimGrid, DimBlock>>>(
        d_input, d_output, DIMENSION[0], DIMENSION[1], DIMENSION[2],
        PERMUTATION[0], PERMUTATION[1], PERMUTATION[2]);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose_shmem_kernel<<<DimGrid, DimBlock>>>(
            d_input, d_output, DIMENSION[0], DIMENSION[1], DIMENSION[2],
            PERMUTATION[0], PERMUTATION[1], PERMUTATION[2]);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    array_check(h_gold, h_output, size);
    kernel_ms /= NUM_REPS;
    std::cout << "       Time (ms): " << kernel_ms << std::endl;
    std::cout << " Bandwidth(GB/s): " << 2 * bytes * 1e-6 / kernel_ms
              << std::endl;
    std::cout << "    Speedup (ms): " << host_ms / kernel_ms << "x"
              << std::endl;


    // ------------------------------------------------------------------------
    // PRINT
    array_print(h_input, size);
    array_print(h_gold, size);
    array_print(h_output, size);


    // ------------------------------------------------------------------------
    // MATRIX TRANSPOSE
    // std::cout << "\nMatrix Transpose" << std::endl;
    // const dim3 DimBlockMatrix(BLOCK_X, BLOCK_Y, 1);
    // const dim3 DimGridMatrix(std::ceil((float)DIMENSION[1] / TILE),
    //                          std::ceil((float)DIMENSION[2] / TILE), 1);
    // CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    // matrix_transpose_kernel<<<DimGridMatrix, DimBlockMatrix>>>(
    //     d_input, d_output, DIMENSION[1], DIMENSION[2]);  // warmup
    // CHECK_CUDA(cudaEventRecord(startEvent, 0));
    // for (int i = 0; i < NUM_REPS; ++i)
    //     matrix_transpose_kernel<<<DimGridMatrix, DimBlockMatrix>>>(
    //         d_input, d_output, DIMENSION[1], DIMENSION[2]);
    // CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    // CHECK_CUDA(cudaEventSynchronize(stopEvent));
    // CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    // CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes,
    // cudaMemcpyDeviceToHost)); array_check(h_gold, h_output, size);
    // std::cout
    // << "         Time (ms): " << kernel_ms << std::endl; std::cout << "
    // Bandwidth(GB / s): "
    //           << 2 * DIMENSION[1] * DIMENSION[3] * sizeof(float) * 1e-6 *
    //                  NUM_REPS / kernel_ms
    //           << std::endl;
    // std::cout << "      Speedup (ms): " << host_ms / kernel_ms << "x"
    //           << std::endl;

    // ------------------------------------------------------------------------
    // CLEAN SHUTDOWN
    delete[] h_input;
    delete[] h_output;
    delete[] h_gold;

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    cudaDeviceReset();

    return 0;
}
