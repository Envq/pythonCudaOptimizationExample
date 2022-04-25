#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>


// ============================================================================
// SETTINGS
const int  NUM_REPS     = 100;
const bool ENABLE_PRINT = false;

const int TILE3D    = 8;
const int TILE2D_v1 = 32;
const int TILE2D_v2 = 8;


// ============================================================================
// CUDA SECTION
inline cudaError_t CHECK_CUDA(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// COPY
__global__ void copy3d_simple_kernel(const float* d_input, float* d_output,
                                     int dimz, int dimy, int dimx) {
    int x     = blockIdx.x * blockDim.x + threadIdx.x;
    int y     = blockIdx.y * blockDim.y + threadIdx.y;
    int z     = blockIdx.z * blockDim.z + threadIdx.z;
    int index = (z * dimy * dimx) + (y * dimx) + x;

    if (z < dimz && y < dimy && x < dimx) {
        d_output[index] = d_input[index];
    }
}

__global__ void copy2d_simple_kernel(const float* d_input, float* d_output,
                                     const int dimy, const int dimx) {
    int x     = blockIdx.x * blockDim.x + threadIdx.x;
    int y     = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * dimx + x;

    if (y < dimy && x < dimx) {
        d_output[index] = d_input[index];
    }
}

// TRANSPOSE SIMPLE
template<int pz, int py, int px>
__global__ void transpose3d_simple_kerneltmplt(const float* d_input,
                                               float* d_output, int dimz,
                                               int dimy, int dimx) {
    int idx[3]  = {blockIdx.z * blockDim.z + threadIdx.z,
                  blockIdx.y * blockDim.y + threadIdx.y,
                  blockIdx.x * blockDim.x + threadIdx.x};
    int iDim[3] = {dimz, dimy, dimx};
    int oDim[3] = {iDim[pz], iDim[py], iDim[px]};
    int odx[3]  = {idx[pz], idx[py], idx[px]};
    int iIndex  = (idx[0] * iDim[1] * iDim[2]) + (idx[1] * iDim[2]) + idx[2];
    int oIndex  = (odx[0] * oDim[1] * oDim[2]) + (odx[1] * oDim[2]) + odx[2];

    if (idx[0] < dimz && idx[1] < dimy && idx[2] < dimx) {
        d_output[oIndex] = d_input[iIndex];
    }
}

void transpose3d_simple_selector(const dim3& DimGrid, const dim3& DimBlock,
                                 const float* d_input, float* d_output,
                                 const int* dim, const int* perm) {
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
        transpose3d_simple_kerneltmplt<0, 1, 2>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        transpose3d_simple_kerneltmplt<0, 2, 1>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2) {
        transpose3d_simple_kerneltmplt<1, 0, 2>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 2 && perm[2] == 0) {
        transpose3d_simple_kerneltmplt<1, 2, 0>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 0 && perm[2] == 1) {
        transpose3d_simple_kerneltmplt<2, 0, 1>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 1 && perm[2] == 0) {
        transpose3d_simple_kerneltmplt<2, 1, 0>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    }
}

__global__ void transpose2d_simple_kernel(const float* d_input, float* d_output,
                                          const int dimy, const int dimx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < dimy && x < dimx) {
        d_output[x * dimy + y] = d_input[y * dimx + x];
    }
}

// TRANSPOSE SHARED MEMORY
template<int pz, int py, int px>
__global__ void transpose3d_shm_kerneltmplt(const float* d_input,
                                            float* d_output, int dimz, int dimy,
                                            int dimx) {
    __shared__ float buffer[TILE3D][TILE3D][TILE3D];

    int iDim[3] = {dimz, dimy, dimx};
    int x       = blockIdx.x * TILE3D + threadIdx.x;
    int y       = blockIdx.y * TILE3D + threadIdx.y;
    int z       = blockIdx.z * TILE3D + threadIdx.z;
    if (z < iDim[0] && y < iDim[1] && x < iDim[2]) {
        int iIndex     = (z * iDim[1] * iDim[2]) + (y * iDim[2]) + x;
        int threads[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
        buffer[threads[pz]][threads[py]][threads[px]] = d_input[iIndex];
    }
    __syncthreads();

    int oDim[3]   = {iDim[pz], iDim[py], iDim[px]};
    int blocks[3] = {blockIdx.z, blockIdx.y, blockIdx.x};
    x             = blocks[px] * TILE3D + threadIdx.x;
    y             = blocks[py] * TILE3D + threadIdx.y;
    z             = blocks[pz] * TILE3D + threadIdx.z;
    if (z < oDim[0] && y < oDim[1] && x < oDim[2]) {
        int oIndex       = (z * oDim[1] * oDim[2]) + (y * oDim[2]) + x;
        d_output[oIndex] = buffer[threadIdx.z][threadIdx.y][threadIdx.x];
    }
}

void transpose3d_shm_selector(const dim3& DimGrid, const dim3& DimBlock,
                              const float* d_input, float* d_output,
                              const int* dim, const int* perm) {
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
        transpose3d_shm_kerneltmplt<0, 1, 2>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        transpose3d_shm_kerneltmplt<0, 2, 1>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2) {
        transpose3d_shm_kerneltmplt<1, 0, 2>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 2 && perm[2] == 0) {
        transpose3d_shm_kerneltmplt<1, 2, 0>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 0 && perm[2] == 1) {
        transpose3d_shm_kerneltmplt<2, 0, 1>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 1 && perm[2] == 0) {
        transpose3d_shm_kerneltmplt<2, 1, 0>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    }
}

__global__ void transpose2d_shm_kernel_v1(const float* d_input, float* d_output,
                                          const int dimy, const int dimx) {
    __shared__ float buffer[TILE2D_v1][TILE2D_v1];

    int x = blockIdx.x * TILE2D_v1 + threadIdx.x;
    int y = blockIdx.y * TILE2D_v1 + threadIdx.y;
    if (y < dimy && x < dimx) {
        buffer[threadIdx.y][threadIdx.x] = d_input[y * dimx + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE2D_v1 + threadIdx.x;
    y = blockIdx.x * TILE2D_v1 + threadIdx.y;
    if (y < dimx && x < dimy) {
        d_output[y * dimy + x] = buffer[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose2d_shm_kernel_v2(const float* d_input, float* d_output,
                                          const int dimy, const int dimx) {
    __shared__ float buffer[TILE2D_v2][TILE2D_v2];

    int x = blockIdx.x * TILE2D_v2 + threadIdx.x;
    int y = blockIdx.y * TILE2D_v2 + threadIdx.y;
    if (y < dimy && x < dimx) {
        buffer[threadIdx.y][threadIdx.x] = d_input[y * dimx + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE2D_v2 + threadIdx.x;
    y = blockIdx.x * TILE2D_v2 + threadIdx.y;
    if (y < dimx && x < dimy) {
        d_output[y * dimy + x] = buffer[threadIdx.x][threadIdx.y];
    }
}

// TRANSPOSE SHARED MEMORY + BANK CONFLICT FREE
template<int pz, int py, int px, int tilez, int tiley, int tilex>
__global__ void transpose3d_shm_bank_kerneltmplt(const float* d_input,
                                                 float* d_output, int dimz,
                                                 int dimy, int dimx) {
    __shared__ float buffer[tilez][tiley][tilex];

    int iDim[3] = {dimz, dimy, dimx};
    int x       = blockIdx.x * TILE3D + threadIdx.x;
    int y       = blockIdx.y * TILE3D + threadIdx.y;
    int z       = blockIdx.z * TILE3D + threadIdx.z;
    if (z < iDim[0] && y < iDim[1] && x < iDim[2]) {
        int iIndex     = (z * iDim[1] * iDim[2]) + (y * iDim[2]) + x;
        int threads[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
        buffer[threads[pz]][threads[py]][threads[px]] = d_input[iIndex];
    }
    __syncthreads();

    int oDim[3]   = {iDim[pz], iDim[py], iDim[px]};
    int blocks[3] = {blockIdx.z, blockIdx.y, blockIdx.x};
    x             = blocks[px] * TILE3D + threadIdx.x;
    y             = blocks[py] * TILE3D + threadIdx.y;
    z             = blocks[pz] * TILE3D + threadIdx.z;
    if (z < oDim[0] && y < oDim[1] && x < oDim[2]) {
        int oIndex       = (z * oDim[1] * oDim[2]) + (y * oDim[2]) + x;
        d_output[oIndex] = buffer[threadIdx.z][threadIdx.y][threadIdx.x];
    }
}

void transpose3d_shm_bank_selector(const dim3& DimGrid, const dim3& DimBlock,
                                   const float* d_input, float* d_output,
                                   const int* dim, const int* perm) {
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
        transpose3d_shm_bank_kerneltmplt<0, 1, 2, 8, 8, 8>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        transpose3d_shm_bank_kerneltmplt<0, 2, 1, 8, 8, 12>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2) {
        transpose3d_shm_bank_kerneltmplt<1, 0, 2, 8, 9, 8>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 2 && perm[2] == 0) {
        transpose3d_shm_bank_kerneltmplt<1, 2, 0, 8, 10, 10>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 0 && perm[2] == 1) {
        transpose3d_shm_bank_kerneltmplt<2, 0, 1, 8, 8, 9>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 1 && perm[2] == 0) {
        transpose3d_shm_bank_kerneltmplt<2, 1, 0, 8, 12, 9>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    }
}

__global__ void transpose2d_shm_bank_kernel_v1(const float* d_input,
                                               float* d_output, const int dimy,
                                               const int dimx) {
    __shared__ float buffer[TILE2D_v1][TILE2D_v1 + 1];

    int x = blockIdx.x * TILE2D_v1 + threadIdx.x;
    int y = blockIdx.y * TILE2D_v1 + threadIdx.y;
    if (y < dimy && x < dimx) {
        buffer[threadIdx.y][threadIdx.x] = d_input[y * dimx + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE2D_v1 + threadIdx.x;
    y = blockIdx.x * TILE2D_v1 + threadIdx.y;
    if (y < dimx && x < dimy) {
        d_output[y * dimy + x] = buffer[threadIdx.x][threadIdx.y];
    }
}
__global__ void transpose2d_shm_bank_kernel_v2(const float* d_input,
                                               float* d_output, const int dimy,
                                               const int dimx) {
    __shared__ float buffer[TILE2D_v2][TILE2D_v2 + 1];

    int x = blockIdx.x * TILE2D_v2 + threadIdx.x;
    int y = blockIdx.y * TILE2D_v2 + threadIdx.y;
    if (y < dimy && x < dimx) {
        buffer[threadIdx.y][threadIdx.x] = d_input[y * dimx + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE2D_v2 + threadIdx.x;
    y = blockIdx.x * TILE2D_v2 + threadIdx.y;
    if (y < dimx && x < dimy) {
        d_output[y * dimy + x] = buffer[threadIdx.x][threadIdx.y];
    }
}


// ============================================================================
// C++ SECTION
void array_init_rand(float* array, int size) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine            generator(seed);
    std::uniform_real_distribution<float> distribution(0, 1);
    for (int i = 0; i < size; i++)
        array[i] = distribution(generator);
}

void array_init_seq(float* array, int size) {
    for (int i = 0; i < size; ++i)
        array[i] = i * 1.0f;
}

void array_print(const float* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

void transpose_cpu(const float* matrix, float* result, int m, int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            result[col * m + row] = matrix[row * n + col];
        }
    }
}

bool array_check(const float* gold, const float* result, int size) {
    for (int i = 0; i < size; ++i) {
        if (result[i] != gold[i]) {
            return false;
        }
    }
    return true;
}

void process(std::string name, bool testbench_mode, const float* gold,
             const float* result, int size, float kernel_ms, float host_ms,
             std::ofstream& file, bool print_speedup) {
    kernel_ms /= NUM_REPS;
    bool  is_correct = array_check(gold, result, size);
    float bandwidth  = 2 * size * sizeof(float) * 1e-6 / kernel_ms;
    float speedup    = host_ms / kernel_ms;

    if (ENABLE_PRINT) {
        array_print(gold, size);
        array_print(result, size);
    }

    if (!testbench_mode) {
        std::cout << name << std::endl;
        std::cout << "            Check: " << (is_correct ? "OK" : "FAIL")
                  << std::endl;
        if (print_speedup) {
            std::cout << "        Time (ms): " << kernel_ms << std::endl;
            std::cout << "     Speedup (ms): " << speedup << "x" << std::endl;
        }
        std::cout << " Bandwidth (GB/s): " << bandwidth << std::endl;
        std::cout << std::endl;
    } else {
        file << name << std::endl;
        file << is_correct << std::endl;
        file << kernel_ms << std::endl;
        file << speedup << std::endl;
        file << bandwidth << std::endl;
    }
}


// ============================================================================
// MAIN
int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------------
    // GET ARGS
    if (argc < 3) {
        std::cout << "call: executable dim_y dim_x" << std::endl;
        std::cout << "example: ./demo_matrix.out 32 32" << std::endl;
        return 0;
    }
    int testbench_mode = false;
    if (argc == 4 && std::string(argv[3]) == "testbench") {
        testbench_mode = true;
    }

    // ------------------------------------------------------------------------
    // GET INFO
    int        dim_y   = std::stoi(argv[1]);
    int        dim_x   = std::stoi(argv[2]);
    int        dim[3]  = {1, dim_y, dim_x};
    int        perm[3] = {0, 2, 1};
    int        size    = dim_y * dim_x;
    const int  bytes   = size * sizeof(float);
    const dim3 DimBlock3D(TILE3D, TILE3D, TILE3D);
    const dim3 DimGrid3D(std::ceil((float)dim_x / DimBlock3D.x),
                         std::ceil((float)dim_y / DimBlock3D.y), 1);
    const dim3 DimBlock2D_v1(TILE2D_v1, TILE2D_v1, 1);
    const dim3 DimGrid2D_v1(std::ceil((float)dim_x / DimBlock2D_v1.x),
                            std::ceil((float)dim_y / DimBlock2D_v1.y), 1);
    const dim3 DimBlock2D_v2(TILE2D_v2, TILE2D_v2, 1);
    const dim3 DimGrid2D_v2(std::ceil((float)dim_x / DimBlock2D_v2.x),
                            std::ceil((float)dim_y / DimBlock2D_v2.y), 1);

    std::string str_tile2d_v1 =
        "(" + std::to_string(TILE2D_v1) + "," + std::to_string(TILE2D_v1) + ")";
    std::string str_tile2d_v2 =
        "(" + std::to_string(TILE2D_v2) + "," + std::to_string(TILE2D_v2) + ")";
    std::string str_tile3d = "(" + std::to_string(TILE3D) + "," +
                             std::to_string(TILE3D) + "," +
                             std::to_string(TILE3D) + ")";

    // ------------------------------------------------------------------------
    // SETUP TIMERS
    float       host_ms, device_ms, kernel_ms;
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

    // array_init_rand(h_input, size);
    array_init_seq(h_input, size);

    // ------------------------------------------------------------------------
    // HOST EXECUTION
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_REPS; ++i)
        transpose_cpu(h_input, h_gold, dim_y, dim_x);
    end     = std::chrono::steady_clock::now();
    host_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() *
              1e-6 / NUM_REPS;

    // ------------------------------------------------------------------------
    // PRINT INFO
    std::ofstream log;
    if (!testbench_mode) {
        std::cout << "size:    " << size << std::endl;
        std::cout << "dimension:   (" << dim_y << ", " << dim_x << ")"
                  << std::endl;
        std::cout << "DimBlock 2D V1:    (" << DimBlock2D_v1.x << ", "
                  << DimBlock2D_v1.y << ", " << DimBlock2D_v1.z << ")"
                  << std::endl;
        std::cout << "DimGrid 2D V1:     (" << DimGrid2D_v1.x << ", "
                  << DimGrid2D_v1.y << ", " << DimGrid2D_v1.z << ")"
                  << std::endl;
        std::cout << "DimBlock 2D V2:    (" << DimBlock2D_v2.x << ", "
                  << DimBlock2D_v2.y << ", " << DimBlock2D_v2.z << ")"
                  << std::endl;
        std::cout << "DimGrid 2D V2:     (" << DimGrid2D_v2.x << ", "
                  << DimGrid2D_v2.y << ", " << DimGrid2D_v2.z << ")"
                  << std::endl;
        std::cout << "DimBlock 3D:    (" << DimBlock3D.x << ", " << DimBlock3D.y
                  << ", " << DimBlock3D.z << ")" << std::endl;
        std::cout << "DimGrid 3D:     (" << DimGrid3D.x << ", " << DimGrid3D.y
                  << ", " << DimGrid3D.z << ")" << std::endl;
        std::cout << "Host Time (ms): " << host_ms << std::endl;
        std::cout << std::endl;
    } else {
        std::string log_name = "logs/logs_matrix/";
        log_name += std::to_string(dim_y) + "x" + std::to_string(dim_x);
        log_name += ".log";
        log.open(log_name, std::ios::out);
        log << host_ms << std::endl;
    }

    // ------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION AND INITIALIZATION
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // ========================================================================
    // COPY 2D BANDWIDTH SIMPLE V1
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    copy2d_simple_kernel<<<DimGrid2D_v1, DimBlock2D_v1>>>(d_input, d_output,
                                                          dim_y,
                                                          dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        copy2d_simple_kernel<<<DimGrid2D_v1, DimBlock2D_v1>>>(d_input, d_output,
                                                              dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Copy 2D simple " + str_tile2d_v1, testbench_mode, h_input,
            h_output, size, kernel_ms, host_ms, log, false);

    // ------------------------------------------------------------------------
    // TRANSPOSE 2D SIMPLE V1
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose2d_simple_kernel<<<DimGrid2D_v1, DimBlock2D_v1>>>(
        d_input, d_output, dim_y,
        dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose2d_simple_kernel<<<DimGrid2D_v1, DimBlock2D_v1>>>(
            d_input, d_output, dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Transpose 2D simple " + str_tile2d_v1, testbench_mode, h_gold,
            h_output, size, kernel_ms, host_ms, log, true);

    // ------------------------------------------------------------------------
    // TRANSPOSE 2D SHARED-MEMORY V1
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose2d_shm_kernel_v1<<<DimGrid2D_v1, DimBlock2D_v1>>>(
        d_input, d_output, dim_y,
        dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose2d_shm_kernel_v1<<<DimGrid2D_v1, DimBlock2D_v1>>>(
            d_input, d_output, dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Transpose 2D with shared-memory " + str_tile2d_v1, testbench_mode,
            h_gold, h_output, size, kernel_ms, host_ms, log, true);

    // ------------------------------------------------------------------------
    // TRANSPOSE 2D SHARED-MEMORY + BANK CONFLICT FREE V1
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose2d_shm_bank_kernel_v1<<<DimGrid2D_v1, DimBlock2D_v1>>>(
        d_input, d_output, dim_y,
        dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose2d_shm_bank_kernel_v1<<<DimGrid2D_v1, DimBlock2D_v1>>>(
            d_input, d_output, dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process(
        "Transpose 2D with shared-memory (bank conflict free) " + str_tile2d_v1,
        testbench_mode, h_gold, h_output, size, kernel_ms, host_ms, log, true);

    // ========================================================================
    // COPY 2D BANDWIDTH SIMPLE V2
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    copy2d_simple_kernel<<<DimGrid2D_v2, DimBlock2D_v2>>>(d_input, d_output,
                                                          dim_y,
                                                          dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        copy2d_simple_kernel<<<DimGrid2D_v2, DimBlock2D_v2>>>(d_input, d_output,
                                                              dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Copy 2D simple " + str_tile2d_v2, testbench_mode, h_input,
            h_output, size, kernel_ms, host_ms, log, false);

    // ------------------------------------------------------------------------
    // TRANSPOSE 2D SIMPLE V2
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose2d_simple_kernel<<<DimGrid2D_v2, DimBlock2D_v2>>>(
        d_input, d_output, dim_y,
        dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose2d_simple_kernel<<<DimGrid2D_v2, DimBlock2D_v2>>>(
            d_input, d_output, dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Transpose 2D simple " + str_tile2d_v2, testbench_mode, h_gold,
            h_output, size, kernel_ms, host_ms, log, true);

    // ------------------------------------------------------------------------
    // TRANSPOSE 2D SHARED-MEMORY V2
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose2d_shm_kernel_v2<<<DimGrid2D_v2, DimBlock2D_v2>>>(
        d_input, d_output, dim_y,
        dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose2d_shm_kernel_v2<<<DimGrid2D_v2, DimBlock2D_v2>>>(
            d_input, d_output, dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Transpose 2D with shared-memory " + str_tile2d_v2, testbench_mode,
            h_gold, h_output, size, kernel_ms, host_ms, log, true);

    // ------------------------------------------------------------------------
    // TRANSPOSE 2D SHARED-MEMORY + BANK CONFLICT FREE V2
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose2d_shm_bank_kernel_v2<<<DimGrid2D_v2, DimBlock2D_v2>>>(
        d_input, d_output, dim_y,
        dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose2d_shm_bank_kernel_v2<<<DimGrid2D_v2, DimBlock2D_v2>>>(
            d_input, d_output, dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process(
        "Transpose 2D with shared-memory (bank conflict free) " + str_tile2d_v2,
        testbench_mode, h_gold, h_output, size, kernel_ms, host_ms, log, true);

    // ========================================================================
    // COPY 3D BANDWIDTH SIMPLE
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    copy3d_simple_kernel<<<DimGrid3D, DimBlock3D>>>(d_input, d_output, 1, dim_y,
                                                    dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        copy3d_simple_kernel<<<DimGrid3D, DimBlock3D>>>(d_input, d_output, 1,
                                                        dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Copy 3D simple " + str_tile3d, testbench_mode, h_input, h_output,
            size, kernel_ms, host_ms, log, false);

    // ------------------------------------------------------------------------
    // TRANSPOSE 3D SIMPLE
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose3d_simple_selector(DimGrid3D, DimBlock3D, d_input, d_output, dim,
                                perm);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose3d_simple_selector(DimGrid3D, DimBlock3D, d_input, d_output,
                                    dim, perm);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Transpose 3D simple " + str_tile3d, testbench_mode, h_gold,
            h_output, size, kernel_ms, host_ms, log, true);

    // ------------------------------------------------------------------------
    // TRANSPOSE 3D SHARED-MEMORY
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose3d_shm_selector(DimGrid3D, DimBlock3D, d_input, d_output, dim,
                             perm);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose3d_shm_selector(DimGrid3D, DimBlock3D, d_input, d_output, dim,
                                 perm);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Transpose shared-memory template " + str_tile3d, testbench_mode,
            h_gold, h_output, size, kernel_ms, host_ms, log, true);

    // ------------------------------------------------------------------------
    // TRANSPOSE 3D SHARED-MEMORY + BANK CONFLICT FREE
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose3d_shm_bank_selector(DimGrid3D, DimBlock3D, d_input, d_output, dim,
                                  perm);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose3d_shm_bank_selector(DimGrid3D, DimBlock3D, d_input, d_output,
                                      dim, perm);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Transpose shared-memory bank-conflict-free template " + str_tile3d,
            testbench_mode, h_gold, h_output, size, kernel_ms, host_ms, log,
            true);


    // ------------------------------------------------------------------------
    // CLEAN SHUTDOWN
    log.close();
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
