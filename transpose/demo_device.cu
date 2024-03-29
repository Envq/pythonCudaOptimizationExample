#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>


// ============================================================================
// SETTINGS
const int  NUM_REPS     = 100;
const bool ENABLE_PRINT = false;
const int  TILE         = 8;


// ============================================================================
// CUDA SECTION
inline cudaError_t CHECK_CUDA(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}


__global__ void transpose_simple_kernel(const float* d_input, float* d_output,
                                        int dimz, int dimy, int dimx, int pz,
                                        int py, int px) {
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

template<int pz, int py, int px>
__global__ void transpose_simple_kerneltmplt(const float* d_input,
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

void transpose_simple_selector(const dim3& DimGrid, const dim3& DimBlock,
                               const float* d_input, float* d_output,
                               const int* dim, const int* perm) {
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
        transpose_simple_kerneltmplt<0, 1, 2>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        transpose_simple_kerneltmplt<0, 2, 1>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2) {
        transpose_simple_kerneltmplt<1, 0, 2>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 2 && perm[2] == 0) {
        transpose_simple_kerneltmplt<1, 2, 0>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 0 && perm[2] == 1) {
        transpose_simple_kerneltmplt<2, 0, 1>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 1 && perm[2] == 0) {
        transpose_simple_kerneltmplt<2, 1, 0>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    }
}


__global__ void transpose_shm_kernel(const float* d_input, float* d_output,
                                     int dimz, int dimy, int dimx, int pz,
                                     int py, int px) {
    __shared__ float buffer[TILE][TILE][TILE];

    int iDim[3] = {dimz, dimy, dimx};
    int x       = blockIdx.x * TILE + threadIdx.x;
    int y       = blockIdx.y * TILE + threadIdx.y;
    int z       = blockIdx.z * TILE + threadIdx.z;
    if (z < iDim[0] && y < iDim[1] && x < iDim[2]) {
        int iIndex     = (z * iDim[1] * iDim[2]) + (y * iDim[2]) + x;
        int threads[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
        buffer[threads[pz]][threads[py]][threads[px]] = d_input[iIndex];
    }
    __syncthreads();

    int oDim[3]   = {iDim[pz], iDim[py], iDim[px]};
    int blocks[3] = {blockIdx.z, blockIdx.y, blockIdx.x};
    x             = blocks[px] * TILE + threadIdx.x;
    y             = blocks[py] * TILE + threadIdx.y;
    z             = blocks[pz] * TILE + threadIdx.z;
    if (z < oDim[0] && y < oDim[1] && x < oDim[2]) {
        int oIndex       = (z * oDim[1] * oDim[2]) + (y * oDim[2]) + x;
        d_output[oIndex] = buffer[threadIdx.z][threadIdx.y][threadIdx.x];
    }
}

template<int pz, int py, int px>
__global__ void transpose_shm_kerneltmplt(const float* d_input, float* d_output,
                                          int dimz, int dimy, int dimx) {
    __shared__ float buffer[TILE][TILE][TILE];

    int iDim[3] = {dimz, dimy, dimx};
    int x       = blockIdx.x * TILE + threadIdx.x;
    int y       = blockIdx.y * TILE + threadIdx.y;
    int z       = blockIdx.z * TILE + threadIdx.z;
    if (z < iDim[0] && y < iDim[1] && x < iDim[2]) {
        int iIndex     = (z * iDim[1] * iDim[2]) + (y * iDim[2]) + x;
        int threads[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
        buffer[threads[pz]][threads[py]][threads[px]] = d_input[iIndex];
    }
    __syncthreads();

    int oDim[3]   = {iDim[pz], iDim[py], iDim[px]};
    int blocks[3] = {blockIdx.z, blockIdx.y, blockIdx.x};
    x             = blocks[px] * TILE + threadIdx.x;
    y             = blocks[py] * TILE + threadIdx.y;
    z             = blocks[pz] * TILE + threadIdx.z;
    if (z < oDim[0] && y < oDim[1] && x < oDim[2]) {
        int oIndex       = (z * oDim[1] * oDim[2]) + (y * oDim[2]) + x;
        d_output[oIndex] = buffer[threadIdx.z][threadIdx.y][threadIdx.x];
    }
}

void transpose_shm_selector(const dim3& DimGrid, const dim3& DimBlock,
                            const float* d_input, float* d_output,
                            const int* dim, const int* perm) {
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
        transpose_shm_kerneltmplt<0, 1, 2>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        transpose_shm_kerneltmplt<0, 2, 1>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2) {
        transpose_shm_kerneltmplt<1, 0, 2>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 2 && perm[2] == 0) {
        transpose_shm_kerneltmplt<1, 2, 0>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 0 && perm[2] == 1) {
        transpose_shm_kerneltmplt<2, 0, 1>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 1 && perm[2] == 0) {
        transpose_shm_kerneltmplt<2, 1, 0>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    }
}


template<int pz, int py, int px, int tilez, int tiley, int tilex>
__global__ void transpose_shm_bank_kerneltmplt(const float* d_input,
                                               float* d_output, int dimz,
                                               int dimy, int dimx) {
    __shared__ float buffer[tilez][tiley][tilex];

    int iDim[3] = {dimz, dimy, dimx};
    int x       = blockIdx.x * TILE + threadIdx.x;
    int y       = blockIdx.y * TILE + threadIdx.y;
    int z       = blockIdx.z * TILE + threadIdx.z;
    if (z < iDim[0] && y < iDim[1] && x < iDim[2]) {
        int iIndex     = (z * iDim[1] * iDim[2]) + (y * iDim[2]) + x;
        int threads[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
        buffer[threads[pz]][threads[py]][threads[px]] = d_input[iIndex];
    }
    __syncthreads();

    int oDim[3]   = {iDim[pz], iDim[py], iDim[px]};
    int blocks[3] = {blockIdx.z, blockIdx.y, blockIdx.x};
    x             = blocks[px] * TILE + threadIdx.x;
    y             = blocks[py] * TILE + threadIdx.y;
    z             = blocks[pz] * TILE + threadIdx.z;
    if (z < oDim[0] && y < oDim[1] && x < oDim[2]) {
        int oIndex       = (z * oDim[1] * oDim[2]) + (y * oDim[2]) + x;
        d_output[oIndex] = buffer[threadIdx.z][threadIdx.y][threadIdx.x];
    }
}

void transpose_shm_bank_selector(const dim3& DimGrid, const dim3& DimBlock,
                                 const float* d_input, float* d_output,
                                 const int* dim, const int* perm) {
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
        transpose_shm_bank_kerneltmplt<0, 1, 2, 8, 8, 8>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        transpose_shm_bank_kerneltmplt<0, 2, 1, 8, 8, 12>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2) {
        transpose_shm_bank_kerneltmplt<1, 0, 2, 8, 9, 8>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 1 && perm[1] == 2 && perm[2] == 0) {
        transpose_shm_bank_kerneltmplt<1, 2, 0, 8, 10, 10>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 0 && perm[2] == 1) {
        transpose_shm_bank_kerneltmplt<2, 0, 1, 8, 8, 9>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
    } else if (perm[0] == 2 && perm[1] == 1 && perm[2] == 0) {
        transpose_shm_bank_kerneltmplt<2, 1, 0, 8, 12, 9>
            <<<DimGrid, DimBlock>>>(d_input, d_output, dim[0], dim[1], dim[2]);
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

void transpose_cpu(const float* input, float* output, const int* iDim,
                   const int* perm) {
    int oDim[] = {iDim[perm[0]], iDim[perm[1]], iDim[perm[2]]};
    for (int z = 0; z < iDim[0]; ++z) {
        for (int y = 0; y < iDim[1]; ++y) {
            for (int x = 0; x < iDim[2]; ++x) {
                int idx[]  = {z, y, x};
                int odx[]  = {idx[perm[0]], idx[perm[1]], idx[perm[2]]};
                int iIndex = (idx[0] * iDim[1] * iDim[2]) + (idx[1] * iDim[2]) +
                             (idx[2]);
                int oIndex = (odx[0] * oDim[1] * oDim[2]) + (odx[1] * oDim[2]) +
                             (odx[2]);
                output[oIndex] = input[iIndex];
            }
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
             const float* result, int size, float device_ms, float host_ms,
             std::ofstream& file) {
    device_ms /= NUM_REPS;
    bool  is_correct = array_check(gold, result, size);
    float speedup    = host_ms / device_ms;

    if (ENABLE_PRINT) {
        array_print(gold, size);
        array_print(result, size);
    }

    if (!testbench_mode) {
        std::cout << name << std::endl;
        std::cout << "            Check: " << (is_correct ? "OK" : "FAIL")
                  << std::endl;
        std::cout << "        Time (ms): " << device_ms << std::endl;
        std::cout << "     Speedup (ms): " << speedup << "x" << std::endl;
        std::cout << std::endl;
    } else {
        file << name << std::endl;
        file << is_correct << std::endl;
        file << device_ms << std::endl;
        file << speedup << std::endl;
    }
}

void print_info(int tile, int size, const int* dim, const int* perm,
                const dim3& DimBlock, const dim3& DimGrid, float host_ms) {
    std::cout << "tile:    " << tile << std::endl;
    std::cout << "size:    " << size << std::endl;
    std::cout << "dimension:   (" << dim[0] << ", " << dim[1] << ", " << dim[2]
              << ")" << std::endl;
    std::cout << "permutation: (" << perm[0] << ", " << perm[1] << ", "
              << perm[2] << ")" << std::endl;
    std::cout << "DimBlock:    (" << DimBlock.x << ", " << DimBlock.y << ", "
              << DimBlock.z << ")" << std::endl;
    std::cout << "DimGrid:     (" << DimGrid.x << ", " << DimGrid.y << ", "
              << DimGrid.z << ")" << std::endl;
    std::cout << "Host Time (ms): " << host_ms << std::endl;
    std::cout << std::endl;
}


// ============================================================================
// MAIN
int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------------
    // GET ARGS
    if (argc < 5) {
        std::cout << "call: executable permutation dim_z dim_y dim_x"
                  << std::endl;
        std::cout << "example: ./demo_device.out 120 32 32 32" << std::endl;
        return 0;
    }
    int testbench_mode = false;
    if (argc == 6 && std::string(argv[5]) == "testbench") {
        testbench_mode = true;
    }

    // ------------------------------------------------------------------------
    // GET INFO
    int dim[3]  = {std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4])};
    int perm[3] = {argv[1][0] - 48, argv[1][1] - 48, argv[1][2] - 48};
    int size    = dim[0] * dim[1] * dim[2];
    const int  bytes = size * sizeof(float);
    const dim3 DimBlock(TILE, TILE, TILE);
    const dim3 DimGrid(std::ceil((float)dim[2] / DimBlock.x),
                       std::ceil((float)dim[1] / DimBlock.y),
                       std::ceil((float)dim[0] / DimBlock.z));

    // ------------------------------------------------------------------------
    // SETUP TIMERS
    float       host_ms, device_ms;
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
        transpose_cpu(h_input, h_gold, dim, perm);
    end     = std::chrono::steady_clock::now();
    host_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() *
              1e-6 / NUM_REPS;

    // ------------------------------------------------------------------------
    // PRINT INFO
    std::ofstream log;
    if (!testbench_mode) {
        print_info(TILE, size, dim, perm, DimBlock, DimGrid, host_ms);
    } else {
        std::string log_name = "logs/logs_device/";
        log_name += std::to_string(dim[0]) + "x" + std::to_string(dim[1]) +
                    "x" + std::to_string(dim[2]);
        log_name += "_";
        log_name += std::to_string(perm[0]) + std::to_string(perm[1]) +
                    std::to_string(perm[2]);
        log_name += ".log";
        log.open(log_name, std::ios::out);
        log << host_ms << std::endl;
    }

    // ------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // ------------------------------------------------------------------------
    // TRANSPOSE SIMPLE
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    transpose_simple_selector(DimGrid, DimBlock, d_input, d_output, dim, perm);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i) {
        CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
        transpose_simple_selector(DimGrid, DimBlock, d_input, d_output, dim,
                                  perm);
        CHECK_CUDA(
            cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    process("Transpose simple template", testbench_mode, h_gold, h_output, size,
            device_ms, host_ms, log);

    // ------------------------------------------------------------------------
    // TRANSPOSE SHARED-MEMORY
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    transpose_shm_selector(DimGrid, DimBlock, d_input, d_output, dim, perm);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i) {
        CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
        transpose_shm_selector(DimGrid, DimBlock, d_input, d_output, dim, perm);
        CHECK_CUDA(
            cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    process("Transpose shared-memory template", testbench_mode, h_gold,
            h_output, size, device_ms, host_ms, log);

    // ------------------------------------------------------------------------
    // TRANSPOSE SHARED-MEMORY + BANK CONFLICT FREE
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    transpose_shm_bank_selector(DimGrid, DimBlock, d_input, d_output, dim,
                                perm);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i) {
        CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
        transpose_shm_bank_selector(DimGrid, DimBlock, d_input, d_output, dim,
                                    perm);
        CHECK_CUDA(
            cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    process("Transpose shared-memory bank-conflict-free template",
            testbench_mode, h_gold, h_output, size, device_ms, host_ms, log);

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
