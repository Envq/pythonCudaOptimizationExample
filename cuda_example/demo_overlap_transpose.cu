#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

// WARNING: NOT WORK

// ============================================================================
// SETTINGS
const int  NUM_REPS     = 1;
const bool ENABLE_PRINT = false;
const int  TILE         = 32;


// ============================================================================
// CUDA SECTION
inline cudaError_t CHECK_CUDA(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void transpose2d_simple_kernel(const float* d_input, float* d_output,
                                          const int dimy, const int dimx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < dimy && x < dimx) {
        d_output[x * dimy + y] = d_input[y * dimx + x];
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

void matrix_print(const float* matrix, int m, int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cout << matrix[row * n + col] << " ";
        }
        std::cout << std::endl;
    }
}

void transpose2d_cpp(const float* matrix, float* result, int m, int n) {
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

void read_chunk(const float* matrix, float* chunk, int m, int n, int size,
                int cy, int cx) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            chunk[row * size + col] =
                matrix[(row + cy * size) * n + (col + cx * size)];
        }
    }
}
void write_chunk(float* matrix, const float* chunk, int m, int n, int size,
                 int cy, int cx) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            matrix[(row + cy * size) * n + (col + cx * size)] =
                chunk[row * size + col];
        }
    }
}


// ============================================================================
// MAIN
int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------------
    // GET ARGS
    if (argc < 3) {
        std::cout << "call: executable dim_y dim_x" << std::endl;
        std::cout << "example: ./demo_overlap.out 32 32" << std::endl;
        return 0;
    }
    int testbench_mode = false;
    if (argc == 4 && std::string(argv[3]) == "testbench") {
        testbench_mode = true;
    }

    // ------------------------------------------------------------------------
    // GET INFO
    int        dim_y = std::stoi(argv[1]);
    int        dim_x = std::stoi(argv[2]);
    int        size  = dim_y * dim_x;
    const int  bytes = size * sizeof(float);
    const dim3 DimBlock(TILE, TILE, 1);
    const dim3 DimGrid(std::ceil((float)dim_x / DimBlock.x),
                       std::ceil((float)dim_y / DimBlock.y), 1);

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
    float *h_input, *h_output, *h_gold;
    h_input  = new float[size]{};
    h_output = new float[size]{};
    // CHECK_CUDA(cudaMallocHost((void**)&h_input, bytes));   // host pinned
    // CHECK_CUDA(cudaMallocHost((void**)&h_output, bytes));  // host pinned
    h_gold = new float[size]{};

    // array_init_rand(h_input, size);
    array_init_seq(h_input, size);

    // ------------------------------------------------------------------------
    // HOST EXECUTION
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_REPS; ++i)
        transpose2d_cpp(h_input, h_gold, dim_y, dim_x);
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
        std::cout << "DimBlock 2D:    (" << DimBlock.x << ", " << DimBlock.y
                  << ", " << DimBlock.z << ")" << std::endl;
        std::cout << "DimGrid 2D:     (" << DimGrid.x << ", " << DimGrid.y
                  << ", " << DimGrid.z << ")" << std::endl;
        std::cout << "Host Time (ms): " << host_ms << std::endl;
        std::cout << std::endl;
    } else {
        // std::string log_name = "logs/logs_device/";
        // log_name += std::to_string(dim[0]) + "x" + std::to_string(dim[1]) +
        //             "x" + std::to_string(dim[2]);
        // log_name += "_";
        // log_name += std::to_string(perm[0]) + std::to_string(perm[1]) +
        //             std::to_string(perm[2]);
        // log_name += ".log";
        // log.open(log_name, std::ios::out);
        // log << host_ms << std::endl;
    }

    // ------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // ------------------------------------------------------------------------
    // TRANSPOSE SIMPLE
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));  // Initialize output
    transpose2d_simple_kernel<<<DimGrid, DimBlock>>>(d_input, d_output, dim_y,
                                                     dim_x);  // warmup
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; ++i)
        transpose2d_simple_kernel<<<DimGrid, DimBlock>>>(d_input, d_output,
                                                         dim_y, dim_x);
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    process("Transpose 2D simple", testbench_mode, h_gold, h_output, size,
            device_ms, host_ms, log);


    // ------------------------------------------------------------------------
    // TRANSPOSE SIMPLE
    float *d_chunk_input, *d_chunk_output;
    CHECK_CUDA(cudaMalloc(&d_chunk_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_chunk_output, bytes));

    const int chunck_size = TILE * TILE;
    const int chunck_byte = chunck_size * sizeof(float);
    float *   h_chunk_input, *h_chunk_output;
    // h_chunk_input  = new float[chunck_size]{};
    // h_chunk_output = new float[chunck_size]{};
    CHECK_CUDA(
        cudaMallocHost((void**)&h_chunk_input, chunck_byte));  // host pinned
    CHECK_CUDA(
        cudaMallocHost((void**)&h_chunk_output, chunck_byte));  // host pinned

    const int NUM_STREAMS = 8;
    const int CHUNK_SIZE  = TILE * TILE;
    const int CHUNK_BYTES = CHUNK_SIZE * sizeof(float);

    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&stream[i]);

    const dim3 DimBlockChunk(TILE, TILE, 1);
    const dim3 DimGridChunk(std::ceil(dim_y / TILE), std::ceil(dim_x / TILE),
                            1);

    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    int streamIndex = 0;
    for (int i = 0; i < DimBlockChunk.y; ++i) {
        for (int j = 0; j < DimBlockChunk.x; ++j) {
            // READ GENERATION
            read_chunk(h_input, h_chunk_input, dim_y, dim_x, TILE, i, j);
            // TRANSPOSE EXECUTION
            CHECK_CUDA(cudaMemcpyAsync(d_chunk_input, h_chunk_input,
                                       CHUNK_BYTES, cudaMemcpyHostToDevice,
                                       stream[streamIndex]));
            transpose2d_simple_kernel<<<DimGridChunk, DimBlockChunk, 0,
                                        stream[streamIndex]>>>(
                d_chunk_input, d_chunk_output, TILE, TILE);
            CHECK_CUDA(cudaMemcpyAsync(h_chunk_output, d_chunk_output,
                                       CHUNK_BYTES, cudaMemcpyDeviceToHost,
                                       stream[streamIndex]));
            // WRITE GENERATION
            cudaDeviceSynchronize();
            write_chunk(h_output, h_chunk_output, dim_x, dim_y, TILE, j, i);
            streamIndex = (streamIndex + 1) % NUM_STREAMS;
        }
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, startEvent, stopEvent));
    process("Transpose 2D simple overlap", testbench_mode, h_gold, h_output,
            size, device_ms, host_ms, log);


    for (int i = 0; i < NUM_STREAMS; ++i)
        CHECK_CUDA(cudaStreamDestroy(stream[i]));


    // ------------------------------------------------------------------------
    // CLEAN SHUTDOWN
    log.close();
    delete[] h_input;
    delete[] h_output;
    // delete[] h_chunk_input;
    // delete[] h_chunk_output;
    cudaFreeHost(h_chunk_input);
    cudaFreeHost(h_chunk_output);
    delete[] h_gold;

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_chunk_input));
    CHECK_CUDA(cudaFree(d_chunk_output));
    cudaDeviceReset();

    return 0;
}
