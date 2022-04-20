#include <cassert>
#include <chrono>
#include <iostream>


// ============================================================================
// GlOBAL VARS
const int N           = 1 << 25;
const int BLOCK       = 256;
const int NUM_STREAMS = 4;

const int SIZE        = N;
const int BYTES       = SIZE * sizeof(float);
const int CHUNK_SIZE  = SIZE / NUM_STREAMS;
const int CHUNK_BYTES = CHUNK_SIZE * sizeof(float);


// ============================================================================
// GPU FUNCTIONS
inline cudaError_t CHECK_CUDA(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void vectorAddKernel(const float* d_input1, const float* d_input2,
                                float* d_output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
        d_output[index] = d_input1[index] + d_input2[index];
}

float launchKernelSeq(const float* h_input1, const float* h_input2,
                      float* h_output) {
    // SETUP TIMERS
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    float kernel_ms;

    // DEVICE MEMORY ALLOCATION
    float *d_input1, *d_input2, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input1, BYTES));
    CHECK_CUDA(cudaMalloc(&d_input2, BYTES));
    CHECK_CUDA(cudaMalloc(&d_output, BYTES));

    // SETUP KERNEL
    const dim3 DimBlock(BLOCK, 1, 1);
    const dim3 DimGrid(std::ceil((float)N / BLOCK), 1, 1);

    // COPY DATA FROM HOST TO DEVICE
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    CHECK_CUDA(cudaMemcpy(d_input1, h_input1, BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input2, h_input2, BYTES, cudaMemcpyHostToDevice));

    // EXECUTE KERNEL
    vectorAddKernel<<<DimGrid, DimBlock>>>(d_input1, d_input2, d_output, N);

    // COPY DATA FROM DEVICE TO HOST
    CHECK_CUDA(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));

    // DEVICE MEMORY DEALLOCATION
    CHECK_CUDA(cudaFree(d_input1));
    CHECK_CUDA(cudaFree(d_input2));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    return kernel_ms;
}


float launchKernelPar1(const float* h_input1, const float* h_input2,
                       float* h_output) {
    // SETUP TIMERS
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    float kernel_ms;

    // SETUP STREAMS
    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&stream[i]);

    // DEVICE MEMORY ALLOCATION
    float *d_input1, *d_input2, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input1, BYTES));
    CHECK_CUDA(cudaMalloc(&d_input2, BYTES));
    CHECK_CUDA(cudaMalloc(&d_output, BYTES));

    // SETUP KERNEL
    const dim3 DimBlock(BLOCK, 1, 1);
    const dim3 DimGrid(std::ceil((float)CHUNK_SIZE / BLOCK), 1, 1);

    // COPY DATA TO DEVICE, EXECUTE KERNERL, COPY DATA TO HOST
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * CHUNK_SIZE;
        CHECK_CUDA(cudaMemcpyAsync(&d_input1[offset], &h_input1[offset],
                                   CHUNK_BYTES, cudaMemcpyHostToDevice,
                                   stream[i]));
        CHECK_CUDA(cudaMemcpyAsync(&d_input2[offset], &h_input2[offset],
                                   CHUNK_BYTES, cudaMemcpyHostToDevice,
                                   stream[i]));
        vectorAddKernel<<<DimGrid, DimBlock, 0, stream[i]>>>(
            &d_input1[offset], &d_input2[offset], &d_output[offset],
            offset + CHUNK_SIZE);
        CHECK_CUDA(cudaMemcpyAsync(&h_output[offset], &d_output[offset],
                                   CHUNK_BYTES, cudaMemcpyDeviceToHost,
                                   stream[i]));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));

    // DEVICE MEMORY DEALLOCATION
    for (int i = 0; i < NUM_STREAMS; ++i)
        CHECK_CUDA(cudaStreamDestroy(stream[i]));
    CHECK_CUDA(cudaFree(d_input1));
    CHECK_CUDA(cudaFree(d_input2));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    return kernel_ms;
}


float launchKernelPar2(const float* h_input1, const float* h_input2,
                       float* h_output) {
    // SETUP TIMERS
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    float kernel_ms;

    // SETUP STREAMS
    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&stream[i]);

    // DEVICE MEMORY ALLOCATION
    float *d_input1, *d_input2, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input1, BYTES));
    CHECK_CUDA(cudaMalloc(&d_input2, BYTES));
    CHECK_CUDA(cudaMalloc(&d_output, BYTES));

    // SETUP KERNEL
    const dim3 DimBlock(BLOCK, 1, 1);
    const dim3 DimGrid(std::ceil((float)CHUNK_SIZE / BLOCK), 1, 1);

    // COPY DATA TO DEVICE, EXECUTE KERNERL, COPY DATA TO HOST
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * CHUNK_SIZE;
        CHECK_CUDA(cudaMemcpyAsync(&d_input1[offset], &h_input1[offset],
                                   CHUNK_BYTES, cudaMemcpyHostToDevice,
                                   stream[i]));
        CHECK_CUDA(cudaMemcpyAsync(&d_input2[offset], &h_input2[offset],
                                   CHUNK_BYTES, cudaMemcpyHostToDevice,
                                   stream[i]));
    }
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * CHUNK_SIZE;
        vectorAddKernel<<<DimGrid, DimBlock, 0, stream[i]>>>(
            &d_input1[offset], &d_input2[offset], &d_output[offset],
            offset + CHUNK_SIZE);
    }
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * CHUNK_SIZE;
        CHECK_CUDA(cudaMemcpyAsync(&h_output[offset], &d_output[offset],
                                   CHUNK_BYTES, cudaMemcpyDeviceToHost,
                                   stream[i]));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));

    // DEVICE MEMORY DEALLOCATION
    for (int i = 0; i < NUM_STREAMS; ++i)
        CHECK_CUDA(cudaStreamDestroy(stream[i]));
    CHECK_CUDA(cudaFree(d_input1));
    CHECK_CUDA(cudaFree(d_input2));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    return kernel_ms;
}


void runTest(const float* h_input1, const float* h_input2, float* h_output,
             const float* h_gold, float host_ms,
             float (*launcher)(const float*, const float*, float*)) {
    // SETUP TIMERS
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    // DEVICE EXECUTION
    start           = std::chrono::steady_clock::now();
    float kernel_ms = launcher(h_input1, h_input2, h_output);
    end             = std::chrono::steady_clock::now();
    float device_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() *
        1e-6;
    // CHECK CORRECTNESS
    bool correct = true;
    for (int i = 0; i < SIZE; ++i) {
        if (h_output[i] != h_gold[i]) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "       i  = " << i << std::endl;
            std::cout << "  gold[i] = " << h_gold[i] << std::endl;
            std::cout << "result[i] = " << h_output[i] << std::endl;
            correct = false;
            break;
        }
    }

    // SPEEDUP
    std::cout << "  Host time (ms): " << host_ms << std::endl;
    std::cout << "Kernel Time (ms): " << kernel_ms << std::endl;
    std::cout << "Device time (ms): " << device_ms << std::endl;
    std::cout << "Speedup kernel: " << host_ms / kernel_ms << "x" << std::endl;
    std::cout << "Speedup global: " << host_ms / device_ms << "x" << std::endl;
}


// ============================================================================
// MAIN
int main(int argc, char* argv[]) {
    // PRINT INFO
    std::cout << "N:           " << N << std::endl;
    std::cout << "BLOCK:       " << BLOCK << std::endl;
    std::cout << "NUM_STREAMS: " << NUM_STREAMS << std::endl;

    // SETUP TIMERS
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    // HOST MEMORY ALLOCATION
    float *h_input1, *h_input2, *h_output;
    CHECK_CUDA(cudaMallocHost((void**)&h_input1, BYTES));  // host pinned
    CHECK_CUDA(cudaMallocHost((void**)&h_input2, BYTES));  // host pinned
    CHECK_CUDA(cudaMallocHost((void**)&h_output, BYTES));  // host pinned
    float* h_gold = new float[N]{};

    // HOST INITIALIZATION
    for (int i = 0; i < N; ++i) {
        h_input1[i] = i * 1.0f;
        h_input2[i] = i * 1.0f;
    }

    // HOST EXECUTION
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < SIZE; ++i) {
        h_gold[i] = h_input1[i] + h_input2[i];
    }
    end = std::chrono::steady_clock::now();
    float host_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() *
        1e-6;

    // RUN TESTs
    std::cout << "\nTest: Sequential" << std::endl;
    runTest(h_input1, h_input2, h_output, h_gold, host_ms, &launchKernelSeq);

    std::cout << "\nTest: Stream1" << std::endl;
    runTest(h_input1, h_input2, h_output, h_gold, host_ms, &launchKernelPar1);

    std::cout << "\nTest: Stream2" << std::endl;
    runTest(h_input1, h_input2, h_output, h_gold, host_ms, &launchKernelPar2);

    // CLEAN SHUTDOWN
    cudaFreeHost(h_input1);
    cudaFreeHost(h_input2);
    cudaFreeHost(h_output);
    delete[] h_gold;
    cudaDeviceReset();
    return 0;
}