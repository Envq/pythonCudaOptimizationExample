#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>


// ============================================================================
// SETTINGS
const std::string testbench_name = "testbench.txt";


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

bool checK_array(const float* result, float* gold, int size) {
    for (int i = 0; i < size; ++i) {
        if (result[i] != gold[i]) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "       i  = " << i << std::endl;
            std::cout << "  gold[i] = " << gold[i] << std::endl;
            std::cout << "result[i] = " << result[i] << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<float> split_float(const std::string& str,
                               const std::string& delim) {
    std::vector<float> tokens;
    size_t             prev = 0, pos = 0;
    do {
        pos = str.find(delim, prev);
        if (pos == std::string::npos)
            pos = str.length();
        std::string token = str.substr(prev, pos - prev);
        if (!token.empty())
            tokens.push_back(std::stof(token));
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());
    return tokens;
}

std::vector<int> split_int(const std::string& str, const std::string& delim) {
    std::vector<int> tokens;
    size_t           prev = 0, pos = 0;
    do {
        pos = str.find(delim, prev);
        if (pos == std::string::npos)
            pos = str.length();
        std::string token = str.substr(prev, pos - prev);
        if (!token.empty())
            tokens.push_back(std::stoi(token));
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());
    return tokens;
}


// ============================================================================
// MAIN
int main(int argc, char* argv[]) {
    std::fstream file(testbench_name, std::ios::in);
    if (file.is_open()) {
        std::string line;
        int         num = 1;
        while (getline(file, line)) {
            printf("Test %d: ", num);
            num++;
            // ----------------------------------------------------------------
            // GET DATA
            auto tokens1 = split_int(line, ",");
            int* dim     = &tokens1[0];

            getline(file, line);
            auto   input   = split_float(line, ",");
            float* h_input = &input[0];

            getline(file, line);
            auto tokens2 = split_int(line, ",");
            int* perm    = &tokens2[0];

            getline(file, line);
            auto   gold   = split_float(line, ",");
            float* h_gold = &gold[0];

            // ----------------------------------------------------------------
            // PROCESS DATA
            int   size = dim[0] * dim[1] * dim[2];
            float h_output[size]{};
            transpose_cpu(h_input, h_output, dim, perm);
            if (!checK_array(h_gold, h_gold, size))
                return 0;
            printf("OK\n");
        }
    }

    file.close();
    return 0;
}
