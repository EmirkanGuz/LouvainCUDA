#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <chrono>

#define THREADS_PER_BLOCK 256

__global__ void computeModularityGain(int *row_ptr, int *col_idx, int *community, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numNodes) return;

    int bestCommunity = community[tid];
    float bestGain = 0.0f;

    for (int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++) {
        int neighbor = col_idx[i];
        int newCommunity = community[neighbor];

        float gain = 0.1f * (newCommunity != bestCommunity);
        if (gain > bestGain) {
            bestGain = gain;
            bestCommunity = newCommunity;
        }
    }
    community[tid] = bestCommunity;
}

__global__ void aggregate_communities(int *row_ptr, int *col_idx, int *d_comm, float *d_weights, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numNodes) return;

    int comm = d_comm[tid];

    for (int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++) {
        int neighbor = col_idx[i];
        int neighborComm = d_comm[neighbor];

        if (comm == neighborComm) {
            atomicAdd(&d_weights[comm], 1.0f);
        }
    }
}


void loadGraphCSR(const std::string &filename, std::vector<int> &row_ptr, std::vector<int> &col_idx, int &numNodes) {
    std::ifstream file(filename);
    std::unordered_map<int, std::vector<int>> adjList;
    int maxNode = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') continue;
        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;

        adjList[u].push_back(v);
        adjList[v].push_back(u);
        maxNode = std::max(maxNode, std::max(u, v));
    }
    file.close();

    numNodes = maxNode + 1;
    row_ptr.resize(numNodes + 1, 0);

    for (int i = 0; i < numNodes; i++) {
        row_ptr[i + 1] = row_ptr[i] + adjList[i].size();
    }
    for (int i = 0; i < numNodes; i++) {
        col_idx.insert(col_idx.end(), adjList[i].begin(), adjList[i].end());
    }
}

void louvainCUDA(std::vector<int> &row_ptr, std::vector<int> &col_idx, std::vector<int> &communities, int numNodes) {
    int *d_row_ptr, *d_col_idx, *d_community;

    cudaMalloc(&d_row_ptr, (numNodes + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, col_idx.size() * sizeof(int));
    cudaMalloc(&d_community, numNodes * sizeof(int));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community, communities.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (numNodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    computeModularityGain<<<blocks, THREADS_PER_BLOCK>>>(d_row_ptr, d_col_idx, d_community, numNodes);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(communities.data(), d_community, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_community);

    std::cout << "CUDA Louvain execution time: " << milliseconds << " ms\n";
}

int main() {
    std::string filename = "soc-LiveJournal1.txt";
    std::vector<int> row_ptr, col_idx;
    int numNodes;

    auto start = std::chrono::high_resolution_clock::now();
    loadGraphCSR(filename, row_ptr, col_idx, numNodes);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = stop - start;
    std::cout << "Graph loading time: " << load_time.count() << " seconds\n";

    std::vector<int> communities(numNodes);
    for (int i = 0; i < numNodes; i++) {
        communities[i] = i;
    }

    louvainCUDA(row_ptr, col_idx, communities, numNodes);

    float *d_weights;
    int *d_newGraph, *d_comm, *d_adj;

    cudaMalloc(&d_weights, numNodes * sizeof(float));
    cudaMemset(d_weights, 0, numNodes * sizeof(float));
    cudaMalloc(&d_newGraph, numNodes * numNodes * sizeof(int));
    cudaMalloc(&d_comm, numNodes * sizeof(int));
    cudaMalloc(&d_adj, col_idx.size() * sizeof(int));

    cudaMemcpy(d_comm, communities.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (numNodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    int *d_row_ptr, *d_col_idx;
    cudaMalloc(&d_row_ptr, (numNodes + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, col_idx.size() * sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    aggregate_communities<<<blocks, THREADS_PER_BLOCK>>>(d_row_ptr, d_col_idx, d_comm, d_weights, numNodes);
    cudaDeviceSynchronize();

    cudaMemcpy(communities.data(), d_comm, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Updated communities (first 10 nodes):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "Node " << i << " -> Community " << communities[i] << "\n";
    }

    std::cout << "Phase 2: Printing community weights...\n";

    std::vector<float> h_weights_out(numNodes);

    cudaDeviceSynchronize();
    cudaMemcpy(h_weights_out.data(), d_weights, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Community weights (first 100):\n";
    for (int i = 0; i < 100; i++) {
        std::cout << "Community " << i << " has total weight: " << h_weights_out[i] << "\n";
    }

    cudaFree(d_weights);
    cudaFree(d_newGraph);
    cudaFree(d_comm);
    cudaFree(d_adj);
    
    return 0;
}