#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <chrono>
#include <algorithm>

#define THREADS_PER_BLOCK 256
const int MAX_ITERATIONS = 1000;
const float MIN_MODULARITY_GAIN = 1e-5f;
const float RESOLUTION = 0.8f;

struct Graph {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    int numNodes;
    float totalEdgeWeight;
};

__global__ void initializeCommunities(int* communities, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes) {
        communities[tid] = tid;
    }
}

__global__ void computeModularityGain(int* row_ptr, int* col_idx, int* community, 
                                     int numNodes, int* changed, float* modularityGain,
                                     float totalEdgeWeight, float resolution) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numNodes) return;

    int currentComm = community[tid];
    int bestCommunity = currentComm;
    float bestGain = 0.0f;

    float nodeDegree = row_ptr[tid+1] - row_ptr[tid];

    float currentMod = 0.0f;
    for (int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++) {
        if (community[col_idx[i]] == currentComm) {
            currentMod += 1.0f;
        }
    }
    currentMod -= resolution * (nodeDegree * nodeDegree) / (2.0f * totalEdgeWeight);

    for (int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++) {
        int neighbor = col_idx[i];
        int newCommunity = community[neighbor];
        if (newCommunity == currentComm) continue;

        float newMod = 0.0f;
        for (int j = row_ptr[tid]; j < row_ptr[tid + 1]; j++) {
            if (community[col_idx[j]] == newCommunity) {
                newMod += 1.0f;
            }
        }
        newMod -= resolution * (nodeDegree * nodeDegree) / (2.0f * totalEdgeWeight);

        float gain = newMod - currentMod;
        if (gain > bestGain) {
            bestGain = gain;
            bestCommunity = newCommunity;
        }
    }

    if (bestGain > MIN_MODULARITY_GAIN && bestCommunity != currentComm) {
        community[tid] = bestCommunity;
        atomicAdd(modularityGain, bestGain);
        atomicExch(changed, 1);
    }
}

__global__ void mergeSmallCommunities(int* row_ptr, int* col_idx, int* community, 
    int numNodes, int minCommunitySize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numNodes) return;

    int currentComm = community[tid];

    int commSize = 0;
    for (int i = 0; i < numNodes; i++) {
        if (community[i] == currentComm) {
            commSize++;
        }
    }

    if (commSize < minCommunitySize) {
        int maxCount = 0;
        int largestNeighborComm = currentComm; 

        const int MAX_TEMP = 100;
        int neighborCounts[MAX_TEMP] = {0};

        for (int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++) {
            int neighborComm = community[col_idx[i]];
            if (neighborComm < MAX_TEMP) {
                neighborCounts[neighborComm]++;
                if (neighborCounts[neighborComm] > maxCount) {
                    maxCount = neighborCounts[neighborComm];
                    largestNeighborComm = neighborComm;
                }
            }
        }

        if (largestNeighborComm != currentComm) {
            community[tid] = largestNeighborComm;
        }
    }
}

Graph loadGraphCSR(const std::string& filename) {
    Graph graph;
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
        graph.totalEdgeWeight += 1.0f;
    }
    file.close();

    graph.numNodes = maxNode + 1;
    graph.row_ptr.resize(graph.numNodes + 1, 0);

    for (int i = 0; i < graph.numNodes; i++) {
        graph.row_ptr[i + 1] = graph.row_ptr[i] + adjList[i].size();
    }
    for (int i = 0; i < graph.numNodes; i++) {
        graph.col_idx.insert(graph.col_idx.end(), adjList[i].begin(), adjList[i].end());
    }

    return graph;
}

void runLouvain(Graph& graph, std::vector<int>& communities) {

    int *d_row_ptr, *d_col_idx, *d_community, *d_changed;
    float *d_modularityGain;
    
    cudaMalloc(&d_row_ptr, (graph.numNodes + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, graph.col_idx.size() * sizeof(int));
    cudaMalloc(&d_community, graph.numNodes * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_modularityGain, sizeof(float));


    cudaMemcpy(d_row_ptr, graph.row_ptr.data(), (graph.numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, graph.col_idx.data(), graph.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community, communities.data(), graph.numNodes * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (graph.numNodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int h_changed;
    float h_modularityGain;


    int iteration = 0;
    float max_gain = 0.0f;
    do {
        h_changed = 0;
        h_modularityGain = 0.0f;
        
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_modularityGain, &h_modularityGain, sizeof(float), cudaMemcpyHostToDevice);

        computeModularityGain<<<blocks, THREADS_PER_BLOCK>>>(
            d_row_ptr, d_col_idx, d_community, graph.numNodes, 
            d_changed, d_modularityGain, graph.totalEdgeWeight, RESOLUTION);
        
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_modularityGain, d_modularityGain, sizeof(float), cudaMemcpyDeviceToHost);

        iteration++;
        std::cout << "Iteration " << iteration << ": Modularity gain = " 
                  << h_modularityGain << std::endl;

        if (max_gain < h_modularityGain)
            max_gain = h_modularityGain;
        else if (h_modularityGain / max_gain < 0.001){
            break;
        }

    } while (h_changed && iteration < MAX_ITERATIONS);

    mergeSmallCommunities<<<blocks, THREADS_PER_BLOCK>>>(
        d_row_ptr, d_col_idx, d_community, graph.numNodes, 3);
    cudaDeviceSynchronize();

    cudaMemcpy(communities.data(), d_community, graph.numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_community);
    cudaFree(d_changed);
    cudaFree(d_modularityGain);
}

void analyzeCommunities(const std::vector<int>& communities) {
    std::unordered_map<int, int> communitySizes;
    for (int comm : communities) {
        communitySizes[comm]++;
    }

    std::vector<std::pair<int, int>> sortedCommunities(communitySizes.begin(), communitySizes.end());
    std::sort(sortedCommunities.begin(), sortedCommunities.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "\nCommunity Distribution:\n";
    std::cout << "Total communities: " << sortedCommunities.size() << "\n";
    
    int topN = std::min(10, (int)sortedCommunities.size());
    float totalNodes = communities.size();
    float coverage = 0.0f;

    for (int i = 0; i < topN; i++) {
        float percent = 100.0f * sortedCommunities[i].second / totalNodes;
        coverage += percent;
        std::cout << "Community " << sortedCommunities[i].first << ": " 
                  << sortedCommunities[i].second << " nodes (" 
                  << percent << "%)\n";
    }

    std::cout << "Top " << topN << " communities cover " << coverage << "% of nodes\n";
    std::cout << "Average community size: " << totalNodes / sortedCommunities.size() << "\n";
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    Graph graph = loadGraphCSR("soc-LiveJournal1.txt");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Graph loaded in " << elapsed.count() << " seconds\n";
    std::cout << "Nodes: " << graph.numNodes << ", Edges: " << graph.col_idx.size()/2 << "\n";

    std::vector<int> communities(graph.numNodes);
    int blocks = (graph.numNodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    int* d_communities;
    cudaMalloc(&d_communities, graph.numNodes * sizeof(int));
    initializeCommunities<<<blocks, THREADS_PER_BLOCK>>>(d_communities, graph.numNodes);
    cudaMemcpy(communities.data(), d_communities, graph.numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_communities);

    start = std::chrono::high_resolution_clock::now();
    runLouvain(graph, communities);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Louvain completed in " << elapsed.count() << " seconds\n";

    analyzeCommunities(communities);

    return 0;
}
