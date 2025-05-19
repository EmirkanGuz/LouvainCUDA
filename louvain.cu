#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <cmath>

#define THREADS_PER_BLOCK 256
const int MAX_ITERATIONS = 1000;
const float MIN_MODULARITY_GAIN = 1e-5f;
const float MODULARITY_THRESHOLD = 0.001f;
const int BUCKET_SIZE = 32;

struct Graph {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    int numNodes;
    float totalEdgeWeight;
};

struct NodeInfo {
    int node_id;
    int degree;
    int community;
};

__global__ void initializeCommunities(int* communities, int* a_c, int* degrees, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes) {
        communities[tid] = tid;
        a_c[tid] = degrees[tid];
    }
}

__device__ float calculateModularityContribution(int node, int community, int* row_ptr, int* col_idx,
                                               int* node_communities, int node_degree,
                                               float totalEdgeWeight, int* a_c) {
    float inv_2m = 1.0f / (2.0f * totalEdgeWeight);
    float mod = 0.0f;
    int internal_edges = 0;

    for (int i = row_ptr[node]; i < row_ptr[node + 1]; i++) {
        int neighbor = col_idx[i];
        if (node_communities[neighbor] == community && neighbor > node) {
            internal_edges++;
        }
    }
    
    mod = 2.0f * internal_edges;
    mod -= (node_degree * a_c[community]) * inv_2m;
    
    return mod;
}

__global__ void computeModularityGainBucketed(int* row_ptr, int* col_idx, int* community,
                                            int numNodes, int* changed,
                                            float totalEdgeWeight, NodeInfo* node_infos,
                                            int* bucket_offsets, int* a_c) {
    int bucket_id = blockIdx.x;
    int tid_in_bucket = threadIdx.x;
    int bucket_start = bucket_offsets[bucket_id];
    int bucket_end = (bucket_id + 1 < gridDim.x) ? bucket_offsets[bucket_id + 1] : numNodes;
    
    __shared__ int shared_changed[THREADS_PER_BLOCK];
    
    if (tid_in_bucket == 0) {
        shared_changed[0] = 0;
    }
    __syncthreads();
    
    for (int node_idx = bucket_start + tid_in_bucket; 
         node_idx < bucket_end; 
         node_idx += blockDim.x) {
        int node = node_infos[node_idx].node_id;
        int currentComm = community[node];
        int bestCommunity = currentComm;
        float bestGain = 0.0f;
        
        int node_degree = row_ptr[node + 1] - row_ptr[node];
        float currentMod = calculateModularityContribution(node, currentComm, row_ptr, col_idx,
                                                         community, node_degree, totalEdgeWeight, a_c);
        
        for (int i = row_ptr[node]; i < row_ptr[node + 1]; i++) {
            int neighbor = col_idx[i];
            int newCommunity = community[neighbor];
            if (newCommunity == currentComm) continue;
            
            float newMod = calculateModularityContribution(node, newCommunity, row_ptr, col_idx,
                                                        community, node_degree, totalEdgeWeight, a_c);
            float gain = newMod - currentMod;
            
            if (gain > bestGain || (gain == bestGain && newCommunity < bestCommunity)) {
                bestGain = gain;
                bestCommunity = newCommunity;
            }
        }
        
        if (bestGain > MIN_MODULARITY_GAIN && bestCommunity != currentComm) {
            atomicSub(&a_c[currentComm], node_degree);
            atomicAdd(&a_c[bestCommunity], node_degree);
            community[node] = bestCommunity;
            shared_changed[0] = 1;
        }
    }
    
    __syncthreads();
    
    if (tid_in_bucket == 0 && shared_changed[0]) {
        atomicExch(changed, 1);
    }
}

__global__ void computeModularity(int* row_ptr, int* col_idx, int* communities,
                                int numNodes, float* modularity, float totalEdgeWeight,
                                NodeInfo* node_infos, int* bucket_offsets, int* a_c) {
    int bucket_id = blockIdx.x;
    int tid_in_bucket = threadIdx.x;
    int bucket_start = bucket_offsets[bucket_id];
    int bucket_end = (bucket_id + 1 < gridDim.x) ? bucket_offsets[bucket_id + 1] : numNodes;
    
    __shared__ float shared_mod[THREADS_PER_BLOCK];
    shared_mod[tid_in_bucket] = 0.0f;
    float inv_2m = 1.0f / (2.0f * totalEdgeWeight);
    
    for (int node_idx = bucket_start + tid_in_bucket; 
         node_idx < bucket_end; 
         node_idx += blockDim.x) {
        int node = node_infos[node_idx].node_id;
        int comm = communities[node];
        int k_i = row_ptr[node + 1] - row_ptr[node];
        
        for (int j = row_ptr[node]; j < row_ptr[node + 1]; j++) {
            int neighbor = col_idx[j];
            if (comm == communities[neighbor] && neighbor > node) {
                int k_j = row_ptr[neighbor + 1] - row_ptr[neighbor];
                shared_mod[tid_in_bucket] += 1.0f - (k_i * k_j) * inv_2m;
            }
        }
    }
    
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid_in_bucket < s) {
            shared_mod[tid_in_bucket] += shared_mod[tid_in_bucket + s];
        }
        __syncthreads();
    }
    
    if (tid_in_bucket == 0) {
        atomicAdd(modularity, shared_mod[0] * inv_2m);
    }
}

void createBuckets(const std::vector<int>& degrees, std::vector<NodeInfo>& node_infos,
                  std::vector<int>& bucket_offsets, int numBuckets) {
    node_infos.resize(degrees.size());
    for (int i = 0; i < degrees.size(); i++) {
        node_infos[i] = {i, degrees[i], i};
    }
    
    std::sort(node_infos.begin(), node_infos.end(),
              [](const NodeInfo& a, const NodeInfo& b) { return a.degree > b.degree; });
    
    int nodesPerBucket = (degrees.size() + numBuckets - 1) / numBuckets;
    bucket_offsets.resize(numBuckets + 1);
    for (int i = 0; i <= numBuckets; i++) {
        bucket_offsets[i] = std::min(i * nodesPerBucket, (int)degrees.size());
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
        graph.totalEdgeWeight += 0.5f;
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

void runLouvainOptimized(Graph& graph, std::vector<int>& communities) {
    std::vector<int> degrees(graph.numNodes);
    for (int i = 0; i < graph.numNodes; i++) {
        degrees[i] = graph.row_ptr[i+1] - graph.row_ptr[i];
    }
    
    int numBuckets = (graph.numNodes + BUCKET_SIZE - 1) / BUCKET_SIZE;
    std::vector<NodeInfo> node_infos;
    std::vector<int> bucket_offsets;
    createBuckets(degrees, node_infos, bucket_offsets, numBuckets);
    
    // Device memory pointers
    int *d_row_ptr, *d_col_idx, *d_community, *d_changed, *d_bucket_offsets, *d_a_c;
    float *d_modularity;
    NodeInfo *d_node_infos;

    // Allocate and copy data to device
    cudaMalloc(&d_row_ptr, (graph.numNodes + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, graph.col_idx.size() * sizeof(int));
    cudaMalloc(&d_community, graph.numNodes * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_modularity, sizeof(float));
    cudaMalloc(&d_node_infos, node_infos.size() * sizeof(NodeInfo));
    cudaMalloc(&d_bucket_offsets, bucket_offsets.size() * sizeof(int));
    cudaMalloc(&d_a_c, graph.numNodes * sizeof(int));

    cudaMemcpy(d_row_ptr, graph.row_ptr.data(), (graph.numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, graph.col_idx.data(), graph.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community, communities.data(), graph.numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_infos, node_infos.data(), node_infos.size() * sizeof(NodeInfo), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bucket_offsets, bucket_offsets.data(), bucket_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize communities and degrees
    int blocks = (graph.numNodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    initializeCommunities<<<blocks, THREADS_PER_BLOCK>>>(d_community, d_a_c, d_row_ptr, graph.numNodes);

    // Main optimization loop
    int h_changed;
    float h_modularity, previous_modularity = 0.0f, h_gain;
    int iteration = 0;
    
    do {
        // Compute current modularity
        cudaMemset(d_modularity, 0, sizeof(float));
        computeModularity<<<numBuckets, THREADS_PER_BLOCK>>>(
            d_row_ptr, d_col_idx, d_community, graph.numNodes,
            d_modularity, graph.totalEdgeWeight, d_node_infos,
            d_bucket_offsets, d_a_c);
        cudaMemcpy(&h_modularity, d_modularity, sizeof(float), cudaMemcpyDeviceToHost);

        // Calculate gain as difference from previous iteration
        h_gain = h_modularity - previous_modularity;
        previous_modularity = h_modularity;

        // Process nodes in buckets
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
        
        computeModularityGainBucketed<<<numBuckets, THREADS_PER_BLOCK>>>(
            d_row_ptr, d_col_idx, d_community, graph.numNodes, 
            d_changed, graph.totalEdgeWeight,
            d_node_infos, d_bucket_offsets, d_a_c);
        
        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

        // Output and check convergence
        std::cout << "Iteration " << iteration 
                  << ": Modularity = " << h_modularity
                  << ", Gain = " << h_gain << std::endl;
        
        if (iteration > 0 && fabsf(h_gain) < MODULARITY_THRESHOLD) {
            std::cout << "Converged with modularity " << h_modularity << std::endl;
            break;
        }
        
    } while (h_changed && iteration++ < MAX_ITERATIONS);

    // Final modularity calculation
    cudaMemset(d_modularity, 0, sizeof(float));
    computeModularity<<<numBuckets, THREADS_PER_BLOCK>>>(
        d_row_ptr, d_col_idx, d_community, graph.numNodes,
        d_modularity, graph.totalEdgeWeight, d_node_infos,
        d_bucket_offsets, d_a_c);
    cudaMemcpy(&h_modularity, d_modularity, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Final Modularity: " << h_modularity << std::endl;
    
    // Copy results back to host
    cudaMemcpy(communities.data(), d_community, graph.numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_community);
    cudaFree(d_changed);
    cudaFree(d_modularity);
    cudaFree(d_node_infos);
    cudaFree(d_bucket_offsets);
    cudaFree(d_a_c);
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
    Graph graph = loadGraphCSR("email-Eu-core.txt");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Graph loaded in " << elapsed.count() << " seconds\n";
    std::cout << "Nodes: " << graph.numNodes << ", Edges: " << graph.col_idx.size()/2 << "\n";

    std::vector<int> communities(graph.numNodes);
    start = std::chrono::high_resolution_clock::now();
    runLouvainOptimized(graph, communities);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Louvain completed in " << elapsed.count() << " seconds\n";

    analyzeCommunities(communities);

    return 0;
}
