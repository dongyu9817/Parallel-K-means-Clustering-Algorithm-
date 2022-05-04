/**
 * Cuda K-means cluster Algorithm
 * Yu Dong (Yu Dong)
 * This is a general CUDA algorithm of a standard k-means iteration with the input data points {P1...n} s
 * The basic idea of this algorithm is:
 * 1: Copy {C1...k} to GPU device
 * 2: Launch the GPU kernel to label {P1...n} to the nearest centroids
 * 3: Copy {L1...n} back to host
 * 4: Calculate the mean for each cluster and update {C1...k}
 */
#include "../kmeans.h"

#include <math.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "CycleTimer.h"

// global constant variable
int threadPerBlock = 128;

static inline int FindNextPower2(int num)
{
    num -= 1;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;
    // num |= num >> 32;
    num += 1;
    return num;
}

__host__ __device__ inline static float compute_distance(points_t point, centroids_t centroid)
{
    int px = point.x;
    int py = point.y;
    int cx = centroid.x;
    int cy = centroid.y;

    float xdiff = px - cx;
    float ydiff = py - cy;

    return sqrt(pow(xdiff, 2) + pow(ydiff, 2));
}

__global__ static void findNearnestNeighbor(points_t *gpu_points_list, centroids_t *gpu_centroids_list, metaInfo_t *gpu_metadata, int *pBlock_changes)
{
    // gpu find the nearnest neighbor parallel function
    // each thread is responsible for one datapoint
    int numPoints = gpu_metadata->numpoints;
    int clusters = gpu_metadata->cluster;
    int dataid = blockDim.x * blockIdx.x + threadIdx.x;
    if (dataid > numPoints)
        return;
    float path_min = 0;
    int org_cluster_id = gpu_points_list[dataid].cluster;
    int cluster_id = 0;
    float path_curr;
    for (int j = 0; j < clusters; ++j)
    {
        path_curr = compute_distance(gpu_points_list[dataid], gpu_centroids_list[j]);
        if (path_min == 0 || path_curr < path_min)
        {
            path_min = path_curr;
            cluster_id = j;
        }
    }
    gpu_points_list[dataid].cluster = cluster_id;
    gpu_points_list[dataid].distance = path_min;
    //printf ("val id %d changes from %d to %d\n", dataid, gpu_points_list[dataid].cluster, gpu_points_list[dataid].distance);
    // use reduction function to get sum of point's cluster change in each block
    extern __shared__ int shared_blockpoints_change[];
    // initialize reduction array
    shared_blockpoints_change[threadIdx.x] = 0;
    if (org_cluster_id != cluster_id)
    {
        shared_blockpoints_change[threadIdx.x] = 1;
    }
    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            shared_blockpoints_change[threadIdx.x] += shared_blockpoints_change[threadIdx.x + i];
        }
        __syncthreads();
    }
    // save sum result in the block reduction array
    if (threadIdx.x == 0)
    {
        pBlock_changes[blockIdx.x] = shared_blockpoints_change[0];
    }
    return;
}


__global__ static void find_change_reduction(int *pBlock_changes, int block_size, int blocks_roundedsize)
{
    extern __shared__ unsigned int shared_reduce[];
    int dataid = threadIdx.x;
    if (dataid >= block_size)
    {
        shared_reduce[dataid] = 0;
    }
    else
    {
        shared_reduce[dataid] = pBlock_changes[dataid];
    }
    __syncthreads();
    for (unsigned int i = blocks_roundedsize / 2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            shared_reduce[threadIdx.x] += shared_reduce[threadIdx.x + i];
        }
        __syncthreads();
    }
    // save sum result in the block reduction array
    if (threadIdx.x == 0)
    {
        pBlock_changes[0] = shared_reduce[0];
    }
    return;
}

double kmeans_cuda(int *n_points, int clusters, points_t **p_list, centroids_t **c_list, int iterations)
{
    // host data
    points_t *points_list = *p_list;
    int num_points = *n_points;
    int  threshold_change = 0;
    *c_list = (centroids_t *)malloc(sizeof(centroids_t) * clusters);
    centroids_t *centroids_list = *c_list;
    metaInfo_t *metaData = (metaInfo_t *)malloc(sizeof(metaInfo_t));
    metaData->numpoints = num_points;
    metaData->iterations = iterations;
    metaData->cluster = clusters;

    // randomly pick centroid based on the cluster number
    // get the initial centroids
    srand(time(0));
    for (int i = 0; i < clusters; ++i)
    {
        int index = rand() % num_points;
        centroids_list[i].x = points_list[index].x;
        centroids_list[i].y = points_list[index].y;
        centroids_list[i].prevx = 0;
        centroids_list[i].prevy = 0;
        centroids_list[i].sum_px = 0;
        centroids_list[i].sum_py = 0;
        centroids_list[i].count = 0;
    }

    // device data on gpu
    points_t *gpu_points_list;
    centroids_t *gpu_centroids_list = *c_list;
    metaInfo_t *gpu_metaData;

    // Copy data from device to GPU
    cudaMalloc(&gpu_points_list, num_points * sizeof(points_t));
    cudaMalloc(&gpu_centroids_list, sizeof(centroids_t) * clusters);
    cudaMalloc(&gpu_metaData, sizeof(metaInfo_t));
    cudaMemcpy(gpu_points_list, points_list, num_points * sizeof(points_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroids_list, centroids_list, sizeof(centroids_t) * clusters, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_metaData, metaData, sizeof(metaInfo_t), cudaMemcpyHostToDevice);

    // initialize CUDA constants
    const unsigned int blocks = (num_points + threadPerBlock - 1) / threadPerBlock;
    const unsigned int blocks_rounded = FindNextPower2(blocks);
    // initialize cluster shared date for each block
    const unsigned int blocks_sharedInfo = threadPerBlock * sizeof(unsigned int);
    const unsigned int blocks_sharedInfo_reduced = blocks_rounded * sizeof(unsigned int);

    // track the number of points that change cluster in each iteration
    int *pBlock_changes;
    cudaMalloc(&pBlock_changes, blocks_rounded * sizeof(unsigned int));

    // update stage
    int curr_iters = 0;
    int cluster_id = 0;
    float delta = 0.0;
    double startTime = CycleTimer::currentSeconds();
    
    do
    {
        // get the nearest centroids
        cudaMemcpy(gpu_centroids_list, centroids_list, sizeof(centroids_t) * clusters, cudaMemcpyHostToDevice);
        findNearnestNeighbor<<<blocks, threadPerBlock, blocks_sharedInfo>>>(gpu_points_list, gpu_centroids_list, gpu_metaData, pBlock_changes);
        //cudaDeviceSynchronize();
        
        cudaMemcpy(points_list, gpu_points_list, num_points * sizeof(points_t), cudaMemcpyDeviceToHost); // for (int i = 0; i < num_points; ++i)

        // update cluster, sum all points in each cluster
        for (int i = 0; i < num_points; ++i)
        {
            
            cluster_id = points_list[i].cluster;
            //printf ("centroids_list data on gpu % d\n", cluster_id);
            centroids_list[cluster_id].sum_px += (points_list[i].x);
            centroids_list[cluster_id].sum_py += (points_list[i].y);
            centroids_list[cluster_id].count += 1;
        }
        
        // compute new centroid points based on arithmetic mean
        for (int i = 0; i < clusters; ++i)
        {
            //printf ("findNearnestNeighbor data on gpu % d\n", centroids_list[i].count);
            int meanx = centroids_list[i].sum_px / centroids_list[i].count;
            int meany = centroids_list[i].sum_py / centroids_list[i].count;
            centroids_list[i].prevx = centroids_list[i].x;
            centroids_list[i].prevy = centroids_list[i].y;
            centroids_list[i].x = meanx;
            centroids_list[i].y = meany;
            centroids_list[i].count = 0;
            centroids_list[i].sum_px = 0;
            centroids_list[i].sum_py = 0;
        }
        // find delta, will compared with threshold in the while condition
        find_change_reduction<<<1, blocks_rounded, blocks_sharedInfo_reduced>>>(pBlock_changes, blocks, blocks_rounded);
        // block change array sums the number of points change cluster in that block
        cudaDeviceSynchronize();
        cudaMemcpy(&threshold_change, pBlock_changes, sizeof(int), cudaMemcpyDeviceToHost);
        // check convergence
        ++curr_iters; 
        delta = (float) threshold_change / num_points;
        // convergence = compute_convergence (centroids_list, clusters);
        // //printf ("new the nearest centroids %d %d\n", centroids_list[0].x, centroids_list[0].y );
        printf ("curr delta %d %.3f \n", curr_iters, delta);

    }while (delta > 0.000001 && curr_iters < 200000);

    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    return overallDuration;
}