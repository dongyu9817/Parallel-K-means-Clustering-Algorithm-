/**
 * Cuda K-means cluster Algorithm
 * Yu Dong (Yu Dong)
 * This is a CUDA-based inequality reinforced k-means algorithm
 * This algorithm can achieve better load balanced by reducing distance calculations.
 * The basic idea of this algorithm is:
 * 1: Calculate inter-centroid distances (ICD) matrix
 * 2: Sort each row of the ICD matrix to derive the ranked index (RID) matrix
 * 3: Copy {C1...k}, ICD, and RID to GPU device
 * 4: Launch the GPU kernel to label {P1...n} to the nearest
 * centroids with the help of ICD, RID, and triangle inequalities
 * 5: Copy {L1...n} back to host
 * 6: Calculate the mean for each cluster and update {C1...k}
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

/**
 * Find the euclidean distance betwen a datapoint and a centroid
 **/
__host__ __device__ inline static float compute_point_distance(points_t point, centroids_t centroid)
{
    int px = point.x;
    int py = point.y;
    int cx = centroid.x;
    int cy = centroid.y;

    float xdiff = px - cx;
    float ydiff = py - cy;

    return sqrt(pow(xdiff, 2) + pow(ydiff, 2));
}

/**
 * Find the euclidean distance between two centroids
 **/
__host__ __device__ inline static float compute_centroid_distance(centroids_t centroid1, centroids_t centroid2)
{
    int cx2 = centroid2.x;
    int cy2 = centroid2.y;
    int cx1 = centroid1.x;
    int cy1 = centroid1.y;

    float xdiff = cx2 - cx1;
    float ydiff = cy2 - cy1;

    return sqrt(pow(xdiff, 2) + pow(ydiff, 2));
}

/**
 * GPU function which is used to find the nearnest centroid of a given data point
 **/
__global__ static void findNearnestNeighbor(points_t *gpu_points_list, centroids_t *gpu_centroids_list, metaInfo_t *gpu_metadata, int *pBlock_changes, float *gpu_icd_matrix, int *gpu_rid_matrix)
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
    int org_dist = compute_point_distance(gpu_points_list[dataid],
                                          gpu_centroids_list[org_cluster_id]);
    int cluster_id = org_cluster_id;
    int curCnt = 0;
    float path_curr;

    int old_cindex = org_cluster_id * clusters;
    for (int j = 1; j < clusters; ++j)
    {
        curCnt = gpu_rid_matrix[old_cindex + j];
        if (gpu_icd_matrix[old_cindex + curCnt] > 2 * org_dist)
        {
            break;
        }
        // failed on triangle inequalities, continue to orginal calculation of distance
        path_curr = compute_point_distance(gpu_points_list[dataid], gpu_centroids_list[curCnt]);

        if (path_min == 0 || path_curr < path_min)
        {
            path_min = path_curr;
            cluster_id = curCnt;
        }
    }
    // assign updated label centroid to the data point
    gpu_points_list[dataid].cluster = cluster_id;
    gpu_points_list[dataid].distance = path_min;
    //printf ("val id %d changes from %d to %d\n", dataid, org_cluster_id, gpu_points_list[dataid].cluster);
    //  use reduction function to get sum of point's cluster change in each block
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

/**
 * Initialize the ICD matrix to store distances between two centroids.
 **/
void build_ICD_matrix(float *icd_matrix, int cluster, centroids_t *c_list)
{
    // icd is a k × k matrix storing the distances between every two centroids
    int c1 = 0;
    int c2 = 0;
    for (c1 = 0; c1 < cluster; ++c1)
    {
        for (c2 = 0; c2 < cluster; ++c2)
        {
            int index = c1 * cluster + c2;
            double dist = compute_centroid_distance(c_list[c1], c_list[c2]);
            icd_matrix[index] = dist;
        }
    }
}

int compare(const void *vala, const void *valb)
{
    const int *a = (const int *)vala;
    const int *b = (const int *)valb;
    if (a[0] == b[0])
        return a[1] - b[1];
    else
        return a[0] - b[0];
}

/**
 * Initialize the RID matrix to store storing results of distances from small to large
 **/
void build_RID_matrix(float *icd_matrix, int *rid_matrix, int cluster)
{
    // RID another k × k matrix,where each row is a permutation of 1, 2, · · · k representing the closeness of the distances from Ci to other centroids
    for (int i = 0; i < cluster; ++i)
    {
        // for each row, create a 2d array to store icd value and its index
        float temp[cluster][2];
        int index = i * cluster;
        for (int j = 0; j < cluster; ++j)
        {
            temp[j][0] = icd_matrix[index + j];
            temp[j][1] = j;
        }
        qsort(temp, cluster, sizeof(*temp), compare);
        // copy sorted centroid index to rid matrix
        for (int j = 0; j < cluster; ++j)
        {
            int sortedindex = temp[j][1];
            rid_matrix[index + j] = sortedindex;
        }
    }
}

/**
 * Main function of cuda kmeans algorithm, called from main.cpp
 **/
double kmeans_cuda_triangle_ineq(int *n_points, int clusters, points_t **p_list, centroids_t **c_list, int iterations)
{
    // host data
    points_t *points_list = *p_list;
    int num_points = *n_points;
    int threshold_change = 0;
    *c_list = (centroids_t *)malloc(sizeof(centroids_t) * clusters);
    centroids_t *centroids_list = *c_list;
    metaInfo_t *metaData = (metaInfo_t *)malloc(sizeof(metaInfo_t));
    metaData->numpoints = num_points;
    metaData->iterations = iterations;
    metaData->cluster = clusters;

    // icd and rid matrix, used for triangle inequalities
    int matrixSize = clusters * clusters;
    float *icd_matrix = (float *)calloc(matrixSize, sizeof(float));
    int *rid_matrix = (int *)calloc(matrixSize, sizeof(int));
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

    // two auxiliary steps (calculating and sorting the inter-centroid distances) are included before the labeling step.

    // device data on gpu
    points_t *gpu_points_list;
    centroids_t *gpu_centroids_list = *c_list;
    metaInfo_t *gpu_metaData;
    float *gpu_icd_matrix;
    int *gpu_rid_matrix;

    // Copy {C1...k}, ICD, and RID to GPU device
    //  Copy data from device to GPU

    cudaMalloc(&gpu_points_list, num_points * sizeof(points_t));
    cudaMalloc(&gpu_centroids_list, sizeof(centroids_t) * clusters);
    cudaMalloc(&gpu_icd_matrix, sizeof(float) * matrixSize);
    cudaMalloc(&gpu_rid_matrix, sizeof(int) * matrixSize);
    cudaMalloc(&gpu_metaData, sizeof(metaInfo_t));

    cudaMemcpy(gpu_points_list, points_list, num_points * sizeof(points_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroids_list, centroids_list, sizeof(centroids_t) * clusters, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_metaData, metaData, sizeof(metaInfo_t), cudaMemcpyHostToDevice);

    // initialize CUDA constants
    // labeling n data points are distributed to a total number of GridSize × BlockSize threads, so each thread is responsible for a subset of t = dn/GridSize/BlockSizee
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
    

    double icdtime =0.0;
    double ridtime =0.0;
    double copytime =0.0;
    double findnearest =0.0;
    double updatecentroid =0.0;

    double startTime = CycleTimer::currentSeconds();
    do
    {
        // point label process:
        // build ICD matrix, store distance between two centroids
        double s1 = CycleTimer::currentSeconds();
        build_ICD_matrix(icd_matrix, clusters, centroids_list);
        double e1 = CycleTimer::currentSeconds();
        icdtime += (e1-s1);
        
        // sort each row of the ICD matrix to derive the ranked index (RID) matrix
        double s2 = CycleTimer::currentSeconds();
        build_RID_matrix(icd_matrix, rid_matrix, clusters);
        double e2 = CycleTimer::currentSeconds();
        ridtime +=  e2 -s2 ;
        
        
        // 3. copy the matrixes to GPU device
        double s3 = CycleTimer::currentSeconds();
        cudaMemcpy(gpu_icd_matrix, icd_matrix, sizeof(float) * matrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_rid_matrix, rid_matrix, sizeof(int) * matrixSize, cudaMemcpyHostToDevice);
        double e3 = CycleTimer::currentSeconds();
        copytime += e3-s3;
        
        // 4. for point Px that was labeled to centroid i, loop through the centrouds following sorted sequence in the ith row of RID matrix.
        // get the nearest centroids
        double s4 = CycleTimer::currentSeconds();
        findNearnestNeighbor<<<blocks, threadPerBlock, blocks_sharedInfo>>>(gpu_points_list, gpu_centroids_list, gpu_metaData, pBlock_changes, gpu_icd_matrix, gpu_rid_matrix);
        
        //printf ("pass findNearnestNeighbor \n");
        // 5. copy back host
        cudaMemcpy(points_list, gpu_points_list, num_points * sizeof(points_t), cudaMemcpyDeviceToHost);
        double e4 = CycleTimer::currentSeconds();
        
        findnearest += e4-s4;
        // 6. find the mean for clusters and update centroids
        double s5 = CycleTimer::currentSeconds();
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
            // printf ("findNearnestNeighbor data on gpu % d\n", centroids_list[i].count);
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
        double e5 = CycleTimer::currentSeconds();
        updatecentroid += e5-s5;
        
        //find delta, will compared with threshold in the while condition
        find_change_reduction<<<1, blocks_rounded, blocks_sharedInfo_reduced>>>(pBlock_changes, blocks, blocks_rounded);
        // block change array sums the number of points change cluster in that block
        cudaDeviceSynchronize();
        cudaMemcpy(&threshold_change, pBlock_changes, sizeof(int), cudaMemcpyDeviceToHost);
        // check convergence
        
        delta = (float)threshold_change / num_points;

        ++curr_iters;
        // convergence = compute_convergence (centroids_list, clusters);
        // //printf ("new the nearest centroids %d %d\n", centroids_list[0].x, centroids_list[0].y );

    } while(delta > 0.000001 && curr_iters < 2000);
    double endTime = CycleTimer::currentSeconds();
    printf ("pass icd time %.3f \n", icdtime);
    printf ("pass rid time %.3f \n", ridtime );
    printf ("pass copy time %.3f \n", copytime);
    printf ("pass find nearest centroid time %.3f \n", findnearest );
    printf ("pass updated centroid time %.3f \n", updatecentroid);
    //delta > 0.000001 && 
    
    double overallDuration = endTime - startTime;
    return overallDuration;
}