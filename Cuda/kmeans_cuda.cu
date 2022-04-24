/**
 * Cuda K-means cluster Algorithm
 * This is the implementation of k-means clustering algorithm in CUDA version
 * Yu Dong (Yu Dong)
 */
#include "kmeans.h"

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

//global constant variable
int threadPerBlock = 128;

static inline int FindNextPower2 (int num) {
    num -= 1;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;
    //num |= num >> 32;
    num += 1;
    return num;
}

__global__ static void findNearnestNeighbor(points_t* gpu_points_list, centroids_t* gpu_centroids_list ) {
 

}

void kmeans_cuda(int *n_points, int clusters, points_t **p_list, centroids_t **c_list, int iterations)
{
    //host data
    points_t *points_list = *p_list;
    int num_points = *n_points;
    *c_list = (centroids_t *)malloc(sizeof(centroids_t) * clusters);
    centroids_t *centroids_list = *c_list;
    
    // randomly pick centroid based on the cluster number
    // get the initial centroids
    srand(time(0));
    for (int i = 0; i < clusters; ++i)
    {
        int index = rand() % num_points;
        centroids_list[i].x = points_list[index].x;
        centroids_list[i].y = points_list[index].y;
        centroids_list[i].prevx =0;
        centroids_list[i].prevy =0;
        centroids_list[i].sum_px =0;
        centroids_list[i].sum_py =0;
        centroids_list[i].count =0;
    }

    //device data on gpu
    points_t *gpu_points_list;
    centroids_t *gpu_centroids_list = *c_list;
    
    //Copy data from device to GPU
    cudaMalloc (&gpu_points_list, num_points* sizeof(points_t));
    cudaMalloc (&gpu_centroids_list, sizeof(centroids_t) * clusters);
    cudaMemcpy (gpu_points_list,points_list, num_points* sizeof(points_t), cudaMemcpyHostToDevice );
    cudaMemcpy (gpu_centroids_list, centroids_list, sizeof(centroids_t) * clusters, cudaMemcpyHostToDevice );
 
    //initialize CUDA constants
    const unsigned int blocks = (num_points + threadPerBlock -1) / threadPerBlock;
    const unsigned int blocks_rounded =FindNextPower2 (blocks);
    //initialize cluster shared date for each block
    const unsigned int blocks_sharedInfo;

    // update stage
    int convergence = 0;
    int curr_iters =0;
    double detla;

    do
    {
        delta = 0.0;
        // get the nearest centroids
        findNearnestNeighbor<<<blocks, threadPerBlock, blocks_sharedInfo >>> (gpu_points_list, gpu_centroids_list );
        cudaDeviceSynchronize();

        for (int i = 0; i < num_points; ++i)
        {
            float  path_min = 0;
            float  path_curr;
            int cluster_id = 0;
            for (int j = 0; j < clusters; ++j)
            {
                path_curr = compute_distance(points_list[i], centroids_list[j]);
                if (path_min == 0 || path_curr < path_min)
                {
                    path_min = path_curr;
                    cluster_id = j;
                    ++delta;
                }
                // update cluster
                points_list[i].cluster = cluster_id;
                points_list[i].distance = path_min;
                centroids_list[cluster_id].sum_px += (points_list[i].x);
                centroids_list[cluster_id].sum_py += (points_list[i].y);
                centroids_list[cluster_id].count += 1;
            }
        }
        //printf ("get the nearest centroids\n");
        // update centroids based on each mean value
        for (int i = 0; i < clusters; ++i)
        {
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
        //printf ("compare the nearest centroids\n");
        // compare nearest centroids and check convergence
        convergence = compute_convergence (centroids_list, clusters);
        // check convergence
        ++curr_iters;
        delta /= num_points;
        //printf ("new the nearest centroids %d %d\n", centroids_list[0].x, centroids_list[0].y );

    } while ( delta > 0.000001 || curr_iters < 200000);

    return;
}