/**
 * Kmeans Clustering
 * Yu Dong (yudong)
 */

#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <omp.h>

typedef struct { 
    int x; //x axis coordinate
    int y; //y axis coordinate
    float distance;
    int cluster; //the cluster id to which a point belongs to
    
} points_t;

typedef struct { 
    int x; //x axis coordinate
    int y; //y axis coordinate
    int prevx;
    int prevy;
    int sum_px;
    int sum_py;
    int count;
    //centroids of each iteration untill convergence
} centroids_t;

typedef struct { 
    int numpoints; //the number of the datapoints
    int iterations; //the number of iterations
    int cluster; //the number of clusters
    //holds the general set up information of the kmeans algorithm
} metaInfo_t;

//structure for the dynamic workload queue
typedef struct { 
    int index;
    points_t task;

} queue_t;

void kmeans_sequential(int* num_points, int clusters, points_t** points_list, centroids_t** centroids_list, int iterations);
double  kmeans_cuda(int* num_points, int clusters, points_t** points_list, centroids_t** centroids_list, int iterations);
double  kmeans_cuda_triangle_ineq(int* num_points, int clusters, points_t** points_list, centroids_t** centroids_list, int iterations);
double  kmeans_cuda_triangle_ineq_loadWithepoch(int* num_points, int clusters, points_t** points_list, centroids_t** centroids_list, int iterations);


const char *get_option_string(const char *option_name, const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

#endif