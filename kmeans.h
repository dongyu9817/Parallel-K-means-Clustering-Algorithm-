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
    int cluster; //the cluster id to which a point belongs to
    
} points_t;

typedef struct { 
    int x; //x axis coordinate
    int y; //y axis coordinate
    //centroids of each iteration untill convergence
} centroids_t;


void kmeans_sequential(int* num_points, int clusters, points_t* points_list, centroids_t* centroids_list, int iterations);

const char *get_option_string(const char *option_name, const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

#endif