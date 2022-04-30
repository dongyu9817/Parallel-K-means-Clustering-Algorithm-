/**
 * Sequential K-means cluster Algorithm
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

float compute_distance(points_t point, centroids_t centroid)
{
    int px = point.x;
    int py = point.y;
    int cx = centroid.x;
    int cy = centroid.y;

    float xdiff = px - cx;
    float ydiff = py - cy;

    return sqrt(pow(xdiff, 2) + pow(ydiff, 2));
}

int compute_convergence (centroids_t* centroids_list, int clusters) {
    for (int i = 0; i < clusters; ++i)
    {
        if (centroids_list[i].x != centroids_list[i].prevx || centroids_list[i].y != centroids_list[i].prevy ) {
            return 0;
        }
    }
    return 1;
}
void kmeans_sequential(int *n_points, int clusters, points_t **p_list, centroids_t **c_list, int iterations)
{
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

    // update stage
    int convergence = 0;
    int curr_iters =0;
    double delta = 0.0;
    do
    {
        delta = 0.0;
        // get the nearest centroids
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
        //printf ("new the nearest centroids %d\n", curr_iters );

    } while (delta > 0.000001 && curr_iters < 2000);

    return;
}
