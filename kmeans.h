/**
 * Kmeans Clustering
 * Yu Dong (yudong)
 */

#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <omp.h>

typedef struct { /* Define the data structure for wire here */
    //number of bends, can be 0,1,2
    int bend;
    int bend_pos[4]; //bend 2 times at most
    //start and endpoints, two arrays
    int start_pos[2];
    int end_pos[2];
    int wire_cost;
} wire_t;

typedef struct { /* Define the data structure for cost element here */
    int val; //val is the numer of wires in a cell
    int wire_in[20];
    omp_lock_t lock;
    int wire_count;
} cost_t;

typedef struct { /* Define the data structure for wire here */
    //board dims
    int dimx;
    int dimy; 
    //start and endpoints, two arrays
    cost_t *cost_table;
} cost_all;

//typedef int cost_t;

const char *get_option_string(const char *option_name, const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

#endif