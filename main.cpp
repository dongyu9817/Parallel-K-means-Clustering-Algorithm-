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
#include <omp.h>
#include <cmath>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

static int _argc;
static const char **_argv;

const char *get_option_string(const char *option_name, const char *default_value)
{
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return _argv[i + 1];
    return default_value;
}

int get_option_int(const char *option_name, int default_value)
{
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return atoi(_argv[i + 1]);
    return default_value;
}

float get_option_float(const char *option_name, float default_value)
{
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return (float)atof(_argv[i + 1]);
    return default_value;
}

static void show_help(const char *program_path)
{
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    //printf("\t-n <num_of_threads> (required)\n");
    printf("\t-k <number of clusters>\n");
    printf("\t-o <output filename>\n");
    printf("\t-c <output centroid filename>\n");
}


void read_inputfile (const char* input_filename, int* num_points, points_t** points_list) {
    FILE *input = fopen(input_filename, "r");

    if (!input)
    {
        printf("Unable to open file: %s.\n", input_filename);
        return;
    }

    fscanf(input, "%d \n", num_points);
    *points_list = (points_t*)malloc(sizeof(points_t)*((*num_points)));

    for (int i = 0; i < *num_points; ++i)
    {
        int xval;
        int yval;
        fscanf(input, "%d %d\n", &xval, &yval);
        (*points_list + i)->x = xval;
        (*points_list + i)->y = yval;
        (*points_list + i)->cluster = -1;
    }
    
    fclose (input);
}

void write_outputfile (const char* output_filename, int* num_points, points_t* points_list) {
    FILE *output = fopen(output_filename, "w");

    if (!output)
    {
        printf("Unable to open file: %s.\n", output_filename);
        return;
    }

    //fprintf(output, "%d \n", *num_points);
	
    for (int i = 0; i < *num_points; ++i)
    {
        	
        fprintf(output, "%d %d %d\n", points_list[i].x, points_list[i].y, points_list[i].cluster);
    }

    fclose (output);
}

void write_centroidfile (const char* output_filename, int cluster, centroids_t* centroids_list) {
    FILE *output = fopen(output_filename, "w");

    if (!output)
    {
        printf("Unable to open file: %s.\n", output_filename);
        return;
    }

    //fprintf(output, "%d \n", *num_points);
	
    for (int i = 0; i < cluster; ++i)
    {	
        fprintf(output, "%d %d\n", centroids_list[i].x, centroids_list[i].y);
    }

    fclose (output);
}


int main(int argc, const char *argv[])
{
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto init_start = Clock::now();
    double init_time = 0;

    _argc = argc - 1;
    _argv = argv + 1;

    const char *input_filename = get_option_string("-f", NULL);
    int clusters = get_option_int("-k", 1);
    const char *output_filename = get_option_string("-o", NULL);
    const char *output_centroid_filename = get_option_string("-c", NULL);
    //output_centroid_filename
    int num_points;
    points_t* points_list;
    centroids_t* centroids_list;
    int iterations;
    // double doubleval = get_option_float("-p", 0.1f);
    // int intval = get_option_int("-i", 5); 

    int error = 0;

    if (input_filename == NULL)
    {
        printf("Error: You need to specify -f.\n");
        error = 1;
    }

    if (error)
    {
        show_help(argv[0]);
        return 1;
    }

    printf("Number of clusters: %d\n", clusters);
    printf("Input file: %s\n", input_filename);
    printf("Output file: %s\n", output_filename);

    read_inputfile (input_filename, &num_points, &points_list);
    printf ("number of points %d %d \n", points_list[300].x, 1);
    init_time += duration_cast<dsec>(Clock::now() - init_start).count();
    printf("Initialization Time: %lf.\n", init_time);

    auto compute_start = Clock::now();
    double compute_time = 0;
    //sequential computation
    kmeans_sequential(&num_points, clusters, &points_list, &centroids_list,  iterations);

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);
    
    write_outputfile(output_filename, &num_points, points_list);
    write_centroidfile (output_centroid_filename, clusters, centroids_list);
}