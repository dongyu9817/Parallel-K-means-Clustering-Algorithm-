/**
 * K-means cluster Algorithm Benchmark 
 * Yu Dong (Yu Dong)
 */

#include "../kmeans.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>


static void usage(const char *program_path)
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


int main(int argc, char **argv)
{

    const char *input_filename = NULL;
    int clusters = 0;
    const char *output_filename = NULL;
    const char *output_centroid_filename = NULL;
    //output_centroid_filename
    int num_points;
    points_t* points_list;
    centroids_t* centroids_list;
    int iterations = 20000;
    // Parse commandline options
    int opt;
    static struct option long_options[] = {
        {"help", 0, 0, '?'},
        {"cluster", 1, 0, 'n'},
        {"inputfile", 1, 0, 'f'},
        {"outputfile", 1, 0, 'o'},
        {"outputCentroid", 1, 0, 'c'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "f:k:o:c:?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'o':
            output_filename = optarg;
            break;
        case 'f':
            input_filename = optarg;
            break;
        case 'c':
            output_centroid_filename = optarg;
            break;
        case 'k':
            clusters = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    double cudaTime = 50000.;


    // double doubleval = get_option_float("-p", 0.1f);
    // int intval = get_option_int("-i", 5); 

    int error = 0;
   
    if (input_filename == NULL || output_filename == NULL || output_centroid_filename == NULL )
    {
        printf("Error: Missing Input Parameter.\n");
        error = 1;
    }

    if (error)
    {
        usage(argv[0]);
        return 1;
    }

    printf("Number of clusters: %d\n", clusters);
    printf("Input file: %s\n", input_filename);
    printf("Output file: %s\n", output_filename);

    read_inputfile (input_filename, &num_points, &points_list);
    printf ("number of points %d %d \n", points_list[300].x, 1);

    //cuda computation

    double gpu_time = kmeans_cuda_triangle_ineq(&num_points, clusters, &points_list, &centroids_list,  iterations);
    cudaTime = std::min(cudaTime, gpu_time);
   
    printf("GPU_time: %.3f ms\n", 1000.f * cudaTime);
    
    write_outputfile(output_filename, &num_points, points_list);
    write_centroidfile (output_centroid_filename, clusters, centroids_list);
}