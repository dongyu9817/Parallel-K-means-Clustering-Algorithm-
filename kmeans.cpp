
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
    FILE *input = fopen(input_filename, "r");

    if (!input)
    {
        printf("Unable to open file: %s.\n", input_filename);
        return 1;
    }

    int dim_x, dim_y;
    int num_of_wires;

    fscanf(input, "%d %d\n", &dim_y, &dim_x);
    fscanf(input, "%d\n", &num_of_wires);
    // num_of_wires = 2;
    wire_t *wires_temp = (wire_t *)calloc(num_of_wires, sizeof(wire_t));
	wire_t *wires = (wire_t *)calloc(num_of_wires, sizeof(wire_t));
	//sort the wire array from small to large
	int array[num_of_wires][2];
    /* Read the grid dimension and wire information from file */
    for (int i = 0; i < num_of_wires; ++i)
    {
        int sx, sy; // start point
        int ex, ey; // end point
        fscanf(input, "%d %d %d %d\n", &sx, &sy, &ex, &ey);
        wire_t *temp = wire_maker(0, sx, sy, ex, ey, 0, 0, 0, 0, 0);
		wires_temp[i] = *temp;
		array[i][0] = abs(ex - sx) * abs(ey - sy);
		array[i][1] = i;
    }
}