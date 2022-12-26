#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;
float **alloc_plate(int height, int width)
{
	float *linear_buff = (float *) malloc(sizeof(float) * height * width);
	
	float **plate = (float **) malloc(sizeof(float *) * height);
	for(int i = 0; i < height; i++)
		plate[i] = linear_buff + (i * width);
	
	return plate;
}

void free_plate(float **plate, int height, int width)
{
	free(plate[0]);
	free(plate);
}

void init_plate(float **plate, int height, int width)
{
	for(int i = 0; i < width; i++)
	{
		plate[0][i] = 100.0; 
		plate[height - 1][i] = 0.0;
	}
	
	for(int i = 1; i < height - 1; i++)
	{
		plate[i][0] = 100.0;
		plate[i][width - 1] = 100.0;

		for(int j = 1; j < width - 1; j++)
			plate[i][j] = (plate[i][j - 1] + plate[i][width - 1] + plate[i - 1][j] + plate[height - 1][j]) / 4;
	}
}

float calc_heat_point(float **plate, int y, int x)
{
	return (plate[y - 1][x] + plate[y + 1][x] + plate[y][x - 1] + plate[y][x + 1]) / 4;
}

void swap_pointers(float ***first, float ***second)
{
	float **tmp = *first;
	*first = *second;
	*second = tmp;
}

std::ostream& writemap(std::ostream& os, float** map, int height, int width)
{
for (int i = 0; i < height; ++i)
{
    for (int j = 0; j < width; ++j)
        {
            os << map[i][j]<<" ";
        }
    os<<"\n";
    }
return os;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
    auto start = high_resolution_clock::now();
	// argumenti
	int height = 100;
	int width = 200;
	float epsilon = 10;

	int id, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n = floor(sqrt(size));
	
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	
	MPI_Group new_group;
	int ranges[3] = {0, n*n - 1, 1};
	MPI_Group_range_incl(world_group, 1, &ranges, &new_group);
	
	MPI_Comm new_world;
	MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_world);
	
	if(new_world == MPI_COMM_NULL)
	{
		MPI_Finalize();
		return 0;
	}

	float **first_plate = alloc_plate(height, width);
	float **second_plate = alloc_plate(height, width);
	init_plate(first_plate, height, width);
	init_plate(second_plate, height, width);

	int x_start = 1 + (int)((double)(id % n) * (width - 2) / n);
	int x_end = 1 + (int)((double)((id % n) + 1) * (width - 2) / n);
	int y_start = 1 + (int)((double)(id / n) * (height - 2) / n);
	int y_end = 1 + (int)((double)((id / n) + 1) * (height - 2) / n);
	
	int iterations = 0;
	
	while(1)
	{
		float local_max_diff = 0.0;
		
		for(int i = y_start; i < y_end; i++)
		{
			for(int j = x_start; j < x_end; j++)
			{
				first_plate[i][j] = calc_heat_point(second_plate, i, j);
				
				float curr_diff = fabs(first_plate[i][j] - second_plate[i][j]);
				
				if(curr_diff > local_max_diff)
					local_max_diff = curr_diff;
			}
		}

		float global_max_diff;
		MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_FLOAT, MPI_MAX, new_world);

		swap_pointers(&first_plate, &second_plate);
		iterations++;

		if(global_max_diff < epsilon)
			break;

		MPI_Request r;
		
		if(id / n > 0)
		{
			MPI_Isend(&second_plate[y_start][x_start], x_end - x_start, MPI_FLOAT, id - n, 0, new_world, &r);
		}
		if(id / n < n - 1)
		{
			MPI_Isend(&second_plate[y_end - 1][x_start], x_end - x_start, MPI_FLOAT, id + n, 0, new_world, &r);
		}
		if(id % n > 0)
		{
			float left_col_send[y_end - y_start];
			for(int i = y_start; i < y_end; i++)
				left_col_send[i - y_start] = second_plate[i][x_start];
			MPI_Isend(left_col_send, y_end - y_start, MPI_FLOAT, id - 1, 0, new_world, &r);
		}
		if(id % n < n - 1)
		{
			float right_col_send[y_end - y_start];
			for(int i = y_start; i < y_end; i++)
				right_col_send[i - y_start] = second_plate[i][x_end - 1];
			MPI_Isend(right_col_send, y_end - y_start, MPI_FLOAT, id + 1, 0, new_world, &r);
		}

		if(id / n > 0)
		{
			MPI_Recv(&second_plate[y_start - 1][x_start], x_end - x_start, MPI_FLOAT, id - n, MPI_ANY_TAG, new_world, MPI_STATUS_IGNORE);
		}
		if(id / n < n - 1)
		{
			MPI_Recv(&second_plate[y_end][x_start], x_end - x_start, MPI_FLOAT, id + n, MPI_ANY_TAG, new_world, MPI_STATUS_IGNORE);
		}
		if(id % n > 0)
		{
			float left_col_recv[y_end - y_start];
			MPI_Recv(left_col_recv, y_end - y_start, MPI_FLOAT, id - 1, MPI_ANY_TAG, new_world, MPI_STATUS_IGNORE);
			for(int i = y_start; i < y_end; i++)
				second_plate[i][x_start - 1] = left_col_recv[i - y_start];
		}
		if(id % n < n - 1)
		{
			float right_col_recv[y_end - y_start];
			MPI_Recv(right_col_recv, y_end - y_start, MPI_FLOAT, id + 1, MPI_ANY_TAG, new_world, MPI_STATUS_IGNORE);
			for(int i = y_start; i < y_end; i++)
				second_plate[i][x_end] = right_col_recv[i - y_start];
		}
	}

	if(id == 0)
	{
		for(int p = 1; p < n*n; p++)
		{
			x_start = 1 + (int)((double)(p % n) * (width - 2) / n);
			x_end = 1 + (int)((double)((p % n) + 1) * (width - 2) / n);
			y_start = 1 + (int)((double)(p / n) * (height - 2) / n);
			y_end = 1 + (int)((double)((p / n) + 1) * (height - 2) / n);

			for(int i = y_start; i < y_end; i++)
			{
				MPI_Recv(&second_plate[i][x_start], x_end - x_start, MPI_FLOAT, p, i, new_world, MPI_STATUS_IGNORE);
			}
		}
    std::fstream of("Map.txt", std::ios::out | std::ios::app);

    if (of.is_open())
    {
        writemap(of, second_plate, height,width);
        of.close();
    }

	}

	else
	{
		for(int i = y_start; i < y_end; i++)
		{
			MPI_Send(&second_plate[i][x_start], x_end - x_start, MPI_FLOAT, 0, i, new_world);
		}
	}

	free_plate(first_plate, height, width);
	free_plate(second_plate, height, width);
    auto end = high_resolution_clock::now();
    auto duration = (end - start);
    cout <<  chrono::duration <double, milli> (duration).count() << endl;
    printf("%d iterations.\n", iterations);
	MPI_Finalize();

	return 0;
}
