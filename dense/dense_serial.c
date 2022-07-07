#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

void dens_mul_serial(int **matrix, int *vector, size_t row_size,
                     size_t col_size, int *output) {
  {
    memset(output, 0, sizeof(int) * col_size);

    for (size_t i = 0; i < col_size; i++)
      for (size_t j = 0; j < row_size; j++)
        output[i] += vector[j] * matrix[i][j];
  }
}

int main(int argc, char **argv) {
  srand(time(NULL));
if (argc < 4)
    return -1;
  int col_size = atoi(argv[1]);
  int row_size = atoi(argv[2]);
  float zero_chance = atof(argv[3]);

  int *matrix[col_size];
  int vector[row_size];
  int output[col_size];
  fill_vector(vector, row_size, 0.2f);
  fill_matrix(matrix, row_size, col_size, zero_chance);

  struct timeval t1, t2;
  double elapsedTime;

  gettimeofday(&t1, NULL);
  dens_mul_serial(matrix, vector, row_size, col_size, output);
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("time: %lf\n", elapsedTime);

  // printf("Matrix is:\n");
  // for (size_t i = 0; i < M; i++, printf("\n"))
  //   for (size_t j = 0; j < N; j++)
  //     printf("%d ", matrix[i][j]);
  // printf("\n");
  //
  // printf("Vector is:\n");
  // for (size_t j = 0; j < N; j++)
  //   printf("%d ", vector[j]);
  // printf("\n");
  //
  // printf("\n");
  // printf("Output is:\n");
  // for (size_t j = 0; j < M; j++)
  //   printf("%d ", output[j]);
  // printf("\n");

  return 0;
}
