#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef ROW_SIZE
#define ROW_SIZE 5
#endif /* !ROW_SIZE */

#ifndef COL_SIZE
#define COL_SIZE 3
#endif /* !COL_SIZE */

void dens_mul_serial(int **matrix, int *vector, size_t row_size,
                     size_t col_size, int *output) {
  memset(output, 0, sizeof(int) * col_size);

  for (size_t i = 0; i < col_size; i++)
    for (size_t j = 0; j < row_size; j++)
      output[i] += vector[j] * matrix[i][j];
}

int main() {
  srand(time(NULL));
  int *matrix[COL_SIZE];
  int vector[ROW_SIZE];
  int output[COL_SIZE];
  fill_matrix(matrix, ROW_SIZE, COL_SIZE, 0.5f);
  fill_vector(vector, ROW_SIZE, 0.8f);
  dens_mul_serial(matrix, vector, ROW_SIZE, COL_SIZE, output);

  printf("Matrix is:\n");
  for (size_t i = 0; i < COL_SIZE; i++, printf("\n"))
    for (size_t j = 0; j < ROW_SIZE; j++)
      printf("%d ", matrix[i][j]);
  printf("\n");

  printf("Vector is:\n");
  for (size_t j = 0; j < ROW_SIZE; j++)
    printf("%d ", vector[j]);
  printf("\n");

  printf("\n");
  printf("Output is:\n");
  for (size_t j = 0; j < COL_SIZE; j++)
    printf("%d ", output[j]);
  printf("\n");

  return 0;
}
