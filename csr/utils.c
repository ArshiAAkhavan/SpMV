#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MOD 5

void fill_vector(int *v, size_t n, float chance) {
  memset(v, 0, sizeof(int) * n);
  int number_of_cols = ((int)(n * (1.0f - chance))) + 1;
  int nnz = rand() % number_of_cols;
  // printf("cols:%d, nnz:%d\n",number_of_cols,nnz);
  for (int i = 0; i < nnz; i++)
    v[rand() % n] = (rand() % MOD);
}

void fill_matrix(int **matrix, size_t row_size, size_t col_size, float chance) {
  for (size_t i = 0; i < col_size; i++) {
    matrix[i] = malloc(sizeof(int) * row_size);
    fill_vector(matrix[i], row_size, chance);
  }
}
