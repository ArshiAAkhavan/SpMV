#include <stdlib.h>
#define MOD 5

void fill_vector(int *v, size_t n, float chance) {
  for (size_t i = 0; i < n; i++) {
    v[i] = (rand() % MOD) * (((rand() % RAND_MAX) * 1.0 / RAND_MAX) < chance);
  }
}

void fill_matrix(int **matrix, size_t row_size, size_t col_size, float chance) {

  for (size_t i = 0; i < col_size; i++) {
    matrix[i] = malloc(sizeof(int) * row_size);
    fill_vector(matrix[i], row_size, chance);
  }
}
