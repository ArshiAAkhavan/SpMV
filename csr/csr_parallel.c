#include "utils.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <time.h>

#ifndef ROW_SIZE
#define ROW_SIZE 5
#endif /* !ROW_SIZE */

#ifndef COL_SIZE
#define COL_SIZE 3
#endif /* !COL_SIZE */

#define NUM_THREADS 16

// csr data format for storing matrices
typedef struct CSR {
  int *cols;
  size_t *row_ptrs;
  int *vals;
  size_t col_size;
  size_t nnz;
} csr_t;

// convert raw matrix to its csr format
void csr_from_raw(int **matrix, size_t row_size, size_t col_size, csr_t *csr) {

  // count nnz values
  size_t nnz = 0;
  for (size_t i = 0; i < col_size; i++)
    for (size_t j = 0; j < row_size; j++)
      nnz += matrix[i][j] != 0;

  printf("nnz is: %ld\n", nnz);
  // construct CSR
  csr->cols = malloc(sizeof(int) * nnz);
  csr->vals = malloc(sizeof(int) * nnz);
  csr->row_ptrs = malloc(sizeof(int) * (col_size + 1));
  csr->nnz = nnz;
  csr->col_size = col_size;
  printf("im here\n");

  // fill the CSR vectors
  size_t counter = 0;
  for (size_t i = 0; i < col_size; i++) {
    csr->row_ptrs[i] = counter;
    for (size_t j = 0; j < row_size; j++) {
      if (matrix[i][j]) {
        csr->cols[counter] = j;
        csr->vals[counter] = matrix[i][j];
        counter++;
      }
    }
  }
  // last row_ptr is not set in the for
  csr->row_ptrs[col_size] = counter;
}

void csr_mul_parallel(csr_t *csr, int *vector, int *output) {
  memset(output, 0, sizeof(int) * csr->col_size);

#pragma omp parallel for schedule(static) shared(output, vector, csr)
  for (size_t i = 0; i < csr->col_size; i++) {
    for (size_t j = csr->row_ptrs[i]; j < csr->row_ptrs[i + 1]; j++) {
      output[i] += vector[csr->cols[j]] * csr->vals[j];
    }
  }
}

int main() {
  omp_set_num_threads(get_nprocs()); // TODO: set this to 1, 4, 16, and 64

#pragma omp parallel
  { volatile char sync_threads = 0; }

  srand(time(NULL));
  int *matrix[COL_SIZE];
  int vector[ROW_SIZE];
  int output[COL_SIZE];
  csr_t csr;

  fill_vector(vector, ROW_SIZE, 0.9f);
  fill_matrix(matrix, ROW_SIZE, COL_SIZE, 0.5f);
  csr_from_raw(matrix, ROW_SIZE, COL_SIZE, &csr);
  csr_mul_parallel(&csr, vector, output);

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
  printf("csr format is:\n");
  printf("r_ptrs:\t");
  for (size_t j = 0; j < csr.col_size + 1; j++)
    printf("%ld ", csr.row_ptrs[j]);

  printf("\ncols:\t");
  for (size_t j = 0; j < csr.nnz; j++)
    printf("%d ", csr.cols[j]);

  printf("\nvals:\t");
  for (size_t j = 0; j < csr.nnz; j++)
    printf("%d ", csr.vals[j]);
  printf("\n");

  printf("\n");
  printf("Output is:\n");
  for (size_t j = 0; j < COL_SIZE; j++)
    printf("%d ", output[j]);
  printf("\n");

  return 0;
}
