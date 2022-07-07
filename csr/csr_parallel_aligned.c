#include "utils.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <time.h>


/* this struct is cash_aligned in a sense that each variable of this type
 * would consume exactly one cache block (on x86 systems), thus, preventing
 * other variable to be placed in the same cache block as it's own cache
 * block.
 * it is usefull when we have an output array which is going to be modified
 * by multiple threads at the same time. since each index of this array is
 * of type `cache_optimized_int`, no false sharing occurs. (notice it does
 * not use any methods of syncronizations, so an index may still be modified
 * by multiple threads)
 */
typedef struct cache_optimized_int {
  int __attribute__((aligned(64))) data;
} data_t;

// csr data format for storing matrices
typedef struct CSR {
  size_t nnz;
  size_t col_size;
  int *cols;
  size_t *row_ptrs;
  int *vals;
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
  csr->row_ptrs = malloc(sizeof(size_t) * (col_size + 1));
  csr->nnz = nnz;
  csr->col_size = col_size;

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

void csr_mul_parallel(csr_t *csr, int *vector, data_t *output) {
  memset(output, 0, sizeof(data_t) * csr->col_size);

#pragma omp parallel for schedule(static) shared(output, vector, csr)
  for (size_t i = 0; i < csr->col_size; i++) {
    for (size_t j = csr->row_ptrs[i]; j < csr->row_ptrs[i + 1]; j++) {
      output[i].data += vector[csr->cols[j]] * csr->vals[j];
    }
  }
}

int main(int argc, char **argv) {
  srand(time(NULL));
  if (argc < 4)
    return -1;
  int col_size = atoi(argv[1]);
  int row_size = atoi(argv[2]);
  float zero_chance = atof(argv[3]);

  omp_set_num_threads(get_nprocs()); // TODO: set this to 1, 4, 16, and 64

#pragma omp parallel
  { volatile char sync_threads = 0; }

  int *matrix[col_size];
  int vector[row_size];
  csr_t csr;
  data_t __attribute__((aligned(64))) output[col_size];

  fill_vector(vector, row_size, 0.2f);
  fill_matrix(matrix, row_size, col_size, zero_chance);
  csr_from_raw(matrix, row_size, col_size, &csr);

  struct timeval t1, t2;
  double elapsedTime;

  gettimeofday(&t1, NULL);
  csr_mul_parallel(&csr, vector, output);
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("time: %lf\n", elapsedTime);

  // printf("Matrix is:\n");
  // for (size_t i = 0; i < COL_SIZE; i++, printf("\n"))
  //   for (size_t j = 0; j < ROW_SIZE; j++)
  //     printf("%d ", matrix[i][j]);
  // printf("\n");
  //
  // printf("Vector is:\n");
  // for (size_t j = 0; j < ROW_SIZE; j++)
  //   printf("%d ", vector[j]);
  // printf("\n");
  //
  // printf("\n");
  // printf("csr format is:\n");
  // printf("r_ptrs:\t");
  // for (size_t j = 0; j < csr.col_size + 1; j++)
  //   printf("%ld ", csr.row_ptrs[j]);
  //
  // printf("\ncols:\t");
  // for (size_t j = 0; j < csr.nnz; j++)
  //   printf("%d ", csr.cols[j]);
  //
  // printf("\nvals:\t");
  // for (size_t j = 0; j < csr.nnz; j++)
  //   printf("%d ", csr.vals[j]);
  // printf("\n");
  //
  // printf("\n");
  // printf("Output is:\n");
  // for (size_t j = 0; j < COL_SIZE; j++)
  //   printf("%d ", output[j].data);
  // printf("\n");
  //
  return 0;
}
