# SpMV

## Methods 

In the following section we would briefly investigate methods used for calculating SpMV

### Naive approach

This method makes no use of the fact that our input matrix is in fact sparse and tries to multiply matrix _M_ to vector _V_
by naively iterating over each cell in matrix.

This implementation can be find in `dense/dense_serial.c` source file.

### CSR
CSR (Compressed Sparse Row) is a well-known data structure used to store sparse matrices

The amount of memory needed for storing a sparse matrix in this format is of `O(nnz+n)` where `nnz` is the 
number of none zero values and n is number of rows in matrix. 

Our CSR format is something like this: 

```C 
typedef struct CSR {
  int *cols;
  int *vals;
  size_t *row_ptrs;

  size_t col_size;
  size_t nnz;
} csr_t;
```
where,
- `col_size` is size of each column which is equal to number of rows.
- `nnz` is number of none zero values in our matrix.
- `vals` is none zero values in our matrix sorted by their row number.
- `cols` is an array of size `nnz` which each index represent column number of their counter part in `vals` array.
- `row_ptrs` is an array containing pointers (indices of `cols` array) determining row number of each value. as an 
example, all values between `row_ptrs[row_idx]` and `row_ptrs[row_idx+1]` indices in `vals`, have row number of `row_idx`.

This data structure has huge advantage when used for storing sparse matrices since it doesn't store any zero value and could reduce data size 
by orders of magnitude (usually `nnz` is of `O(n)` which means we need `O(n)` memory as oppose to `O(m*n)` in naive implementation).

This decrease in storage size also means that we need to do less computation for SpMV since we only have access to `nnz` values and not the whole matrix

`csr/csr_serial.c` implements a serial implementation of SpMV for matrices stored with `csr_t` data format.
as shown in the source file, after converting matrix to its `csr_t` format, all we have to do is do this simple for loop: 

```C 
  for (size_t i = 0; i < csr->col_size; i++) {
    for (size_t j = csr->row_ptrs[i]; j < csr->row_ptrs[i + 1]; j++) {
      output[i] += vector[csr->cols[j]] * csr->vals[j];
    }
  }
```






