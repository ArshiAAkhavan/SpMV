#include "utils.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

using namespace std;

const short AVX512_VLEN = 512;
const short AVX256_VLEN = 256;

const short PACK_SIZE = AVX256_VLEN / (8 * sizeof(input_t));

typedef struct EllpackSlice {
  int cap;
  input_t *vals;
  addr_t *cols;
  vector<int> pack_ptr;
  vector<pair<int, addr_t>> rows;
} ellpack_t;

void ellpack_from_raw(vector<vector<input_t>> &matrix, addr_t row_size,
                      addr_t col_size, ellpack_t &ellpack) {
  // sort the matrix by the nnz for each row
  // pairs are the combined value of <nnz_c , row_idx>
  // of their respective row
  ellpack.rows.reserve(col_size);
  for (addr_t i = 0; i < col_size; i++) {
    int nnz = 0;
    for (addr_t j = 0; j < row_size; j++)
      nnz += matrix[i][j] != 0;
    ellpack.rows.push_back(make_pair(nnz, i));
  }
  sort(ellpack.rows.begin(), ellpack.rows.end(),
       std::less<pair<int, addr_t>>());

  // calculate data_size for each pack
  int cap = 0;
  ellpack.pack_ptr.push_back(cap);
  for (addr_t i = PACK_SIZE - 1; i < col_size; i += PACK_SIZE) {
    cap += ellpack.rows[i].first * PACK_SIZE;
    ellpack.pack_ptr.push_back(cap);
  }
  cap += ellpack.rows.back().first * PACK_SIZE;
  ellpack.pack_ptr.push_back(cap);

  ellpack.cap = cap;
  // zero by default;
  ellpack.vals = (input_t *)aligned_alloc(32, cap * sizeof(input_t));
  ellpack.cols = (addr_t *)aligned_alloc(32, cap * sizeof(addr_t));
  fill_n(ellpack.vals, cap, 0);
  fill_n(ellpack.cols, cap, 0);

  // constructing ELLPACK_SLICED
  addr_t curr_pack_idx = 0;
  /* for each pack do the following:
   *    for each row in the pack:
   *        for each nz value in this row, place the val and its coll_idx
   *        in curr_pack_ptr* PACK_SIZE * nnz + row_ext where:
   *            curr_pack_ptr: start index of the current pack in vals
   *            nnz: nnz of the values currenctly seen in this row
   *            row_ext: offset of this value in its pack's row_slice
   * note:
   *    for every iteration of the `while` loop, curr_pack_ptr must be
   *    incremented by `PACK_SIZE` * max nnz seen in rows of curresponding
   *    pack, but since rows are sorted increasing, the last row in the pack
   *    has no less nnz (if not more) that other rows, hence, we can easy
   *    increment `curr_pack_ptr` by `PACK_SIZE` * nnz of last row in pack.
   */
  while (curr_pack_idx * PACK_SIZE < ellpack.rows.size()) {
    for (int row_ofs = 0; row_ofs < PACK_SIZE; row_ofs++) {
      if (curr_pack_idx * PACK_SIZE + row_ofs >= ellpack.rows.size())
        break;

      addr_t row_idx = ellpack.rows[curr_pack_idx * PACK_SIZE + row_ofs].second;
      int nnz = 0;
      for (addr_t col_idx = 0; col_idx < row_size; col_idx++) {
        if (matrix[row_idx][col_idx]) {
          auto curr_pack_ptr = ellpack.pack_ptr[curr_pack_idx];
          auto idx = curr_pack_ptr + PACK_SIZE * (nnz) + row_ofs;

          ellpack.vals[idx] = matrix[row_idx][col_idx];
          ellpack.cols[idx] = col_idx;
          nnz++;
        }
      }
    }
    curr_pack_idx++;
  }
}

void ellpack_mul_simd(ellpack_t &ellpack, vector<input_t> &vect,
                      vector<input_t> &out) {
  out.assign(ellpack.rows.size(), 0);
  input_t *vec = (input_t *)aligned_alloc(32, vect.size() * sizeof(input_t));
  input_t *res =
      (input_t *)aligned_alloc(32, ellpack.rows.size() * sizeof(input_t));
  fill_n(res, ellpack.rows.size(), 0);

  for (addr_t i = 0; i < vect.size(); i++)
    vec[i] = vect[i];

  for (addr_t pck_idx = 0; pck_idx < ellpack.pack_ptr.size() - 1; pck_idx++) {
    for (addr_t pck_ofs = ellpack.pack_ptr[pck_idx];
         pck_ofs < ellpack.pack_ptr[pck_idx + 1]; pck_ofs += PACK_SIZE) {
      addr_t row_ofs = pck_idx * PACK_SIZE;

      __m256i m256_idx =
          _mm256_load_si256((__m256i const *)(ellpack.cols + pck_ofs));
      __m256 m256_val = _mm256_load_ps((ellpack.vals + pck_ofs));
      __m256 m256_prv = _mm256_load_ps((res + row_ofs));
      __m256 m256_vec = _mm256_i32gather_ps(vec, m256_idx, 4);

      /* sadly my system didn't have support for `mm_fmadd` which was
       * introduced in avx512vl
       */
      __m256 m256_mul = _mm256_mul_ps(m256_vec, m256_val);
      __m256 m256_out = _mm256_add_ps(m256_mul, m256_prv);

      /* sadly my system didn't have support for `mm_i32scatter` which was
       * introduced in avx512vl
       * so we just directly load to an intermediate array and then do
       * the scatter by hand
       */
      _mm256_store_ps((res + row_ofs), m256_out);
    }
  }

  /* sadly my system didn't have support for `mm_i32scatter` which was
   * introduced in avx512vl
   */
  for (addr_t i = 0; i < ellpack.rows.size(); i++) {
    addr_t row_idx = ellpack.rows[i].second;
    out[row_idx] = res[i];
  }
}

int main(int argc, char **argv) {
  srand(time(NULL));
  if (argc < 3)
    return -1;
  int col_size = atoi(argv[1]);
  int row_size = atoi(argv[2]);

  vector<vector<input_t>> mat;
  vector<input_t> vec;
  vector<input_t> out;
  fill_matrix(mat, row_size, col_size, 0.5f);
  fill_vector(vec, row_size, 0.8f);

  ellpack_t ellpack;
  ellpack_from_raw(mat, row_size, col_size, ellpack);

  struct timeval t1, t2;
  double elapsedTime;

  gettimeofday(&t1, NULL);
  ellpack_mul_simd(ellpack, vec, out);
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("time: %lf\n", elapsedTime);

  // printf("Matrix is:\n");
  // for (auto row : mat) {
  //   for (auto cell : row) {
  //     cout << cell << " ";
  //   }
  //   cout << endl;
  // }
  // cout << endl;
  //
  // cout << "Vector is:" << endl;
  // for (auto cell : vec) {
  //   cout << cell << " ";
  // }
  // cout << endl;
  //
  // cout << "ellpack.vals is:" << endl;
  // for (int i = 0; i < ellpack.cap; i++) {
  //   cout << ellpack.vals[i] << " ";
  // }
  // cout << endl;
  //
  // cout << "ellpack.cols is:" << endl;
  // for (int i = 0; i < ellpack.cap; i++) {
  //   cout << ellpack.cols[i] << " ";
  // }
  // cout << endl;
  //
  // cout << "ellpack.pack_ptr is:" << endl;
  // for (auto cell : ellpack.pack_ptr) {
  //   cout << cell << " ";
  // }
  // cout << endl;
  // cout << endl;
  // cout << "Output is:" << endl;
  // for (auto cell : out) {
  //   cout << cell << " ";
  // }
  // cout << endl;

  return 0;
}
