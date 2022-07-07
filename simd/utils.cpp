#include "utils.h"
#include <random>
#include <vector>

#define MOD 5
using namespace std;

// v.reserve(len);
// v.assign(len, 0);
// int nnz = rand() % ((int)(len * (1.0f - chance)));
// for (int i = 0; i < nnz; i++)
//   v[rand() % len] = (rand() % MOD);
void fill_vector(vector<input_t> &v, const int len, float chance) {
  v.reserve(len);
  for (int i = 0; i < len; i++) {
    v.push_back((rand() % MOD) *
                (((rand() % RAND_MAX) * 1.0 / RAND_MAX) < chance));
  }
}

void fill_matrix(vector<vector<input_t>> &m, const int row_size,
                 const int col_size, float chance) {
  m.reserve(col_size);
  for (int i = 0; i < col_size; i++) {
    vector<input_t> v;
    fill_vector(v, row_size, chance);
    m.push_back(v);
  }
}
