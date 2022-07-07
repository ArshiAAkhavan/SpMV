#include <vector>
using namespace std;
typedef float input_t;
typedef int addr_t;
void fill_vector(vector<input_t> &v, const int len, float chance);
void fill_matrix(vector<vector<input_t>> &m, const int row_size,
                 const int col_size, float chance);
