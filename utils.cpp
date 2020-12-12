#include "utils.h"

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> utils::to_row_format(const Eigen::SparseMatrix<double, Eigen::RowMajor>& M)
{
  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<double> vals;

  for( int i = 0; i < M.outerSize(); ++i )
    for( typename Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(M, i); it; ++it ) {
      rows.push_back(it.row());
      cols.push_back(it.col());
      vals.push_back(it.value());
    }

  return make_tuple(rows, cols, vals);
}


Eigen::SparseMatrix<double, Eigen::RowMajor> utils::create_sparse(vector<int>& rows, vector<int>& cols, vector<double>& vals, int size, int density)
{
    

  Eigen::SparseMatrix<double, Eigen::RowMajor> result(size, size);
  result.reserve(Eigen::VectorXi::Constant(size, density)); // TODO: verificar se é assim (ou com int)

  for( int i = 0; i < vals.size(); ++i )
    result.insert(rows[i], cols[i]) = vals[i];
  result.makeCompressed();

  return result;
}


Eigen::SparseMatrix<double, Eigen::RowMajor> utils::create_sparse(const vector<utils::SparseData>& X, int size, int density)
{

  Eigen::SparseMatrix<double, Eigen::RowMajor> result(size, size);
  result.reserve(Eigen::VectorXi::Constant(size, density));

  for( int i = 0; i < X.size(); ++i )
    for( int j = 0; j < X[i].data.size(); ++j ) {
      result.coeffRef(i, X[i].indices[j]) = X[i].data[j];
      result.coeffRef(X[i].indices[j], i) = X[i].data[j];
    }


  return result;
}

long utils::tau_rand_int(vector<long>& state)
{

    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ ((((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19);
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ ((((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25);
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ ((((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11);

    return state[0] ^ state[1] ^ state[2];

}

double utils::rdist(const vector<double>& x, const vector<double>& y)
{
    double result = 0.0;
    int dim = x.size();

    for( int i = 0; i < dim; ++i ) {
        double diff = x[i]-y[i];
        result += diff*diff;
    }

    return result;
}

double utils::clip(double value)
{
    if( value > 4.0 )
      return 4.0;
    else if( value < -4.0 )
      return -4.0;
    else 
      return value;
}


vector<vector<double>> utils::pairwise_distances(vector<vector<double>>& X)
{

  int n = X.size();
  int d = X[0].size();


  vector<vector<double>> pd(n, vector<double>(n, 0.0));


  // TODO: add possibility for other distance functions
  #pragma omp parallel for 
  for( int i = 0; i < n; ++i ) {
    for( int j = i+1; j < n; ++j ) {

      double distance = 0;

      for( int k = 0; k < d; ++k ) {
       // distance += (X[i*d + k]-X[j*d + k])*(X[i*d + k]-X[j*d + k]);
        distance += (X[i][k]-X[j][k])*(X[i][k]-X[j][k]);
      }

      pd[i][j] = sqrt(distance);
      pd[j][i] = pd[i][j];
    }
  }

  return pd;
}