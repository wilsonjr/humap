#ifndef UTILS_H
#define UTILS_H

#include <tuple>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>
#include <Eigen/Sparse>

#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace std;

namespace utils {


/**
 * Storage for a row in a sparse matrix representation
 */
struct SparseData
{
  SparseData() {}

  SparseData(vector<double> data_, vector<int> indices_): data(data_), indices(indices_) {} 


  /**
  * Adds non-zero value to representation
  *
  * @param index column index in the sparse matrix
  * @param value the non-zero value
  */
  void push(int index, double value) {
    data.push_back(value);
    indices.push_back(index);
  }

  // non-zero values
  vector<double> data;

  // column indices
  vector<int> indices;
};


/**
 * Computes a array of linearly spaced numbers
 *
 * ...
 *
 * @tparam T the type of the range
 * @param start_in value representing the range begin
 * @param start_in value representing the range end
 * @param num_in number of values in the returned array
 * @return Container with the resulting values
 */
template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

/**
 * Computes the sorting indices of an array
 *
 * ...
 *
 * @tparam T the type of the Container
 * @param data Container to compute sorting array
 * @param reserve indicates whether to sort increasingly or not
 * @return Container with the resulting sorting indices
 */
template<typename T>
std::vector<int> argsort(const std::vector<T>& data, bool reverse=false) {

  std::vector<int> v(data.size());

  std::iota(v.begin(), v.end(), 0);
  if( reverse ) 
    std::sort(v.begin(), v.end(), [&](int i, int j){ return data[i] > data[j]; });
  else
    std::sort(v.begin(), v.end(), [&](int i, int j){ return data[i] < data[j]; });

  return v;
}


/**
 * Rearrages an array based on indices
 *
 * ...
 *
 * @tparam T the type of the Container
 * @param data Container to be rearranged
 * @param indices Container with indices
 * @return Container with rearranged values
 */
template<typename T>
std::vector<T> arrange_by_indices(const std::vector<T>& data, std::vector<int>& indices) 
{
  std::vector<T> v(indices.size());

  for( int i = 0; i < indices.size(); ++i ) {
    v[i] = data[indices[i]];
  }

  return v;
}


// Converts an Eigen::Matrix to tuple format
std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> to_row_format(const Eigen::SparseMatrix<double, Eigen::RowMajor>& M);

// Creates an Eigen::Matrix from containers
Eigen::SparseMatrix<double, Eigen::RowMajor> create_sparse(vector<int>& rows, vector<int>& cols, vector<double>& vals, int size, int density);

// Creates an Eigen::Matrix from SparseData
Eigen::SparseMatrix<double, Eigen::RowMajor> create_sparse(const vector<SparseData>& X, int size, int density);

// Computes the squared distance of two points
double rdist(const vector<double>& x, const vector<double>& y);

// Clip a gradient value
double clip(double value);

// Computes the pairwise distance for a matrix of points
vector<vector<double>> pairwise_distances(vector<vector<double>>& X);


}

#endif