// Author: Wilson Estécio Marcílio Júnior <wilson_jr@outlook.com>

/*
 *
 * Copyright (c) 2021, Wilson Estécio Marcílio Júnior (São Paulo State University)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the São Paulo State University.
 * 4. Neither the name of the São Paulo State University nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY WILSON ESTÉCIO MARCÍLIO JÚNIOR ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL WILSON ESTÉCIO MARCÍLIO JÚNIOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef UTILS_H
#define UTILS_H

#include <tuple>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <iostream>
#include <algorithm>

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

// output verbosity
void log(bool verbose, const string& message);


}

#endif