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

  SparseData(vector<float> data_, vector<int> indices_): data(data_), indices(indices_) {} 


  /**
  * Adds non-zero value to representation
  *
  * @param index column index in the sparse matrix
  * @param value the non-zero value
  */
  void push(int index, float value) {
    data.push_back(value);
    indices.push_back(index);
  }

  // non-zero values
  vector<float> data;

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
std::vector<float> linspace(T start_in, T end_in, int num_in)
{

  std::vector<float> linspaced;

  float start = static_cast<float>(start_in);
  float end = static_cast<float>(end_in);
  float num = static_cast<float>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  float delta = (end - start) / (num - 1);

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

// template <typename T>
// void save_vector(std::ofstream &ofs, const std::vector<T> &v) {
//     size_t len = v.size();
//     ofs.write((char*)&len, sizeof(len));

//     ofs.write((char*)v.data(), len * sizeof(T));
// }

// template <typename T>
// void load_vector(std::ifstream &ifs, std::vector<T> &v) {
//     size_t len;
//     ifs.read((char*)&len, sizeof(len));

//     v.resize(len);
//     ifs.read((char*)v.data(), len * sizeof(T));
// }

// template <typename T>
// void save_scalar(std::ofstream &ofs, const T &val) {
//     ofs.write((char*)&val, sizeof(T));
// }

// template <typename T>
// void load_scalar(std::ifstream &ifs, T &val) {
//     ifs.read((char*)&val, sizeof(T));
// }

// void save_vector_of_vectors(std::ofstream &ofs, const std::vector<std::vector<int>> &v) {
//     // Save the length of the outer vector
//     size_t len = v.size();
//     ofs.write((char*)&len, sizeof(len));

//     // Save each inner vector
//     for (const auto &inner_v : v) {
//         save_vector(ofs, inner_v);
//     }
// }

// template<typename T>
// void load_matrix(std::ifstream &ifs, std::vector<std::vector<T>> &v) {
//     // Load the length of the outer vector
//     size_t len;
//     ifs.read((char*)&len, sizeof(len));

//     // Resize the outer vector to the correct length
//     v.resize(len);

//     // Load each inner vector
//     for (auto &inner_v : v) {
//         load_vector(ifs, inner_v);
//     }
// }

// Converts an Eigen::Matrix to tuple format
std::tuple<std::vector<int>, std::vector<int>, std::vector<float>> to_row_format(const Eigen::SparseMatrix<float, Eigen::RowMajor>& M);

// Creates an Eigen::Matrix from containers
Eigen::SparseMatrix<float, Eigen::RowMajor> create_sparse(vector<int>& rows, vector<int>& cols, vector<float>& vals, int size, int density);

// Creates an Eigen::Matrix from SparseData
Eigen::SparseMatrix<float, Eigen::RowMajor> create_sparse(const vector<SparseData>& X, int size, int density);

// Computes the squared distance of two points
float rdist(const vector<float>& x, const vector<float>& y);

// Clip a gradient value
float clip(float value);

// Computes the pairwise distance for a matrix of points
vector<vector<float>> pairwise_distances(vector<vector<float>>& X);

// output verbosity
void log(bool verbose, const string& message);

std::string encode_pos(int a, int b);
}

#endif