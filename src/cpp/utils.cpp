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


#include "utils.h"

/**
* Converts a Eigen::SparseMatrix to row-based tuple format
*
* @param M Eigen::SparseMatrix with values
* @return std::tuple containing three lists representing the row indices, column indices, and the respective non-zero values
*
*/
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

/**
* Creates a Eigen::SparseMatrix from indices and values
*
* @param rows Container representing with the row indices of non-zero values
* @param cols Container representing with the col indices of non-zero values
* @param vals Container representing with the non-zero values
* @param size int representing the matrix number of rows
* @param density int representing the max number of non-zero values in a row
* @return Eigen::SparseMatrix containing the sparse matrix representing organized by rows
*
*/
Eigen::SparseMatrix<double, Eigen::RowMajor> utils::create_sparse(vector<int>& rows, vector<int>& cols, vector<double>& vals, int size, int density)
{
    

  Eigen::SparseMatrix<double, Eigen::RowMajor> result(size, size);
  result.reserve(Eigen::VectorXi::Constant(size, density)); 

  for( int i = 0; i < vals.size(); ++i )
    result.insert(rows[i], cols[i]) = vals[i];
  result.makeCompressed();

  return result;
}

/**
* Creates a Eigen::SparseMatrix from SparseData
*
* @param X SparseData with indices and non-zero values
* @param size int representing the matrix number of rows
* @param density int representing the max number of non-zero values in a row
* @return Eigen::SparseMatrix
*
*/
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

/**
* Computes the distance between two points without squared root
*
* @param x Container representing the first point
* @param y Container representing the second point
* @return the squared distance between two points
*
*/
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

/**
* Clips a value between -4 and 4 (used in the embedding minimization)
*
* @param value A gradient value
* @return the clipped value between -4 and 4
*
*/
double utils::clip(double value)
{
    if( value > 4.0 )
      return 4.0;
    else if( value < -4.0 )
      return -4.0;
    else 
      return value;
}

/**
* Computes the pairwise distance for a matrix
*
* @param X Container with shape (n_samples, n_features)
* @return Container with shape (n_samples, n_samples) representing the pairwise distance
*
*/
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
        distance += (X[i][k]-X[j][k])*(X[i][k]-X[j][k]);
      }

      pd[i][j] = sqrt(distance);
      pd[j][i] = pd[i][j];
    }
  }

  return pd;
}