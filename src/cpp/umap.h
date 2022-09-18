// Author: Wilson Estécio Marcílio Júnior <wilson_jr@outlook.com>
// Code adapted from UMAP's official implementation: https://github.com/lmcinnes/umap implemented by Leland McInnes

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


#ifndef UMAP_H
#define UMAP_H

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <cmath>
#include <limits>
#include <typeinfo>
#include <omp.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


#include "utils.h"
#include "uniform_distribution.h"

#include "external/efanna/index_graph.h"
#include "external/efanna/index_random.h"
#include "external/efanna/index_kdtree.h"
#include "external/efanna/util.h"

namespace py = pybind11;

using namespace std;
using namespace efanna2e;

namespace umap {


static double SMOOTH_K_TOLERANCE = 1e-5;
static double MIN_K_DIST_SCALE = 1e-3;


/**
*  Class for generating random values during random walks
*/
class RandomGenerator
{
public:
    
	static RandomGenerator& Instance() {
        static RandomGenerator s;
        return s;
    }

    std::mt19937 & get() {
		mt.seed(0);
        return mt;
    }

private:
    RandomGenerator() {
        std::random_device rd;

		// if
        auto seed = 0;// std::chrono::high_resolution_clock::now().time_since_epoch().count();
        mt.seed(0);
    }
    ~RandomGenerator() {}

    RandomGenerator(RandomGenerator const&) = delete;
    RandomGenerator& operator= (RandomGenerator const&) = delete;

    std::mt19937 mt;
};

/**
* Class to store data points throughout the hierarchy
*
*/
class Matrix 
{

public:

	Matrix() {}

	/**
	* Constructs a Matrix using a sparse representation
	*
	* @param eigen_sparse_ Eigen::SparseMatrix representing data points 
	*/
	Matrix(Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_sparse_): eigen_sparse(eigen_sparse_), sparse(true)
	{
		shape_.push_back(eigen_sparse_.rows());
		shape_.push_back(eigen_sparse_.cols());
	}

	/**
	* Constructs a Matrix using a dense representation
	*
	* @param dense_matrix_ Container representing data points 
	*/
	Matrix(vector<vector<double>> dense_matrix_): dense_matrix(dense_matrix_), sparse(false) {
		shape_.push_back(dense_matrix_.size());
		shape_.push_back(dense_matrix_[0].size());
	}

	/**
	* Constructs a Matrix using a Container of SparseData
	* 
	* @param sparse_matrix_ Container of SparseData representing the columns with non-zero elements
	*/
	Matrix(vector<utils::SparseData> sparse_matrix_): sparse_matrix(sparse_matrix_), sparse(true) {
		shape_.push_back(sparse_matrix_.size());
		shape_.push_back(-1); // TODO: how to define?
	}

	/**
	* Constructs a Matrix using a Container of SparseData
	*
	* @param sparse_matrix_ Container of SparseData representing the columns with non-zero elements
	* @param dim int representing the matrix dimensionality
	*/
	Matrix(vector<utils::SparseData> sparse_matrix_, int dim): sparse_matrix(sparse_matrix_), sparse(true) {
		shape_.push_back(sparse_matrix_.size());
		shape_.push_back(dim); 
	}

	// get a matrix row
	vector<double> get_row(int i);

	// get the matrix size
	int size() { return shape_[0]; }

	
	/**
	* Returns the matrix shape
	*
	* @param index int representing the dimension
	* @return int representing the shape of the specified dimension
	*/
	int shape(int index) { return shape_[index]; }

	// returns the C-like float array of the matrix (only for dense representation)
	float* data_f();

	// returns the C-like double array of the matrix (only for dense representation)
	double* data();

	// check if the matrix is sparse
	bool is_sparse() const { return sparse; }

	vector<int> shape_;
	
	vector<utils::SparseData> sparse_matrix;

	Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_sparse;
	
	vector<vector<double>> dense_matrix;

private:

	bool sparse = false;
};


/**
* UMAP class for embedding high-dimensional data in low-dimensional spaces.
*
* This is a C++ implementation heavily based on the official implementation of Leland's algorithm available on github.
*/
class UMAP
{
public:
	
	/**
	* Constructs a UMAP object
	*
	* @param metric_ string representing the metric for distance computation. Currently, we only support euclidean.
	* @param n_neighbors_ int representing the number of neighbors for k nearest neighbor computation.
	* @param min_dist_ double The effective minimum distance between embedded points.
	* @param knn_algorithm string representin the knn algorithm
	* @param init_ string with the type of low-embedding initialization
	*/
	UMAP(string metric_, int n_neighbors_, double min_dist_=0.15, string knn_algorithm="NNDescent", string init_="Spectral", bool reproducible_=false): 
		metric(metric_), 
		verbose(false),
		n_neighbors(n_neighbors_),
		min_dist(min_dist_),
		init(init_),
		local_connectivity(1.0),
		try_reproducible(reproducible_)
	{
		knn_args["knn_algorithm"] = knn_algorithm;

		knn_args["nlist"] = "100";
		knn_args["nTrees"] = "50";
		knn_args["mLevel"] = "8";

		// hard-coded
		// TODO: find an intelligent way to define these parameters
		knn_args["L"] = std::to_string(n_neighbors_); //"100";
		knn_args["iter"] = n_neighbors_ >= 50 ? "3" : "10";
		knn_args["S"] = std::to_string((int)(n_neighbors_ >= 50 ? 0.3*n_neighbors_ : n_neighbors_)); 
		knn_args["R"] = std::to_string((int)(n_neighbors_ >= 50 ? 0.3*n_neighbors_ : n_neighbors_)); 
	}

	/**
	* Get the graph created after kernel computation 
	*
	* @return Eigen::SparseMatrix with the kernel graph
	*/
	Eigen::SparseMatrix<double, Eigen::RowMajor>& get_graph() { return this->graph_; }

	/**
	* Get the knn distances 
	* 
	* @return vector<vector<double>> with the knn distances for each data point
	*/
	vector<vector<double>>& knn_dists() { return _knn_dists; }

	/**
	* Get the knn indices 
	*
	* @return vector<vector<int>> with the knn indices for each data point
	*/
	vector<vector<int>>& knn_indices() { return _knn_indices; }

	/**
	* Get the sigmas found for each data point during kernel computation 
	*
	* @return vector<double> with the sigmas of each data point
	*/
	vector<double> sigmas() { return this->_sigmas; }

	/**
	*  Get the distance of each data point to its closest neighbor 
	*
	* @return vector<double> with the distance of each data point to its closest neighbor
	*/
	vector<double> rhos() { return this->_rhos; }
	
	// fit dataset using an array of SparseData
	void fit(const vector<utils::SparseData>& X);
	
	// fit dataset using a SparseMatrix
	void fit(const Eigen::SparseMatrix<double, Eigen::RowMajor>& X);

	// fit dataset using a dense matrix
	void fit(vector<vector<double>> X);

	// receives the fit operation
	void fit(const Matrix& X);

	void set_ab_parameters(double a, double b) {
		this->a = a;
		this->b = b;
	} 

	// set which datapoints are free during optimization
	void set_free_datapoints(vector<bool> free_datapoints) {
		this->_free_datapoints = free_datapoints;
	}

	// set how free is the fixing data points
	void set_fixing_term(double fixing_term) {
		this->_fixing_term = fixing_term;
	}

	bool is_reproducible() {
		return try_reproducible;
	}

	// fit the dataset in fact
	void prepare_for_fitting(Matrix& X);


	// produces initial low-dimensional representation using Spectral Embedding
	vector<vector<double>> spectral_layout(Matrix& data, const Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, int dim);


	vector<double> make_epochs_per_sample(const vector<double>& weights, int n_epochs);

	// optimize the low-dimensional representation to slowly converge to UMAP projection
	vector<vector<double>> optimize_layout_euclidean(vector<vector<double>>& head_embedding, vector<vector<double>>& tail_embedding,
								   const vector<int>& head, const vector<int>& tail, int n_epochs, int n_vertices, 
								   const vector<double>& epochs_per_sample);

	bool 										 verbose;
	string                                       metric;
	vector<int>                                  rows;
	vector<int>                                  cols;
	vector<double>                               vals;
	vector<double> 								 sum_vals;
	vector<double> 							     vals_transition;
	map<string, string> 						 knn_args;
	Eigen::SparseMatrix<double, Eigen::RowMajor> transition_matrix;
	
private:
	
	
	bool low_memory;
	bool _sparse_data;
	bool force_approximation_algorithm = false;
	bool try_reproducible = false;

	int n_epochs;
	int n_components;
	int random_state = 0;
	int n_neighbors, _n_neighbors;	
	
	double _a, _b;
	double local_connectivity;
	double a = -1.0, b = -1.0;
	double _fixing_term = 0.01;
	double _initial_alpha = 1.0;
	double repulsion_strength = 1.0;
	double negative_sample_rate = 5.0;	
	double spread = 1.0, min_dist = 0.001;

	string init = "Spectral";

	Matrix dataset;
	Matrix pairwise_distance;

	vector<bool>		   _free_datapoints;
	vector<double> 		   _sigmas; 
	vector<double>         _rhos; 
	vector<vector<int>>    _knn_indices;
	vector<vector<double>> _knn_dists;
	vector<vector<double>> embedding_;

	Eigen::SparseMatrix<double, Eigen::RowMajor> graph_; 

	

	// method for Spectral Embedding
	vector<vector<double>> component_layout(umap::Matrix& data, int n_components, vector<int>& component_labels, int dim);

	// method for Spectral Embedding
	vector<vector<double>> multi_component_layout(umap::Matrix& data,  const Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, int n_components, 
												  vector<int>& component_labels, int dim);

	// optimize the layout for one epoch
	void optimize_euclidean_epoch(vector<vector<double>>& head_embedding, vector<vector<double>>& tail_embedding,
								   const vector<int>& head, const vector<int>& tail, int n_vertices, 
								   const vector<double>& epochs_per_sample, double a, double b, 
								   double gamma, int dim, bool move_other, double alpha, vector<double>& epochs_per_negative_sample,
								   vector<double>& epoch_of_next_negative_sample, vector<double>& epoch_of_next_sample, 
								   int n);

	void optimize_euclidean_epoch_reproducible(vector<vector<double>>& head_embedding, vector<vector<double>>& tail_embedding,
										   const vector<int>& head, const vector<int>& tail, int n_vertices, 
										   const vector<double>& epochs_per_sample, double a, double b, 
										   double gamma, int dim, bool move_other, double alpha, vector<double>& epochs_per_negative_sample,
										   vector<double>& epoch_of_next_negative_sample, 
										   vector<double>& epoch_of_next_sample, 
										   int n);

};





tuple<double, double> find_ab_params(double spread, double min_dist);

// construct the graph representing the similarity between each data sample
tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, vector<double>, vector<double>> fuzzy_simplicial_set(
	umap::Matrix& X, int n_neighbors, double random_state, string metric, 
	vector<vector<int>>& knn_indices, vector<vector<double>>& knn_dists,
	double local_connectivity=1.0, bool apply_set_operations=true, bool verbose=false, umap::UMAP* obj=0);

// use the knn distances to find the kernel parameters
tuple<vector<double>, vector<double>> smooth_knn_dist(vector<vector<double>>& distances,
	double k, int n_iter=64, double local_connectivity=1.0, double bandwidth=1.0);

// find the nearest neighbors
tuple<vector<vector<int>>, vector<vector<double>>> nearest_neighbors(umap::Matrix& X,
	int n_neighbors, string metric, map<string, string> knn_args, bool verbose=false,bool reproducible=false);

// compute the affinities after find knn, sigma and rho values
tuple<vector<int>, vector<int>, vector<double>, vector<double>> compute_membership_strenghts(
	vector<vector<int>>& knn_indices, vector<vector<double>>& knn_dists, 
	vector<double>& sigmas, vector<double>& rhos);

// computes the pairwise distance between data points
std::vector<std::vector<double>> pairwise_distances(Matrix& X, string metric="euclidean");







}

#endif