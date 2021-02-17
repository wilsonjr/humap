#ifndef UMAP_H
#define UMAP_H

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

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


#include "utils.h"

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







// ???????
class Matrix 
{

public:

	Matrix() {}

	// Matrix(const Matrix& a) {
	// 	Matrix();
	// 	this->shape_ = a.shape_;
	// 	this->sparse_matrix = a.sparse_matrix;
	// 	this->dense_matrix = a.dense_matrix;
	// 	this->py_matrix = a.py_matrix;
	// }
	Matrix(Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_sparse_): eigen_sparse(eigen_sparse_), sparse(true)
	{
		shape_.push_back(eigen_sparse_.rows());
		shape_.push_back(eigen_sparse_.cols());


	}


	Matrix(vector<vector<double>> dense_matrix_): dense_matrix(dense_matrix_), sparse(false) {
		shape_.push_back(dense_matrix_.size());
		shape_.push_back(dense_matrix_[0].size());
	}


	Matrix(vector<utils::SparseData> sparse_matrix_): sparse_matrix(sparse_matrix_), sparse(true) {
		shape_.push_back(sparse_matrix_.size());
		shape_.push_back(-1); // TODO: como definir?
	}

	Matrix(vector<utils::SparseData> sparse_matrix_, int dim): sparse_matrix(sparse_matrix_), sparse(true) {
		shape_.push_back(sparse_matrix_.size());
		shape_.push_back(dim); 
	}


	// Matrix(py::array_t<double> dense_matrix_): py_matrix(dense_matrix_), sparse(false) {
	// 	Matrix();
	// 	matrix_buffer = dense_matrix_.request();
	// 	matrix_ptr = (double*) matrix_buffer.ptr;
	// 	shape_.push_back(matrix_buffer.shape[0]);
	// 	shape_.push_back(matrix_buffer.shape[1]);

	// }

	vector<double> get_row(int i) {
		


		if( sparse ) {
			// cout << "this is the shape: " << shape_[1] << endl;
			// cout << "this is the sparse_matrix.size(): " << sparse_matrix.size() << endl;
			// cout << "this is the number of indices: " << sparse_matrix[i].data.size() << endl;
			// cout << "this is the number of indices: " << sparse_matrix[i].indices.size() << endl;
			// cout << "this is the row I am asking for: " << i << endl;



			vector<double> row(shape_[1], 0.0);

			// cout << "creating matrix" << endl;
			for( int count = 0; count < sparse_matrix[i].indices.size(); ++count ) {
				int index = sparse_matrix[i].indices[count];			
				row[index] = sparse_matrix[i].data[count];
				// cout << index << " -> " << row.size() << endl;

			}
			// cout << "created matrix" << endl;
			return row;
		} else {
			return dense_matrix[i];
		}
	}


	int size() {
		return shape_[0];
	}

	int shape(int index) {
		return shape_[index];
	}

	float* data_f() {
		if( sparse )
			return nullptr;

		float* d = new float[dense_matrix.size()*dense_matrix[0].size()];

		for( int i = 0; i < dense_matrix.size(); ++i )
			for( int j = 0; j < dense_matrix[0].size(); ++j )
				*(d + i*dense_matrix[0].size() + j) = (float) dense_matrix[i][j];

		return d;
	}

	double* data() {
		if( sparse )
			return nullptr;

		double* d = new double[dense_matrix.size()*dense_matrix[0].size()];

		for( int i = 0; i < dense_matrix.size(); ++i )
			for( int j = 0; j < dense_matrix[0].size(); ++j )
				*(d + i*dense_matrix[0].size() + j) = dense_matrix[i][j];

		return d;
	}


	 bool is_sparse() const { return sparse; }

	// double operator[](int index) {
	// 	if( sparse )
	// 		throw runtime_error("[] operator not supported for sparse matrix.");

	// 	return matrix_ptr[index];
	// }

	vector<int> shape_;
	
	vector<utils::SparseData> sparse_matrix;

	Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_sparse;
	
	vector<vector<double>> dense_matrix;


	

private:

	bool sparse = false;
};

class UMAP
{
public:

	UMAP(): 
		metric("euclidean"), 
		verbose(true),
		n_neighbors(15),
		local_connectivity(1.0)
	{	
		cout << "Constructor 1" << endl;

		knn_args["knn_algorithm"] = "FAISS_IVFFlat";

		// TODO: make it dinamic!
		knn_args["nprobes"] = "3";
		knn_args["nlist"] = "100";

		knn_args["nTrees"] = "50";
		knn_args["mLevel"] = "8";

		knn_args["L"] = "100";
		knn_args["iter"] = "10";
		knn_args["S"] = "50";
		knn_args["R"] = "50";
	}

	// TODO take care of these parameters
	// add list of parameters
	UMAP(string metric_, string knn_algorithm="FAISS_IVFFlat"): 
		metric(metric_), 
		verbose(true),
		n_neighbors(15),
		local_connectivity(1.0)
	{
		cout << "Constructor 2" << endl;

		knn_args["knn_algorithm"] = knn_algorithm;

		// TODO: make it dinamic!
		knn_args["nprobes"] = "3";
		knn_args["nlist"] = "100";

		knn_args["nTrees"] = "50";
		knn_args["mLevel"] = "8";

		knn_args["L"] = "100";
		knn_args["iter"] = "10";
		knn_args["S"] = "50";
		knn_args["R"] = "50";
	}

	UMAP(string metric_, int n_neighbors_, double min_dist_=0.15, string knn_algorithm="FAISS_IVFFlat"): 
		metric(metric_), 
		verbose(true),
		n_neighbors(n_neighbors_),
		min_dist(min_dist_),
		local_connectivity(1.0)
	{
		cout << "Constructor 3: " << knn_algorithm << endl;
		cout << "min_dist: " << min_dist << endl;

		knn_args["knn_algorithm"] = knn_algorithm;

		// TODO: make it dinamic!
		knn_args["nprobes"] = "3";
		knn_args["nlist"] = "100";

		knn_args["nTrees"] = "50";
		knn_args["mLevel"] = "8";

		knn_args["L"] = "100";
		knn_args["iter"] = "3";
		knn_args["S"] = "30";
		knn_args["R"] = "30";
	}


	void fit(Matrix X);

	string getName() { return name; }

	Eigen::SparseMatrix<double, Eigen::RowMajor>& get_graph() {
		return this->graph_;
	}

	vector<vector<double>>& knn_dists() {
		return _knn_dists;
	}

	vector<vector<int>>& knn_indices() {
		return _knn_indices;
	}

	vector<double> sigmas() {
		return this->_sigmas;
	}

	vector<double> rhos() {
		return this->_rhos;
	}
	

	void fit_hierarchy_sparse(const vector<utils::SparseData>& X);
	void fit_hierarchy_sparse(const Eigen::SparseMatrix<double, Eigen::RowMajor>& X);
	// void fit_hierarchy_dense(py::array_t<double> X);
	void fit_hierarchy_dense(vector<vector<double>> X);
	void fit_hierarchy(const Matrix& X);

	void set_ab_parameters(double a, double b) {
		this->a = a;
		this->b = b;
	} 

	void prepare_for_fitting(Matrix& X);
	void fit(py::array_t<double> X);
	vector<vector<double>> fit_transform(py::array_t<double> X);

	vector<int>    rows;
	vector<int>    cols;
	vector<double>  vals;
	vector<double> sum_vals;
	vector<double> vals_transition;
	string metric;

	map<string, string> knn_args;

	Eigen::SparseMatrix<double, Eigen::RowMajor> transition_matrix;

	vector<vector<double>> spectral_layout(Matrix& data, const Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, int dim);
	vector<double> make_epochs_per_sample(const vector<double>& weights, int n_epochs);

	vector<vector<double>> optimize_layout_euclidean(vector<vector<double>>& head_embedding, vector<vector<double>>& tail_embedding,
								   const vector<int>& head, const vector<int>& tail, int n_epochs, int n_vertices, 
								   const vector<double>& epochs_per_sample, vector<long>& rng_state);





private:

	double _a, _b;
	
	
	int n_neighbors, _n_neighbors;
	vector<vector<int>> _knn_indices;
	vector<vector<double>> _knn_dists;
	bool angular_rp_forest;
	double set_op_mix_ratio;
	double local_connectivity;
	bool verbose;
	Eigen::SparseMatrix<double, Eigen::RowMajor> graph_; 

	vector<double> _sigmas; 
	vector<double> _rhos; 

	bool _sparse_data;

	double a = -1.0, b = -1.0;
	double spread = 1.0, min_dist = 0.001;

	Matrix dataset;

	Matrix pairwise_distance;

	vector<vector<double>> embedding_;
	int n_components;
	double _initial_alpha = 1.0;
	double repulsion_strength = 1.0;
	double negative_sample_rate = 5.0;
	int n_epochs;

	bool force_approximation_algorithm = false;
	bool low_memory;
	int random_state = 0;


	string init = "spectral";

	string name = "C++ implementation of UMAP";


	vector<vector<double>> component_layout(umap::Matrix& data, int n_components, 
														 vector<int>& component_labels, int dim);
	vector<vector<double>> multi_component_layout(umap::Matrix& data, 
	const Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, int n_components, 
	vector<int>& component_labels, int dim);


	void optimize_euclidean_epoch(vector<vector<double>>& head_embedding, vector<vector<double>>& tail_embedding,
										   const vector<int>& head, const vector<int>& tail, int n_vertices, 
										   const vector<double>& epochs_per_sample, double a, double b, vector<long>& rng_state, 
										   double gamma, int dim, bool move_other, double alpha, vector<double>& epochs_per_negative_sample,
										   vector<double>& epoch_of_next_negative_sample, vector<double>& epoch_of_next_sample, int n);

};





tuple<double, double> find_ab_params(double spread, double min_dist);

tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, vector<double>, vector<double>> fuzzy_simplicial_set(
	umap::Matrix& X, int n_neighbors, double random_state, string metric, 
	vector<vector<int>>& knn_indices, vector<vector<double>>& knn_dists,
	bool angular=false, double set_op_mix_ratio=1.0, double local_connectivity=1.0,
	bool apply_set_operations=true, bool verbose=false, umap::UMAP* obj=0);

tuple<vector<double>, vector<double>> smooth_knn_dist(vector<vector<double>>& distances,
	double k, int n_iter=64, double local_connectivity=1.0, double bandwidth=1.0);

tuple<vector<vector<int>>, vector<vector<double>>> nearest_neighbors(umap::Matrix& X,
	int n_neighbors, string metric, bool angular, double random_state, map<string, string> knn_args, bool verbose=false);

tuple<vector<int>, vector<int>, vector<double>, vector<double>> compute_membership_strenghts(
	vector<vector<int>>& knn_indices, vector<vector<double>>& knn_dists, 
	vector<double>& sigmas, vector<double>& rhos);

std::vector<std::vector<double>> pairwise_distances(Matrix& X, string metric="euclidean");




}

#endif