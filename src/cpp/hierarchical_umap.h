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


#ifndef HIERARCHICAL_UMAP_H
#define HIERARCHICAL_UMAP_H

#include <omp.h>
#include <map>
#include <stack>
#include <queue>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <time.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "umap.h"

#include "uniform_distribution_double.h"

using namespace std;
using namespace umap;

namespace py = pybind11;

namespace humap {

// converts py array to dense representation
vector<vector<double>> convert_to_vector(const py::array_t<double>& v);

// creates a sparse object from rows, columns, and values
vector<utils::SparseData> create_sparse(int n, const vector<int>& rows, const vector<int>& cols, const vector<double>& vals);

// returns how many times each data point was an endpoint after a markov chain
vector<int>  markov_chain(vector<vector<int>>& knn_indices, vector<double>& vals, vector<int>& cols, int num_walks, int walk_length, bool reproducible); 

// returns the endpoint after a random walk
int random_walk(int vertex, int n_neighbors, vector<double>& vals, vector<int>& cols, int walk_length, 
	            uniform_real_distribution<double>& unif, std::mt19937& rng);


// returns the max neighborhood after markov chain
int markov_chain(vector<vector<int>>& knn_indices, vector<double>& vals, vector<int>& cols, 
	             int num_walks, int walk_length, vector<int>& landmarks, int influence_neighborhood, 
				 vector<vector<int>>& neighborhood, vector<vector<int>>& association, bool reproducible);

// returns the endpoint after a random walk
int random_walk(int vertex, int n_neighbors, vector<double>& vals, vector<int>& cols, 
				int walk_length, uniform_real_distribution<double>& unif, mt19937& rng, vector<int>& is_landmark);	


/**
* Metadata to store information of each hierarchy level during fitting
*
*/
struct Metadata {

	/**
	* Construct Metadata class
	*
	* @param indices_ Container representing the indices selected from the level below
	* @param owners_ Container representing which data point owns it
	* @param strength_ Container representing the association strength
	* @param association_ Container representing which data points in the level below are associated with it (TODO: use the inverse)
	* @param size_ int representing the number of data points in the level
	*/	
	Metadata(vector<int> indices_, vector<int> owners_, vector<double> strength_, vector<vector<int>> association_, int size_)
	: indices(indices_), owners(owners_), strength(strength_), association(association_), size(size_)
	{	 	
	}

	int size;

	vector<int> indices;
	vector<int> owners;
	vector<int> count_influence;

	vector<double> strength;

	vector<vector<int>> association;	 
};

/**
* Storage for sparse matrix during similarity computation
*
*/
struct SparseComponents
{
	SparseComponents(vector<int> rows_, vector<int> cols_, vector<double> vals_): rows(rows_), cols(cols_), vals(vals_) 
	{
	}

	vector<int> cols;
	vector<int> rows;
	vector<double> vals;
};



/**
* Hierarchical UMAP
*
*/
class HierarchicalUMAP
{
public:
	/**
	* Constructs HierarchicalUMAP
	*
	* @param similarity_method_ string representing the similarity method (we only support 'euclidean' right now)
	* @param percents_ py::array_t<double> representing the percentage of points in each hierarchy level after the first level (whole dataset)
	* @param n_neighbors_ int representing the number of neighbors for knn computation
	* @param min_dist_ double representing the minimum distance between manifold structures
	* @param knn_algorithm_ string representing which knn algorithm to use
	* @param init_ string representing the initialization of low-dimensional representation
	* @param verbose_ bool controling the verbosity of HUMAP	
	*/
	HierarchicalUMAP(string similarity_method_, py::array_t<double> percents_, int n_neighbors_=15, double min_dist_=0.15, 
					 string knn_algorithm_="NNDescent", string init_="Spectral", bool verbose_=false, bool reproducible_=false) 
	: similarity_method(similarity_method_), n_neighbors(n_neighbors_), min_dist(min_dist_), 
		knn_algorithm(knn_algorithm_), percent_glue(0.0), init(init_), verbose(verbose_), reproducible(reproducible_) {

		percents = vector<double>((double*)percents_.request().ptr, (double*)percents_.request().ptr + percents_.request().shape[0]);
	}

	HierarchicalUMAP() {}


	// fits the hierarchy on X
	void fit(py::array_t<double> X, py::array_t<int> y);

	// returns the hierarchy level labels 
	py::array_t<int> get_labels(int level);

	// returns the subset X associated to the hierarchy level
	Eigen::SparseMatrix<double, Eigen::RowMajor> get_data(int level);

	// returns the embedding of the hierarchy level
	py::array_t<double> get_embedding(int level);

	// generates and returns the embedding of the hierarchy level
	py::array_t<double> transform(int level);

	// returns the indices of the embedding corresponding to the hierarchy level below
	py::array_t<int> get_indices(int level);

	// returns the influence of each data point in a hierarchy level on the level below it
	py::array_t<int> get_influence(int level);

	// returns the original indices of a hierarchy level
	py::array_t<int> get_original_indices(int level);

	// generates the embedding for the hierarchy level below based on a set of classes
	py::array_t<double> project(int level, py::array_t<int> c);	

	// generates the embeddding for the hierarchy level below based on a set of indices
	py::array_t<double> project_indices(int level, py::array_t<int> indices);

	// get the labels of the embedded subset
	py::array_t<int> get_labels_selected() { return py::cast(this->labels_selected); }

	// get the influence of the embedded subset
	py::array_t<int> get_influence_selected() { return py::cast(this->influence_selected); }

	// get the indices of the embedded subset
	py::array_t<int> get_indices_selected() { return py::cast(this->indices_selected); }

	// sets the number of random walks for landmark selection
	void set_landmarks_nwalks(int value) { this->landmarks_nwalks = value; }

	// sets the walk length for landmark selection
	void set_landmarks_wl(int value) { this->landmarks_wl = value; }

	// sets the number of random walks for similarity computation
	void set_influence_nwalks(int value) { this->influence_nwalks = value; }

	// sets the walk length for similarity computation
	void set_influence_wl(int value) { this->influence_wl = value; }

	// sets the number of neighbors used in influence neighborhood
	void set_influence_neighborhood(int value) { this->influence_neighborhood = value; }


	void set_distance_similarity(bool value) { this->distance_similarity = value; }

	// sets the ab parameters computed using Python
	void set_ab_parameters(double a, double b) { this->a = a; this->b = b; }

	// defines how the embedding will be performed
	void set_focus_context(bool value) { this->focus_context = value; }

	// set how free is the fixed data points across level
	void set_fixing_term(double fixing_term) { this->_fixing_term = fixing_term; }

	// fix datapoints
	void set_fixed_datapoints(py::array_t<double> fixed) { this->fixed_datapoints = convert_to_vector(fixed); }

	// file
	void set_info_file(string filename) { 
		this->output_filename = filename; 
		this->output_file.open(filename); 
	}

	void set_random_state(int random_state) { this->random_state = random_state; }
	void set_n_epochs(int n_epochs) { this->n_epochs = n_epochs; }

	// set statistics
	void dump_info(string info);
		
private:

	int n_neighbors;
	int n_epochs = 500;
	int n_components = 2;
	int random_state = 0;

	int landmarks_nwalks = 10;
	int landmarks_wl = 10;
	
	int influence_nwalks = 20;
	int influence_wl = 30;
	int influence_neighborhood = 0;

	bool verbose;
	bool focus_context = false;
	bool distance_similarity = false;
	bool reproducible;
	
	double min_dist = 0.15;
	double a = -1.0, b = -1.0;
	double percent_glue = 0.0;
	double _fixing_term = 0.01;

	string output_filename = "";
	ofstream output_file;
	string init = "Spectral";
	string similarity_method;
	string knn_algorithm;

	vector<int>                    labels_selected;
	vector<int>                    influence_selected;
	vector<int>                    indices_selected;
	vector<int>                    indices_fixed;
	vector<bool>                   free_datapoints;
	vector<double> 				   percents;
	vector<vector<int>>            hierarchy_y;
	vector<vector<int>>            original_indices;
	vector<vector<int>>            _indices;
	vector<vector<double>> 		   _sigmas;
	vector<vector<double>> 		   fixed_datapoints;
	vector<vector<int>>            level_landmarks;
	vector<vector<vector<double>>> embeddings;

	vector<Metadata> metadata;

	vector<umap::UMAP> reducers;
	
	vector<umap::Matrix> hierarchy_X;
	vector<umap::Matrix> dense_backup;

	// finds which data point influence the one passed as parameter
	int influenced_by(int level, int index);
	
	// append information of a landmark to the similarity data structure
	void add_similarity(int index, int i, vector<vector<int>>& neighborhood, std::vector<std::vector<int> >& indices, 
						int* mapper, double* elements, vector<vector<int>>& indices_nzeros, int n, double max_incidence, vector<vector<int>>& association);

	// create a sparse represention after similarity computaiton 
	SparseComponents create_sparse(int n, int n_neighbors, double* elements, vector<vector<int>>& indices_nzeros);

	// compute the similarity among landmarks
	SparseComponents sparse_similarity(int level, int n, int n_neighbors, vector<int>& greatest, vector<vector<int>>& neighborhood,
									   double max_incidence, vector<vector<int>>& association);

	// update the position of a landmark based on its surroundings	
	vector<double> update_position(int i, vector<int>& neighbors, umap::Matrix& X);

	// performs the embedding on the dataset X using the graph force 
	vector<vector<double>> embed_data(int level, Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, umap::Matrix& X);

	// associates points to landmarks
	void associate_to_landmarks(int n, int n_neighbors, int* indices, vector<int>& cols, 
								vector<double>& strength, vector<int>& owners, vector<int>& indices_landmark, 
								vector<vector<int>>& association, vector<int>& count_influence, vector<int>& is_landmark, 
								vector<vector<double>>& knn_dists);

	// associates points to landmarks
	void associate_to_landmarks(int n, int n_neighbors, vector<int>& landmarks, vector<int>& cols, 
								vector<double>& strength, vector<int>& owners, vector<int>& indices, 
								vector<vector<int>>& association, vector<int>& count_influence, 
								vector<int>& is_landmark, vector<vector<double>>& knn_dists );

	// searches for a point owner using the knn structure computed in UMAP
	int depth_first_search(int n_neighbors, int* neighbors, vector<int>& cols, 
	  					   vector<double>& strength, vector<int>& owners, vector<int>& is_landmark);

	// returns the influence of each index
	vector<int> get_influence_by_indices(int level, vector<int> indices);

	// helper function to project indices
	py::array_t<double> project_data(int level, vector<int> selected_indices);

};

}

#endif