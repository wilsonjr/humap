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
#include <time.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "umap.h"


using namespace std;

namespace py = pybind11;

namespace humap {



struct Metadata {

	 // Metadata(vector<int> indices_, vector<int> owners_, vector<double> strength_)
	 // : indices(indices_), owners(owners_), strength(strength_)
	 // {	 	
	 // }

	  Metadata(vector<int> indices_, vector<int> owners_, vector<double> strength_, vector<vector<int>> association_, int size_)
	 : indices(indices_), owners(owners_), strength(strength_), association(association_), size(size_)
	 {	 	
	 }

 	 vector<int> indices;
	 vector<int> owners;
	 vector<double> strength;
	 int size;
	 vector<vector<int>> association;
	 vector<int> count_influence;


	 // vector<int> indices;
	 // vector<int> owners;
	 // vector<double> strength;
};

class StringRef
{
private:
    char const*     begin_;
    int             size_;

public:
    int size() const { return size_; }
    char const* begin() const { return begin_; }
    char const* end() const { return begin_ + size_; }

    StringRef( char const* const begin, int const size )
        : begin_( begin )
        , size_( size )
    {}
};

struct SparseComponents
{
	SparseComponents(vector<int> rows_, vector<int> cols_, vector<double> vals_): rows(rows_), cols(cols_), vals(vals_) {

	}

	vector<int> cols;
	vector<int> rows;
	vector<double> vals;

};

class HierarchicalUMAP
{

public:
	HierarchicalUMAP(string similarity_method_, py::array_t<double> percents_, int n_neighbors_=15, double min_dist_=0.15,string knn_algorithm_="FAISS_IVFFlat", double percent_glue_ = 0.0,bool verbose_=false) 
	: similarity_method(similarity_method_), n_neighbors(n_neighbors_), min_dist(min_dist_), knn_algorithm(knn_algorithm_), percent_glue(percent_glue_), verbose(verbose_) {

		percents = vector<double>((double*)percents_.request().ptr, (double*)percents_.request().ptr + percents_.request().shape[0]);

	}

	HierarchicalUMAP() {
		cout << "Consegui instanciar" << endl;
	}

	void fit(py::array_t<double> X, py::array_t<int> y);
	py::array_t<int> get_labels(int level);
	Eigen::SparseMatrix<double, Eigen::RowMajor> get_data(int level);
	py::array_t<double> get_embedding(int level);
	py::array_t<double> transform(int level);
	py::array_t<int> get_indices(int level);
	py::array_t<double> get_sigmas(int level);
	py::array_t<int> get_influence(int level);
	py::array_t<int> get_original_indices(int level);

	py::array_t<double> project(int level, py::array_t<int> c);	
	py::array_t<double> project_indices(int level, py::array_t<int> indices);
	py::array_t<double> project_data(int level, vector<int> selected_indices);

	py::array_t<int> get_labels_selected() { return py::cast(this->labels_selected); }
	py::array_t<int> get_influence_selected() { return py::cast(this->influence_selected); }
	py::array_t<int> get_indices_selected() { return py::cast(this->indices_selected); }

	void set_landmarks_nwalks(int value) { this->landmarks_nwalks = value; }
	void set_landmarks_wl(int value) { this->landmarks_wl = value; }
	void set_influence_nwalks(int value) { this->influence_nwalks = value; }
	void set_influence_wl(int value) { this->influence_wl = value; }
	void set_influence_neighborhood(int value) { this->influence_neighborhood = value; }
	void set_distance_similarity(bool value) { this->distance_similarity = value; }
	void set_path_increment(bool value) { this->path_increment = value; }


	tuple<py::array_t<double>, py::array_t<double>> explain(int n_walks, int walk_length, int max_hops, py::array_t<int> indices);
	
private:

	vector<vector<vector<double>>> embeddings;
	vector<umap::UMAP> reducers;
	vector<umap::Matrix> hierarchy_X;
	vector<umap::Matrix> dense_backup;
	vector<vector<int>> hierarchy_y;
	vector<Metadata> metadata;

	vector<double> percents;
	string similarity_method;
	string knn_algorithm;


	int landmarks_nwalks = 10;
	int landmarks_wl = 10;
	int influence_nwalks = 20;
	int influence_wl = 30;
	int influence_neighborhood = 30;

	bool distance_similarity = true;
	bool path_increment = true;

	int n_neighbors;
	bool verbose;
	int n_epochs = 500;
	int n_components = 2;
	double min_dist = 0.15;

	double percent_glue =0.0;

	int random_state = 0;

	vector<vector<int>> original_indices;
	vector<vector<int>> _indices;
	vector<vector<double>> _sigmas;
	vector<int> labels_selected;
	vector<int> influence_selected;
	vector<int> indices_selected;

	int influenced_by(int level, int index);
	
	void add_similarity(int index, int i, int n_neighbors, vector<int>& cols,  
					Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, vector<vector<double>>& knn_dists,
					std::vector<std::vector<double> >& membership_strength, std::vector<std::vector<int> >& indices,
					std::vector<std::vector<double> >& distance, int* mapper, 
					double* elements, vector<vector<int>>& indices_nzeros, int n);
	
	SparseComponents create_sparse(int n, int n_neighbors, double* elements, vector<vector<int>>& indices_nzeros);

	void add_similarity2(int index, int i, int n_neighbors, vector<int>& cols, vector<double>& vals,
					Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, vector<vector<double>>& knn_dists,
					std::vector<std::vector<double> >& membership_strength, std::vector<std::vector<int> >& indices,
					std::vector<std::vector<double> >& distance, int* mapper, 
					double* elements, vector<vector<int>>& indices_nzeros, int n);
	
	SparseComponents create_sparse2(int level, int n, int n_neighbors, double* elements, vector<vector<int>>& indices_nzeros);


	void add_similarity3(int index, int i, vector<vector<int>>& neighborhood, std::vector<std::vector<int> >& indices, 
						 int* mapper, double* elements, vector<vector<int>>& indices_nzeros, int n, double max_incidence, vector<vector<int>>& association);


	SparseComponents create_sparse3(int n, int n_neighbors, double* elements, vector<vector<int>>& indices_nzeros);

	SparseComponents sparse_similarity(int level, int n, int n_neighbors, vector<int>& greatest,  
									vector<vector<int>>& neighborhood,
									double max_incidence, vector<vector<int>>& association);

	SparseComponents sparse_similarity(int level, int n, int n_neighbors, vector<int>& greatest, vector<int> &cols, vector<double>& vals,

									   Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, vector<vector<double>>& knn_dists);

	vector<double> update_position(int i, int n_neighbors, vector<int>& cols, vector<double>& vals, umap::Matrix& X, 
		                          Eigen::SparseMatrix<double, Eigen::RowMajor>& graph);
	vector<double> update_position(int i, vector<int>& neighbors, umap::Matrix& X);

	vector<vector<double>> embed_data(int level, Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, umap::Matrix& X);


	void associate_to_landmarks(int n, int n_neighbors, int* indices, vector<int>& cols, const vector<double>& sigmas,
								   vector<double>& strength, vector<int>& owners, vector<int>& indices_landmark, vector<vector<int>>& association, vector<int>& count_influence, vector<int>& is_landmark, 
								   Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, vector<vector<double>>& knn_dists);
	void associate_to_landmarks(int n, int n_neighbors, vector<int>& landmarks, vector<int>& cols, 
		vector<double>& strength, vector<int>& owners, vector<int>& indices, vector<vector<int>>& association, vector<int>& count_influence, vector<int>& is_landmark, vector<vector<double>>& knn_dists );
	int depth_first_search(int n_neighbors, int* neighbors, vector<int>& cols, const vector<double>& sigmas,
								  vector<double>& strength, vector<int>& owners, vector<int>& is_landmark);
	int dfs(int u, int n_neighbors, bool* visited, vector<int>& cols, const vector<double>& sigmas,
				   vector<double>& strength, vector<int>& owners, vector<int>& is_landmark);

	vector<int> get_influence_by_indices(int level, vector<int> indices);

	// void associate_to_landmarks(int n, int n_neighbors, int* indices, vector<int>& cols, const vector<double>& sigmas,
	// 							   double* strength, int* owners, int* indices_landmark, vector<vector<int>>& association, vector<int>& is_landmark, 
	// 							   Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, vector<vector<double>>& knn_dists);
	// int depth_first_search(int n_neighbors, int* neighbors, vector<int>& cols, const vector<double>& sigmas,
	// 							  double* strength, int* owners, vector<int>& is_landmark);
	// int dfs(int u, int n_neighbors, bool* visited, vector<int>& cols, const vector<double>& sigmas,
	// 			   double* strength, int* owners, vector<int>& is_landmark);


	// void associate_to_landmarks(int n, int n_neighbors, vector<int>& landmarks, vector<int>& cols, double* strength, int* owners, int* indices, vector<vector<int>>& association, vector<int>& is_landmark, vector<vector<double>>& knn_dists );



};




std::map<std::string, double> convert_dict_to_map(py::dict dictionary);
void split_string( string const& str, vector<StringRef> &result, char delimiter = ' ');
void tokenize(std::string &str, char delim, std::vector<std::string> &out);
vector<vector<double>> convert_to_vector(const py::array_t<double>& v);
vector<utils::SparseData> create_sparse(int n, 
										const vector<int>& rows, 
										const vector<int>& cols, 
										const vector<double>& vals);
void softmax(vector<double>& input, size_t size);
double sigmoid(double input);


vector<int>  markov_chain(vector<vector<int>>& knn_indices, 
						 vector<double>& vals, vector<int>& cols, 
						 int num_walks, int walk_length, vector<double>& sum_vals, bool path_increment); 

int random_walk(int vertex, int n_neighbors,
				vector<double>& vals, vector<int>& cols, 
				int current_step, int walk_length, vector<int>& endpoint,
				std::uniform_real_distribution<double>& unif, 
				std::default_random_engine& rng, vector<double>& sum_vals, bool path_increment);



int markov_chain(vector<vector<int>>& knn_indices, vector<vector<double>>& knn_dists, 
	vector<double>& vals, vector<int>& cols, Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, 
	int num_walks, int walk_length, vector<double>& sum_vals,
	vector<int>& landmarks, int influence_neighborhood, vector<vector<int>>& neighborhood, vector<vector<int>>& association);

int random_walk(int vertex, 
				   int n_neighbors, vector<double>& vals, vector<int>& cols, Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
				int current_step, int walk_length, uniform_real_distribution<double>& unif, 
														mt19937& rng, vector<double>& sum_vals, vector<int>& is_landmark);


vector<vector<double>> explain_neighborhoods(vector<vector<int>>& knn_indices, 
													Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
												    int num_walks, int walk_length, 
												    vector<double>& sum_vals, 
												    vector<vector<double>>& X);

void random_walk_explain(int vertex, vector<vector<int>>& knn_indices, 
								Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
								int current_step, int walk_length, 
								std::uniform_real_distribution<double>& unif,
								std::default_random_engine& rng, 
								vector<double>& sum_vals, 
								vector<vector<double>>& X,
								vector<vector<double>>& result);

vector<vector<double>> explain_neighborhoods(int index, vector<vector<int>>& knn_indices, 
													Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
												    int num_walks, int walk_length, 
												    vector<double>& sum_vals, 
												    vector<vector<double>>& X);



tuple<vector<vector<double>>, vector<double>> explain_neighborhoods(int index, int max_hops, vector<vector<int>>& knn_indices, 
													Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
												    int num_walks, int walk_length, 
												    vector<double>& sum_vals, 
												    vector<vector<double>>& X);

bool has(vector<int>& indices, int index);

class RandomGenerator
{
public:
    static RandomGenerator& Instance() {
        static RandomGenerator s;
        return s;
    }
    std::mt19937 & get() {
        return mt;
    }

private:
    RandomGenerator() {
        std::random_device rd;

        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        mt.seed(seed);
    }
    ~RandomGenerator() {}

    RandomGenerator(RandomGenerator const&) = delete;
    RandomGenerator& operator= (RandomGenerator const&) = delete;

    std::mt19937 mt;
};



}





#endif