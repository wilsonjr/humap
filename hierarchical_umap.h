#ifndef HIERARCHICAL_UMAP_H
#define HIERARCHICAL_UMAP_H

#include <map>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "umap.h"


using namespace std;

namespace py = pybind11;

namespace humap {



struct Metadata {

	 // Metadata(vector<int> indices_, vector<int> owners_, vector<float> strength_)
	 // : indices(indices_), owners(owners_), strength(strength_)
	 // {	 	
	 // }

	  Metadata(int* indices_, int* owners_, float* strength_, size_t size_)
	 : indices(indices_), owners(owners_), strength(strength_), size(size_)
	 {	 	
	 }

 	 int* indices;
	 int* owners;
	 float* strength;
	 int size;


	 // vector<int> indices;
	 // vector<int> owners;
	 // vector<float> strength;
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
	SparseComponents(vector<int> rows_, vector<int> cols_, vector<float> vals_): rows(rows_), cols(cols_), vals(vals_) {

	}

	vector<int> cols;
	vector<int> rows;
	vector<float> vals;

};

class HierarchicalUMAP
{

public:
	HierarchicalUMAP(string similarity_method_, py::array_t<float> percents_, int n_neighbors_=15, string knn_algorithm_="FAISS_IVFFlat", bool verbose_=false) 
	: similarity_method(similarity_method_), n_neighbors(n_neighbors_), knn_algorithm(knn_algorithm_), verbose(verbose_) {

		percents = vector<float>((float*)percents_.request().ptr, (float*)percents_.request().ptr + percents_.request().shape[0]);

	}

	HierarchicalUMAP() {
		cout << "Consegui instanciar" << endl;
	}

	void fit(py::array_t<float> X, py::array_t<int> y);
	py::array_t<int> get_labels(int level);
	Eigen::SparseMatrix<float, Eigen::RowMajor> get_data(int level);
	py::array_t<float> get_embedding(int level);
	py::array_t<int> get_indices(int level);
	py::array_t<float> get_sigmas(int level);




private:

	vector<vector<vector<float>>> embeddings;
	vector<umap::UMAP> reducers;
	vector<umap::Matrix> hierarchy_X;
	vector<vector<int>> hierarchy_y;
	vector<Metadata> metadata;

	vector<float> percents;
	string similarity_method;
	string knn_algorithm;


	int n_neighbors;
	bool verbose;
	int n_epochs = -1;
	int n_components = 2;

	int random_state = 0;

	vector<vector<int>> _indices;
	vector<vector<float>> _sigmas;


	
	
	void add_similarity(int index, int i, int n_neighbors, vector<int>& cols,  
					Eigen::SparseMatrix<float, Eigen::RowMajor>& graph, vector<vector<float>>& knn_dists,
					std::vector<std::vector<float> >& membership_strength, std::vector<std::vector<int> >& indices,
					std::vector<std::vector<float> >& distance, int* mapper, 
					float* elements, vector<vector<int>>& indices_nzeros, int n);
	
	SparseComponents create_sparse(int n, int n_neighbors, float* elements, vector<vector<int>>& indices_nzeros);

	SparseComponents sparse_similarity(int n, int n_neighbors, vector<int>& greatest, vector<int> &cols, 

									   Eigen::SparseMatrix<float, Eigen::RowMajor>& graph, vector<vector<float>>& knn_dists);

	vector<float> update_position(int i, int n_neighbors, vector<int>& cols, vector<float>& vals, umap::Matrix& X, 
		                          Eigen::SparseMatrix<float, Eigen::RowMajor>& graph);






};

void associate_to_landmarks(int n, int n_neighbors, int* indices, vector<int>& cols, vector<float>& sigmas,
								   float* strength, int* owners, int last_landmark, 
								   Eigen::SparseMatrix<float, Eigen::RowMajor>& graph);
int depth_first_search(int n_neighbors, int* neighbors, vector<int>& cols, vector<float>& sigmas,
							  float* strength, int* owners, int last_landmark);
int dfs(int u, int n_neighbors, bool* visited, vector<int>& cols, vector<float>& sigmas,
			   float* strength, int* owners, int last_landmark);



std::map<std::string, float> convert_dict_to_map(py::dict dictionary);
void split_string( string const& str, vector<StringRef> &result, char delimiter = ' ');
void tokenize(std::string &str, char delim, std::vector<std::string> &out);
vector<vector<float>> convert_to_vector(const py::array_t<float>& v);
vector<utils::SparseData> create_sparse(int n, const vector<int>& rows, const vector<int>& cols, const vector<float>& vals);
void softmax(vector<double>& input, size_t size);
double sigmoid(double input);

}





#endif