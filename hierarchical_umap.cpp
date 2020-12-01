#include "hierarchical_umap.h"

namespace py = pybind11;
using namespace std;


vector<utils::SparseData> humap::create_sparse(int n, const vector<int>& rows, const vector<int>& cols, const vector<float>& vals)
{
	vector<utils::SparseData> sparse(n, utils::SparseData());

	for( int i = 0; i < rows.size(); ++i )
		sparse[rows[i]].push(cols[i], vals[i]);


	return sparse;
}

vector<vector<float>> humap::convert_to_vector(const py::array_t<float>& v)
{
	py::buffer_info bf = v.request();
	float* ptr = (float*) bf.ptr;

	vector<vector<float>> vec(bf.shape[0], vector<float>(bf.shape[1], 0.0));
	for (int i = 0; i < vec.size(); ++i)
	{
		for (int j = 0; j < vec[0].size(); ++j)
		{
			vec[i][j] = ptr[i*vec[0].size() + j];
		}
	}

	return vec;
}


std::map<std::string, float> humap::convert_dict_to_map(py::dict dictionary) 
{
	std::map<std::string, float> result;

	for( std::pair<py::handle, py::handle> item: dictionary ) {

		std::string key = item.first.cast<std::string>();
		float value = item.second.cast<float>();

		result[key] = value; 

	}
	return result;
}

void humap::split_string( string const& str, vector<humap::StringRef> &result, char delimiter)
{
    enum State { inSpace, inToken };

    State state = inSpace;
    char const*     pTokenBegin = 0;    // Init to satisfy compiler.
    for( string::const_iterator it = str.begin(); it != str.end(); ++it )
    {
        State const newState = (*it == delimiter? inSpace : inToken);
        if( newState != state )
        {
            switch( newState )
            {
            case inSpace:
                result.push_back( StringRef( pTokenBegin, &*it - pTokenBegin ) );
                break;
            case inToken:
                pTokenBegin = &*it;
            }
        }
        state = newState;
    }
    if( state == inToken )
    {
        result.push_back( StringRef( pTokenBegin, &*str.end() - pTokenBegin ) );
    }
}


int humap::HierarchicalUMAP::dfs(int u, int n_neighbors, bool* visited, vector<int>& cols, const vector<float>& sigmas,
			   vector<float>& strength, vector<int>& owners, vector<int>& is_landmark)
{
	visited[u] = true;

	// cout << "dfs::checking for landmark" << endl;
	if( is_landmark[u] != -1 ) //sigmas[u] >= sigmas[last_landmark] )
		return u;
	else if( owners[u] != -1 )
		return owners[u];

	// cout << "dfs::constructing neighbor list" << endl;
	int* neighbors = new int[n_neighbors*sizeof(int)];
	for( int j = 0; j < n_neighbors; ++j ) {
		neighbors[j] = cols[u*n_neighbors + j];
	}	

	// cout << "dfs::checking for landmark" << endl;

	for( int i = 1; i < n_neighbors; ++i ) {
		int neighbor = neighbors[i];

		if( !visited[neighbor] ) {

			int landmark = this->dfs(neighbor, n_neighbors, visited, cols, sigmas, 
									  strength, owners, is_landmark);
			if( landmark != -1 ) {
				free(neighbors);
				neighbors = nullptr;
				return landmark;
			}
		}
	}


	if( neighbors )
		free(neighbors);
	return -1;


}

int humap::HierarchicalUMAP::depth_first_search(int n_neighbors, int* neighbors, vector<int>& cols, const vector<float>& sigmas,
							  vector<float>& strength, vector<int>& owners, vector<int>& is_landmark)
{
	bool* visited = new bool[sigmas.size()*sizeof(bool)];
	fill(visited, visited+sigmas.size(), false);

	for( int i = 1; i < n_neighbors; ++i ) {

		int neighbor = *(neighbors + i);

		if( !*(visited + neighbor) ) {

			int landmark = this->dfs(neighbor, n_neighbors, visited, cols, sigmas, strength, owners, is_landmark);
			if( landmark != -1 )  
				return landmark;
		}
	}

	return -1;
}

void humap::HierarchicalUMAP::associate_to_landmarks(int n, int n_neighbors, vector<int>& landmarks, vector<int>& cols, 
	vector<float>& strength, vector<int>& owners, vector<int>& indices, 


	vector<vector<int>>& association, vector<int>& is_landmark, 

	vector<int>& count_influence, vector<vector<float>>& knn_dists )
{
	int* neighbors = new int[n_neighbors*sizeof(int)];

	for( int i = 0; i < n; ++i ) {

		int landmark = landmarks[i];

		owners[landmark] = landmark;
		strength[landmark] = 0;
		indices[landmark] = i;
		association[landmark].push_back(i);
		count_influence[i]++;

		for( int j = 0; j < n_neighbors; ++j ) {
			neighbors[j] = cols[landmark*n_neighbors + j];
		}



		for( int j = 1; j < n_neighbors; ++j ) {

			int neighbor = neighbors[j];

			if( is_landmark[neighbor] != -1 ) {
				// owners[neighbor] = neighbor;
				// indices[neighbor] = is_landmark[neighbor];
				// strength[neighbor] = 0;

				// association[neighbor].push_back(is_landmark[neighbor]);
				// cout << "2.1" << endl;
				// cout << neighbor << " < " << is_landmark.size() << endl;
				// cout << is_landmark[neighbor] <<" < " << count_influence.size() << endl;
				// count_influence[is_landmark[neighbor]]++;
				// cout << "2.2" << endl;
				continue;
			}

			if( owners[neighbor] == -1 ) { // is not associated to any landmark
				owners[neighbor] = landmark;
				indices[neighbor] = i;
				strength[neighbor] = knn_dists[landmark][j];
				association[neighbor].push_back(i);
				count_influence[i]++;
			} 
			else if( knn_dists[landmark][j] < strength[neighbor] ) {
				owners[neighbor] = landmark;
				indices[neighbor] = i;
				association[neighbor].push_back(i);
				strength[neighbor] = knn_dists[landmark][j];				
			}
		}

	}

	free(neighbors);

}


void humap::HierarchicalUMAP::associate_to_landmarks(int n, int n_neighbors, int* indices, vector<int>& cols, const vector<float>& sigmas,
								   vector<float>& strength, vector<int>& owners, vector<int>& indices_landmark, vector<vector<int>>& association, vector<int>& count_influence, vector<int>& is_landmark, 
								   Eigen::SparseMatrix<float, Eigen::RowMajor>& graph, vector<vector<float>>& knn_dists)
{
	int* neighbors = new int[n_neighbors*sizeof(int)];
	int count_search = 0;
	for( int i = 0; i < n; ++i ) {

		int index = indices[i];
		if( is_landmark[index] != -1 || owners[index] != -1 )
			continue;

		// for( int j = 0; j < n_neighbors; ++j ) {
		// 	neighbors[j] = cols[index*n_neighbors + j];
		// }

		// int nn = neighbors[1];

		bool found = false;
		
		for( int j = 1; j < n_neighbors && !found; ++j ) {

			int nn = cols[index*n_neighbors + j];


			if( is_landmark[nn] != -1 ) {
				strength[index] = knn_dists[nn][j];//graph.coeffRef(nn, index);
				owners[index] = nn;
				indices_landmark[index] = is_landmark[nn];
				association[index].push_back(is_landmark[nn]);
				count_influence[is_landmark[nn]]++;
				found = true;
			} else if( owners[nn] != -1 ) {
				// TODO: this is an estimative
				strength[index] = strength[nn]; //graph.coeffRef(*(owners + nn), index);
				owners[index] = owners[nn];			
				indices_landmark[index] = is_landmark[owners[nn]];	
				association[index].push_back(is_landmark[owners[nn]]);
				count_influence[is_landmark[owners[nn]]]++;
				found = true;
			}
		}

		if( !found ) {

			count_search++;
			int landmark = this->depth_first_search(n_neighbors, neighbors, cols, sigmas, 
					                                     strength, owners, is_landmark);
			// cout << "ENTREI AQUI: " << landmark << endl;
			if( landmark != -1 ) {
				// cout << "Found a landmark :)" << endl;
				// TODO: need to compute when needed
				strength[index] = 0;//knn_dists[landmark]
				owners[index] = landmark;
				indices_landmark[index] = is_landmark[landmark];	
				association[index].push_back(is_landmark[landmark]);
				count_influence[is_landmark[landmark]]++;
			} else {
				cout << "Did not find a landmark :(" << endl;
				throw runtime_error("Did not find a landmark");
			}


		}

		// for( int j = 1; j < n_neighbors && !found; ++j ) {

		// 	int nn = cols[index*n_neighbors + j];


		// 	if( is_landmark[nn] ) {
		// 		*(strength + index) = graph.coeffRef(nn, index);
		// 		*(owners + index) = nn;
		// 		found = true;
		// 	} else if( *(strength + nn) != -1.0 ) {
		// 		*(strength + index) = graph.coeffRef(*(owners + nn), index);
		// 		*(owners + index) = *(owners + nn);				
		// 		found = true;
		// 	}
		// }






		// if( is_landmark[nn] ) {
		// 	// cout << "Associating with a landmark" << endl;
		// 	*(strength + index) = graph.coeffRef(nn, index);
		// 	*(owners + index) = nn;
		// } else {
		// 	// cout << "hello, estou aqui : 0" << endl;
		// 	if( *(strength + nn) != -1.0 ) {
		// 		// cout << "Associating with already found" << endl;
		// 		*(strength + index) = graph.coeffRef(*(owners + nn), index);
		// 		*(owners + index) = *(owners + nn);
		// 	} else {

				
		// 	}
		// }



		// if( is_landmark[nn] ) {
		// 	// cout << "Associating with a landmark" << endl;
		// 	*(strength + index) = graph.coeffRef(nn, index);
		// 	*(owners + index) = nn;
		// } else {
		// 	// cout << "hello, estou aqui : 0" << endl;
		// 	if( *(strength + nn) != -1.0 ) {
		// 		// cout << "Associating with already found" << endl;
		// 		*(strength + index) = graph.coeffRef(*(owners + nn), index);
		// 		*(owners + index) = *(owners + nn);
		// 	} else {

		// 		int landmark = this->depth_first_search(n_neighbors, neighbors, cols, sigmas, 
		// 			                                     strength, owners, is_landmark);
		// 		// cout << "ENTREI AQUI: " << landmark << endl;
		// 		if( landmark != -1 ) {
		// 			// cout << "Found a landmark :)" << endl;
		// 			*(strength + index) = graph.coeffRef(landmark, index);
		// 			*(owners + index) = landmark;
		// 		} else {
		// 			cout << "Did not find a landmark :(" << endl;
		// 			throw runtime_error("Did not find a landmark");
		// 		}
		// 	}
		// }
	}
	free(neighbors);
}

void humap::HierarchicalUMAP::add_similarity(int index, int i, int n_neighbors, vector<int>& cols,  
					Eigen::SparseMatrix<float, Eigen::RowMajor>& graph, vector<vector<float>>& knn_dists,
					std::vector<std::vector<float> >& membership_strength, std::vector<std::vector<int> >& indices,
					std::vector<std::vector<float> >& distance, int* mapper, 
					float* elements, vector<vector<int>>& indices_nzeros, int n)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;
	// cout << "falha 3" << endl;
	std::vector<int> neighbors;

	for( int j = 1; j < n_neighbors; ++j ) {
		neighbors.push_back(cols[i*n_neighbors + j]);
	}
	// cout << "falha 4" << endl;
	//#pragma omp parallel for default(shared) schedule(dynamic, 50)
	// #pragma omp parallel for default(shared) schedule(dynamic, 100)
	for( int j = 0; j < neighbors.size(); ++j ) {
		// cout << "falha 6" << endl;
		int neighbor = neighbors[j];

		if( membership_strength[neighbor].size() == 0 ) {
			// cout << "falha 5" << endl;

			// #pragma omp critical(initing_neighbors)
			// {
				membership_strength[neighbor].push_back(graph.coeffRef(i, neighbor));
				// cout << "falha 7" << endl;	
				distance[neighbor].push_back(knn_dists[i][j+1]);//(knn_dists[i*n_neighbors + (j+1)]);
				// cout << "falha 8" << endl;
				indices[neighbor].push_back(i);
			// }
		} else {
			float ms2 = graph.coeffRef(i, neighbor);
			float d2 = knn_dists[i][j+1];//[i*n_neighbors + (j+1)];
			int ind2 = i;
				
			for( int count = 0; count < membership_strength[neighbor].size(); ++count ) {

				float ms1 = membership_strength[neighbor][count];
				float d1 = distance[neighbor][count];
				int ind1 = indices[neighbor][count];
	

				float s = (std::min(ms1*d1, ms2*d2)/std::max(ms1*d1, ms2*d2))/(n_neighbors-1);

				// test as map
				if( *(mapper + ind1) != -1 ) {

					int u = *(mapper + ind1);
					int v = *(mapper + ind2);

					*(elements + u*n + v) += s;
					*(elements + v*n + u) += s;

					indices_nzeros[u].push_back(v);
					indices_nzeros[v].push_back(u);

				}
			}
			membership_strength[neighbor].push_back(ms2);
			distance[neighbor].push_back(d2);
			indices[neighbor].push_back(ind2);
		}
	}
}

void humap::tokenize(std::string &str, char delim, std::vector<std::string> &out)
{
	size_t start;
	size_t end = 0;

	while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
	{
		end = str.find(delim, start);
		out.push_back(str.substr(start, end - start));
	}
}

humap::SparseComponents humap::HierarchicalUMAP::create_sparse(int n, int n_neighbors, float* elements, vector<vector<int>>& indices_nzeros)
{

	vector<int> cols;
	vector<int> rows;
	vector<float> vals;

	int* current = new int[n*sizeof(int)];
	fill(current, current+n, 0);

	for( int i = 0; i < n; ++i ) {
		bool flag = true;
		for( int j = 0; j < indices_nzeros[i].size(); ++j ) {
			int index = indices_nzeros[i][j];

			if( *(current + index) )
				continue;

			*(current + index) = 1;
			if( *(elements + i*n + index) != 0.0 ) {				
				rows.push_back(i);
				cols.push_back(index);
				if( i == index )
					flag =false;
				vals.push_back(1 - *(elements + i*n + index));			
			}
		}

 		for( int j = 0; j < n_neighbors+5; ++j ) {
			if( *(elements + i*n + j) == 0.0 && i != j) {				
				rows.push_back(i);
				cols.push_back(j);
				vals.push_back(1);
			} 
		}

		for( int j = 0; j < indices_nzeros[i].size(); ++j ){
			*(current + indices_nzeros[i][j]) = 0;
		}
		 if(  flag ) {
		 	rows.push_back(i);
				cols.push_back(i);
				vals.push_back(0);
		 }
	}

	return humap::SparseComponents(rows, cols, vals);
}



humap::SparseComponents humap::HierarchicalUMAP::sparse_similarity(int n, int n_neighbors, vector<int>& greatest, vector<int> &cols, 
																   Eigen::SparseMatrix<float, Eigen::RowMajor>& graph, vector<vector<float>>& knn_dists) 
{

	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	std::vector<std::vector<int> > indices_sim;
	std::vector<std::vector<float> > membership_strength;
	std::vector<std::vector<float> > distance_sim;

	int* mapper = new int[n * sizeof(int)];
	fill(mapper, mapper+n, -1);

	for( int i = 0; i < greatest.size(); ++i )
		mapper[greatest[i]] = i;


	for( int i = 0; i < n; ++i ) {
		indices_sim.push_back(std::vector<int>());
		membership_strength.push_back(std::vector<float>());
		distance_sim.push_back(std::vector<float>());
	}

	float* elements = new float[greatest.size()*greatest.size()*sizeof(float)];
	fill(elements, elements+greatest.size()*greatest.size(), 0.0);


	int* non_zeros = new int[greatest.size() * sizeof(int)];
	fill(non_zeros, non_zeros+greatest.size(), 0);

	vector<vector<int>> indices_nzeros(greatest.size(), vector<int>());

	

	for( int i = 0; i < greatest.size(); ++i ) {


		add_similarity(i, greatest[i], n_neighbors, cols, graph, knn_dists, membership_strength, 
			indices_sim, distance_sim, mapper, elements, indices_nzeros, greatest.size());

		// data_dict[std::to_string(i)+'_'+std::to_string(i)] = 1.0;
	}
	auto begin = clock::now();
	humap::SparseComponents sc = this->create_sparse(greatest.size(), n_neighbors, elements, indices_nzeros);
	sec duration = clock::now() - begin;
	cout << "Time for sparse components: " << duration.count() << endl;


	free(elements);
	free(non_zeros);
	free(mapper);

	return sc;
}


vector<float> humap::HierarchicalUMAP::update_position(int i, int n_neighbors, vector<int>& cols, vector<float>& vals,
													   umap::Matrix& X, Eigen::SparseMatrix<float, Eigen::RowMajor>& graph)
{
	std::vector<int> neighbors;

	for( int j = 0; j < n_neighbors; ++j ) {
		neighbors.push_back(cols[i*n_neighbors + j]);
	}


	vector<float> u = X.dense_matrix[i];

	vector<float> mean_change(X.shape(1), 0);
	for( int j = 0; j < neighbors.size(); ++j ) {
		int neighbor = neighbors[j];

		vector<float> v = X.dense_matrix[neighbor];

		vector<float> temp(v.size(), 0.0);
		for( int k = 0; k < temp.size(); ++k )
			temp[k] = graph.coeffRef(i, neighbor)*(v[k]-u[k]);

		transform(mean_change.begin(), mean_change.end(), 
				  temp.begin(), mean_change.begin(), plus<float>());
	}

	transform(mean_change.begin(), mean_change.end(), mean_change.begin(), [n_neighbors](float& c){
		return c/(n_neighbors);
	});

	transform(u.begin(), u.end(), mean_change.begin(), u.begin(), plus<float>());

	return u;
}

double humap::sigmoid(double input) 
{

	return 1.0/(1.0 + exp(-input));
}

void humap::softmax(vector<double>& input, size_t size) {

	vector<double> exp_values(input.size(), 0.0);
	for( int i = 0; i < exp_values.size(); ++i ) {
		exp_values[i] = exp(input[i]);
	}
	double sum_exp = accumulate(exp_values.begin(), exp_values.end(), 0.0);
	for( int i = 0; i < exp_values.size(); ++i ) {
		input[i] = exp(input[i])/sum_exp;
	}	
}


void humap::HierarchicalUMAP::fit(py::array_t<float> X, py::array_t<int> y)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;


	// const auto before = clock::now();

	// cout << "Size: " << X.request().shape[0] << ", " << X.request().shape[1] << endl;
	// cout << "Similarity Method: " << this->similarity_method << endl;
	// cout << "Number of levels: " << this->percents.size() << endl;
	// for( int level = 0; level < this->percents.size(); ++level )
	// 	printf("Level %d: %.2f of the instances from level %d.\n", (level+1), this->percents[level], level);
	// cout << "KNN algorithm: " << this->knn_algorithm << endl;


	// const sec duration = clock::now() - before;
	// cout << "it took "  << duration.count() << "s" << endl;


	// return;
	auto hierarchy_before = clock::now();


	auto before = clock::now();
	umap::Matrix first_level(humap::convert_to_vector(X));
	sec duration = clock::now() - before;

	this->hierarchy_X.push_back(first_level);
	this->hierarchy_y.push_back(vector<int>((int*)y.request().ptr, (int*)y.request().ptr + y.request().shape[0]));

	// if( this->knn_algorithm == "FAISS_IVFFlat" && first_level.size() < 10000 ) {
	// 	this->knn_algorithm = "FAISS_Flat";
	// }

	umap::UMAP reducer = umap::UMAP("euclidean", this->n_neighbors, this->knn_algorithm);

	if( this->verbose ) {
		cout << "\n\n*************************************************************************" << endl;
		cout << "*********************************LEVEL 0*********************************" << endl;
		cout << "*************************************************************************" << endl;
	}

	before = clock::now();
	reducer.fit_hierarchy(this->hierarchy_X[0]);
	duration = clock::now() - before;
	cout << "Fitting for first level: " << duration.count() << endl;
	cout << endl;

	this->reducers.push_back(reducer);

	// int* indices = new int[sizeof(int)*this->hierarchy_X[0].size()];
	vector<int> indices(this->hierarchy_X[0].size(), 0);
	iota(indices.begin(), indices.end(), 0);

	// int* owners = new int[sizeof(int)*this->hierarchy_X[0].size()];
	// fill(owners, owners+this->hierarchy_X[0].size(), -1);
	vector<int> owners(this->hierarchy_X[0].size(), -1);


	// float* strength = new float[sizeof(float)*this->hierarchy_X[0].size()];
	// fill(strength, strength+this->hierarchy_X[0].size(), -1.0);
	vector<float> strength(this->hierarchy_X[0].size(), -1.0);

	vector<vector<int>> association(this->hierarchy_X[0].size(), vector<int>());

	this->metadata.push_back(humap::Metadata(indices, owners, strength, association, this->hierarchy_X[0].size()));

	this->original_indices.push_back(indices);


	Eigen::SparseMatrix<float, Eigen::RowMajor> graph = this->reducers[0].get_graph();

	vector<vector<float>> knn_dists = this->reducers[0].knn_dists();

	for( int level = 0; level < this->percents.size(); ++level ) {
		// this->n_neighbors = (int) (1.5*this->n_neighbors);

		auto level_before = clock::now();

		int n_elements = (int) (this->percents[level] * this->hierarchy_X[level].size());
		



		auto begin_sampling = clock::now();
		vector<float> values(this->reducers[level].sigmas().size(), 0.0);
		float mean_sigma = 0.0;
		float sum_sigma = 0.0;
		float max_sigma = -1;

		for( int i = 0; i < values.size(); ++i ) {
			max_sigma = max(max_sigma, this->reducers[level].sigmas()[i]);
			sum_sigma += fabs(this->reducers[level].sigmas()[i]);
 		}

		vector<double> probs(this->reducers[level].sigmas().size(), 0.0);
		for( int i = 0; i < values.size(); ++i ) {
			// probs[i] = this->reducers[level].sigmas()[i];
			// probs[i] = this->reducers[level].sigmas()[i]/max_sigma;
			probs[i] = this->reducers[level].sigmas()[i]/sum_sigma;
			
			// probs[i] = humap::sigmoid(this->reducers[level].sigmas()[i]);
 		}
		
   		humap::softmax(probs, this->reducers[level].sigmas().size());

		py::module np = py::module::import("numpy");
		py::object choice = np.attr("random");
		py::object indices_candidate = choice.attr("choice")(py::cast(probs.size()), py::cast(n_elements), py::cast(false),	py::cast(probs));
		vector<int> possible_indices = indices_candidate.cast<vector<int>>();

   		vector<int> s_indices = utils::argsort(possible_indices);
   		for( int i = 1; i < s_indices.size(); ++i ) {
   			if( possible_indices[s_indices[i-1]] == possible_indices[s_indices[i]] ) {
   				cout << "OLHA ACHEI UM REPETIDO :)" << endl;
   			}
   		}

		vector<int> greatest = possible_indices;

		vector<int> orig_inds(greatest.size(), 0);

		for( int i = 0; i < orig_inds.size(); ++i )
			orig_inds[i] = this->original_indices[level][greatest[i]];
		this->original_indices.push_back(orig_inds);

		this->_sigmas.push_back(this->reducers[level].sigmas());

		this->_indices.push_back(greatest);

		umap::Matrix data;	

		if( this->verbose ) {
			cout << "\n\n*************************************************************************" << endl;
			cout << "*********************************LEVEL "<< (level+1) << "*********************************" << endl;
			cout << "*************************************************************************" << endl;
		}
		
		if( this->verbose )
			cout << "Level " << (level+1) << ": " << n_elements << " data samples." << endl;


		if( this->verbose )
			cout << "Computing level" << endl;


		if( this->similarity_method == "similarity" ) {

			vector<vector<float>> dense;


			for( int i = 0; i < greatest.size(); ++i ) {
					
				vector<float> row = update_position(greatest[i], this->n_neighbors, this->reducers[level].cols, 
					this->reducers[level].vals, this->hierarchy_X[level], graph);
				dense.push_back(row);
			}


			data = umap::Matrix(dense);
			reducer = umap::UMAP("euclidean", this->n_neighbors, this->knn_algorithm);



		} else if( this->similarity_method == "precomputed" ) {

			auto sparse_before = clock::now();
			SparseComponents triplets = this->sparse_similarity(this->hierarchy_X[level].size(), this->n_neighbors,
																 greatest, this->reducers[level].cols,
																 graph, knn_dists);
			sec sparse_duration = clock::now() - sparse_before;

			cout << "Sparse Matrix: " << sparse_duration.count() << endl;

			auto eigen_before = clock::now();
			// Eigen::SparseMatrix<float, Eigen::RowMajor> sparse = utils::create_sparse(triplets.rows, triplets.cols, triplets.vals,
			// 																		  n_elements, n_neighbors*4);


			vector<utils::SparseData> sparse = humap::create_sparse(n_elements, triplets.rows, triplets.cols, triplets.vals);
			sec eigen_duration = clock::now() - eigen_before;
			cout << "Constructing eigen matrix: " << eigen_duration.count() << endl;
			cout << endl;

			data = umap::Matrix(sparse, greatest.size());
			reducer = umap::UMAP("precomputed", this->n_neighbors, this->knn_algorithm);
		} else {


			reducer = umap::UMAP("euclidean", this->n_neighbors, this->knn_algorithm);


		}


		if( this->verbose )
			cout << "Fitting hierarchy level" << endl; 



		this->metadata[level].count_influence = vector<int>(greatest.size(), 0);

		auto fit_before = clock::now();
		reducer.fit_hierarchy(data);
		sec fit_duration = clock::now() - fit_before;
		cout << "Fitting level " << (level+1) << ": " << fit_duration.count() << endl;
		cout << endl;

		auto associate_before = clock::now();
		cout << level << ": METADATA PONTO SIZE: " << this->metadata[level].size << endl;
		vector<int> is_landmark(this->metadata[level].size, -1);
		for( int i = 0; i < greatest.size(); ++i ) 
		{
			is_landmark[greatest[i]] = i;
		}

		this->associate_to_landmarks(greatest.size(), this->n_neighbors, greatest, this->reducers[level].cols, 
			this->metadata[level].strength, this->metadata[level].owners, this->metadata[level].indices, 
			this->metadata[level].association, is_landmark, this->metadata[level].count_influence, knn_dists);

		sec associate_duration = clock::now() - associate_before;
		cout << "Associate landmark: " << associate_duration.count() << endl;
		cout << endl;


		auto use_before = clock::now();
		int n = 0;
		for( int i = 0; i < this->metadata[level].size; ++i ) {
			if( this->metadata[level].owners[i] == -1 )
				n++;
		}
		cout << "NUMBER OF ELEMENTS WITH NO OWNER: " << n << endl;
		int* indices_not_associated = new int[sizeof(int)*n];
		for( int i = 0, j = 0; i < this->metadata[level].size; ++i )
			if( this->metadata[level].owners[i] == -1.0 )
				*(indices_not_associated + j++) = i;

		this->associate_to_landmarks(n, this->n_neighbors, indices_not_associated, this->reducers[level].cols,
									  this->reducers[level].sigmas(), this->metadata[level].strength,
									  this->metadata[level].owners, this->metadata[level].indices, 
									  this->metadata[level].association, this->metadata[level].count_influence, 
									  is_landmark, graph, knn_dists);

		sec use_duration = clock::now() - use_before;
		cout << "Use landmark: " << use_duration.count() << endl;
		cout << endl;


		cout << "I have " << this->metadata[level].size << " owners" << endl;
		cout << "See the first 10: " << level << endl;
		for( int i = 0; i < 10; ++i ) {
			cout << i << " " << this->metadata[level].indices[i] << " (" << this->metadata[level].owners[i] << ") -> " << this->metadata[level].strength[i] << endl;


		}
		cout << endl;

		if( this->verbose )
			cout << "Appending information for the next hierarchy level" << endl;

		// int* new_owners = new int[sizeof(int)*greatest.size()];
		// fill(new_owners, new_owners+greatest.size(), -1);
		vector<int> new_owners(greatest.size(), -1);

		// float* new_strength = new float[sizeof(float)*greatest.size()];
		// fill(new_strength, new_strength+greatest.size(), -1.0);
		vector<float> new_strength(greatest.size(), -1.0);

		vector<vector<int>> new_association(greatest.size(), vector<int>());

		this->metadata.push_back(Metadata(greatest, new_owners, new_strength, new_association, greatest.size()));//vector<int>(greatest.size(), -1), vector<float>(greatest.size(), -1.0)));
		this->reducers.push_back(reducer);
		this->hierarchy_X.push_back(data);
		this->hierarchy_y.push_back(utils::arrange_by_indices(this->hierarchy_y[level], greatest));


		knn_dists = reducer.knn_dists();
		graph = reducer.get_graph();


		sec level_duration = clock::now() - level_before;
		cout << "Level construction: " << level_duration.count() << endl;
		cout << endl;
	}




	sec hierarchy_duration = clock::now() - hierarchy_before;
	cout << endl;
	cout << "Hierarchy construction: " << hierarchy_duration.count() << endl;
	cout << endl;



	for( int i = 0; i < this->hierarchy_X.size(); ++i ) {
		// cout << "chguei aqui 1 " << endl;
		Eigen::SparseMatrix<float, Eigen::RowMajor> graph = this->reducers[i].get_graph();		
		vector<vector<float>> result = this->embed_data(i, graph, this->hierarchy_X[i]);
		// int n_vertices = graph.cols();
		// // cout << "chguei aqui 2 " << endl;

		// if( this->n_epochs == -1 ) {
		// 	if( graph.rows() <= 10000 )
		// 		n_epochs = 500;
		// 	else 
		// 		n_epochs = 200;

		// }
		// // cout << "chguei aqui 3 " << endl;
		// if( !graph.isCompressed() )
		// 	graph.makeCompressed();
		// // graph = graph.pruned();
		// // cout << "chguei aqui 4 " << endl;
		// float max_value = graph.coeffs().maxCoeff();
		// graph = graph.pruned(max_value/(float)n_epochs, 1.0);


		// // cout << "chguei aqui 5 " << endl;


		// vector<vector<float>> embedding = this->reducers[i].spectral_layout(this->hierarchy_X[i], graph, this->n_components);


		// // cout << "chguei aqui 6 " << endl;
		// vector<int> rows, cols;
		// vector<float> data;

		
		// // cout << "chguei aqui 7 " << endl;
		// tie(rows, cols, data) = utils::to_row_format(graph);

		// // cout << "chguei aqui 8 " << endl;
		// vector<float> epochs_per_sample = this->reducers[i].make_epochs_per_sample(data, this->n_epochs);
		// // cout << "\n\nepochs_per_sample: " << epochs_per_sample.size() << endl;

		// // cout << "chguei aqui 9 " << endl;
		// // for( int j = 0; j < 20; ++j )
		// // 	printf("%.4f ", epochs_per_sample[j]);

		// // cout << endl << endl;



		// // cout << "chguei aqui 10 " << endl;
		// vector<float> min_vec, max_vec;
		// // printf("min and max values:\n");
		// for( int j = 0; j < this->n_components; ++j ) {


		// 	// float min_v = embedding[0][j];
		// 	// float max_v = embedding[0][j];
		// 	// for( int i = 0; i < embedding.size(); ++i ) {
		// 	// 	// cout << embedding[i][j] << " ";
		// 	// 	min_v = min(min_v, embedding[i][j]);
		// 	// 	max_v = max(max_v, embedding[i][j]);
		// 	// }


		// 	// cout << endl;
		// 	// min_vec.push_back(min_v);
		// 	// max_vec.push_back(max_v);

		// 	min_vec.push_back((*min_element(embedding.begin(), embedding.end(), 
		// 		[j](vector<float> a, vector<float> b) {							
		// 			return a[j] < b[j];
		// 		}))[j]);
		// 	// cout <<" ****** 2" << endl;
		// 	max_vec.push_back((*max_element(embedding.begin(), embedding.end(), 
		// 		[j](vector<float> a, vector<float> b) {
		// 			return a[j] < b[j];
		// 		}))[j]);
		// 	// cout <<" ****** 3" << endl;

		// }
		// // cout << "chguei aqui 11 " << endl;
		// vector<float> max_minus_min(this->n_components, 0.0);
		// transform(max_vec.begin(), max_vec.end(), min_vec.begin(), max_minus_min.begin(), [](float a, float b){ return a-b; });
		// // cout << "chguei aqui 12 " << endl;

		// for( int j = 0; j < embedding.size(); ++j ) {

		// 	transform(embedding[j].begin(), embedding[j].end(), min_vec.begin(), embedding[j].begin(), 
		// 		[](float a, float b) {
		// 			return 10*(a-b);
		// 		});

		// 	transform(embedding[j].begin(), embedding[j].end(), max_minus_min.begin(), embedding[j].begin(),
		// 		[](float a, float b) {
		// 			return a/b;
		// 		});
		// }
		// // cout << "chguei aqui 13 " << endl;

		// // printf("\n\nNOISED EMBEDDING:\n");

		// // for( int j = 0; j < 10; ++j ) {
		// // 	if(  j < 10 )
		// // 		printf("%.4f %.4f\n", embedding[j][0], embedding[j][1]);
		// // }
		// // printf("------------------------------------------------------------------\n");

		// py::module scipy_random = py::module::import("numpy.random");
		// py::object randomState = scipy_random.attr("RandomState")(this->random_state);

		// py::object rngStateObj = randomState.attr("randint")(numeric_limits<int>::min(), numeric_limits<int>::max(), 3);
		// vector<long> rng_state = rngStateObj.cast<vector<long>>();

		// printf("Embedding a level with %d data samples\n", embedding.size());
		// before = clock::now(); 
		// vector<vector<float>> result = this->reducers[i].optimize_layout_euclidean(
		// 	embedding,
		// 	embedding,
		// 	rows,
		// 	cols,
		// 	this->n_epochs,
		// 	n_vertices,
		// 	epochs_per_sample,
		// 	rng_state);
		// duration = clock::now() - before;
		// cout << endl << "It took " << duration.count() << " to embed." << endl;

		this->embeddings.push_back(result);

	}

}

int humap::HierarchicalUMAP::influenced_by(int level, int index)
{
	if( level < 0 ) {
		return 1;
	} else if( level == 0 ) {
		return this->metadata[level].count_influence[index];
	} else {
		int s = 0;
		for( int i = 0; i < this->metadata[level].size; ++i )
			if( this->metadata[level].indices[i] == index )
				s += this->influenced_by(level-1, i);
		return s;
	}


}

vector<int> humap::HierarchicalUMAP::get_influence_by_indices(int level, vector<int> indices) 
{
	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");
	vector<int> influence(indices.size(), 0);

	for( int i = 0; i < indices.size(); ++i ) {
		influence[i] = influenced_by(level-1, indices[i]);
	}

	return influence;
}

py::array_t<int> humap::HierarchicalUMAP::get_influence(int level)
{
	if( level >= this->hierarchy_X.size() || level <= 0 )
		throw new runtime_error("Level out of bounds.");

	vector<int> influence(this->metadata[level].size, 0);
	// cout << "creating influence " << endl;
	// for( int i = 0; i < this->metadata[level-1].size; ++i ) {
	// 	cout << "size: " << this->metadata[level-1].association[i].size() << endl;
	// 	for( int j = 0; j < this->metadata[level-1].association[i].size(); ++j )
	// 		influence[this->metadata[level-1].association[i][j]] += 1;	
	// }


	// for( int i = 0; i < this->metadata[level-1].size; ++i ) {
	// 	influence[this->metadata[level-1].indices[i]] += 1;
	// }

	for( int i = 0; i < this->metadata[level].size; ++i ) {
		influence[i] = influenced_by(level-1, i);
	}

	// cout << "level: " << level << endl;
	// int sum = 0;


	// for( int i = 0; i < this->metadata[level-1].count_influence.size(); ++i ) {
	// 	// cout << i << ": " << this->metadata[level-1].count_influence[i] << endl;
	// 	sum += this->metadata[level-1].count_influence[i];
	// }
	// cout << "count_influence: " << sum << endl;


	return py::cast(influence);
}

py::array_t<float> humap::HierarchicalUMAP::get_sigmas(int level)
{
	if( level >= this->hierarchy_X.size()-1 || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->_sigmas[level]);
}

py::array_t<int> humap::HierarchicalUMAP::get_indices(int level)
{
	if( level >= this->hierarchy_X.size()-1 || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->_indices[level]);
}

py::array_t<int> humap::HierarchicalUMAP::get_original_indices(int level)
{
	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->original_indices[level]);
}



py::array_t<int> humap::HierarchicalUMAP::get_labels(int level)
{
	if( level == 0 )  
		throw new runtime_error("Sorry, we won't me able to return all the labels!");

	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");


	return py::cast(this->hierarchy_y[level]);
}

py::array_t<float> humap::HierarchicalUMAP::get_embedding(int level)
{
	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->embeddings[level]);
}	


Eigen::SparseMatrix<float, Eigen::RowMajor> humap::HierarchicalUMAP::get_data(int level)
{
	if( level == 0 )  
		throw new runtime_error("Sorry, we won't me able to return all dataset! Please, project using UMAP.");

	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return utils::create_sparse(this->hierarchy_X[level].sparse_matrix, this->hierarchy_X[level].size(), (int) this->n_neighbors*2.5);
}

vector<vector<float>> humap::HierarchicalUMAP::embed_data(int level, Eigen::SparseMatrix<float, Eigen::RowMajor>& graph, umap::Matrix& X)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;
	int n_vertices = graph.cols();
	// cout << "chguei aqui 2 " << endl;

	if( this->n_epochs == -1 ) {
		if( graph.rows() <= 10000 )
			n_epochs = 500;
		else 
			n_epochs = 200;

	}
	// cout << "chguei aqui 3 " << endl;
	if( !graph.isCompressed() )
		graph.makeCompressed();
	// graph = graph.pruned();
	// cout << "chguei aqui 4 " << endl;
	float max_value = graph.coeffs().maxCoeff();
	graph = graph.pruned(max_value/(float)n_epochs, 1.0);


	// cout << "chguei aqui 5 " << endl;


	vector<vector<float>> embedding = this->reducers[level].spectral_layout(X, graph, this->n_components);


	// cout << "chguei aqui 6 " << endl;
	vector<int> rows, cols;
	vector<float> data;

	
	// cout << "chguei aqui 7 " << endl;
	tie(rows, cols, data) = utils::to_row_format(graph);

	// cout << "chguei aqui 8 " << endl;
	vector<float> epochs_per_sample = this->reducers[level].make_epochs_per_sample(data, this->n_epochs);
	// cout << "\n\nepochs_per_sample: " << epochs_per_sample.size() << endl;

	// cout << "chguei aqui 9 " << endl;
	// for( int j = 0; j < 20; ++j )
	// 	printf("%.4f ", epochs_per_sample[j]);

	// cout << endl << endl;



	// cout << "chguei aqui 10 " << endl;
	vector<float> min_vec, max_vec;
	// printf("min and max values:\n");
	for( int j = 0; j < this->n_components; ++j ) {

		min_vec.push_back((*min_element(embedding.begin(), embedding.end(), 
			[j](vector<float> a, vector<float> b) {							
				return a[j] < b[j];
			}))[j]);
		// cout <<" ****** 2" << endl;
		max_vec.push_back((*max_element(embedding.begin(), embedding.end(), 
			[j](vector<float> a, vector<float> b) {
				return a[j] < b[j];
			}))[j]);
		// cout <<" ****** 3" << endl;

	}
	// cout << "chguei aqui 11 " << endl;
	vector<float> max_minus_min(this->n_components, 0.0);
	transform(max_vec.begin(), max_vec.end(), min_vec.begin(), max_minus_min.begin(), [](float a, float b){ return a-b; });
	// cout << "chguei aqui 12 " << endl;

	for( int j = 0; j < embedding.size(); ++j ) {

		transform(embedding[j].begin(), embedding[j].end(), min_vec.begin(), embedding[j].begin(), 
			[](float a, float b) {
				return 10*(a-b);
			});

		transform(embedding[j].begin(), embedding[j].end(), max_minus_min.begin(), embedding[j].begin(),
			[](float a, float b) {
				return a/b;
			});
	}

	py::module scipy_random = py::module::import("numpy.random");
	py::object randomState = scipy_random.attr("RandomState")(this->random_state);

	py::object rngStateObj = randomState.attr("randint")(numeric_limits<int>::min(), numeric_limits<int>::max(), 3);
	vector<long> rng_state = rngStateObj.cast<vector<long>>();

	printf("Embedding a level with %d data samples\n", embedding.size());
	auto before = clock::now(); 
	vector<vector<float>> result = this->reducers[level].optimize_layout_euclidean(
		embedding,
		embedding,
		rows,
		cols,
		this->n_epochs,
		n_vertices,
		epochs_per_sample,
		rng_state);
	sec duration = clock::now() - before;
	cout << endl << "It took " << duration.count() << " to embed." << endl;

	return result;
}	

py::array_t<float> humap::HierarchicalUMAP::project(int level, py::array_t<int> c)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;
	py::buffer_info bf = c.request();
	int* classes = (int*) bf.ptr;
	cout << "classes 1" << endl;

	vector<int> selected_indices;
	for( int i = 0; i < this->hierarchy_y[level].size(); ++i ) {
		bool flag = false;
		for( int j = 0; j < bf.shape[0]; ++j )
			if( this->hierarchy_y[level][i] == classes[j] ) {
				flag = true;
				break;
			}

		if( flag )
			selected_indices.push_back(i);
	}	
	cout << "classes: " << endl;
	for( int i = 0; i < bf.shape[0]; ++i )
		cout << classes[i] << endl;

	cout << endl;
	cout << "selected indices :) " << selected_indices.size() << endl;


	vector<bool> is_in_it(this->metadata[level-1].size, false);
	vector<int> indices_next_level;
	vector<int> labels;
	map<int, int> mapper;
	cout << "hello 1" << endl;
	for( int i = 0; i < this->metadata[level-1].size; ++i ) {
		int landmark = this->metadata[level-1].indices[i];

		for( int j = 0; j < selected_indices.size(); ++j )
			if( landmark == selected_indices[j] && !is_in_it[i] ) {
				labels.push_back(this->hierarchy_y[level-1][i]);
				indices_next_level.push_back(i);
				mapper[i] = indices_next_level.size()-1;
				is_in_it[i] = true;
				break;
			}
	}
	this->labels_selected = labels;
	cout << "hello 2" << endl;
	this->influence_selected = this->get_influence_by_indices(level-1, indices_next_level);
	this->indices_selected = indices_next_level;
	cout << "hello 3" << endl;
	if( this->hierarchy_X[level-1].is_sparse() ) {
		umap::Matrix X = this->hierarchy_X[level-1];
		vector<utils::SparseData> new_X;

		for( int i = 0; i < indices_next_level.size(); ++i ) {

			utils::SparseData sd = X.sparse_matrix[indices_next_level[i]];
			vector<float> data = sd.data;
			vector<int> indices = sd.indices;

			vector<float> new_data;
			vector<int> new_indices;

			for( int j = 0; j < indices.size(); ++j ) {

				if( mapper.count(indices[j]) > 0 ) {
					new_data.push_back(data[j]);
					new_indices.push_back(mapper[indices[j]]);
				}
			}

			new_X.push_back(utils::SparseData(new_data, new_indices));
		}

		auto manipulation = clock::now();

		Eigen::SparseMatrix<float, Eigen::RowMajor> graph = this->reducers[level-1].get_graph();
		Eigen::SparseMatrix<float, Eigen::RowMajor> new_graph(indices_next_level.size(), indices_next_level.size());
		new_graph.reserve(Eigen::VectorXi::Constant(indices_next_level.size(), this->n_neighbors*2+5));

		for( int i = 0; i < indices_next_level.size(); ++i ) {
			// cout<<i<<endl;
			int k = indices_next_level[i];
			for( Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
				if( mapper.count(it.col()) >0 ) {
					new_graph.insert(i, mapper[it.col()]) = it.value();
				
				}
			}

		}
		new_graph.makeCompressed();
		sec manipulation_duration = clock::now() - manipulation;
		cout << "duration of manipulation " << manipulation_duration.count() << endl;

		umap::Matrix nX = umap::Matrix(new_X, indices_next_level.size());

		// auto projection = clock::now();
		// umap::UMAP reducer = umap::UMAP("precomputed", min_neighbors, this->knn_algorithm);
		// reducer.fit_hierarchy(nX);
		// sec projection_duration = clock::now() - projection;
		// cout << "duration of projection " << projection_duration.count() << endl;

		return py::cast(this->embed_data(level-1, new_graph, nX));
	} else {


		umap::Matrix X = this->hierarchy_X[level-1];
		vector<vector<float>> new_X;
		cout << "passei 1" << endl;
		for( int i = 0; i < indices_next_level.size(); ++i ) {

			vector<float> dd = X.dense_matrix[indices_next_level[i]];
			new_X.push_back(dd);

			// vector<float> data = sd.data;
			// vector<int> indices = sd.indices;

			// vector<float> new_data;
			// vector<int> new_indices;

			// for( int j = 0; j < indices.size(); ++j ) {

			// 	if( mapper.count(indices[j]) > 0 ) {
			// 		new_data.push_back(data[j]);
			// 		new_indices.push_back(mapper[indices[j]]);
			// 	}
			// }

			// new_X.push_back(utils::SparseData(new_data, new_indices));
		}
		cout << "passei 2" << endl;
		auto manipulation = clock::now();

		Eigen::SparseMatrix<float, Eigen::RowMajor> graph = this->reducers[level-1].get_graph();
		Eigen::SparseMatrix<float, Eigen::RowMajor> new_graph(indices_next_level.size(), indices_next_level.size());
		new_graph.reserve(Eigen::VectorXi::Constant(indices_next_level.size(), this->n_neighbors*2+5));
		cout << "passei 3" << endl;
		for( int i = 0; i < indices_next_level.size(); ++i ) {
			// cout<<i<<endl;
			int k = indices_next_level[i];
			for( Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
				if( mapper.count(it.col()) >0 ) {
					new_graph.insert(i, mapper[it.col()]) = it.value();
				
				}
			}

		}
		new_graph.makeCompressed();
		cout << "passei 4" << endl;
		sec manipulation_duration = clock::now() - manipulation;
		cout << "duration of manipulation " << manipulation_duration.count() << endl;

		umap::Matrix nX = umap::Matrix(new_X);
		cout << "passei 5" << endl;
		// auto projection = clock::now();
		// umap::UMAP reducer = umap::UMAP("precomputed", min_neighbors, this->knn_algorithm);
		// reducer.fit_hierarchy(nX);
		// sec projection_duration = clock::now() - projection;
		// cout << "duration of projection " << projection_duration.count() << endl;

		return py::cast(this->embed_data(level-1, new_graph, nX));




	}
	vector<vector<float>> vec;
	return py::cast(vec);
}