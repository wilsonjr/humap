#include "hierarchical_umap.h"

namespace py = pybind11;
using namespace std;


vector<utils::SparseData> humap::create_sparse(int n, const vector<int>& rows, const vector<int>& cols, const vector<double>& vals)
{
	vector<utils::SparseData> sparse(n, utils::SparseData());

	for( int i = 0; i < rows.size(); ++i )
		sparse[rows[i]].push(cols[i], vals[i]);


	return sparse;
}

vector<vector<double>> humap::convert_to_vector(const py::array_t<double>& v)
{
	py::buffer_info bf = v.request();
	double* ptr = (double*) bf.ptr;

	vector<vector<double>> vec(bf.shape[0], vector<double>(bf.shape[1], 0.0));
	for (int i = 0; i < vec.size(); ++i)
	{
		for (int j = 0; j < vec[0].size(); ++j)
		{
			vec[i][j] = ptr[i*vec[0].size() + j];
		}
	}

	return vec;
}


std::map<std::string, double> humap::convert_dict_to_map(py::dict dictionary) 
{
	std::map<std::string, double> result;

	for( std::pair<py::handle, py::handle> item: dictionary ) {

		std::string key = item.first.cast<std::string>();
		double value = item.second.cast<double>();

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


int humap::HierarchicalUMAP::dfs(int u, int n_neighbors, bool* visited, vector<int>& cols, const vector<double>& sigmas,
			   vector<double>& strength, vector<int>& owners, vector<int>& is_landmark)
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
				if( neighbors ) {
					free(neighbors);
					neighbors = 0;
				}
				return landmark;
			}
		}
	}


	if( neighbors ) {
		free(neighbors);
		neighbors = 0;
	}
	return -1;


}

int humap::HierarchicalUMAP::depth_first_search(int n_neighbors, int* neighbors, vector<int>& cols, const vector<double>& sigmas,
							  vector<double>& strength, vector<int>& owners, vector<int>& is_landmark)
{
	bool* visited = new bool[sigmas.size()*sizeof(bool)];
	fill(visited, visited+sigmas.size(), false);
	int* neighbors_search = new int[n_neighbors*sizeof(int)];

	for( int i = 1; i < n_neighbors; ++i ) {

		int neighbor = *(neighbors + i);
		int landmark = -1;

		if( !*(visited + neighbor) ) {
			

			stack<int> p;
			p.push(neighbor);
			while( !p.empty() ) {
				int u = p.top();
				p.pop();

				visited[u] = true;

				if( is_landmark[u] != -1 ) {
					landmark = u;
					break;
				} else if( owners[u] != -1 ) {
					landmark = owners[u];
					break;
				}

				for( int j = 0; j < n_neighbors; ++j ) 
					neighbors_search[j] = cols[u*n_neighbors + j];

				for( int j = 1; j < n_neighbors; ++j ) {
					int v = neighbors_search[j];
					if( !visited[v] )
						p.push(v);
				}
			}





			// int landmark = this->dfs(neighbor, n_neighbors, visited, cols, sigmas, strength, owners, is_landmark);
			// if( landmark != -1 )  
				// return landmark;
		}

		if( landmark != -1 ) {
			free(neighbors);
			neighbors = 0;
			return landmark;
		}

	}

	return -1;
}

void humap::HierarchicalUMAP::associate_to_landmarks(int n, int n_neighbors, vector<int>& landmarks, vector<int>& cols, 
	vector<double>& strength, vector<int>& owners, vector<int>& indices, 


	vector<vector<int>>& association, vector<int>& is_landmark, 

	vector<int>& count_influence, vector<vector<double>>& knn_dists )
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

	if( neighbors ) {
		free(neighbors);
		neighbors = 0;
	}

}


void humap::HierarchicalUMAP::associate_to_landmarks(int n, int n_neighbors, int* indices, vector<int>& cols, const vector<double>& sigmas,
								   vector<double>& strength, vector<int>& owners, vector<int>& indices_landmark, vector<vector<int>>& association, vector<int>& count_influence, vector<int>& is_landmark, 
								   Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, vector<vector<double>>& knn_dists)
{
	int* neighbors = new int[n_neighbors*sizeof(int)];
	int count_search = 0;
	for( int i = 0; i < n; ++i ) {

		int index = indices[i];
		 if( is_landmark[index] != -1 ||  owners[index] != -1 )
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
	if( neighbors ) {
		free(neighbors);
		neighbors = 0;
	}
}

void humap::HierarchicalUMAP::add_similarity(int index, int i, int n_neighbors, vector<int>& cols,  
					Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, vector<vector<double>>& knn_dists,
					std::vector<std::vector<double> >& membership_strength, std::vector<std::vector<int> >& indices,
					std::vector<std::vector<double> >& distance, int* mapper, 
					double* elements, vector<vector<int>>& indices_nzeros, int n)
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
			double ms2 = graph.coeffRef(i, neighbor);
			double d2 = knn_dists[i][j+1];//[i*n_neighbors + (j+1)];
			int ind2 = i;
				
			for( int count = 0; count < membership_strength[neighbor].size(); ++count ) {

				double ms1 = membership_strength[neighbor][count];
				double d1 = distance[neighbor][count];
				int ind1 = indices[neighbor][count];

				// cout << "strenghts: " << ms1 << " " << ms2 << endl;
				

				// double s = (std::min(ms1*d1, ms2*d2)/std::max(ms1*d1, ms2*d2))/(n_neighbors-1);
				double s = (std::min(ms1*d1, ms2*d2)/std::max(ms1*d1, ms2*d2))/(n_neighbors-1);

				// double s = ((std::min(ms1, ms2)/std::max(ms1,ms2))/(n_neighbors-1) + 
				// 		  (std::min(d1, d2)/std::max(d1,d2))/(n_neighbors-1))/2.0;
				// double s = 1/(n_neighbors-1);
				// double s = (std::min(d1, d2)/std::max(d1, d2))/(n_neighbors-1);

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

void humap::HierarchicalUMAP::add_similarity2(int index, int i, int n_neighbors, vector<int>& cols, vector<double>& vals,
					Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, vector<vector<double>>& knn_dists,
					std::vector<std::vector<double> >& membership_strength, std::vector<std::vector<int> >& indices,
					std::vector<std::vector<double> >& distance, int* mapper, 
					double* elements, vector<vector<int>>& indices_nzeros, int n)
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
			double ms2 = graph.coeffRef(i, neighbor);
			double d2 = knn_dists[i][j+1];//[i*n_neighbors + (j+1)];
			int ind2 = i;
			double val2 = vals[i*n_neighbors + (j+1)];
				
			for( int count = 0; count < membership_strength[neighbor].size(); ++count ) {

				double ms1 = membership_strength[neighbor][count];
				double d1 = distance[neighbor][count];
				int ind1 = indices[neighbor][count];
				double val1 = vals[neighbor*n_neighbors + count];


				// double s = std::min(d1, d2) / std::max(d1, d2);
				double s = d1 + d2;



				// test as map
				if( *(mapper + ind1) != -1 ) {

					int u = *(mapper + ind1);
					int v = *(mapper + ind2);


					// *(elements + u*n + v) += s;
					// *(elements + v*n + u) += s;

					*(elements + u*n + v) = min(s, *(elements + u*n + v));
					*(elements + v*n + u) = min(s, *(elements + v*n + u));

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

void humap::HierarchicalUMAP::add_similarity3(int index, int i, vector<vector<int>>& neighborhood, 
											  std::vector<std::vector<int> >& indices,
											  int* mapper, double* elements, vector<vector<int>>& indices_nzeros, int n, 
											  double max_incidence, vector<vector<int>>& association)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	// cout << "i: " << i << ", neighborhood.size()" << neighborhood.size() << endl;

	std::vector<int> neighbors = neighborhood[index];
	// cout << "add_similarity3 1" << endl;

	//#pragma omp parallel for default(shared) schedule(dynamic, 50)
	// #pragma omp parallel for default(shared) schedule(dynamic, 100)
	for( int j = 0; j < neighbors.size(); ++j ) {
		// cout << "falha 6" << endl;
		int neighbor = neighbors[j];

		if( indices[neighbor].size() == 0 ) {
			// cout << "add_similarity3 2" << endl;
			// distance[neighbor].push_back(rw_distances[index][j]);
			// cout << "add_similarity3 3" << endl;
			indices[neighbor].push_back(i);
			// cout << "add_similarity3 4" << endl;

		} else {
			// cout << "add_similarity3 5" << endl;
			// double d2 = rw_distances[index][j];
			int ind2 = i;
			// cout << "add_similarity3 7" << endl;
				
			for( int count = 0; count < indices[neighbor].size(); ++count ) {
				// cout << "add_similarity3 8" << endl;
				// double d1 = distance[neighbor][count];
				int ind1 = indices[neighbor][count];
				
				// cout << "add_similarity3 9" << endl;
				// double s = std::min(d1, d2)/std::max(d1, d2) / max_incidence;
				//double s = d1+d2;
				// 
			
				


				// cout << "olha esse s: " << s << endl;
				// cout << "add_similarity3 10" << endl;
				// test as map
				if( *(mapper + ind1) != -1 ) {
					// cout << "add_similarity3 11" << endl;

					int u = *(mapper + ind1);
					int v = *(mapper + ind2);

				// 		cout << "neighbor: " << neighbor << endl;
				// cout << "ind1: " << ind1 << endl;
				// cout << "ind2: " << ind2 << endl;
				// cout << "association.shape: " << association.size() << " x " << association[0].size() << endl;

					//double s = (1.0 / max_incidence);

					double s = 0.0;
					if( this->distance_similarity ) {
						s = (std::min(association[u][neighbor], association[v][neighbor])/std::max(association[u][neighbor], association[v][neighbor]))/max_incidence;
					} else {
						s = (1.0 / max_incidence);
					}
						

					// cout << "ind1: " << ind1 << endl;
					// cout << "ind2: " << ind2 << endl;
					// cout << "u: " << u << endl;
					// cout << "v: " << v << endl;

					// *(elements + u*n + v) -= eps;
					// *(elements + v*n + u) -= eps;

					// cout << "add_similarity3 12" << endl;
					// cout << "u: " << u << endl;
					// cout << "v: " << v << endl;
					// cout << "n: " << n << endl;
					// cout << "n*n: " << (n*n) << endl;

					*(elements + u*n + v) += s;
					*(elements + v*n + u) += s;


					// cout << "add_similarity3 13" << endl;
					// *(elements + u*n + v) = min(s, *(elements + u*n + v));
					// *(elements + v*n + u) = min(s, *(elements + v*n + u));

					indices_nzeros[u].push_back(v);
					indices_nzeros[v].push_back(u);
					// cout << "add_similarity3 14" << endl;

				}
			}
			// cout << "add_similarity3 15" << endl;
			// distance[neighbor].push_back(d2);
			indices[neighbor].push_back(ind2);
			// cout << "add_similarity3 16" << endl;
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

humap::SparseComponents humap::HierarchicalUMAP::create_sparse(int n, int n_neighbors, double* elements, vector<vector<int>>& indices_nzeros)
{

	vector<int> cols;
	vector<int> rows;
	vector<double> vals;

	int* current = new int[n*sizeof(int)];
	fill(current, current+n, 0);
	double max_found = -1.0;		
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
					flag = false;
				vals.push_back(1.0 - *(elements + i*n + index));			
				max_found = max(max_found, 1.0 - *(elements + i*n + index));
			}
		}

 		for( int j = 0; j < n_neighbors+5; ++j ) {
			if( *(elements + i*n + j) == 0.0 && i != j) {				
				rows.push_back(i);
				cols.push_back(j);
				vals.push_back(1.0);
				// vals.push_back(-1.0);
			} 
		}

		for( int j = 0; j < indices_nzeros[i].size(); ++j ){
			*(current + indices_nzeros[i][j]) = 0;
		}
	 	if( flag ) {
		 	rows.push_back(i);
			cols.push_back(i);
			vals.push_back(0);
	 	}
	}


	return humap::SparseComponents(rows, cols, vals);
}

humap::SparseComponents humap::HierarchicalUMAP::create_sparse2(int level, int n, int n_neighbors, double* elements, vector<vector<int>>& indices_nzeros)
{

	vector<int> cols;
	vector<int> rows;
	vector<double> vals;

	double eps = 0.0001;
	double base_value = eps*(n_neighbors-1);
	base_value = 1;
	int* current = new int[n*sizeof(int)];
	fill(current, current+n, 0);

	for( int i = 0; i < n; ++i ) {
		bool flag = true;
		for( int j = 0; j < indices_nzeros[i].size(); ++j ) {
			int index = indices_nzeros[i][j];

			if( *(current + index) )
				continue;

			*(current + index) = 1;
			if( *(elements + i*n + index) != base_value ) {				
				rows.push_back(i);
				cols.push_back(index);
				if( i == index )
					flag = false;

				int u = this->original_indices[level][i];
				int v = this->original_indices[level][index];
				double d = sqrt(utils::rdist(this->hierarchy_X[0].get_row(u), this->hierarchy_X[0].get_row(v)));
				// double d = sqrt(utils::rdist(this->dense_backup[level].get_row(i), this->dense_backup[level].get_row(index)));

				// vals.push_back(d);
				// vals.push_back(*(elements + i*n + index));			
				vals.push_back(d/ *(elements + i*n + index));
			}
		}






 		for( int j = 0; j < n_neighbors+5; ++j ) {
			if( *(elements + i*n + j) == base_value && i != j) {	
				int u = this->original_indices[level][i];
				int v = this->original_indices[level][j];			
				double d = sqrt(utils::rdist(this->hierarchy_X[0].get_row(u), this->hierarchy_X[0].get_row(v)));
				rows.push_back(i);
				cols.push_back(j);

				// double d = sqrt(utils::rdist(this->dense_backup[level].get_row(i), this->dense_backup[level].get_row(j)));


				// vals.push_back(d);
				vals.push_back(d/ *(elements + i*n + j));
				//vals.push_back(numeric_limits<double>::max());
				//vals.push_back(1);
			} 
		}

		for( int j = 0; j < indices_nzeros[i].size(); ++j ){
			*(current + indices_nzeros[i][j]) = 0;
		}
		 if( flag ) {
		 		rows.push_back(i);
				cols.push_back(i);
				vals.push_back(0);
		 }
	}

	return humap::SparseComponents(rows, cols, vals);
}

humap::SparseComponents humap::HierarchicalUMAP::create_sparse3(int n, int n_neighbors, double* elements, vector<vector<int>>& indices_nzeros)
{

	vector<int> cols;
	vector<int> rows;
	vector<double> vals;

	int* current = new int[n*sizeof(int)];
	fill(current, current+n, 0);
		double max_found = -1.0;

	for( int i = 0; i < n; ++i ) {
		bool flag = true;

		for( int j = 0; j < indices_nzeros[i].size(); ++j ) {
			int index = indices_nzeros[i][j];

			if( *(current + index) )
				continue;

			*(current + index) = 1;
			if( *(elements + i*n + index) != numeric_limits<double>::max() ) {				
				rows.push_back(i);
				cols.push_back(index);
				if( i == index )
					flag = false;
				max_found = max(max_found, *(elements + i*n + index));
				vals.push_back(*(elements + i*n + index));			
			}
		}

 		for( int j = 0; j < n_neighbors+5; ++j ) {
			if( *(elements + i*n + j) == numeric_limits<double>::max() && i != j) {				
				rows.push_back(i);
				cols.push_back(j);
				vals.push_back(numeric_limits<double>::max());
			} 
		}

		for( int j = 0; j < indices_nzeros[i].size(); ++j ){
			*(current + indices_nzeros[i][j]) = 0;
		}

	 	if( flag ) {
		 	rows.push_back(i);
			cols.push_back(i);
			vals.push_back(0);
	 	}
	}

	for( int i = 0; i < vals.size(); ++i )
		if( vals[i] == -1.0 ) 
			vals[i] = max_found;

	return humap::SparseComponents(rows, cols, vals);
}

humap::SparseComponents humap::HierarchicalUMAP::sparse_similarity(int level, int n, int n_neighbors, vector<int>& greatest,  
																vector<vector<int>>& neighborhood, 
																double max_incidence, vector<vector<int>>& association) 
{

	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	std::vector<std::vector<int> > indices_sim;

	int* mapper = new int[n * sizeof(int)];
	fill(mapper, mapper+n, -1);

	for( int i = 0; i < greatest.size(); ++i )
		mapper[greatest[i]] = i;


	for( int i = 0; i < n; ++i ) {
		indices_sim.push_back(std::vector<int>());
	}
	// cout << "sparse_similarity 1" << endl;
	double* elements = new double[greatest.size()*greatest.size()*sizeof(double)];
	fill(elements, elements+greatest.size()*greatest.size(), 0);

	// cout << "sparse_similarity 2" << endl;
	int* non_zeros = new int[greatest.size() * sizeof(int)];
	fill(non_zeros, non_zeros+greatest.size(), 0);

	vector<vector<int>> indices_nzeros(greatest.size(), vector<int>());
	// cout << "sparse_similarity 3" << endl;

	for( int i = 0; i < greatest.size(); ++i ) {
		this->add_similarity3(i, greatest[i], neighborhood, indices_sim,
							  mapper, elements, indices_nzeros, greatest.size(), max_incidence, association);
	}

	cout << "creating sparse" << endl;
	auto begin = clock::now();
 	humap::SparseComponents sc = this->create_sparse(greatest.size(), n_neighbors, elements, indices_nzeros);
 	// humap::SparseComponents sc = this->create_sparse3(greatest.size(), n_neighbors, elements, indices_nzeros);
	sec duration = clock::now() - begin;
	cout << "Time for sparse components: " << duration.count() << endl;


	if( elements ) {
		free(elements);
		elements = 0;
	}
	if( non_zeros ) {

		free(non_zeros);
		non_zeros = 0;
	}
	if( mapper ) {
		free(mapper);
		mapper = 0;
	}

	return sc;
}

humap::SparseComponents humap::HierarchicalUMAP::sparse_similarity(int level, int n, int n_neighbors, 
																	vector<int>& greatest, vector<int> &cols, 
																	vector<double>& vals,
																   Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, 
																   vector<vector<double>>& knn_dists) 
{

	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	std::vector<std::vector<int> > indices_sim;
	std::vector<std::vector<double> > membership_strength;
	std::vector<std::vector<double> > distance_sim;

	int* mapper = new int[n * sizeof(int)];
	fill(mapper, mapper+n, -1);

	for( int i = 0; i < greatest.size(); ++i )
		mapper[greatest[i]] = i;


	for( int i = 0; i < n; ++i ) {
		indices_sim.push_back(std::vector<int>());
		membership_strength.push_back(std::vector<double>());
		distance_sim.push_back(std::vector<double>());
	}

	double eps = 0.0001;
	double base_value = eps*(n_neighbors-1);
	base_value = 1;
	double* elements = new double[greatest.size()*greatest.size()*sizeof(double)];
	// fill(elements, elements+greatest.size()*greatest.size(), 0);
	fill(elements, elements+greatest.size()*greatest.size(), base_value);


	int* non_zeros = new int[greatest.size() * sizeof(int)];
	fill(non_zeros, non_zeros+greatest.size(), 0);

	vector<vector<int>> indices_nzeros(greatest.size(), vector<int>());

	

	for( int i = 0; i < greatest.size(); ++i ) {


		 add_similarity2(i, greatest[i], n_neighbors, cols, vals, graph, knn_dists, membership_strength, 
		 	indices_sim, distance_sim, mapper, elements, indices_nzeros, greatest.size());
		// add_similarity(i, greatest[i], n_neighbors, cols, graph, knn_dists, membership_strength, 
			// indices_sim, distance_sim, mapper, elements, indices_nzeros, greatest.size());

		// data_dict[std::to_string(i)+'_'+std::to_string(i)] = 1.0;

		
	}

	cout << "creating sparse" << endl;
	auto begin = clock::now();
	 humap::SparseComponents sc = this->create_sparse2(level, greatest.size(), n_neighbors, elements, indices_nzeros);
 	// humap::SparseComponents sc = this->create_sparse(greatest.size(), n_neighbors, elements, indices_nzeros);
	sec duration = clock::now() - begin;
	cout << "Time for sparse components: " << duration.count() << endl;


	if( elements ) {
		free(elements);
		elements = 0;
	}
	if( non_zeros ) {

		free(non_zeros);
		non_zeros = 0;
	}
	if( mapper ) {
		free(mapper);
		mapper = 0;
	}

	return sc;
}





vector<double> humap::HierarchicalUMAP::update_position(int i, int n_neighbors, vector<int>& cols, vector<double>& vals,
													   umap::Matrix& X, Eigen::SparseMatrix<double, Eigen::RowMajor>& graph)
{
	std::vector<int> neighbors;

	for( int j = 0; j < n_neighbors; ++j ) {
		neighbors.push_back(cols[i*n_neighbors + j]);
	}


	vector<double> u = X.dense_matrix[i];

	vector<double> mean_change(X.shape(1), 0);
	for( int j = 0; j < neighbors.size(); ++j ) {
		int neighbor = neighbors[j];

		vector<double> v = X.dense_matrix[neighbor];

		vector<double> temp(v.size(), 0.0);
		for( int k = 0; k < temp.size(); ++k ) {
			temp[k] = graph.coeffRef(i, neighbor)*(v[k]-u[k]);
			// temp[k] = (v[k]-u[k]);
		}

		std::transform(mean_change.begin(), mean_change.end(), 
				  temp.begin(), mean_change.begin(), plus<double>());
	}

	std::transform(mean_change.begin(), mean_change.end(), mean_change.begin(), [n_neighbors](double& c){
		return c/(n_neighbors-1);
	});

	std::transform(u.begin(), u.end(), mean_change.begin(), u.begin(), plus<double>());

	return u;
}


vector<double> humap::HierarchicalUMAP::update_position(int i, vector<int>& neighbors, umap::Matrix& X)
{

	vector<double> u = X.dense_matrix[i];

	vector<double> mean_change(X.shape(1), 0);
	for( int j = 0; j < neighbors.size(); ++j ) {
		int neighbor = neighbors[j];

		vector<double> v = X.dense_matrix[neighbor];

		vector<double> temp(v.size(), 0.0);
		for( int k = 0; k < temp.size(); ++k ) {
			temp[k] = (v[k]-u[k]);
			// temp[k] = (v[k]-u[k]);
		}

		std::transform(mean_change.begin(), mean_change.end(), 
				  temp.begin(), mean_change.begin(), plus<double>());
	}
	int n_neighbors = (int) neighbors.size();
	std::transform(mean_change.begin(), mean_change.end(), mean_change.begin(), [n_neighbors](double& c){
		return c/(n_neighbors);
	});

	std::transform(u.begin(), u.end(), mean_change.begin(), u.begin(), plus<double>());

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

// TODO: salvar e verificar como não utilizar Eigen::SparseMatrix
int humap::random_walk(int vertex, int n_neighbors, vector<double>& vals, vector<int>& cols,
						int current_step, int walk_length, vector<int>& endpoint, std::uniform_real_distribution<double>& unif, 
						std::default_random_engine& rng, vector<double>& sum_vals, bool path_increment) 
{
	// cout << "random_walk: sai de " << vertex << endl;
	// cout << "------------------------------------" << endl;
	int begin_vertex = vertex;
	for( int step = 0; step < walk_length; ++step ) {
			// cout << "1" << endl;

		double c = unif(rng);

		// cout << "c_random_walk: " << c << endl;

		int next_vertex = vertex;
		// cout << "2" << endl;
		double incremental_prob = 0.0;

		// for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, vertex); it; ++it ) {
			
			// cout << "Sum_vals: " << sum_vals.size() << ", vertex: " << vertex << endl;
			// cout << "sum_vals: " << sum_vals[vertex] << ", it.value()/sum_vals: " << (it.value()/sum_vals[vertex]) << endl;

		// 	incremental_prob += (it.value()/sum_vals[vertex]);

		// 	if( c < incremental_prob ) {
		// 		next_vertex = it.col();
		// 		break;
		// 	}
		// }
		int mult = vertex*n_neighbors;
		for( int it = 0; it < n_neighbors; ++it ) {
			incremental_prob += (vals[mult + it]);///sum_vals[vertex]);
			if( c < incremental_prob ) {
				next_vertex = cols[mult + it];
				break;
			}
		}


		if( next_vertex == vertex ) {
			return -1;
		}		

		//if( path_increment && step < walk_length-1 )
		//	endpoint[next_vertex]++;
		vertex = next_vertex;

	}
	// if( vertex == begin_vertex )
	// 	return -1;


	return vertex;
}


vector<int> humap::markov_chain(vector<vector<int>>& knn_indices, 
								vector<double>& vals, vector<int>& cols, 
							 	int num_walks, int walk_length, vector<double>& sum_vals, bool path_increment) 
{	
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	auto begin_sampling = clock::now();

	vector<int> endpoint(knn_indices.size(), 0);

	// std::mt19937_64 rng;	
	// uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	// std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32)};	
	

	std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> unif(0.0, 1.0);

	#pragma omp parallel for// default(shared) 
	for( int i = 0; i < knn_indices.size(); ++i ) {

		// perform num_walks random walks for this vertex
		for( int walk = 0; walk < num_walks; ++walk ) {
			int vertex = humap::random_walk(i, knn_indices[0].size(), vals, cols, 0, walk_length, endpoint, unif, rng, sum_vals, path_increment);
			// #pragma omp critical(update_elements)
			// {
				if( vertex != -1 )
					endpoint[vertex]++;
			// }	
		}

	}

	sec end_sampling = clock::now() - begin_sampling;
	cout << "Sampling: " << end_sampling.count() << endl;

	return endpoint;
}


int humap::random_walk(int vertex, int n_neighbors, vector<double>& vals, vector<int>& cols, 
				Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
	   				  int current_step, int walk_length, uniform_real_distribution<double>& unif, 
					  mt19937& rng, vector<double>& sum_vals, vector<int>& is_landmark)
{

// cout << "random_walk 1" << endl;
	for( int step = 0;  step < walk_length; ++step ) {
		// cout << "random_walk 2" << endl;
		double c = unif(rng);
		int next_vertex = vertex;
		double incremental_prob = 0.0;
		// cout << "random_walk 3: " << sum_vals.size() << ", " << vertex  << endl;
		// for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, vertex); it; ++it ) {
			
		// 	incremental_prob += (it.value()/sum_vals[vertex]);

		// 	if( c < incremental_prob ) {
		// 		next_vertex = it.col();
		// 		break;
		// 	}
		// }
		int mult = vertex*n_neighbors;
		for( int it = 0; it < n_neighbors; ++it ) {
			incremental_prob += (vals[mult + it]);///sum_vals[vertex]);
			
			if( c < incremental_prob ) {
				next_vertex = cols[mult + it];
				break;
			}
		} 
		// cout << "random_walk 4: " << vertex << ", " << next_vertex << endl;
		
		if( next_vertex == vertex )
			return -1;
		// cout << "random_walk 5" << endl;
		if( is_landmark[next_vertex] != -1 )
			return next_vertex;
		// cout << "random_walk 6" << endl;
		vertex = next_vertex;
	}
	// cout << "random_walk 7" << endl;
	return -1;
}

bool humap::has(vector<int>& indices, int index)
{	
	bool flag = false;
#pragma omp parallel for
	for( int i = 0; i < indices.size(); ++i )
		if( indices[i] == index ) {
			flag = true;
			i += indices.size();
		}

	return flag;
}

int humap::markov_chain(vector<vector<int>>& knn_indices, 
	vector<vector<double>>& knn_dists, 
	vector<double>& vals, vector<int>& cols, Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
	int num_walks, int walk_length, vector<double>& sum_vals,
	vector<int>& landmarks, int influence_neighborhood, vector<vector<int>>& neighborhood, vector<vector<int>>& association)
{	

	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	auto begin_influence = clock::now();
	
	vector<int> is_landmark(knn_indices.size(), -1);
	for( int i = 0; i < landmarks.size(); ++i ) {
		is_landmark[landmarks[i]] = i;
	}
	
	neighborhood = vector<vector<int>>(landmarks.size(), vector<int>());
	association = vector<vector<int>>(landmarks.size(), vector<int>(knn_indices.size(), 0));
	

	std::mt19937& rng = RandomGenerator::Instance().get();
	std::uniform_real_distribution<double> unif(0.0, 1.0);
	int max_neighborhood = -1;


	cout << "Adding neighborhood of " << influence_neighborhood  << "neighbors."<< endl;
	auto begin_neighborhood = clock::now();
	if( influence_neighborhood > 1 ) {
		#pragma omp parallel for
		for( int i = 0; i < knn_indices.size(); ++i ) {
			if( is_landmark[i] != -1 )
				continue;

			for(int j = 1; j < influence_neighborhood; ++j ) {
				#pragma omp critical
				{
					if( is_landmark[knn_indices[i][j]] != -1 ) {

						int index = is_landmark[knn_indices[i][j]];
						// neighborhood[index][i] = 1;

						neighborhood[index].push_back(i);
						max_neighborhood = max(max_neighborhood, (int) neighborhood[index].size());

						association[index][i] = 1;

					}
				}
				
			}
		}
	}
	sec end_neighborhood = clock::now() - begin_neighborhood;


	cout << "Adding neighborhood: " << end_neighborhood.count() << endl;

	#pragma omp parallel for 
	for( int i = 0; i < is_landmark.size(); ++i ) {
		// cout << "i: " << i<< endl;
		 // cout << "Passei 1: " << i << endl;
	

		if( is_landmark[i] != -1 ) 
			continue;

		 // cout << "Passei 2: " << i << endl;
		for( int walk = 0; walk < num_walks; ++walk ) {
			// cout << "walk: " << walk << endl;
			 // cout << "Passei 3: " << i << endl;
			int vertex = humap::random_walk(i, knn_indices[0].size(), vals, cols, graph, 0, walk_length, unif, rng, sum_vals, is_landmark);
			 // cout << "Passei 4: " << i << " " << vertex <<  endl; 
			#pragma omp critical(update_information)
			{
				if(  vertex != -1 ) {				
					int index = is_landmark[vertex];
					if( !association[index][i] ) {
						neighborhood[index].push_back(i);
						max_neighborhood = max(max_neighborhood, (int) neighborhood[index].size());
					} 
					association[index][i]++;
				} 	
			}
		}
	}


	sec end_influence = clock::now() - begin_influence;
	cout << "Computed already: " << end_influence.count() << endl;
	// for( int i = 0; i < neighborhood.size(); ++i )
	// 	for( int j = 0; j < neighborhood[i].size(); ++j )
	// 		if( neighborhood[i][j] )
	// 			sparse_neighborhood[i].push_back(j);





	return max_neighborhood;

}


void humap::random_walk_explain(int vertex, vector<vector<int>>& knn_indices, 
								Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
								int current_step, int walk_length, 
								std::uniform_real_distribution<double>& unif,
								std::default_random_engine& rng, 
								vector<double>& sum_vals, 
								vector<vector<double>>& X,
								vector<vector<double>>& result)
{
	double count = 0.0;
	int original_vertex = vertex;
	vector<double> mean(result[0].size(), 0);
	vector<double> diff(result[0].size(), 0);
	vector<vector<double>> vertices;

	for( int step = 0; step < walk_length; ++step ) {

		double c = unif(rng);
		int next_vertex = vertex;
		double incremental_prob = 0.0;

		for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, vertex); it; ++it ) {

			incremental_prob += (it.value()/sum_vals[vertex]);
			if( c < incremental_prob ) {
				next_vertex = it.col();
				break;
			}
		}

		if( next_vertex == vertex )
			break;

		count += 1.0;

		std::transform(X[next_vertex].begin(), X[next_vertex].end(), X[original_vertex].begin(),
					   diff.begin(), [](double a, double b){ return std::fabs(a-b); });


		std::transform(mean.begin(), mean.end(), diff.begin(), mean.begin(), [](double a, double b){ return a+b; });
		vertices.push_back(diff);


		std::transform(X[next_vertex].begin(), X[next_vertex].end(), X[original_vertex].begin(),
					   result[next_vertex].begin(), [](double a, double b){ return std::min(a, b)/std::max(a,b); });

	}

	
	for( int i = 0; i < mean.size(); ++i )
		mean[i] /= count;
	// result[original_vertex] = mean;
	for( int i = 0; i < vertices.size(); ++i ) {
		for( int j = 0; j < vertices[i].size(); ++j )
			result[original_vertex][j] += (vertices[i][j] - mean[j])*(vertices[i][j] - mean[j]);
	}

	for( int i = 0; i < result[original_vertex].size(); ++i )
		result[original_vertex][i] = 1;//std::sqrt(result[original_vertex][i]/(count));
}

vector<vector<double>> humap::explain_neighborhoods(vector<vector<int>>& knn_indices, 
													Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
												    int num_walks, int walk_length, 
												    vector<double>& sum_vals, 
												    vector<vector<double>>& X)
{
	vector<vector<double>> results(X.size(), vector<double>(X[0].size(), 0.0));

	std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> unif(0.0, 1.0);

	#pragma omp parallel for 
	for( int i = 0; i < knn_indices.size(); ++i ) {
		cout << "explaining data point: " << i << endl; 
		for( int walk = 0; walk < num_walks; ++walk ) {
			humap::random_walk_explain(i, knn_indices, graph, 0, walk_length, unif, rng, sum_vals, X, results);
		}
	}		


	return results;
}

vector<vector<double>> humap::explain_neighborhoods(int index, vector<vector<int>>& knn_indices, 
													Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
												    int num_walks, int walk_length, 
												    vector<double>& sum_vals, 
												    vector<vector<double>>& X)
{
	vector<vector<double>> results(X.size(), vector<double>(X[0].size(), 0.0));

	std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> unif(0.0, 1.0);

	#pragma omp parallel for 
	for( int walk = 0; walk < num_walks; ++walk ) {
		humap::random_walk_explain(index, knn_indices, graph, 0, walk_length, unif, rng, sum_vals, X, results);
	}

	return results;
}


tuple<vector<vector<double>>, vector<double>> humap::explain_neighborhoods(int index, int max_hops, vector<vector<int>>& knn_indices, 
													Eigen::SparseMatrix<double, Eigen::RowMajor>& graph,
												    int num_walks, int walk_length, 
												    vector<double>& sum_vals, 
												    vector<vector<double>>& X)
{
	vector<vector<double>> results(X.size(), vector<double>(X[0].size(), 0.0));

	vector<bool> visited(knn_indices.size(), false);
	vector<bool> distance(knn_indices.size(), 0);

	queue<int> q;
	q.push(index);

	distance[index] = 0;
	double E = 0.0;


	while( !q.empty() ) {

		int u = q.front();
		q.pop();

		visited[u] = true;
		if( distance[u]+1 < max_hops )
			continue;

		for( int i = 0; i < knn_indices[u].size(); ++i ) {

			int v = knn_indices[u][i];
			if( !visited[v] ) {
				distance[v] = distance[u]+1;
				q.push(v);

				std::transform(X[index].begin(), X[index].end(), X[v].begin(), results[v].begin(), [](double a, double b){
					return fabs(a-b);
				});
				E += 1.0;
			}

		}
	}

	vector<double> max_f(X[0].size(), -1.0);
	for( int i = 0; i < results.size(); ++i ) {
		for( int j = 0; j < results[0].size(); ++j )
			max_f[j] = max(max_f[j], results[i][j]);
	}

	vector<double> importance(max_f.size(), 0.0);
	for( int i = 0; i < importance.size(); ++i ) {
		for( int j = 0; j < results.size(); ++j ) 
			importance[i] += (1 - results[j][i]/max_f[i]);
		importance[i] /= E;
	}  


	return make_tuple(results, importance);
}


tuple<py::array_t<double>, py::array_t<double>> humap::HierarchicalUMAP::explain(int n_walks, int walk_length, int max_hops, py::array_t<int> indices)
{
	if( this->reducers.size() == 0 ) 
		throw new runtime_error("Have you called 'fit'?");

	vector<int> to_explain((int*)indices.request().ptr, (int*)indices.request().ptr + indices.request().shape[0]);

	vector<vector<vector<double>>> explanations;
	vector<vector<double>> importances;

	for( int i = 0; i < to_explain.size(); ++i ) {

		int index = to_explain[i];

		vector<vector<double>> local_neighborhood;
		vector<double> importance;
		tie(local_neighborhood, importance) = explain_neighborhoods(index, max_hops, this->reducers[0].knn_indices(),
																		  this->reducers[0].transition_matrix,
																		  n_walks, walk_length, this->reducers[0].sum_vals,
																		  this->hierarchy_X[0].dense_matrix);

		explanations.push_back(local_neighborhood);
		importances.push_back(importance);
	}



	// vector<vector<double>> explanations = explain_neighborhoods(this->reducers[0].knn_indices(),
	// 														    this->reducers[0].transition_matrix,
	// 														    n_walks, walk_length, this->reducers[0].sum_vals, 
	// 														    this->hierarchy_X[0].dense_matrix);
	py::array_t<double> exp = py::cast(explanations);
	py::array_t<double> imp = py::cast(importances);

	return std::make_tuple(exp, imp);
}


void humap::HierarchicalUMAP::fit(py::array_t<double> X, py::array_t<int> y)
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
	this->dense_backup.push_back(first_level);
	this->hierarchy_y.push_back(vector<int>((int*)y.request().ptr, (int*)y.request().ptr + y.request().shape[0]));

	// if( this->knn_algorithm == "FAISS_IVFFlat" && first_level.size() < 10000 ) {
	// 	this->knn_algorithm = "FAISS_Flat";
	// }

	umap::UMAP reducer = umap::UMAP("euclidean", this->n_neighbors, this->min_dist, this->knn_algorithm);

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


	// double* strength = new double[sizeof(double)*this->hierarchy_X[0].size()];
	// fill(strength, strength+this->hierarchy_X[0].size(), -1.0);
	vector<double> strength(this->hierarchy_X[0].size(), -1.0);

	vector<vector<int>> association(this->hierarchy_X[0].size(), vector<int>());

	this->metadata.push_back(humap::Metadata(indices, owners, strength, association, this->hierarchy_X[0].size()));

	this->original_indices.push_back(indices);


	Eigen::SparseMatrix<double, Eigen::RowMajor> graph = this->reducers[0].get_graph();

	vector<vector<double>> knn_dists = this->reducers[0].knn_dists();

	for( int level = 0; level < this->percents.size(); ++level ) {
		// this->n_neighbors = (int) (1.5*this->n_neighbors);

		auto level_before = clock::now();

		int n_elements = (int) (this->percents[level] * this->hierarchy_X[level].size());
		



		// auto begin_sampling = clock::now();
		// vector<double> values(this->reducers[level].sigmas().size(), 0.0);
		// double mean_sigma = 0.0;
		// double sum_sigma = 0.0;
		// double max_sigma = -1;
		// int min_sigma_index = 0;
		// double min_sigma = this->reducers[level].sigmas()[0];
		// for( int i = 0; i < values.size(); ++i ) {
		// 	if( min_sigma > this->reducers[level].sigmas()[i] ) {
		// 		min_sigma = this->reducers[level].sigmas()[i];
		// 		min_sigma_index = i;
		// 	}
		// 	max_sigma = max(max_sigma, this->reducers[level].sigmas()[i]);
		// 	sum_sigma += fabs(this->reducers[level].sigmas()[i]);
		// 	mean_sigma += this->reducers[level].sigmas()[i];
 	// 	}
 	// 	mean_sigma /= (double)this->reducers[level].sigmas().size();
 	// 	double sdd = 0.0;
 	// 	for( int i = 0; i < values.size(); ++i ) {

 	// 		sdd += ((this->reducers[level].sigmas()[i] - mean_sigma)*(this->reducers[level].sigmas()[i] - mean_sigma));			

 	// 	}

 	// 	sdd = sqrt(sdd/((double)values.size()-1.0));
 	// 	sec end_sampling = clock::now() - begin_sampling;
 	// 	cout << "Sampled in " << end_sampling.count() << " seconds." << endl;

 		// auto begin_preparation = clock::now();
		// vector<double> probs(this->reducers[level].sigmas().size(), 0.0);
		// cout << "WARNING: using sigma" << endl; 
		// double t_min = -1;
		// double t_max = 1;

		// std::mt19937_64 rng;
	 //    // initialize the random number generator with time-dependent seed
	 //    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	 //    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
	 //    rng.seed(ss);
	 //    // initialize a uniform distribution between 0 and 1
	 //    std::uniform_real_distribution<double> unif(0, 1);
	 //    vector<int> land_indices;

		// for( int i = 0; i < values.size(); ++i ) {
		// 	// if( this->reducers[level].sigmas()[i] <= 0.0 )
		// 	// // 	cout << "ACHEI OLHA: " << this->reducers[level].sigmas()[i] << endl;
		// 	// if( i < 10 )
		// 	// 	cout << "sigma("<<i<<"): " << this->reducers[level].sigmas()[i] << endl; 
		// 	probs[i] = -this->reducers[level].sigmas()[i];
		// 	// probs[i] = this->reducers[level].sigmas()[i]/max_sigma;
		// 	// probs[i] = this->reducers[level].sigmas()[i]/sum_sigma;
			
		// 	// probs[i] = humap::sigmoid(this->reducers[level].sigmas()[i]);

		// 	// probs[i] = ((this->reducers[level].sigmas()[i] - mean_sigma)/sdd)*(-1.0);
		// 	// double m = this->reducers[level].sigmas()[i];

		// 	// probs[i] = ((m-min_sigma)/(max_sigma-min_sigma)) * (t_max - t_min) + t_min;
		// 	// probs[i] *= -1;


		// 	// if( i < 10 || probs[i] > 9 )
		// 	// 	cout << "sigma2("<<i<<"): " << probs[i] << endl; 
 	// 	}
 		// sec end_preparation = clock::now() - begin_preparation;
 		// cout << "Prepared in " << end_preparation.count() << " seconds." << endl;
 		// vector<int> selected_land(this->reducers[level].sigmas().size(), 0);
 		// int count_selected = 0;

 		// while( count_selected < n_elements ) {
 		// 	for( int i = 0; i < values.size() && count_selected < n_elements; ++i ) {
 		// 		if( selected_land[i] )
 		// 			continue;

 		// 		double currentRandomNumber = unif(rng);
			// 	cout << "currentNumber: " << currentRandomNumber << ", sigma: " << 1.0-(this->reducers[level].sigmas()[i]/max_sigma) << endl;
			// 	cout << "count_selected: " << count_selected << endl;
			// 	if( currentRandomNumber <= 1.0-((double)this->reducers[level].sigmas()[i]/(double)max_sigma)) {
			// 	// if( currentRandomNumber <= probs[i] ) {
			// 		land_indices.push_back(i);
			// 		selected_land[i] = 1;
			// 		count_selected++;
			// 	}
 		// 	}
 		// }

 		// cout << "Computing random walks" << endl;
 		auto begin_random_walk = clock::now();
 		vector<int> landmarks;
 		landmarks = humap::markov_chain(this->reducers[level].knn_indices(),
 										this->reducers[level].vals_transition, this->reducers[level].cols,
 													// this->reducers[level].transition_matrix, 
										this->landmarks_nwalks, this->landmarks_wl, 
										this->reducers[level].sum_vals, this->path_increment);
 		sec end_random_walk = clock::now() - begin_random_walk;
 		cout << "Landmark selection: " << end_random_walk.count() << endl;
 		vector<int> inds_lands;
 		
 		vector<int> sorted_landmarks = utils::argsort(landmarks, true);
 		
 		double glue_percent = this->percent_glue;
 		int n_random_elements = (int) n_elements*glue_percent;

 		int begin = 0;
 		for( int i = 0; i < n_elements - n_random_elements; ++i, ++begin )
 			inds_lands.push_back(sorted_landmarks[i]);
 		cout<< inds_lands.size() << " selected landmarks!" << endl << endl;
 		// if( n_random_elements > 0 ) {
 		// 	double max_land = *max_element(landmarks.begin()+begin, landmarks.end());
 		// 	vector<double> probs_land;
			// for( int i = begin; i < landmarks.size(); ++i ) {
 		// 		probs_land.push_back(-this->reducers[level].sigmas()[sorted_landmarks[i]]);
 		// 		// probs_land.push_back(landmarks[sorted_landmarks[i]]/max_land);
			// }

 		// 	humap::softmax(probs_land, probs_land.size());

 		// 	py::module np = py::module::import("numpy");
 		// 	py::object choice = np.attr("random");

 		// 	py::object indices_candidate = choice.attr("choice")(py::cast(probs_land.size()), py::cast(n_random_elements), py::cast(false), py::cast(probs_land));
 		// 	vector<int> possible_landmarks = indices_candidate.cast<vector<int>>();


 		// 	cout << "ADDING " << possible_landmarks.size() << " random landmarks." << endl;
 		// 	for( int i = 0; i < possible_landmarks.size(); ++i ) 
 		// 		inds_lands.push_back(sorted_landmarks[begin + possible_landmarks[i]]);
 		// } 
 		
 		// cout << "**************************************************************************" << endl << endl;
 		// cout << "NUMBER OF LANDMARKS USING RANDOM WALK: " << inds_lands.size() << ", " << n_elements << endl;
		

 		// cout << "NUMBER OF LANDMARKS: " << landmarks.size() << endl;

 		
 			
 		vector<vector<int>> neighborhood;

 		vector<vector<int>> association;
 		double max_incidence; //(double) (this->reducers[level].sigmas().size() - inds_lands.size());
 		auto influence_begin = clock::now();
 		max_incidence = humap::markov_chain(this->reducers[level].knn_indices(),
 																	  this->reducers[level].knn_dists(),
 																	  this->reducers[level].vals_transition,
 																	  this->reducers[level].cols,
 																	  this->reducers[level].transition_matrix,
 																	  this->influence_nwalks, this->influence_wl,  
 																	  this->reducers[level].sum_vals,
 																	  inds_lands, this->influence_neighborhood,
 																	  neighborhood, association);
 		sec influence_time = clock::now() - influence_begin;
 		cout << "Influence neighborhood: " << influence_time.count() << endl;

 		//max_incidence =(double) (this->reducers[level].sigmas().size() - inds_lands.size());

		// py::module np = py::module::import("numpy");
		// py::object choice = np.attr("random");
		// cout << "probs size: " << probs.size() << endl;
		// cout << "n_elements: " << n_elements << endl;
		// py::object indices_candidate = choice.attr("choice")(py::cast(probs.size()), py::cast(n_elements), py::cast(false),	py::cast(probs));
		// vector<int> possible_indices = indices_candidate.cast<vector<int>>();

  //  		vector<int> s_indices = utils::argsort(possible_indices);
  //  		for( int i = 1; i < s_indices.size(); ++i ) {
  //  			if( possible_indices[s_indices[i-1]] == possible_indices[s_indices[i]] ) {
  //  				cout << "OLHA ACHEI UM REPETIDO :)" << endl;
  //  			}
  //  		}

  //  		vector<int> g_indices = utils::argsort(this->reducers[level].sigmas());


		// vector<int> greatest = possible_indices;
		// vector<int> greatest = land_indices;
		// vector<int> greatest = landmarks;
		vector<int> greatest = inds_lands;

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

			vector<vector<double>> dense;


			for( int i = 0; i < greatest.size(); ++i ) {
					
				// vector<double> row = update_position(greatest[i], this->n_neighbors, this->reducers[level].cols, 
				// 	this->reducers[level].vals, this->hierarchy_X[level], this->reducers[level].transition_matrix);


				vector<double> row = this->update_position(i, neighborhood[i], this->hierarchy_X[level]);

				dense.push_back(row);
			}


			data = umap::Matrix(dense);
			reducer = umap::UMAP("euclidean", this->n_neighbors, this->min_dist, this->knn_algorithm);



		} else if( this->similarity_method == "precomputed" ) {	 
			// cout << "passei 1" << endl;
			// vector<vector<double>> dense;
			// cout << "passei 2" << endl;

			// for( int i = 0; i < greatest.size(); ++i ) {
			// 	// cout << "update_position " << i << endl;
			// 	vector<double> row = update_position(greatest[i], this->n_neighbors, this->reducers[level].cols, 
			// 		this->reducers[level].vals, this->dense_backup[level], this->reducers[level].transition_matrix);
			// 	// cout << "done " << i << endl;
			// 	dense.push_back(row);
			// }
			// cout << "hello to aqui "<< endl;

			// this->dense_backup.push_back(umap::Matrix(dense));


			// cout << "passei dense_backup" << endl;

			auto sparse_before = clock::now();
			// SparseComponents triplets = this->sparse_similarity(level+1, this->hierarchy_X[level].size(), this->n_neighbors,
			// 													 greatest, this->reducers[level].cols, this->reducers[level].vals,
			// 													 this->reducers[level].get_graph(), this->reducers[level].knn_dists());

			SparseComponents triplets = this->sparse_similarity(level+1, this->hierarchy_X[level].size(), this->n_neighbors,
																greatest, neighborhood, max_incidence, association); 


			sec sparse_duration = clock::now() - sparse_before;

			cout << "Sparse Matrix: " << sparse_duration.count() << endl;




			



			auto eigen_before = clock::now();
			// Eigen::SparseMatrix<double, Eigen::RowMajor> sparse = utils::create_sparse(triplets.rows, triplets.cols, triplets.vals,
			// 																		  n_elements, n_neighbors*4);


			vector<utils::SparseData> sparse = humap::create_sparse(n_elements, triplets.rows, triplets.cols, triplets.vals);
			sec eigen_duration = clock::now() - eigen_before;
			cout << "Constructing eigen matrix: " << eigen_duration.count() << endl;
			cout << endl;

			data = umap::Matrix(sparse, greatest.size());
			reducer = umap::UMAP("precomputed", this->n_neighbors, this->min_dist, this->knn_algorithm);
		} else {


			reducer = umap::UMAP("euclidean", this->n_neighbors, this->min_dist, this->knn_algorithm);


		}


		if( this->verbose )
			cout << "\n*****Fitting hierarchy level*****\n" << endl; 



		this->metadata[level].count_influence = vector<int>(greatest.size(), 0);

		auto fit_before = clock::now();
		reducer.fit_hierarchy(data);
		sec fit_duration = clock::now() - fit_before;
		cout << "Fitting level " << (level+1) << ": " << fit_duration.count() << endl;
		cout << endl;

		auto associate_before = clock::now();
		vector<int> is_landmark(this->metadata[level].size, -1);
		for( int i = 0; i < greatest.size(); ++i ) 
		{
			is_landmark[greatest[i]] = i;
		}

		this->associate_to_landmarks(greatest.size(), this->n_neighbors, greatest, this->reducers[level].cols, 
			this->metadata[level].strength, this->metadata[level].owners, this->metadata[level].indices, 
			this->metadata[level].association, is_landmark, this->metadata[level].count_influence, this->reducers[level].knn_dists());

		sec associate_duration = clock::now() - associate_before;
		cout << "Associate landmark: " << associate_duration.count() << endl;


		auto use_before = clock::now();
		int n = 0;
		for( int i = 0; i < this->metadata[level].size; ++i ) {
			if( this->metadata[level].owners[i] == -1 )
				n++;
		}

		int* indices_not_associated = new int[sizeof(int)*n];
		for( int i = 0, j = 0; i < this->metadata[level].size; ++i )
			if( this->metadata[level].owners[i] == -1.0 )
				*(indices_not_associated + j++) = i;

		this->associate_to_landmarks(n, this->n_neighbors, indices_not_associated, this->reducers[level].cols,
									  this->reducers[level].sigmas(), this->metadata[level].strength,
									  this->metadata[level].owners, this->metadata[level].indices, 
									  this->metadata[level].association, this->metadata[level].count_influence, 
									  is_landmark, this->reducers[level].get_graph(), this->reducers[level].knn_dists());

		sec use_duration = clock::now() - use_before;
		cout << "Associating remaining to landmarks: " << use_duration.count() << endl;
		// cout << endl;


		// cout << "I have " << this->metadata[level].size << " owners" << endl;
		// cout << "See the first 10: " << level << endl;
		// for( int i = 0; i < 10; ++i ) {
		// 	cout << i << " " << this->metadata[level].indices[i] << " (" << this->metadata[level].owners[i] << ") -> " << this->metadata[level].strength[i] << endl;


		// }
		cout << endl;

		if( this->verbose )
			cout << "Appending information for the next hierarchy level" << endl;

		vector<int> new_owners(greatest.size(), -1);
		vector<double> new_strength(greatest.size(), -1.0);
		vector<vector<int>> new_association(greatest.size(), vector<int>());

		this->metadata.push_back(Metadata(greatest, new_owners, new_strength, new_association, greatest.size()));//vector<int>(greatest.size(), -1), vector<double>(greatest.size(), -1.0)));
		this->reducers.push_back(reducer);
		this->hierarchy_X.push_back(data);
		this->hierarchy_y.push_back(utils::arrange_by_indices(this->hierarchy_y[level], greatest));


		// knn_dists = reducer.knn_dists();
		// graph = reducer.get_graph();

		sec level_duration = clock::now() - level_before;
		cout << "Level construction: " << level_duration.count() << endl;
		cout << endl;

		// this->n_neighbors = (int) (1.2 * this->n_neighbors);

	}




	sec hierarchy_duration = clock::now() - hierarchy_before;
	cout << endl;
	cout << "Hierarchy construction: " << hierarchy_duration.count() << endl;
	cout << endl;


	for( int i = 0; i < this->hierarchy_X.size(); ++i ) {
		this->embeddings.push_back(vector<vector<double>>());
	}


	// for( int i = 0; i < this->hierarchy_X.size(); ++i ) {
	// 	vector<vector<double>> result = this->embed_data(i, this->reducers[i].get_graph(), this->hierarchy_X[i]);
	// 	this->embeddings.push_back(result);
	// }
}

py::array_t<double> humap::HierarchicalUMAP::transform(int level) 
{
	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	vector<vector<double>> result = this->embed_data(level, this->reducers[level].get_graph(), this->hierarchy_X[level]);


	return py::cast(result);
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

py::array_t<double> humap::HierarchicalUMAP::get_sigmas(int level)
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

py::array_t<double> humap::HierarchicalUMAP::get_embedding(int level)
{
	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->embeddings[level]);
}	


Eigen::SparseMatrix<double, Eigen::RowMajor> humap::HierarchicalUMAP::get_data(int level)
{
	if( level == 0 )  
		throw new runtime_error("Sorry, we won't me able to return all dataset! Please, project using UMAP.");

	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return utils::create_sparse(this->hierarchy_X[level].sparse_matrix, this->hierarchy_X[level].size(), (int) this->n_neighbors*2.5);
}

vector<vector<double>> humap::HierarchicalUMAP::embed_data(int level, Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, umap::Matrix& X)
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
	double max_value = graph.coeffs().maxCoeff();
	graph = graph.pruned(max_value/(double)n_epochs, 1.0);


	// cout << "chguei aqui 5 " << endl;


	vector<vector<double>> embedding = this->reducers[level].spectral_layout(X, graph, this->n_components);


	// cout << "chguei aqui 6 " << endl;
	vector<int> rows, cols;
	vector<double> data;

	
	// cout << "chguei aqui 7 " << endl;
	tie(rows, cols, data) = utils::to_row_format(graph);

	// cout << "chguei aqui 8 " << endl;
	vector<double> epochs_per_sample = this->reducers[level].make_epochs_per_sample(data, this->n_epochs);
	// cout << "\n\nepochs_per_sample: " << epochs_per_sample.size() << endl;

	// cout << "chguei aqui 9 " << endl;
	// for( int j = 0; j < 20; ++j )
	// 	printf("%.4f ", epochs_per_sample[j]);

	// cout << endl << endl;



	// cout << "chguei aqui 10 " << endl;
	vector<double> min_vec, max_vec;
	// printf("min and max values:\n");
	for( int j = 0; j < this->n_components; ++j ) {

		min_vec.push_back((*min_element(embedding.begin(), embedding.end(), 
			[j](vector<double> a, vector<double> b) {							
				return a[j] < b[j];
			}))[j]);
		// cout <<" ****** 2" << endl;
		max_vec.push_back((*max_element(embedding.begin(), embedding.end(), 
			[j](vector<double> a, vector<double> b) {
				return a[j] < b[j];
			}))[j]);
		// cout <<" ****** 3" << endl;

	}
	// cout << "chguei aqui 11 " << endl;
	vector<double> max_minus_min(this->n_components, 0.0);
	std::transform(max_vec.begin(), max_vec.end(), min_vec.begin(), max_minus_min.begin(), [](double a, double b){ return a-b; });
	// cout << "chguei aqui 12 " << endl;

	for( int j = 0; j < embedding.size(); ++j ) {

		std::transform(embedding[j].begin(), embedding[j].end(), min_vec.begin(), embedding[j].begin(), 
			[](double a, double b) {
				return 10*(a-b);
			});

		std::transform(embedding[j].begin(), embedding[j].end(), max_minus_min.begin(), embedding[j].begin(),
			[](double a, double b) {
				return a/b;
			});
	}

	py::module scipy_random = py::module::import("numpy.random");
	py::object randomState = scipy_random.attr("RandomState")(this->random_state);

	py::object rngStateObj = randomState.attr("randint")(numeric_limits<int>::min(), numeric_limits<int>::max(), 3);
	vector<long> rng_state = rngStateObj.cast<vector<long>>();

	printf("Embedding a level with %d data samples\n", embedding.size());
	auto before = clock::now(); 
	vector<vector<double>> result = this->reducers[level].optimize_layout_euclidean(
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

py::array_t<double> humap::HierarchicalUMAP::project_indices(int level, py::array_t<int> indices)
{
	
	py::buffer_info bf = indices.request();
	int* inds = (int*) bf.ptr;

	vector<int> selected_indices(inds, inds+bf.shape[0]);
	cout << "Number of indices: " << selected_indices.size() << endl;

	return this->project_data(level, selected_indices);
}

py::array_t<double> humap::HierarchicalUMAP::project(int level, py::array_t<int> c)
{
	py::buffer_info bf = c.request();
	int* classes = (int*) bf.ptr;
	cout << "classes 1" << endl;

	vector<int> selected_indices;
	cout << "Passei " << this->hierarchy_y.size() << ", " << level << endl;
	for( int i = 0; i < this->hierarchy_y[level].size(); ++i ) {
		bool flag = false;
		cout << "passei 2>  " << bf.shape[0] << endl;
		for( int j = 0; j < bf.shape[0]; ++j ) {
			cout << "passei 3> " << this->hierarchy_y[level].size() << ", " <<  i << ", " << j << endl; 
			if( this->hierarchy_y[level][i] == classes[j] ) {
				flag = true;
				break;
			}
		}
		if( flag )
			selected_indices.push_back(i);
	}	
	cout << "classes: " <<  bf.shape[0] << endl;
	for( int i = 0; i < bf.shape[0]; ++i )
		cout << classes[i] << endl;
	cout << "passei" << endl;
	cout << endl;
	cout << "selected indices :) " << selected_indices.size() << endl;
	return this->project_data(level, selected_indices);
}

py::array_t<double> humap::HierarchicalUMAP::project_data(int level, vector<int> selected_indices)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;
	


	vector<bool> is_in_it(this->metadata[level-1].size, false);
	vector<int> indices_next_level;
	vector<int> labels;
	map<int, int> mapper;
	
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
	
	this->influence_selected = this->get_influence_by_indices(level-1, indices_next_level);
	this->indices_selected = indices_next_level;
	
	if( this->hierarchy_X[level-1].is_sparse() ) {
		umap::Matrix X = this->hierarchy_X[level-1];
		vector<utils::SparseData> new_X;
		int min_neighbors = 39993;
		for( int i = 0; i < indices_next_level.size(); ++i ) {

			utils::SparseData sd = X.sparse_matrix[indices_next_level[i]];
			vector<double> data = sd.data;
			vector<int> indices = sd.indices;

			vector<int> assigned(indices_next_level.size(), 0);


			vector<double> new_data;
			vector<int> new_indices;

			for( int j = 0; j < indices.size(); ++j ) {

				if( mapper.count(indices[j]) > 0 ) {
					new_data.push_back(data[j]);
					new_indices.push_back(mapper[indices[j]]);
					assigned[mapper[indices[j]]] = 1;
				}
			}

			for( int j = 0; j < assigned.size(); ++j ) {
				if( !assigned[j] ) {
					new_data.push_back(1.0);
					new_indices.push_back(j);
				}
			}


			min_neighbors = min(min_neighbors, (int)new_indices.size());
			new_X.push_back(utils::SparseData(new_data, new_indices));
		}

		auto manipulation = clock::now();

		Eigen::SparseMatrix<double, Eigen::RowMajor> graph = this->reducers[level-1].get_graph();
		Eigen::SparseMatrix<double, Eigen::RowMajor> new_graph(indices_next_level.size(), indices_next_level.size());
		new_graph.reserve(Eigen::VectorXi::Constant(indices_next_level.size(), this->n_neighbors*2+5));

		for( int i = 0; i < indices_next_level.size(); ++i ) {
			// cout<<i<<endl;
			int k = indices_next_level[i];
			for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
				if( mapper.count(it.col()) > 0 ) {
					new_graph.insert(i, mapper[it.col()]) = it.value();
				
				}
			}

		}
		new_graph.makeCompressed();
		sec manipulation_duration = clock::now() - manipulation;
		cout << "duration of manipulation " << manipulation_duration.count() << endl;

		umap::Matrix nX = umap::Matrix(new_X, indices_next_level.size());

		return py::cast(this->embed_data(level-1, new_graph, nX));

		// auto projection = clock::now();
		// cout << "NEIGHBORS: " << min_neighbors << endl;
		// // umap::UMAP reducer = umap::UMAP("precomputed", min_neighbors, this->min_dist, this->knn_algorithm);
		// //umap::UMAP reducer = umap::UMAP("precomputed", this->n_neighbors, this->min_dist, this->knn_algorithm);

		// umap::UMAP reducer = umap::UMAP("precomputed", this->n_neighbors, this->min_dist, this->knn_algorithm);
		// reducer.fit_hierarchy(nX);
		// sec projection_duration = clock::now() - projection;
		// cout << "duration of projection " << projection_duration.count() << endl;
		// return py::cast(this->embed_data(level-1, reducer.get_graph(), nX));

		
	} else {


		umap::Matrix X = this->hierarchy_X[level-1];
		vector<vector<double>> new_X;
		cout << "passei 1" << endl;
		for( int i = 0; i < indices_next_level.size(); ++i ) {

			vector<double> dd = X.dense_matrix[indices_next_level[i]];
			new_X.push_back(dd);

			// vector<double> data = sd.data;
			// vector<int> indices = sd.indices;

			// vector<double> new_data;
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

		pair<int,int> max_neighbor = *std::max_element(mapper.begin(), mapper.end(), [](const pair<int,int>& a, const pair<int, int>& b) {
			return a.second < b.second;
		});
		cout << "olha: " << max_neighbor.second << " "  << (this->n_neighbors*2+5) << endl;
		Eigen::SparseMatrix<double, Eigen::RowMajor> graph = this->reducers[level-1].get_graph();
		Eigen::SparseMatrix<double, Eigen::RowMajor> new_graph(indices_next_level.size(), indices_next_level.size());
		new_graph.reserve(Eigen::VectorXi::Constant(indices_next_level.size(), max_neighbor.second+5));
		cout << "passei 3" << endl;
		for( int i = 0; i < indices_next_level.size(); ++i ) {
			// cout<<i<<endl;
			int k = indices_next_level[i];
			for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
				if( mapper.count(it.col()) >0 ) {
					new_graph.insert(i, mapper[it.col()]) = it.value();
					if( i >= indices_next_level.size() )
						cout << "ATENCAO, i >= indices_next_level.size()" << endl;
					if( mapper[it.col()] >= max_neighbor.second+5 )
						cout << "ATENCAO, mapper[it.col()] >= this->n_neighbors*2+5" << endl;

				}	
			}

		}
		cout << "passei 3.5" << endl;
		new_graph.makeCompressed();
		cout << "passei 4" << endl;
		sec manipulation_duration = clock::now() - manipulation;
		cout << "duration of manipulation " << manipulation_duration.count() << endl;

		umap::Matrix nX = umap::Matrix(new_X);

		return py::cast(this->embed_data(level-1, new_graph, nX));


		// cout << "passei 5" << endl;
		// auto projection = clock::now();
		// umap::UMAP reducer = umap::UMAP("euclidean", 15, this->min_dist, this->knn_algorithm);
		// reducer.fit_hierarchy(nX);
		// sec projection_duration = clock::now() - projection;
		// cout << "duration of projection " << projection_duration.count() << endl;
		// return py::cast(this->embed_data(level-1, reducer.get_graph(), nX));
		




	}
	vector<vector<double>> vec;
	return py::cast(vec);
}