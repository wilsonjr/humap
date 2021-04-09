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


#include "hierarchical_umap.h"

namespace py = pybind11;
using namespace std;

/**
* Create a sparse representation 
*
* @param n int representing the dataset size
* @param rows Container representing the row indices 
* @param cols Container representing the column indices
* @param vals Container representing the non-zero values
* @return Container of SparseData
*/
vector<utils::SparseData> humap::create_sparse(int n, const vector<int>& rows, const vector<int>& cols, const vector<double>& vals)
{
	vector<utils::SparseData> sparse(n, utils::SparseData());

	for( int i = 0; i < rows.size(); ++i )
		sparse[rows[i]].push(cols[i], vals[i]);


	return sparse;
}


/**
* Convert py array to dense representation
* 
* @param v py::array_t containing the datataset
* @return Container with dense representation of the dataset
*/
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



/**
* Performs depth first search on a point neighborhood
*
* @param n_neighbors int representing the number of neighbors
* @param neighbors int* representing the neighbors of each data point
* @param cols Container representing the neighborhoods
* @param strength Container representing the strength of the edge
* @param owners Container the landmark owners
* @param is_landmark Container to specify whether a point is a landmark or not
* @return int representing the landmark
*/
int humap::HierarchicalUMAP::depth_first_search(int n_neighbors, int* neighbors, vector<int>& cols, vector<double>& strength, vector<int>& owners, vector<int>& is_landmark)
{
	bool* visited = new bool[is_landmark.size()*sizeof(bool)];
	fill(visited, visited+is_landmark.size(), false);
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
		}

		if( landmark != -1 ) {
			free(neighbors);
			neighbors = 0;
			return landmark;
		}

	}

	return -1;
}


/**
* Associate data points to landmarks
* @param n int representing the number of data points in the level
* @param n_neighbors int representing the number of neighbors
* @param landmarks Container representing the list of landmarks
* @param cols Container representing the neighborhoods
* @param strength Container representing the strength of the edge
* @param owners Container representing the landmark owners
* @param indices Container representing the indices of the level 
* @param association Container reepresenting the data point associated to a landmark
* @param is_landmark Container to specify whether a point is a landmark or not
* @param count_influence Container to store the number of data points influenced by each landmark
* @param knn_dists Container with the knn distances
*/
void humap::HierarchicalUMAP::associate_to_landmarks(int n, int n_neighbors, vector<int>& landmarks, vector<int>& cols, 
													 vector<double>& strength, vector<int>& owners, vector<int>& indices, 
													 vector<vector<int>>& association, vector<int>& is_landmark, 
													 vector<int>& count_influence, vector<vector<double>>& knn_dists)
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

/**
* Associate data points to landmarks
* @param n int representing the number of data points in the level
* @param n_neighbors int representing the number of neighbors
* @param indices int* representing the data points not associated with any landmark
* @param cols Container representing the neighborhoods
* @param strength Container representing the strength of the edge
* @param owners Container representing the landmark owners
* @param indices_landmark Container representing the landmark indices
* @param association Container representing the data point associated to a landmark
* @param count_influence Container to store the number of data points influenced by each landmark
* @param is_landmark Container to specify whether a point is a landmark or not
* @param knn_dists Container with the knn distances
*/
void humap::HierarchicalUMAP::associate_to_landmarks(int n, int n_neighbors, int* indices, vector<int>& cols, 
								   					 vector<double>& strength, vector<int>& owners, vector<int>& indices_landmark, 
								   					 vector<vector<int>>& association, vector<int>& count_influence, vector<int>& is_landmark, 
								   					 vector<vector<double>>& knn_dists)
{
	int* neighbors = new int[n_neighbors*sizeof(int)];
	int count_search = 0;
	for( int i = 0; i < n; ++i ) {

		int index = indices[i];
		 if( is_landmark[index] != -1 ||  owners[index] != -1 )
			continue;

		bool found = false;
		
		for( int j = 1; j < n_neighbors && !found; ++j ) {

			int nn = cols[index*n_neighbors + j];


			if( is_landmark[nn] != -1 ) {
				strength[index] = knn_dists[nn][j];
				owners[index] = nn;
				indices_landmark[index] = is_landmark[nn];
				association[index].push_back(is_landmark[nn]);
				count_influence[is_landmark[nn]]++;
				found = true;
			} else if( owners[nn] != -1 ) {
				// TODO: this is an estimative
				strength[index] = strength[nn]; 
				owners[index] = owners[nn];			
				indices_landmark[index] = is_landmark[owners[nn]];	
				association[index].push_back(is_landmark[owners[nn]]);
				count_influence[is_landmark[owners[nn]]]++;
				found = true;
			}
		}

		if( !found ) {

			count_search++;
			int landmark = this->depth_first_search(n_neighbors, neighbors, cols, strength, owners, is_landmark);
			if( landmark != -1 ) {
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

		
	}
	if( neighbors ) {
		free(neighbors);
		neighbors = 0;
	}
}

/**
* Increment the similarity in the data structure for a landmark 
*
* @param index int representing the iteration index
* @param index int representing the landmark index
* @param neighborhood Container representing the representation neighborhood of each landmark
* @param indices Container representing the indices of intersection
* @param mapper int* to map each indice to assoation Container
* @param elements double* to store the similarities
* @param indices_nzeros Container to store the location of non-zero elements
* @param n int representing the length of elements
* @param max_incidence double representing the greater number of neighbors in the representation neighborhood
* @param association Container representing the association between landmark and point
*/
void humap::HierarchicalUMAP::add_similarity(int index, int i, vector<vector<int>>& neighborhood, 
											  std::vector<std::vector<int> >& indices,
											  int* mapper, double* elements, vector<vector<int>>& indices_nzeros, int n, 
											  double max_incidence, vector<vector<int>>& association)
{
	std::vector<int> neighbors = neighborhood[index];

	//#pragma omp parallel for default(shared) schedule(dynamic, 50)
	// #pragma omp parallel for default(shared) schedule(dynamic, 100)
	for( int j = 0; j < neighbors.size(); ++j ) {
		int neighbor = neighbors[j];

		if( indices[neighbor].size() == 0 ) {
			indices[neighbor].push_back(i);
		} else {
			int ind2 = i;
				
			for( int count = 0; count < indices[neighbor].size(); ++count ) {
				int ind1 = indices[neighbor][count];

				if( *(mapper + ind1) != -1 ) {
					int u = *(mapper + ind1);
					int v = *(mapper + ind2);

				
					double s = 0.0;
					if( this->distance_similarity ) {
						s = (std::min(association[u][neighbor], association[v][neighbor])/std::max(association[u][neighbor], association[v][neighbor]))/max_incidence;
					} else {
						s = (1.0 / max_incidence);
					}						

					*(elements + u*n + v) += s;
					*(elements + v*n + u) += s;

					indices_nzeros[u].push_back(v);
					indices_nzeros[v].push_back(u);

				}
			}

			indices[neighbor].push_back(ind2);
		}
	}
}

/**
* Create a sparse representation from the similarity computed among landmarks
*
* @param n int representing the number of landmarks
* @param n_neighbors int representing the number of neighbors
* @param elements double* representing the similarity values
* @param indices_nzeros Container with indices of non-zero values for each matrix row
* @return SparseComponents with cols, rows, and non-zero values
*/
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


/**
* 
*
* @param level int representing the hierarchical level
* @param n int representing the number of landmarks
* @param n_neighbors int representing the number of neighbors
* @param greatest Container representing the landmarks
* @param neighborhood Container representing the neighborhood
* @param max_incidence double representing the maximum neighborhood
* @param association Container representing the representation neighborhood
* @return SparseComponents with sparse representation of similarity among landmarks
*/
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
	
	double* elements = new double[greatest.size()*greatest.size()*sizeof(double)];
	fill(elements, elements+greatest.size()*greatest.size(), 0);
	
	int* non_zeros = new int[greatest.size() * sizeof(int)];
	fill(non_zeros, non_zeros+greatest.size(), 0);

	vector<vector<int>> indices_nzeros(greatest.size(), vector<int>());

	for( int i = 0; i < greatest.size(); ++i ) {
		this->add_similarity(i, greatest[i], neighborhood, indices_sim,
							  mapper, elements, indices_nzeros, greatest.size(), max_incidence, association);
	}

 	humap::SparseComponents sc = this->create_sparse(greatest.size(), n_neighbors, elements, indices_nzeros);

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

/**
* Function to update a point position throughout hierarchy levels
* WARNING: In development
* 
* @param i int representing the landmark index
* @param neighbors Container representing the list of neighbors
* @param X Matrix representing the matrix
*/
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

/**
* Computes a random walk on the neighboring graph for sampling selection
*
* @param vertex int representing start point
* @param n_neighbors int representing the number of neighbors
* @param vals Container representing the graph strength
* @param cols Container representing the neighborhood
* @param walk_length int representing the max hops in the random walk
* @param unif uniform_real_distribution
* @param rng default_random_engine
* @return int representing the endpoint
*/
int humap::random_walk(int vertex, int n_neighbors, vector<double>& vals, vector<int>& cols,
					   int walk_length, std::uniform_real_distribution<double>& unif, 
					   std::default_random_engine& rng) 
{

	int begin_vertex = vertex;
	for( int step = 0; step < walk_length; ++step ) {
		double c = unif(rng);
		int next_vertex = vertex;
		double incremental_prob = 0.0;

		int mult = vertex*n_neighbors;
		for( int it = 0; it < n_neighbors; ++it ) {
			incremental_prob += (vals[mult + it]);
			if( c < incremental_prob ) {
				next_vertex = cols[mult + it];
				break;
			}
		}

		if( next_vertex == vertex ) {
			return -1;
		}		

		vertex = next_vertex;
	}

	return vertex;
}

/**
* Performs a markov chain in the neighborhood graph for sampling selection
*
* @param knn_indices Container representing the neighborhood graph
* @param vals Container representing the graph strength
* @param cols Container representing the neighborhood
* @param num_walks int representing the number of random walks
* @param walk_length int representing the walk length
* @return Container representing how many times each landmark was the endpoint
*/
vector<int> humap::markov_chain(vector<vector<int>>& knn_indices, 
								vector<double>& vals, vector<int>& cols, 
							 	int num_walks, int walk_length) 
{	
	vector<int> endpoint(knn_indices.size(), 0);

	std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> unif(0.0, 1.0);

	#pragma omp parallel for// default(shared) 
	for( int i = 0; i < knn_indices.size(); ++i ) {

		// perform num_walks random walks for this vertex
		for( int walk = 0; walk < num_walks; ++walk ) {
			int vertex = humap::random_walk(i, knn_indices[0].size(), vals, cols, walk_length, unif, rng);
			if( vertex != -1 )
				endpoint[vertex]++;
		}

	}

	return endpoint;
}

/**
* Computes a random walk on the neighboring graph for constructing representation neighborhood
*
* @param vertex int representing start point
* @param n_neighbors int representing the number of neighbors
* @param vals Container representing the graph strength
* @param cols Container representing the neighborhood
* @param walk_length int representing the max hops in the random walk
* @param unif uniform_real_distribution
* @param rng default_random_engine
* @param is_landmark Container storing landmarks information
* @return int representing the endpoint
*/
int humap::random_walk(int vertex, int n_neighbors, vector<double>& vals, vector<int>& cols, 				
	   				   int walk_length, uniform_real_distribution<double>& unif, 
					   mt19937& rng, vector<int>& is_landmark)
{

	for( int step = 0;  step < walk_length; ++step ) {
		double c = unif(rng);
		int next_vertex = vertex;
		double incremental_prob = 0.0;

		// for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, vertex); it; ++it ) {
			
		// 	incremental_prob += (it.value()/sum_vals[vertex]);

		// 	if( c < incremental_prob ) {
		// 		next_vertex = it.col();
		// 		break;
		// 	}
		// }
		int mult = vertex*n_neighbors;
		for( int it = 0; it < n_neighbors; ++it ) {
			incremental_prob += (vals[mult + it]);
			
			if( c < incremental_prob ) {
				next_vertex = cols[mult + it];
				break;
			}
		} 
		
		if( next_vertex == vertex )
			return -1;

		if( is_landmark[next_vertex] != -1 )
			return next_vertex;

		vertex = next_vertex;
	}
	return -1;
}

/**
* Performs a markov chain in the neighborhood graph for constructing representation neighborhood
*
* @param knn_indices Container representing the neighborhood graph
* @param vals Container representing the graph strength
* @param cols Container representing the neighborhood
* @param num_walks int representing the number of random walks
* @param walk_length int representing the walk length
* @param landmarks Container storing the landmarks
* @param influence_neighborhood int representing how many local neighbors to add in the representation neighborhood
* @param neighborhood Container to store the representation neighborhood
* @param association Container to store the force of association (how many times a landmark was the endpoint of a random walks)
* @return int with the maximum representation neighborhood
*/
int humap::markov_chain(vector<vector<int>>& knn_indices, 
						vector<double>& vals, vector<int>& cols,
						int num_walks, int walk_length, 
						vector<int>& landmarks, int influence_neighborhood, 
						vector<vector<int>>& neighborhood, vector<vector<int>>& association)
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


	#pragma omp parallel for 
	for( int i = 0; i < is_landmark.size(); ++i ) {	

		if( is_landmark[i] != -1 ) 
			continue;

		for( int walk = 0; walk < num_walks; ++walk ) {
			int vertex = humap::random_walk(i, knn_indices[0].size(), vals, cols, walk_length, unif, rng, is_landmark);
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

	return max_neighborhood;
}


/**
* Fits the hierarchy 
*
* @param X py::array_t with the dataset
* @param y py::array_t with the labels
*/
void humap::HierarchicalUMAP::fit(py::array_t<double> X, py::array_t<int> y)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	auto hierarchy_before = clock::now();


	auto before = clock::now();
	umap::Matrix first_level(humap::convert_to_vector(X));
	sec duration = clock::now() - before;

	this->hierarchy_X.push_back(first_level);
	this->dense_backup.push_back(first_level);
	this->hierarchy_y.push_back(vector<int>((int*)y.request().ptr, (int*)y.request().ptr + y.request().shape[0]));

	if( this->verbose ) {
		cout << "\n\n*************************************************************************" << endl;
		cout << "*********************************LEVEL 0*********************************" << endl;
		cout << "*************************************************************************" << endl << endl;
		cout << "Level 0 with "  << first_level.size() << " data samples." << endl;
		cout << "Fitting the first hierarchy level... "; 
	}


	
	umap::UMAP reducer = umap::UMAP("euclidean", this->n_neighbors, this->min_dist, this->knn_algorithm, this->init);
	reducer.set_ab_parameters(this->a, this->b);
	
	before = clock::now();
	reducer.fit(this->hierarchy_X[0]);
	duration = clock::now() - before;
	if( this->verbose ) {
		cout << "done in " << duration.count() << " seconds." << endl;
	}


	

	this->reducers.push_back(reducer);

	vector<int> indices(this->hierarchy_X[0].size(), 0);
	iota(indices.begin(), indices.end(), 0);
	vector<int> owners(this->hierarchy_X[0].size(), -1);
	vector<double> strength(this->hierarchy_X[0].size(), -1.0);
	vector<vector<int>> association(this->hierarchy_X[0].size(), vector<int>());

	this->metadata.push_back(humap::Metadata(indices, owners, strength, association, this->hierarchy_X[0].size()));

	this->original_indices.push_back(indices);


	Eigen::SparseMatrix<double, Eigen::RowMajor> graph = this->reducers[0].get_graph();

	vector<vector<double>> knn_dists = this->reducers[0].knn_dists();

	for( int level = 0; level < this->percents.size(); ++level ) {
		

		auto level_before = clock::now();

		int n_elements = (int) (this->percents[level] * this->hierarchy_X[level].size());		

		if( this->verbose ) {
			cout << "\n\n*************************************************************************" << endl;
			cout << "*********************************LEVEL "<< (level+1) << "*********************************" << endl;
			cout << "*************************************************************************" << endl << endl;

			
			cout << "Level " << (level+1) << ": " << n_elements << " data samples." << endl;
		}




		/*
			COMPUTING RANDOM WALK FOR SAMPLING SELETION
 		*/
 		auto begin_random_walk = clock::now();
 		if( this->verbose ) {
 			cout << "Computing random walks for sampling selection... ";
 		} 		

 		vector<int> landmarks;
 		landmarks = humap::markov_chain(this->reducers[level].knn_indices(),
 										this->reducers[level].vals_transition, 
 										this->reducers[level].cols,
										this->landmarks_nwalks, 
										this->landmarks_wl); 

 		sec end_random_walk = clock::now() - begin_random_walk;
 		cout << "done in " << end_random_walk.count() << " seconds." << endl;

 		vector<int> inds_lands; 		
 		vector<int> sorted_landmarks = utils::argsort(landmarks, true); 		
 		for( int i = 0; i < n_elements; ++i )
 			inds_lands.push_back(sorted_landmarks[i]);
 		 



 		/*
			COMPUTING RANDOM WALK FOR CONSTRUCTING REPRESENTATION NEIGHBORHOOD
 		*/
 		auto influence_begin = clock::now();
 		if( this->verbose ) {
 			cout << "Computing random walks for constucting representation neighborhood... ";
 		}


 		vector<vector<int>> neighborhood;
 		vector<vector<int>> association;
 		double max_incidence; //(double) (this->reducers[level].sigmas().size() - inds_lands.size());

 		max_incidence = humap::markov_chain(this->reducers[level].knn_indices(),
										    this->reducers[level].vals_transition,
										    this->reducers[level].cols,
										    this->influence_nwalks, this->influence_wl,  
										    //this->reducers[level].sum_vals,
										    inds_lands, this->influence_neighborhood,
										    neighborhood, association);
 		sec influence_time = clock::now() - influence_begin;
 		cout << "done in " << influence_time.count() << " seconds." << endl;

 			
 		level_landmarks.push_back(inds_lands);


 		/*
			STORE INFORMATION ABOUT ORIGINAL INDICES AND LANDMARKS
 		*/
		vector<int> greatest = inds_lands;

		vector<int> orig_inds(greatest.size(), 0);

		for( int i = 0; i < orig_inds.size(); ++i )
			orig_inds[i] = this->original_indices[level][greatest[i]];

		this->original_indices.push_back(orig_inds);
		this->_sigmas.push_back(this->reducers[level].sigmas());
		this->_indices.push_back(greatest);


		/*
			COMPUTE SIMILARITY AMONG THE LANDMARKS
		*/
		if( this->verbose ) {
			cout << "Computing similarity among landmarks... ";
		}

		umap::Matrix data;			
		auto similarity_before = clock::now();		
		if( this->similarity_method == "similarity" ) {

			vector<vector<double>> dense;
			for( int i = 0; i < greatest.size(); ++i ) {
				vector<double> row = this->update_position(i, neighborhood[i], this->hierarchy_X[level]);
				dense.push_back(row);
			}

			data = umap::Matrix(dense);
			reducer = umap::UMAP("euclidean", this->n_neighbors, this->min_dist, this->knn_algorithm, this->init);
			reducer.set_ab_parameters(this->a, this->b);


		} else if( this->similarity_method == "precomputed" ) {	 
			
			SparseComponents triplets = this->sparse_similarity(level+1, this->hierarchy_X[level].size(), this->n_neighbors,
																greatest, neighborhood, max_incidence, association); 
		
			vector<utils::SparseData> sparse = humap::create_sparse(n_elements, triplets.rows, triplets.cols, triplets.vals);
			data = umap::Matrix(sparse, greatest.size());
			reducer = umap::UMAP("precomputed", this->n_neighbors, this->min_dist, this->knn_algorithm, this->init);
			reducer.set_ab_parameters(this->a, this->b);

		} else {

			reducer = umap::UMAP("euclidean", this->n_neighbors, this->min_dist, this->knn_algorithm, this->init);
			reducer.set_ab_parameters(this->a, this->b);
		}
		sec similarity_after = clock::now() - similarity_before;
		if( this->verbose ) {
			cout << "done in " << similarity_after.count() << " seconds." << endl;
		}


		/*
			FITTING HIERARCHY LEVEL			
		*/
		if( this->verbose ) {
			cout << "Fitting the hierarchy level... "; 
		}

		this->metadata[level].count_influence = vector<int>(greatest.size(), 0);

		auto fit_before = clock::now();
		reducer.fit(data);
		sec fit_duration = clock::now() - fit_before;

		cout << "done in " << fit_duration.count() << " seconds." << endl;


		/*
			ASSOCIATING DATA POINTS TO LANDMARKS
		*/
		if( this->verbose ) {
			cout << "Associating data points to landmarks... "; 
		}

		auto associate_before = clock::now();
		vector<int> is_landmark(this->metadata[level].size, -1);
		for( int i = 0; i < greatest.size(); ++i ) 
		{
			is_landmark[greatest[i]] = i;
		}

		this->associate_to_landmarks(greatest.size(), this->n_neighbors, greatest, this->reducers[level].cols, 
			this->metadata[level].strength, this->metadata[level].owners, this->metadata[level].indices, 
			this->metadata[level].association, is_landmark, this->metadata[level].count_influence, this->reducers[level].knn_dists());

		
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
									  this->metadata[level].strength, this->metadata[level].owners, this->metadata[level].indices, 
									  this->metadata[level].association, this->metadata[level].count_influence, 
									  is_landmark, this->reducers[level].knn_dists());

		sec associate_duration = clock::now() - associate_before;
		cout << "done in " << associate_duration.count() << " seconds." << endl;

		/*
			STORE INFORMATION FOR THE NEXT HIERARCHY LEVEL
		*/
		if( this->verbose ) {
			cout << "Storing information for the next hierarchy level... ";
		}
		auto information_before = clock::now();

		vector<int> new_owners(greatest.size(), -1);
		vector<double> new_strength(greatest.size(), -1.0);
		vector<vector<int>> new_association(greatest.size(), vector<int>());

		this->metadata.push_back(Metadata(greatest, new_owners, new_strength, new_association, greatest.size()));//vector<int>(greatest.size(), -1), vector<double>(greatest.size(), -1.0)));
		this->reducers.push_back(reducer);
		this->hierarchy_X.push_back(data);
		this->hierarchy_y.push_back(utils::arrange_by_indices(this->hierarchy_y[level], greatest));

		sec information_after = clock::now() - information_before;
		if( this->verbose ) {
			cout << "done in " << information_after.count() << " seconds." << endl;
		}

		sec level_duration = clock::now() - level_before;
		cout << endl << "Level construction: " << level_duration.count() << endl;
		cout << endl;

	}




	sec hierarchy_duration = clock::now() - hierarchy_before;
	cout << endl;
	cout << "Hierarchy construction in " << hierarchy_duration.count() << " seconds." << endl;
	cout << endl;


	for( int i = 0; i < this->hierarchy_X.size(); ++i ) {
		this->embeddings.push_back(vector<vector<double>>());
	}
}

/**
* Generate the embedding for a hierarchical level
*
* @param level int representing the hierarchical level
* @return py::array_t containing the embedding 
*/
py::array_t<double> humap::HierarchicalUMAP::transform(int level) 
{
	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	if( this->fixed_datapoints.size() != 0 && level < this->hierarchy_X.size()-1 ) {

		this->indices_fixed = vector<int>(level_landmarks[level].begin(), level_landmarks[level].end());

		this->free_datapoints = vector<bool>(this->metadata[level].indices.size(), true);
		for( int i = 0; i < this->indices_fixed.size(); ++i ) {
			this->free_datapoints[indices_fixed[i]] = false;
		}
	}

	vector<vector<double>> result = this->embed_data(level, this->reducers[level].get_graph(), this->hierarchy_X[level]);

	return py::cast(result);
}

/**
* Get the landmark influencing the data point
*
* @param level int represeting the hierarchical level
* @param index int representing the data point index
* @return int with the landmark index
*/
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


/**
* Get the indices of landmarks influencing the indices
*
* @param level int representing the hierarchical level
* @param indices Container with the data point indices
* @return Container with the list of landmarks
*/
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

/**
* Gets the influence of each landmark in a hierarchy level
*
* @param level int representing the level
* @return Container with list of influence
*/
py::array_t<int> humap::HierarchicalUMAP::get_influence(int level)
{
	if( level >= this->hierarchy_X.size() || level <= 0 )
		throw new runtime_error("Level out of bounds.");

	vector<int> influence(this->metadata[level].size, 0);

	for( int i = 0; i < this->metadata[level].size; ++i ) {
		influence[i] = influenced_by(level-1, i);
	}

	return py::cast(influence);
}

/**
* Gets the indices of the landmarks in a hierarchy level with respect to the level below it
*	
* @param level int representing the level
* @return Container with the list of indices
*/
py::array_t<int> humap::HierarchicalUMAP::get_indices(int level)
{
	if( level >= this->hierarchy_X.size()-1 || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->_indices[level]);
}

/**
* Get the original indices of a hierarchy level
*
* @param level int representing the hierarchy level
* @return Container with the original indices
*/
py::array_t<int> humap::HierarchicalUMAP::get_original_indices(int level)
{
	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->original_indices[level]);
}

/**
* Gets the labels for a hierarchy level
*
* @param level int representing the hierarchy level
* @return Container with the labels 
*/
py::array_t<int> humap::HierarchicalUMAP::get_labels(int level)
{
	if( level == 0 )  
		throw new runtime_error("Sorry, we won't be able to return all the labels!");

	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->hierarchy_y[level]);
}

/**
* Gets the embedding of a hierarchy level
*
* @param level int representing the hierarchy level
* @return py::array_t with the embedding
*/
py::array_t<double> humap::HierarchicalUMAP::get_embedding(int level)
{
	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return py::cast(this->embeddings[level]);
}	

/**
* Gets the data points data of a hierarchy level
*
* @param level int representing the hierarchy level
* @return Eigen::SparseMatrix representing the subset of data in the hierarchy level
*/
Eigen::SparseMatrix<double, Eigen::RowMajor> humap::HierarchicalUMAP::get_data(int level)
{
	if( level == 0 )  
		throw new runtime_error("Sorry, we won't me able to return all dataset! Please, project using UMAP.");

	if( level >= this->hierarchy_X.size() || level < 0 )
		throw new runtime_error("Level out of bounds.");

	return utils::create_sparse(this->hierarchy_X[level].sparse_matrix, this->hierarchy_X[level].size(), (int) this->n_neighbors*2.5);
}

/**
* Embed a subset of data 
*
* @param level int representing the hierarchy level
* @param graph Eigen::SparseMatrix representing the graph forces
* @param X Matrix representing the subset of data
* @return Container with embed data
*/
vector<vector<double>> humap::HierarchicalUMAP::embed_data(int level, Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, umap::Matrix& X)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	auto before = clock::now(); 

	int n_vertices = graph.cols();
	if( this->n_epochs == -1 ) {
		if( graph.rows() <= 10000 )
			n_epochs = 500;
		else 
			n_epochs = 200;

	}
	
	if( !graph.isCompressed() )
		graph.makeCompressed();
	
	double max_value = graph.coeffs().maxCoeff();
	graph = graph.pruned(max_value/(double)n_epochs, 1.0);

	/*
		COMPUTE INITIAL LOW-DIMENSIONAL REPRESENTATION
	*/
	if( this->verbose ) {
		cout << "Initing low-dimensional representation... ";
	}
	
	auto tic = clock::now();
	vector<vector<double>> embedding = this->reducers[level].spectral_layout(X, graph, this->n_components);
	sec toc = clock::now() - tic; 

	this->reducers[level].set_free_datapoints(this->free_datapoints);
	this->reducers[level].set_fixing_term(this->_fixing_term);

	if( this->free_datapoints.size() != 0 ) {
		for( int i = 0; i < this->indices_fixed.size(); ++i ) {
			embedding[this->indices_fixed[i]] = this->fixed_datapoints[i];
		}
	}

	if( this->verbose ) {
		cout << "done in " << toc.count() << " seconds." << endl;
		cout << "Fixing term: " << this->_fixing_term << endl;
	}


	
	vector<int> rows, cols;
	vector<double> data;	
	tie(rows, cols, data) = utils::to_row_format(graph);
	
	vector<double> epochs_per_sample = this->reducers[level].make_epochs_per_sample(data, this->n_epochs);
	
	vector<double> min_vec, max_vec;
	for( int j = 0; j < this->n_components; ++j ) {

		min_vec.push_back((*min_element(embedding.begin(), embedding.end(), 
			[j](vector<double> a, vector<double> b) {							
				return a[j] < b[j];
			}))[j]);
		max_vec.push_back((*max_element(embedding.begin(), embedding.end(), 
			[j](vector<double> a, vector<double> b) {
				return a[j] < b[j];
			}))[j]);
	}
	
	vector<double> max_minus_min(this->n_components, 0.0);
	std::transform(max_vec.begin(), max_vec.end(), min_vec.begin(), max_minus_min.begin(), [](double a, double b){ return a-b; });
	

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

	if( this->verbose ) {
		cout << "Embedding level " << level << " with " << embedding.size() << " data samples.\n" << endl << endl;
	}

	this->reducers[level].verbose = this->verbose;
	

	vector<vector<double>> result = this->reducers[level].optimize_layout_euclidean(
		embedding,
		embedding,
		rows,
		cols,
		this->n_epochs,
		n_vertices,
		epochs_per_sample);


	sec duration = clock::now() - before;
	if( this->verbose ) {
		cout << endl << "It took " << duration.count() << " to embed." << endl;
	}

	// makes sure a level only influences on the level below it
	this->free_datapoints = vector<bool>();
	this->fixed_datapoints = vector<vector<double>>();


	return result;
}	


/**
* Embed a subset of data based on indices
*
* @param level int representing the hierarchy level
* @param indices py:::array_t representing the landmark indices
* @return py::array_t with embed data
*/
py::array_t<double> humap::HierarchicalUMAP::project_indices(int level, py::array_t<int> indices)
{
	
	py::buffer_info bf = indices.request();
	int* inds = (int*) bf.ptr;

	vector<int> selected_indices(inds, inds+bf.shape[0]);

	return this->project_data(level, selected_indices);
}

/**
* Embed a subset of data based on lables
*
* @param level int representing the hierarchy level
* @param c py:::array_t representing the labels
* @return py::array_t with embed data
*/
py::array_t<double> humap::HierarchicalUMAP::project(int level, py::array_t<int> c)
{
	py::buffer_info bf = c.request();
	int* classes = (int*) bf.ptr;

	vector<int> selected_indices;
	for( int i = 0; i < this->hierarchy_y[level].size(); ++i ) {
		bool flag = false;
		
		for( int j = 0; j < bf.shape[0]; ++j ) {
			if( this->hierarchy_y[level][i] == classes[j] ) {
				flag = true;
				break;
			}
		}

		if( flag )
			selected_indices.push_back(i);
	}	

	return this->project_data(level, selected_indices);
}

/**
* Selects the subset of data from hierarchy level below based on the selected indices
*
* @param level int representing the hierarchy level
* @param selected_indices Container representing the landmarks
* @return py::array_t with embed data
*/
py::array_t<double> humap::HierarchicalUMAP::project_data(int level, vector<int> selected_indices)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;
	
	vector<bool> is_in_it(this->metadata[level-1].size, false);
	vector<int> indices_next_level;
	vector<int> labels;
	map<int, int> mapper;

	vector<int> correspond_values;
	vector<int> landmark_order;
	for( int i = 0; i < this->metadata[level-1].size; ++i ) {
		int landmark = this->metadata[level-1].indices[i];

		for( int j = 0; j < selected_indices.size(); ++j ) {
			
			if( landmark == selected_indices[j] && !is_in_it[i] ) {
				labels.push_back(this->hierarchy_y[level-1][i]);
				indices_next_level.push_back(i);
				mapper[i] = indices_next_level.size()-1;
				is_in_it[i] = true;

				if( this->original_indices[level-1][i] == this->original_indices[level][selected_indices[j]]) {
					correspond_values.push_back(indices_next_level.size()-1);
					landmark_order.push_back(landmark);				
				}

				break;
			}

			// cout << "nao" << endl;

		}
	}

	if( this->fixed_datapoints.size() != 0 ) {

		cout << "Fixed datapoints is not empty! " << endl;
		cout << "I will have " << this->fixed_datapoints.size() << " data points in the next optimization. " << endl;
		cout << "Correspond values has " << correspond_values.size() << "/" << selected_indices.size() << " indices, from " << indices_next_level.size() << " to be projected. " << endl;
		
		this->free_datapoints = vector<bool>(indices_next_level.size(), true);
		vector<int> indices_cor = utils::argsort(landmark_order);
		this->indices_fixed = vector<int>();

		for( int i = 0; i < indices_cor.size(); ++i ) {
			this->free_datapoints[correspond_values[indices_cor[i]]] = false;
			this->indices_fixed.push_back(correspond_values[indices_cor[i]]);
			// this->free_datapoints[selected_indices[i]] = false;
		}

		// this->indices_fixed = selected_indices;
	}

	this->labels_selected = labels;	
	this->influence_selected = this->get_influence_by_indices(level-1, indices_next_level);
	this->indices_selected = indices_next_level;

	if( this->hierarchy_X[level-1].is_sparse() && !this->focus_context  ) {

		umap::Matrix X = this->hierarchy_X[level-1];
		vector<utils::SparseData> new_X;

		int min_neighbors = 99999;
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

		Eigen::SparseMatrix<double, Eigen::RowMajor> graph = this->reducers[level-1].get_graph();
		Eigen::SparseMatrix<double, Eigen::RowMajor> new_graph(indices_next_level.size(), indices_next_level.size());


		pair<int,int> max_neighbor = *std::max_element(mapper.begin(), mapper.end(), [](const pair<int,int>& a, const pair<int, int>& b) {
			return a.second < b.second;
		});

		new_graph.reserve(Eigen::VectorXi::Constant(indices_next_level.size(), max_neighbor.second+5));
	
		for( int i = 0; i < indices_next_level.size(); ++i ) {
	
			int k = indices_next_level[i];
			for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
				if( mapper.count(it.col()) > 0 ) {
					new_graph.insert(i, mapper[it.col()]) = it.value();
					if( i >= indices_next_level.size() )
						cout << "ATTENTION: i >= indices_next_level.size()" << endl;
					if( mapper[it.col()] >= max_neighbor.second+5 )
						cout << "ATTENTION: mapper[it.col()] >= this->n_neighbors*2+5" << endl;

				}	
			}

		}


		// new_graph.reserve(Eigen::VectorXi::Constant(indices_next_level.size(), this->n_neighbors*2+5));

		// for( int i = 0; i < indices_next_level.size(); ++i ) {
		// 	int k = indices_next_level[i];
		// 	for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
		// 		if( mapper.count(it.col()) > 0 ) {
		// 			new_graph.insert(i, mapper[it.col()]) = it.value();
				
		// 		}
		// 	}

		// }
		new_graph.makeCompressed();

		umap::Matrix nX = umap::Matrix(new_X, indices_next_level.size());

		return py::cast(this->embed_data(level-1, new_graph, nX));
		
	} if( this->hierarchy_X[level-1].is_sparse() && this->focus_context ) {

		if( this->verbose )
			cout << "Using Focus+Context strategy" << endl;

		umap::Matrix X = this->hierarchy_X[level-1];
		vector<utils::SparseData> new_X;

		int min_neighbors = 99999;
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



		// preprocessing for current level
		map<int, int> new_mapper;
		vector<int> indices_to_iterate;
		int N_level = this->hierarchy_y[level].size();
		vector<int> additional_y;
		std::vector<int>::iterator it;

		for( int i = 0, j = 0; i < N_level; ++i ) {
			it = find(selected_indices.begin(), selected_indices.end(), i);
			if( it == selected_indices.end() ) {
				new_mapper[i] = new_X.size() + j++;
				indices_to_iterate.push_back(i);
				additional_y.push_back(this->hierarchy_y[level][i]);
			}
		}

		this->labels_selected.insert(this->labels_selected.end(), additional_y.begin(), additional_y.end());

		for( int i = 0; i < indices_to_iterate.size(); ++i ) {

			int index = indices_to_iterate[i];
			utils::SparseData sd = this->hierarchy_X[level].sparse_matrix[index];

			// verificar se precisa disso tudo out somente n_neighbors
			vector<int> assigned(indices_to_iterate.size() + indices_next_level.size(), 0);

			vector<double> data = sd.data;
			vector<int> indices = sd.indices;

			vector<double> new_data;
			vector<int> new_indices;

			for( int j = 0; j < indices.size(); ++j ) {

				if( new_mapper.count(indices[j]) > 0 ) {
					new_data.push_back(data[j]);
					new_indices.push_back(new_mapper[indices[j]]);
					assigned[new_mapper[indices[j]]] = 1;
				}
			}




			for( int j = 0; j < assigned.size(); ++j ) {
				if( !assigned[j] ) {
					new_data.push_back(1.0);
					new_indices.push_back(j);		
				}
			}

			min_neighbors = min(min_neighbors, (int) new_indices.size());
			new_X.push_back(utils::SparseData(new_data, new_indices));

		}

		Eigen::SparseMatrix<double, Eigen::RowMajor> graph = this->reducers[level-1].get_graph();
		Eigen::SparseMatrix<double, Eigen::RowMajor> graph2 = this->reducers[level].get_graph();

		Eigen::SparseMatrix<double, Eigen::RowMajor> new_graph(new_X.size(), new_X.size());

		pair<int, int> max_neighbor = *std::max_element(new_mapper.begin(), new_mapper.end(), 
		[](const pair<int, int>& a, const pair<int, int>& b) {
			return a.second < b.second;
		});

		new_graph.reserve(Eigen::VectorXi::Constant(new_X.size(), max_neighbor.second+5));

		for( int i = 0; i < indices_next_level.size(); ++i ) {

			int k = indices_next_level[i];
			
			for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
				if( mapper.count(it.col()) > 0 ) {
					new_graph.insert(i, mapper[it.col()]) = it.value();
					if( i >= new_X.size() )
						cout << "ATTENTION: i >= new_X.size()" << endl;
					if( mapper[it.col()] >= max_neighbor.second+5 )
						cout << "ATTENTION: mapper[it.col()] >= max_neighbor.second+5" << endl;
				}
			}
		}

		for( int i = 0; i < indices_to_iterate.size(); ++i ) {
			
			int k = indices_to_iterate[i];

			for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph2, k); it; ++it ) {
				if( new_mapper.count(it.col()) > 0 ) {
					new_graph.insert(indices_next_level.size() + i, new_mapper[it.col()]) = it.value();

					if( i >= new_X.size() )
						cout << "ATTENTION: i >= new_X.size()" << endl;
					if( new_mapper[it.col()] >= max_neighbor.second+5 )
						cout << "ATTENTION: new_mapper[it.col()] >= max_neighbor.second+5" << endl;					
				}
			}
		}


		new_graph.makeCompressed();

		umap::Matrix nX = umap::Matrix(new_X, new_X.size());

		return py::cast(this->embed_data(level-1, new_graph, nX));
	} else {


		umap::Matrix X = this->hierarchy_X[level-1];
		vector<vector<double>> new_X;

		for( int i = 0; i < indices_next_level.size(); ++i ) {
			vector<double> dd = X.dense_matrix[indices_next_level[i]];
			new_X.push_back(dd);
		}
	
		pair<int,int> max_neighbor = *std::max_element(mapper.begin(), mapper.end(), [](const pair<int,int>& a, const pair<int, int>& b) {
			return a.second < b.second;
		});
	
		Eigen::SparseMatrix<double, Eigen::RowMajor> graph = this->reducers[level-1].get_graph();
		Eigen::SparseMatrix<double, Eigen::RowMajor> new_graph(indices_next_level.size(), indices_next_level.size());
		new_graph.reserve(Eigen::VectorXi::Constant(indices_next_level.size(), max_neighbor.second+5));
	
		for( int i = 0; i < indices_next_level.size(); ++i ) {
	
			int k = indices_next_level[i];
			for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
				if( mapper.count(it.col()) >0 ) {
					new_graph.insert(i, mapper[it.col()]) = it.value();
					if( i >= indices_next_level.size() )
						cout << "ATTENTION: i >= indices_next_level.size()" << endl;
					if( mapper[it.col()] >= max_neighbor.second+5 )
						cout << "ATTENTION: mapper[it.col()] >= this->n_neighbors*2+5" << endl;

				}	
			}

		}

		new_graph.makeCompressed();
		
		umap::Matrix nX = umap::Matrix(new_X);

		return py::cast(this->embed_data(level-1, new_graph, nX));
	}

	return py::cast(vector<vector<double>>());
}

