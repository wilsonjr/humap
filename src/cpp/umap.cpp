#include "umap.h"
#include "utils.h"

namespace py = pybind11;
using namespace std;

void umap::UMAP::optimize_euclidean_epoch(vector<vector<double>>& head_embedding, vector<vector<double>>& tail_embedding,
										   const vector<int>& head, const vector<int>& tail, int n_vertices, 
										   const vector<double>& epochs_per_sample, double a, double b, vector<long>& rng_state, 
										   double gamma, int dim, bool move_other, double alpha, vector<double>& epochs_per_negative_sample,
										   vector<double>& epoch_of_next_negative_sample, vector<double>& epoch_of_next_sample, int n)
{
	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<int> dist(0, head_embedding.size());



	#pragma omp parallel for
	for( int i = 0; i < epochs_per_sample.size(); ++i ) {
		// printf("2\n");
		if( epoch_of_next_sample[i] <= n ) {
			int j = head[i];
			int k = tail[i];
			// printf("5\n");

			vector<double>* current = &head_embedding[j];
			// printf("6\n");
			
			vector<double>* other = &tail_embedding[k];
			// printf("7\n");
			

			double dist_squared = utils::rdist((*current), (*other));
			// printf("8\n");

			double grad_coeff = 0.0;

			if( dist_squared > 0.0 ) {
				// printf("9\n");
				grad_coeff = -2.0 * a * b * pow(dist_squared, b-1.0);
				// printf("10\n");
				grad_coeff /= a * pow(dist_squared, b) + 1.0;
				// printf("11\n");

			}

			// printf("12\n");
			for( int d = 0; d < dim; ++d ) {
				// printf("13\n");
				double grad_d = utils::clip(grad_coeff * ((*current)[d] - (*other)[d]));
				// printf("14\n");
				(*current)[d] += (grad_d * alpha);
				// printf("15\n");
				if( move_other ) {
					// printf("16\n");
					(*other)[d] += (-grad_d * alpha);
				}

			}

			// printf("17\n");
			epoch_of_next_sample[i] += epochs_per_sample[i];
			// printf("18\n");
			int n_neg_samples = (int) ((n-epoch_of_next_negative_sample[i])/epochs_per_negative_sample[i]);
			// printf("19\n");
			for( int p = 0; p < n_neg_samples; ++p ) {
				// int k = utils::tau_rand_int(rng_state);
				int k = dist(engine);
				// if( k < 0 )
				// 	k *= -1;
				// printf("21\n");
				k = k % n_vertices;
				// printf("22 > %d\n", k);
				other = &tail_embedding[k];
				// printf("23\n");
				dist_squared = utils::rdist(*current, *other);
				// printf("24\n");
				if( dist_squared > 0.0 ) {
					// printf("25\n");
					grad_coeff = 2.0 * gamma * b;
					// printf("26\n");
					grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1.0);
					// printf("27\n");
				} else if( j == k ) {
					// printf("28\n");
					continue;
				} else {
					// printf("29\n");
					grad_coeff = 0.0;
				}
				// printf("30\n");
				for( int d = 0; d < dim; ++d ) {
					// printf("31\n");
					double grad_d = 0.0;
					// printf("32\n");
					if( grad_coeff > 0.0 ) {
						// printf("33\n");
						grad_d = utils::clip(grad_coeff * ((*current)[d] - (*other)[d]));
					}
					else {
						// printf("34\n");
						grad_d = 4.0;
					}
					// printf("35\n");
					(*current)[d] += (grad_d * alpha);
					// printf("36\n");
				}

			}
			// printf("37\n");
			epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i]);
			// printf("38\n");
		}
	}
}

vector<vector<double>> umap::UMAP::optimize_layout_euclidean(vector<vector<double>>& head_embedding, vector<vector<double>>& tail_embedding,
										   const vector<int>& head, const vector<int>& tail, int n_epochs, int n_vertices, 
										   const vector<double>& epochs_per_sample, vector<long>& rng_state)
{
	double a = this->_a;
	double b = this->_b;
	double gamma = this->repulsion_strength;
	double initial_alpha = this->_initial_alpha;
	double negative_sample_rate = this->negative_sample_rate;


	int dim = head_embedding[0].size();
	bool move_other = head_embedding.size() == tail_embedding.size();
	double alpha = initial_alpha;

	// cout << "a: " << a << endl;
	// cout << "b: " << b << endl;
	// cout << "gamma: " << gamma << endl;
	// cout << "initial_alpha: " << initial_alpha << endl;
	// cout << "negative_sample_rate: " << negative_sample_rate << endl;
	// cout << "dim: " << dim << endl;
	// cout << "move_other: " << move_other << endl;
	// cout << "alpha: " << alpha << endl;




	vector<double> epochs_per_negative_sample(epochs_per_sample.size(), 0.0);
	transform(epochs_per_sample.begin(), epochs_per_sample.end(), epochs_per_negative_sample.begin(), 
		[negative_sample_rate](double a) {
			return a/negative_sample_rate;
		});

	// cout << "epochs_per_negative_sample" << endl;
	// for( int i = 0; i < 10; ++i )
	// 	printf("%.4f ", epochs_per_negative_sample[i]);
	// cout << endl;

	vector<double> epoch_of_next_negative_sample(epochs_per_negative_sample.begin(), epochs_per_negative_sample.end());
	// cout << "epoch_of_next_negative_sample" << endl;
	// for( int i = 0; i < 10; ++i )
	// 	printf("%.4f ", epoch_of_next_negative_sample[i]);
	// cout << endl;

	vector<double> epoch_of_next_sample(epochs_per_sample.begin(), epochs_per_sample.end());
	// cout << "epoch_of_next_sample" << endl;
	// for( int i = 0; i < 10; ++i )
	// 	printf("%.4f ", epoch_of_next_sample[i]);
	// cout << endl;


	for( int epoch = 0; epoch < n_epochs; ++epoch ) {


		this->optimize_euclidean_epoch(
			head_embedding,
			tail_embedding,
			head,
			tail,
			n_vertices,
			epochs_per_sample,
			a, 
			b,
			rng_state,
			gamma,
			dim,
			move_other,
			alpha,
			epochs_per_negative_sample,
			epoch_of_next_negative_sample,
			epoch_of_next_sample,
			epoch);

		alpha = initial_alpha * (1.0 - ((double)epoch/(double)n_epochs));


		if( this->verbose && epoch % (int)(n_epochs/10) == 0)
			printf("\tcompleted %d / %d epochs\n", epoch, n_epochs);

	}
	printf("\tcompleted %d epochs\n", n_epochs);


	return head_embedding;

}

vector<double> umap::UMAP::make_epochs_per_sample(const vector<double>& weights, int n_epochs)
{

	vector<double> result(weights.size(), -1.0);
	vector<double> n_samples(weights.size(), 0.0);

	double max_weight = *max_element(weights.begin(), weights.end());

	transform(weights.begin(), weights.end(), n_samples.begin(), [n_epochs, max_weight](double weight){ 
		return n_epochs*(weight/max_weight); 
	});

	transform(result.begin(), result.end(), n_samples.begin(), result.begin(), [n_epochs](double r, double s) {
		if( s > 0.0 )
			return (double)n_epochs / s;
		else
			return r;
	});


	return result;
}

vector<vector<double>> umap::UMAP::component_layout(umap::Matrix& data, int n_components, 
														 vector<int>& component_labels, int dim)
{


	// cout << "Entrei 1" << endl;
	// cout << "n_components " << n_components << endl;
	// cout << "data.is_sparse(): " << data.is_sparse() << endl;
	// cout << "data.shape(1) " << data.shape(1) << endl;

	vector<vector<double>> component_centroids(n_components, vector<double>(data.shape(1), 0.0));

	// cout << "Entrei 2" << endl;
	vector<vector<double>> distance_matrix;

	if( this->metric == "precomputed" ) {
		cout << "precomputed" << endl;
		distance_matrix = vector<vector<double>>(n_components, vector<double>(n_components, 0.0));


		// cout << "shape distance_matrix: " << n_components << " x " << n_components << endl;
		for( int c_i = 0; c_i < n_components; ++c_i ) {
			// cout << "passei 1" << endl;
			vector<vector<double>> dm_i;
			for( int i = 0; i < component_labels.size(); ++i ) {
				// cout << "passei 2 (data.shape: " << data.shape(1) << ", " <<  component_labels.size() << ")" << endl;
				if( component_labels[i] == c_i )
					dm_i.push_back(data.get_row(i));
				// cout << "passei 3" << endl;
			}
			string linkage = "min";
			// cout << "passei 4" << endl;
			for( int c_j = c_i+1; c_j < n_components; ++c_j  ) {

				double dist = 0.0;
				for( int j = 0; j < dm_i.size(); ++j ) {
					for( int i = 0; i < component_labels.size(); ++i ) {
						if( component_labels[i] == c_j ) {
							dist = min(dist, dm_i[j][i]);
						}

					}
				}
				// cout << "passei 5" << endl;
				distance_matrix[c_i][c_j] = dist;
				distance_matrix[c_j][c_i] = dist;
				// cout << "passei 6" << endl;
			}
		}
		// cout << "passei aqui irmao" << endl;

	} else {
	
		for( int label = 0; label < n_components; ++label ) {
	
			vector<double> sum_v(data.shape(1), 0.0);
			double count = 0.0;
			for( int i = 0; i < component_labels.size(); ++i ) {

				if( component_labels[i] == label ) {
					vector<double> row = data.get_row(i);
					for( int j = 0; j < row.size(); ++j  )
						sum_v[j] += row[j];
					count++;
				}
			}
	
			for( int j = 0; j < sum_v.size(); ++j  )
				sum_v[j] /= count;
			component_centroids[label] = sum_v;
		}
		distance_matrix = utils::pairwise_distances(component_centroids);
	
	}
	for( int i = 0; i < distance_matrix.size(); ++i ) 
		for( int j = 0; j < distance_matrix[i].size(); ++j ) 
			distance_matrix[i][j] = exp(-(distance_matrix[i][j]*distance_matrix[i][j]));
	
	py::module manifold = py::module::import("sklearn.manifold");
	py::object SpectralEmbedding = manifold.attr("SpectralEmbedding")(py::arg("n_components")=dim, py::arg("affinity")="precomputed");
	py::object embedding = SpectralEmbedding.attr("fit_transform")(py::cast(distance_matrix));

	vector<vector<double>> component_embedding = embedding.cast<vector<vector<double>>>();
	double max_v = component_embedding[0][0];
	for( int i = 0; i < component_embedding.size(); ++i ) {
		for( int j = 0; j < component_embedding[i].size(); ++j ) {
			max_v = max(max_v, component_embedding[i][j]);
		}
	}

	for( int i = 0; i < component_embedding.size(); ++i ) {
		for( int j = 0; j < component_embedding[i].size(); ++j ) {
			component_embedding[i][j] = component_embedding[i][j]/max_v;
		}
	}

	return component_embedding;
}

vector<vector<double>> umap::UMAP::multi_component_layout(umap::Matrix& data, 
	const Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, int n_components, 
	vector<int>& component_labels, int dim)
{


	vector<vector<double>> result(graph.rows(), vector<double>(dim, 0.0));

	vector<vector<double>> meta_embedding;
	if( n_components > 2*dim ) {
		// cout << "compoent layout " << endl;
		meta_embedding = this->component_layout(data, n_components, component_labels, dim);

		cout << "n_components > 2*dim" << endl;
		cout << "meta_embedding.shape: " << meta_embedding.size() << " x " << meta_embedding[0].size() << endl;
	} else {
		// cout << "compoent 1 " << endl;
		int k = (int)ceil(n_components/2.0);
		// cout << "compoent 2 " << endl;
		vector<vector<double>> base(k, vector<double>(k, 0.0));
		for( int i = 0; i < k; ++i )
			base[i][i] = 1;
		// cout << "compoent 3 " << endl;
		int count = 0;
		for( int i = 0; i < k && count < n_components; ++i, ++count ) {
			meta_embedding.push_back(base[i]);
		}
		// cout << "compoent 4 " << endl;
		for( int i = 0; i < k && count < n_components; ++i, ++count ) {
			transform(base[i].begin(), base[i].end(), base[i].begin(), [](double& c) { return -c; });
			meta_embedding.push_back(base[i]);
		}

		cout << "n_components <= 2*dim" << endl;
		cout << "meta_embedding.shape: " << meta_embedding.size() << " x " << meta_embedding[0].size() << endl;

	}



	for( int label = 0; label < n_components; ++label ) {

		int occurences = count(component_labels.begin(), component_labels.end(), label);
		// cout << "occurences: " << occurences << endl;
		Eigen::SparseMatrix<double, Eigen::ColMajor> tempCol(occurences, graph.cols());
		// cout << "created tempCol" << endl;
		tempCol.reserve(Eigen::VectorXi::Constant(graph.cols(), occurences));
		// cout << "reserved tempCol" << endl;
		for( int k=0, row = -1; k < graph.outerSize(); ++k) {
				
			if( component_labels[k] != label )
				continue;
			row++;
			for( Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(graph, k); it; ++it ) {
				// cout << "trying to insert tempCol" << endl;
				
				tempCol.insert(row, it.col()) = it.value();
				// cout << "inserted tempCol" << endl;
			}
		}
		// cout << "added tempCol" << endl;
		tempCol.makeCompressed();
		// cout << "compressed tempCol" << endl;


		Eigen::SparseMatrix<double, Eigen::RowMajor> component_graph(occurences, occurences);
		// cout << "created component_graph" << endl;
		component_graph.reserve(Eigen::VectorXi::Constant(occurences, occurences));
		// cout << "reserved component_graph" << endl;
		for( int k = 0, col = -1; k < tempCol.outerSize(); ++k) {
				
			if( component_labels[k] != label )
				continue;
			col++;
			for( Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator it(tempCol, k); it; ++it ) {

				// cout << it.row() << " >= 0 && " << it.row() << " < " << component_graph.rows() << endl;
				// cout << it.col() << " >= 0 && " << it.col() << " < " << component_graph.cols() << endl;

				component_graph.insert(it.row(), col) = it.value();
			}
		}
		// cout << "added component_graph" << endl;
		component_graph.makeCompressed();
		// cout << "compressed component_graph" << endl;


		// TODO: do this right...
		double min_dist = 999999.0;
		for( int i = 0; i < meta_embedding.size(); ++i ) {
			double distance = 0.0;
			for( int k = 0; k < meta_embedding[i].size(); ++k ) {
				distance += (meta_embedding[label][k]-meta_embedding[i][k])*(meta_embedding[label][k]-meta_embedding[i][k]);				
			}
			distance = sqrt(distance);
			if( distance > 0.0 ) {
				min_dist = min(min_dist, distance);
			}
		}  
		double data_range = min_dist / 2.0;


		if( component_graph.rows() < 2*dim ) {

			py::module scipy_random = py::module::import("numpy.random");
			py::object randomState = scipy_random.attr("RandomState")(this->random_state);
			vector<int> size = {(int)component_graph.rows(), dim};
			py::object noiseObj = randomState.attr("uniform")(py::arg("low")=-data_range, py::arg("high")=data_range, 
														     py::arg("size")=size);
			vector<vector<double>> noise = noiseObj.cast<vector<vector<double>>>();


			int row=0;

			for( int i = 0; i < component_labels.size(); ++i ) {

				if( component_labels[i] == label ) {
					for( int j = 0; j < noise[row].size(); ++j ) {
						result[i][j] = noise[row][j] + meta_embedding[label][j];
					}
					row++;
				}



			}

			continue;


		}

		// cout << "PASSEI 1 " << endl;


		Eigen::VectorXd diag_eigen = component_graph * Eigen::VectorXd::Ones(component_graph.cols());
		vector<double> diag_data(&diag_eigen[0], diag_eigen.data() + diag_eigen.size());
		vector<double> temp(diag_data.size(), 0.0);
		for( int i = 0; i < temp.size(); ++i )
			temp[i] = 1.0/sqrt(diag_data[i]);
	
		// cout << "PASSEI 2 " << endl;

		py::module scipy_sparse = py::module::import("scipy.sparse");
		// cout << "PASSEI 3 " << endl;
		py::object Iobj = scipy_sparse.attr("identity")(component_graph.rows(), py::arg("format") = "csr");
		// cout << "PASSEI 4 " << endl;
		py::object Dobj = scipy_sparse.attr("spdiags")(py::cast(temp), 0, component_graph.rows(), component_graph.rows(), 
			py::arg("format") = "csr");
		// cout << "PASSEI 5 " << endl;
		Eigen::SparseMatrix<double, Eigen::RowMajor> I = Iobj.cast<Eigen::SparseMatrix<double, Eigen::RowMajor>>();
		// cout << "PASSEI 6 " << endl;
		Eigen::SparseMatrix<double, Eigen::RowMajor> D = Dobj.cast<Eigen::SparseMatrix<double, Eigen::RowMajor>>();
		// cout << "PASSEI 7 " << endl;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L = I - D * component_graph * D;
		// cout << "PASSEI 8 " << endl;
		int k = dim+1;
		int num_lanczos_vectors = max(2*k+1, (int)sqrt(component_graph.rows()));

		try {
			// cout << "PASSEI 9 " << endl;
			py::module scipy_sparse_linalg = py::module::import("scipy.sparse.linalg");	
			py::object eigen;
			// cout << "PASSEI 10 " << endl;
			eigen = scipy_sparse_linalg.attr("eigsh")(L, k, //nullptr, nullptr, 
				py::arg("which") ="SM", py::arg("v0") = py::cast(vector<int>(L.rows(), 1.0)), 
				py::arg("ncv") = num_lanczos_vectors, py::arg("tol") = 1e-4, py::arg("maxiter") = graph.rows()*5);
			
			// cout << "PASSEI 11 " << endl;
			py::object eigenval = eigen.attr("__getitem__")(0);
			py::object eigenvec = eigen.attr("__getitem__")(1);
			vector<double> eigenvalues = eigenval.cast<vector<double>>();
			vector<vector<double>> eigenvectors = eigenvec.cast<vector<vector<double>>>();
			// cout << "PASSEI 12 " << endl;
			vector<int> order_all = utils::argsort(eigenvalues);
			vector<int> order(order_all.begin()+1, order_all.begin()+k);
			// cout << "PASSEI 13 " << endl;
	
			vector<vector<double>> component_embedding(eigenvectors.size());
			// cout << "PASSEI 14 " << endl;
			double max_value = -1.0;
			for( int i = 0; i < eigenvectors.size(); ++i ) {
				component_embedding[i] = utils::arrange_by_indices(eigenvectors[i], order);

				// if(  i < 10 )
				// 	printf("%.4f %.4f\n", spectral_embedding[i][0], spectral_embedding[i][1]);

				max_value = max(max_value, abs(*max_element(component_embedding[i].begin(), component_embedding[i].end(), 
					[](double a, double b) { 
						return abs(a) < abs(b);
					})));
			}
			// cout << "PASSEI 15 " << endl;
			double expansion = data_range/max_value;
			for( int i = 0; i < component_embedding.size(); ++i ) {
				transform(component_embedding[i].begin(), component_embedding[i].end(), 
					component_embedding[i].begin(), [expansion](double &c){ return c*expansion; });
			}

			// cout << "PASSEI 16 " << endl;
			// cout << component_embedding.size() << ", " << component_labels.size() << ", "
				 // << result.size() << endl;
			for( int i = 0, index = 0; i < component_labels.size(); ++i ) {
				if( component_labels[i] == label ) {
					transform(component_embedding[index].begin(), component_embedding[index].end(), 
					 meta_embedding[label].begin(), result[i].begin(), plus<double>());					
					index++;
				}
			}
			// cout << "PASSEI 17 " << endl;

		} catch(...) {
			wcout << "WARNING: spectral initialisation failed! The eigenvector solver\n" <<
                "failed. This is likely due to too small an eigengap. Consider\n" <<
                "adding some noise or jitter to your data.\n\n" <<
                "Falling back to random initialisation!" << endl;

			py::module scipy_random = py::module::import("numpy.random");
			py::object randomState = scipy_random.attr("RandomState")(this->random_state);
			vector<int> size = {(int)component_graph.rows(), dim};
			py::object noiseObj = randomState.attr("uniform")(py::arg("low")=-data_range, py::arg("high")=data_range, 
														     py::arg("size")=size);
			vector<vector<double>> noise = noiseObj.cast<vector<vector<double>>>();


			int row=0;

			for( int i = 0; i < component_labels.size(); ++i ) {

				if( component_labels[i] == label ) {
					for( int j = 0; j < noise[row].size(); ++j ) {
						result[i][j] = noise[row][j] + meta_embedding[label][j];
					}
					row++;
				}



			}
		}		


	}

	return result;


}

vector<vector<double>> umap::UMAP::spectral_layout(umap::Matrix& data, 
	const Eigen::SparseMatrix<double, Eigen::RowMajor>& graph, int dim)
{
	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;
	// random initialization
	cout << "init: " << this->init << endl;

	if( this->init != "Spectral" ) {
		cout << "Random initialization" << endl;
		py::module scipy_random = py::module::import("numpy.random");
		py::object randomState = scipy_random.attr("RandomState")(this->random_state);
		vector<int> size = {(int)graph.rows(), dim};
		py::object noiseObj = randomState.attr("uniform")(py::arg("low")=-10, py::arg("high")=10, py::arg("size")=size);


		return noiseObj.cast<vector<vector<double>>>();
	} 
	// else {
	// 	cout << "Spectral initialization" << endl;
	// 	py::module sklearn_manifold = py::module::import("sklearn.manifold");

	// 	py::object spectralEmbedding = sklearn_manifold.attr("SpectralEmbedding")(py::arg("n_components")=2, py::arg("affinity")="precomputed");
	// 	py::object embedding = spectralEmbedding.attr("fit_transform")(Eigen::MatrixXd(graph));

	// 	return embedding.cast<vector<vector<double>>>();
	// }



	int n_samples = graph.rows();

	// cout << "spectral 1" << endl;


	py::module csgraph = py::module::import("scipy.sparse.csgraph");
	py::object connected_components = csgraph.attr("connected_components")(graph);

	// cout << "spectral 2" << endl;
	int n_components = connected_components.attr("__getitem__")(0).cast<int>();
	vector<int> labels = connected_components.attr("__getitem__")(1).cast<vector<int>>();
	// cout << "spectral 3: " << n_components << endl;
	// cout << "n_components: " << n_components << endl;
	// cout << "labels[:10]" << endl;
	// for( int i = 0; i < 10; ++i )
	// 	cout << labels[i] << " ";
	// cout << endl;

	if( n_components > 1) {
		cout << "WARNING: found more than one component." << endl;
		vector<vector<double>> spectral_embedding = this->multi_component_layout(data, graph, n_components, labels, dim);
		cout << "Consegui passar de multi_component_layout" << endl;
		double max_value = spectral_embedding[0][0];
		for( int i = 0; i < spectral_embedding.size(); ++i )
			for( int j = 0; j <spectral_embedding[i].size(); ++j )
				max_value = max(max_value, spectral_embedding[i][j]);


		py::module scipy_random = py::module::import("numpy.random");
		py::object randomState = scipy_random.attr("RandomState")(this->random_state);
		vector<int> size = {(int)graph.rows(), n_components};
		py::object noiseObj = randomState.attr("normal")(py::arg("scale")=0.0001, py::arg("size")=size);

		vector<vector<double>> noise = noiseObj.cast<vector<vector<double>>>();
		double expansion = 10.0/max_value;
		for( int i = 0; i < spectral_embedding.size(); ++i ) {
			transform(spectral_embedding[i].begin(), spectral_embedding[i].end(), 
				spectral_embedding[i].begin(), [expansion](double &c){ return c*expansion; });
		}
		for( int i = 0; i < spectral_embedding.size(); ++i ) {
			transform(spectral_embedding[i].begin(), spectral_embedding[i].end(), 
				noise[i].begin(), spectral_embedding[i].begin(), plus<double>());
		}

		return spectral_embedding;
	}
	// cout << "spectral 4" << endl;
	Eigen::VectorXd result = graph * Eigen::VectorXd::Ones(graph.cols());
	// cout << "spectral 5" << endl;
	vector<double> diag_data(&result[0], result.data() + result.size());
	// vector<double> diag_data;


	// for( int i = 0; i < graph.cols(); ++i ) {
	// 	double s = 0;
	// 	for( int j  =0; j< graph.rows(); ++j )
	// 		s += graph.coeff(j, i);

	// 	diag_data.push_back(s);
	// }
	// cout << "diag_data[:10]" << endl;
	// for( int i = 0; i < 10; ++i )
		// cout << diag_data[i] << " ";
	// cout << endl;


	// cout << "spectral 6" << endl;
	vector<double> temp(diag_data.size(), 0.0);
	// cout << "spectral 7" << endl;
	for( int i = 0; i < temp.size(); ++i )
		temp[i] = 1.0/sqrt(diag_data[i]);
	// cout << "spectral 8" << endl;
	


	py::module scipy_sparse = py::module::import("scipy.sparse");
	// cout << "spectral 9" << endl;
	py::object Iobj = scipy_sparse.attr("identity")(graph.rows(), py::arg("format") = "csr");
	// cout << "spectral 10" << endl;
	py::object Dobj = scipy_sparse.attr("spdiags")(py::cast(temp), 0, graph.rows(), graph.rows(), py::arg("format") = "csr");
	// cout << "spectral 11" << endl;
	Eigen::SparseMatrix<double, Eigen::RowMajor> I = Iobj.cast<Eigen::SparseMatrix<double, Eigen::RowMajor>>();
	// cout << "spectral 12" << endl;
	Eigen::SparseMatrix<double, Eigen::RowMajor> D = Dobj.cast<Eigen::SparseMatrix<double, Eigen::RowMajor>>();
	// cout << "spectral 13" << endl;
	Eigen::SparseMatrix<double, Eigen::RowMajor> L = I-D*graph*D;
	// cout << "spectral 14" << endl;
	

	// for( int i = 0; i < 20; ++i ) {
	// 	auto row = L.row(i);
	// 	cout  << i << ": ";
	// 	cout << row.head(20);
	// 	cout << endl;
	// }


	int k = dim+1;
	int num_lanczos_vectors = max(2*k+1, (int)sqrt(graph.rows()));
	// cout << "spectral 15" << endl;
	// cout << "k: " << k << endl;
	// cout << "num_lanczos_vectors: " << num_lanczos_vectors << endl;


	try {
		py::module scipy_sparse_linalg = py::module::import("scipy.sparse.linalg");	
		py::object eigen;

		// cout << "spectral 16" << endl;
		if( L.rows() < 2000000 ) {

			auto tic = clock::now();
			// Eigen::MatrixXd dMat = Eigen::MatrixXd(L);
			// cout << "rows: " << L.rows() << " x cols: "  << L.cols() << endl;
			// Eigen::EigenSolver<Eigen::SparseMatrix<double, Eigen::RowMajor>> es(L);
			// Eigen::MatrixXd A = Eigen::MatrixXd::Random(6,6);
			// cout << "Here is a random 6x6 matrix, A:" << endl << A << endl << endl;
			 
			// Eigen::EigenSolver<Eigen::MatrixXd> es(L);
			// cout << "The eigenvalues of A are:" << endl << es.eigenvalues() << endl;
			// cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;
			sec toc = clock::now() - tic;
			// cout << "Sparse to dense with c++: " << toc.count() << endl;


			tic = clock::now();

			eigen = scipy_sparse_linalg.attr("eigsh")(L, k, //nullptr, nullptr, 
				py::arg("which") ="SM", py::arg("v0") = py::cast(vector<int>(L.rows(), 1.0)), 
				py::arg("ncv") = num_lanczos_vectors, py::arg("tol") = 1e-4, py::arg("maxiter") = graph.rows()*5);

			toc = clock::now() - tic;
			cout << "Eigen with python: " << toc.count() << endl;

			
			// cout << "The eigenvalues of A are:" << endl << es.eigenvalues() << endl;
			// cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;

			// cout << "spectral 17" << endl;
		} else {
		// cout << "spectral 18" << endl;
			throw new runtime_error("L.rows() >= 2000000. Not implemented yet.");

		}




// cout << "spectral 19" << endl;
		py::object eigenval = eigen.attr("__getitem__")(0);
		py::object eigenvec = eigen.attr("__getitem__")(1);
// cout << "spectral 20" << endl;
		vector<double> eigenvalues = eigenval.cast<vector<double>>();
		vector<vector<double>> eigenvectors = eigenvec.cast<vector<vector<double>>>();
		// cout << "eigenvalues" << endl;
		// for( int i = 0; i < 10; ++i )
		// 	printf("%.4f ", eigenvalues[i]);
		// cout << endl << endl;

		// cout << "eigenvectors" << endl;
		// for( int i = 0; i < 10; ++i )
		// {
			// for(int j = 0; j < eigenvectors[i].size(); ++j )
				// printf("%.4f ", eigenvectors[i][j]);
			// cout << endl;
		// }
// cout << "spectral 21" << endl;
		vector<int> order_all = utils::argsort(eigenvalues);
		vector<int> order(order_all.begin()+1, order_all.begin()+k);
// cout << "spectral 22" << endl;
		vector<vector<double>> spectral_embedding(eigenvectors.size());
// cout << "spectral 23" << endl;
		// printf("embedding: \n");
		double max_value = -1.0;
		for( int i = 0; i < eigenvectors.size(); ++i ) {
			spectral_embedding[i] = utils::arrange_by_indices(eigenvectors[i], order);
// 
			// if(  i < 10 )
				// printf("%.4f %.4f\n", spectral_embedding[i][0], spectral_embedding[i][1]);

			max_value = max(max_value, 
				abs(*max_element(spectral_embedding[i].begin(), spectral_embedding[i].end(), [](double a, double b) { return abs(a) < abs(b);})));
		}
// cout << "spectral 24" << endl;
		// printf("\nmax_value: %.4f\n", max_value);

		py::module scipy_random = py::module::import("numpy.random");
		py::object randomState = scipy_random.attr("RandomState")(this->random_state);
		vector<int> size = {(int)graph.rows(), n_components};
		py::object noiseObj = randomState.attr("normal")(py::arg("scale")=0.0001, py::arg("size")=size);
// cout << "spectral 25" << endl;
		vector<vector<double>> noise = noiseObj.cast<vector<vector<double>>>();
// cout << "spectral 26" << endl;
		double expansion = 10.0/max_value;
// cout << "spectral 27" << endl;
		for( int i = 0; i < spectral_embedding.size(); ++i ) {
			transform(spectral_embedding[i].begin(), spectral_embedding[i].end(), 
				spectral_embedding[i].begin(), [expansion](double &c){ return c*expansion; });
		}
// cout << "spectral 28" << endl;
		for( int i = 0; i < spectral_embedding.size(); ++i ) {
			transform(spectral_embedding[i].begin(), spectral_embedding[i].end(), 
				noise[i].begin(), spectral_embedding[i].begin(), plus<double>());
		}
// cout << "spectral 29" << endl;
		// printf("\n\n\nFINAL EMBEDDING:\n\n");

		// for( int i = 0; i < 10; ++i ) {
		// 	if(  i < 10 )
		// 		printf("%.4f %.4f\n", spectral_embedding[i][0], spectral_embedding[i][1]);
		// }






		return spectral_embedding;

	} catch(...) {
		wcout << "WARNING (Spectral Layout): spectral initialisation failed! The eigenvector solver\n" <<
                "failed. This is likely due to too small an eigengap. Consider\n" <<
                "adding some noise or jitter to your data.\n\n" <<
                "Falling back to random initialisation!" << endl;
                
		py::module scipy_random = py::module::import("numpy.random");
		py::object randomState = scipy_random.attr("RandomState")(this->random_state);
		vector<int> size = {(int)graph.rows(), dim};
		py::object noiseObj = randomState.attr("uniform")(py::arg("low")=-10, py::arg("high")=10, py::arg("size")=size);
		return noiseObj.cast<vector<vector<double>>>();
	}


}

vector<vector<double>> umap::pairwise_distances(umap::Matrix& X, std::string metric)
{

  int n = X.shape(0);
  int d = X.shape(1);


  vector<vector<double>> pd(n, vector<double>(n, 0.0));


  // TODO: add possibility for other distance functions
  for( int i = 0; i < n; ++i ) {
    for( int j = i+1; j < n; ++j ) {

      double distance = 0;

      for( int k = 0; k < d; ++k ) {
       // distance += (X[i*d + k]-X[j*d + k])*(X[i*d + k]-X[j*d + k]);
      	distance += (X.dense_matrix[i][k]-X.dense_matrix[j][k])*(X.dense_matrix[i][k]-X.dense_matrix[j][k]);
      }

      pd[i][j] = sqrt(distance);
      pd[j][i] = pd[i][j];
    }
  }

  return pd;
}

tuple<vector<int>, vector<int>, vector<double>, vector<double>> umap::compute_membership_strenghts(
	vector<vector<int>>& knn_indices, vector<vector<double>>& knn_dists, 
	vector<double>& sigmas, vector<double>& rhos) 
{

	int n_samples = knn_indices.size();
	int n_neighbors = knn_indices[0].size();
	int size = n_samples*n_neighbors;

	vector<int> rows(size, 0);
	vector<int> cols(size, 0);
	vector<double> vals(size, 0.0);


	vector<double> sum_vals(n_samples, 0.0);


	for( int i = 0; i < n_samples; ++i )
	{
		double sum = 0.0;
	
		for( int j = 0; j < n_neighbors; ++j )
		{
			if( knn_indices[i][j] == -1 )
				continue;
			double val = 0.0;
			if( knn_indices[i][j] == i )
				val = 0.0;
			else if( knn_dists[i][j]-rhos[i] <= 0.0 || sigmas[i] == 0.0 )
				val = 1.0;
			else 
				val = exp(-((knn_dists[i][j] - rhos[i])/ (sigmas[i])));

			rows[i * n_neighbors + j] = i;
            cols[i * n_neighbors + j] = knn_indices[i][j];
            vals[i * n_neighbors + j] = val;
            sum += val;
		}

		sum_vals[i] = sum;
	}

	return make_tuple(rows, cols, vals, sum_vals);

}

tuple<vector<double>, vector<double>> umap::smooth_knn_dist(vector<vector<double>>& distances,
	double k, int n_iter, double local_connectivity, double bandwidth)
{

	double target = log2(k) * bandwidth;
	vector<double> rho(distances.size(), 0);
	vector<double> result(distances.size(), 0);

	double mean_distances = 0.0;

	#pragma omp parallel for
	for (int i = 0; i < distances.size(); ++i)
	{
		mean_distances += accumulate( distances[i].begin(), distances[i].end(), 0.0);
	}
	mean_distances /= (distances.size()*distances[0].size());

	// cout << "target: " << target << endl;
	// cout << "mean_distances: " << mean_distances << endl;
	#pragma omp parallel for
	for( int i = 0; i < distances.size(); ++i ) {

		double lo = 0.0;
		double hi = numeric_limits<double>::max();
		double mid = 1.0;

		vector<double> ith_distances = distances[i];
		vector<double> non_zero_dists;
		copy_if(ith_distances.begin(), ith_distances.end(), back_inserter(non_zero_dists), [](double d){ return d > 0.0; });

		if( non_zero_dists.size() >= local_connectivity ) {

			int index = (int) floor(local_connectivity);
			double interpolation = local_connectivity - (double)index;


			if( index > 0 ) {

				rho[i] = non_zero_dists[index-1];

				if( interpolation > umap::SMOOTH_K_TOLERANCE ) {

					rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index-1]);

				}

			} else {

				rho[i] = interpolation * non_zero_dists[0];

			}

		} else if( non_zero_dists.size() > 0 ) {

			rho[i] = *max_element(non_zero_dists.begin(), non_zero_dists.end());

		}

		// cout << "n_iter " << n_iter << endl;
		// cout << "local_connectivity " << local_connectivity << endl;
		// cout << "bandwidth " << bandwidth << endl;
		// cout << "rho[i]3 " <<  rho[i] << endl;
		for( int n = 0; n < n_iter; ++n ) {

			double psum = 0.0;

			for( int j = 1;  j < distances[0].size(); ++j ) {

				double d = distances[i][j] - rho[i];
				psum += (d > 0.0 ? exp(-(d/mid)) : 1.0);
			}

			if( fabs(psum-target) < umap::SMOOTH_K_TOLERANCE )
				break;

			if( psum > target ) {

				hi = mid;
				mid = (lo+hi)/2.0;
			} else {
				lo = mid;
				if( hi == numeric_limits<double>::max() )
					mid *= 2;
				else
					mid = (lo+hi)/2.0;
			}
		}
		result[i] = mid;

		if( rho[i] > 0.0 ) {
			double mean_ith_distances = accumulate(ith_distances.begin(), ith_distances.end(), 0.0)/ith_distances.size();
			if( result[i] < umap::MIN_K_DIST_SCALE*mean_ith_distances )
				result[i] = umap::MIN_K_DIST_SCALE*mean_ith_distances;

		} else {
			if( result[i] < umap::MIN_K_DIST_SCALE*mean_distances )
				result[i] = umap::MIN_K_DIST_SCALE*mean_distances;
		}




	}

	return make_tuple(result, rho);
}

tuple<vector<vector<int>>, vector<vector<double>>> umap::nearest_neighbors(umap::Matrix& X,
	int n_neighbors, string metric, bool angular, double random_state, map<string, string> knn_args, bool verbose)
{

	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;
	if( verbose )
		cout << "Finding nearest neighbors" << endl;

	vector<vector<int>> knn_indices;
	vector<vector<double>> knn_dists;

	if( metric == "precomputed" ) {

		knn_indices = vector<vector<int>>(X.size(), vector<int>(n_neighbors, 0));
		knn_dists = vector<vector<double>>(X.size(), vector<double>(n_neighbors, 0.0));

		// #pragma omp parallel for default(shared)		
		for( int i = 0; i < knn_indices.size(); ++i )
		{
			vector<double> row_data = X.dense_matrix[i];
			vector<int> sorted_indices = utils::argsort(row_data);
			vector<int> row_nn_data_indices(sorted_indices.begin(), sorted_indices.begin()+n_neighbors);

			knn_indices[i] = row_nn_data_indices;
			knn_dists[i] = utils::arrange_by_indices<double>(row_data, row_nn_data_indices);

		}

		
	} else {
		string algorithm = knn_args["knn_algorithm"];

		if( algorithm == "FLANN" ) {

			py::module pyflann = py::module::import("pyflann");
			py::module flann = pyflann.attr("FLANN")();
			py::array_t<double> data = py::cast(X.dense_matrix);
			py::object result = flann.attr("nn")(
				data, data, n_neighbors,
				py::arg("checks")=128
				,
				py::arg("trees")=3
				,
				py::arg("iterations")=15
				);//, py::arg("checks")=256, py::arg("trees")=3);


			py::object knn_indices_ = result.attr("__getitem__")(0);
			py::object knn_dists_ = result.attr("__getitem__")(1);

			knn_dists = knn_dists_.cast<vector<vector<double>>>();
			knn_indices = knn_indices_.cast<vector<vector<int>>>();

		} else if( algorithm == "ANNOY" ) {

			py::module annoy = py::module::import("annoy");
			py::object t = annoy.attr("AnnoyIndex")(
				py::cast(X.shape(1)), py::cast("euclidean")
				);

			for( int i = 0; i < X.shape(0); ++i ) {
				py::array_t<double> values = py::cast(X.dense_matrix[i]);
				t.attr("add_item")(py::cast(i), values);				
			}

			t.attr("build")(100, py::arg("n_jobs")=-1);


			for( int i = 0; i < X.shape(0); ++i ) {
				py::object result = t.attr("get_nns_by_item")(
					py::cast(i), n_neighbors, py::arg("include_distances")=true
					);

				vector<int> indices = result.attr("__getitem__")(0).cast<vector<int>>();
				vector<double> dists = result.attr("__getitem__")(1).cast<vector<double>>();

				knn_dists.push_back(dists);
				knn_indices.push_back(indices);
			}


			for( int i = 0; i < 5; ++i ) {
				cout << i <<" ";
				for( int j = 0; j < knn_dists[i].size(); ++j ) {

					cout << knn_dists[i][j]  << "(" <<  knn_indices[i][j] << "); ";

				}
				cout << endl;
			}
			cout << endl;






		} else if( algorithm == "NNDescent" ) {
			auto prep_before = clock::now();
			cout << "L: " << knn_args["L"] << endl;
			cout << "iter: " << knn_args["iter"] << endl;
			cout << "S: " << knn_args["S"] << endl;
			cout << "R: " << knn_args["R"] << endl;
			unsigned L = (unsigned) stoi(knn_args["L"]);
			unsigned iter = (unsigned) stoi(knn_args["iter"]);
			unsigned S = (unsigned) stoi(knn_args["S"]);
			unsigned R = (unsigned) stoi(knn_args["R"]);
			unsigned K = (unsigned) n_neighbors-1;


			efanna2e::IndexRandom init_index(X.shape(1), X.shape(0));
			efanna2e::IndexGraph index(X.shape(1), X.shape(0), efanna2e::L2, (efanna2e::Index*)(&init_index));

			efanna2e::Parameters params;
			params.Set<unsigned>("K", K); // the number of neighbors to construct the neighbor graph
			params.Set<unsigned>("L", L); // how many neighbors will I check for refinement?
			params.Set<unsigned>("iter", iter); // the number of iterations
			params.Set<unsigned>("S", S); // how many numbers of points in the leaf node; candidate pool size
			params.Set<unsigned>("R", R); 
			sec preparation = clock::now() - prep_before;
			cout << "Preparation time: " << preparation.count() << endl;
			auto before = clock::now();
			float* data = X.data_f();
			sec duration = clock::now() - before;
			cout << "Constructing array time: " << duration.count() << endl;
			before = clock::now();
			index.Build(X.shape(0), data, params);
			duration = clock::now() - before;
			

			delete data;

			knn_indices = vector<vector<int>>(X.size(), vector<int>(n_neighbors, 0));
			knn_dists = vector<vector<double>>(X.size(), vector<double>(n_neighbors, 0.0));

			#pragma omp parallel for
			for( int i = 0; i < X.shape(0); ++i ) 
			{
				knn_dists[i][0] = 0.0;
				knn_indices[i][0] = i;

				for( int j = 0; j < K; ++j ) {
					knn_dists[i][j+1] = (double) index.graph_[i].pool[j].distance;
					knn_indices[i][j+1] = index.graph_[i].pool[j].id;

				}	
			}

			cout << "Building and Creating time: " << duration.count() << endl;

		} else if( algorithm == "KDTree_NNDescent" ) {


			unsigned nTrees = (unsigned) stoi(knn_args["nTrees"]);
			unsigned mLevel = (unsigned) stoi(knn_args["mLevel"]);
			unsigned L = (unsigned) stoi(knn_args["L"]);
			unsigned iter = (unsigned) stoi(knn_args["iter"]);
			unsigned S = (unsigned) stoi(knn_args["S"]);
			unsigned R = (unsigned) stoi(knn_args["R"]);
			unsigned K = (unsigned) n_neighbors;



			float* data = X.data_f();
			unsigned ndims = (unsigned) X.shape(1);
			unsigned nsamples = (unsigned) X.shape(0);


			cout << "ndims: " << ndims << endl;
			cout << "nsamples: " << nsamples << endl;

			cout << "nTrees: " << knn_args["nTrees"] << endl;
			cout << "mLevel: " << knn_args["mLevel"] << endl;

			cout << "L: " << knn_args["L"] << endl;
			cout << "iter: " << knn_args["iter"] << endl;
			cout << "S: " << knn_args["S"] << endl;
			cout << "R: " << knn_args["R"] << endl;



			float* data_aligned = efanna2e::data_align(data, nsamples, ndims);
			efanna2e::IndexKDtree index_kdtree(ndims, nsamples, efanna2e::L2, nullptr);

			efanna2e::Parameters params_kdtree;
			params_kdtree.Set<unsigned>("K", K);
			params_kdtree.Set<unsigned>("nTrees", nTrees);
			params_kdtree.Set<unsigned>("mLevel", mLevel);

			index_kdtree.Build(nsamples, data_aligned, params_kdtree);


			efanna2e::IndexRandom init_index(ndims, nsamples);
			efanna2e::IndexGraph index_nndescent(ndims, nsamples, efanna2e::L2, (efanna2e::Index*)(&init_index));

			index_nndescent.final_graph_ = index_kdtree.final_graph_;
			// index_kdtree.Save("mnist.graph");
			// index_nndescent.Load("mnist.graph");


			efanna2e::Parameters params_nndescent;
			params_nndescent.Set<unsigned>("K", K);
			params_nndescent.Set<unsigned>("L", L);
			params_nndescent.Set<unsigned>("iter", iter);
			params_nndescent.Set<unsigned>("S", S);
			params_nndescent.Set<unsigned>("R", R);

			index_nndescent.RefineGraph(data_aligned, params_nndescent);

		


			knn_indices = vector<vector<int>>(X.size(), vector<int>(n_neighbors, 0));
			knn_dists = vector<vector<double>>(X.size(), vector<double>(n_neighbors, 0.0));

			// #pragma omp parallel for
			for( int i = 0; i < nsamples; ++i ) 
			{

				knn_dists[i][0] = 0.0;
				knn_indices[i][0] = i;
				for( int j = 0; j < K-1; ++j ) {
					knn_dists[i][j+1] = (double) index_nndescent.graph_[i].pool[j].distance;
					knn_indices[i][j+1] = index_nndescent.graph_[i].pool[j].id;

				}	
				// cout << endl;

			}
			free(data_aligned);
		}

	}

	return make_tuple(knn_indices, knn_dists);
}


tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, vector<double>, vector<double>> umap::fuzzy_simplicial_set(
	umap::Matrix& X, int n_neighbors, double random_state, string metric, 
	vector<vector<int>>& knn_indices, vector<vector<double>>& knn_dists,
	bool angular, double set_op_mix_ratio, double local_connectivity,
	bool apply_set_operations, bool verbose, umap::UMAP* obj)
{

	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;

	if( knn_indices.size() == 0 || knn_dists.size() == 0 ) {
		tie(knn_indices, knn_dists) = umap::nearest_neighbors(X, n_neighbors, metric, angular, random_state, obj->knn_args, verbose);
	} 



	// cout << "knn_indices:" << endl;
	// for( int i = 0; i < 5; ++i ) {
	// 	for( int j = 0; j < knn_indices[i].size(); ++j )
	// 		cout << knn_indices[i][j] << " ";

	// 	cout << endl;
	// }

	// cout << "\nknn_dists:" << endl;
	// for( int i = 0; i < 5; ++i ) {
	// 	for( int j = 0; j < knn_dists[i].size(); ++j )
	// 		cout << knn_dists[i][j] << " ";

	// 	cout << endl;
	// }




	vector<double> sigmas, rhos;
	auto tic = clock::now();
	tie(sigmas, rhos) = umap::smooth_knn_dist(knn_dists, (double) n_neighbors, 64, local_connectivity);
	sec toc = clock::now() - tic;
	cout << "Smooth knn dist: " << toc.count() << endl;
	// cout << "sigmas" << endl;
	// for (int i = 0; i < 20; ++i)
	// {
	// 	cout << sigmas[i] << " ";
	// }
	// cout << endl;

	// cout << "rhos" << endl;
	// for (int i = 0; i < 20; ++i)
	// {
	// 	cout << rhos[i] << " ";
	// }
	// cout << endl;

	// cout << endl << endl;
	

	vector<int> rows, cols;
	vector<double> vals, sum_vals;
	tic = clock::now();
	tie(rows, cols, vals, sum_vals) = umap::compute_membership_strenghts(knn_indices, knn_dists, sigmas, rhos);
	toc = clock::now() - tic;
	cout << "Compute membership strenghts: " << toc.count() << endl;

	if( obj ) {

		obj->rows = rows;
		obj->cols = cols;
		obj->vals = vals;
		obj->sum_vals = sum_vals;
	}
	vector<double> vals_transition(vals.begin(), vals.end());

	// cout << "rows: " << endl;
	// for( int i = 0; i < 20; ++i ) {
	// 	cout << rows[i] << " ";
	// }
	// cout << endl;

	// cout << "cols: " << endl;
	// for( int i = 0; i < 20; ++i ) {
	// 	cout << cols[i] << " ";
	// }
	// cout << endl;

	// cout << "vals: " << endl;
	// for( int i = 0; i < 20; ++i ) {
	// 	cout << vals[i] << " ";
	// }
	// cout << endl;

	// cout << endl << endl;

	// TODO: evaluate the complexity of this method


	tic = clock::now();
	Eigen::SparseMatrix<double, Eigen::RowMajor> result(X.size(), X.size());
	result.reserve(Eigen::VectorXi::Constant(X.size(), 2*n_neighbors)); // TODO: verificar se é assim (ou com int)
	// result.reserve(rows*2*n_neighbors);
	
	#pragma omp parallel for
	for( int i = 0; i < vals.size(); ++i ) {
		result.insert(rows[i], cols[i]) = vals[i];
		vals_transition[i] = vals_transition[i]/sum_vals[rows[i]];
	}
	obj->vals_transition = vals_transition;
	result.makeCompressed();

	toc = clock::now() - tic;
	cout << "Creating sparse matrix (result): " << toc.count() << endl;


	tic = clock::now();
	obj->transition_matrix = 1.0*result;//Eigen::SparseMatrix<double, Eigen::RowMajor>(result);
	toc = clock::now() - tic;
	cout << "Creating transition_matrix: " << toc.count() << endl;

	if( apply_set_operations ) {

		tic = clock::now();
		Eigen::SparseMatrix<double, Eigen::RowMajor> transpose = result.transpose();
		result = 0.5 * (result + transpose);
		toc = clock::now() - tic;
		cout << "Applying set operations: " << toc.count() << endl;
	}


	// for( int i = 0; i < 20; ++i ) {
	// 	auto row = result.row(i);
	// 	cout  << i << ": ";
	// 	cout << row.head(20);
	// 	cout << endl;
	// }




	return make_tuple(result, sigmas, rhos);

}

tuple<double, double> umap::find_ab_params(double spread, double min_dist) 
{

	// py::initialize_interpreter();

	// py::scoped_interpreter guard;

	py::module np = py::module::import("numpy");
	py::object random = np.attr("random");
	py::module scipy = py::module::import("scipy.optimize");

	py::module curve_module = py::module::import("CurveModule");

	vector<double> xv = utils::linspace<double>(0, spread*3, 300);
	vector<double> yv = vector<double>(xv.size(), 0.0);


	for( int i = 0; i < yv.size(); ++i )
	{
		if( xv[i] < min_dist )
		{
			yv[i] = 1.0;
		} else {
			yv[i] = exp(-(xv[i]-min_dist)/spread);
		}
	}

	py::array_t<double> pyXV = py::cast(xv);
	py::array_t<double> pyYV = py::cast(yv);

	py::function pyCurve = curve_module.attr("fitting_curve");

	py::function curve_fit = scipy.attr("curve_fit");

	py::object ret = curve_fit(pyCurve, pyXV, pyYV);

	py::object ab = ret.attr("__getitem__")(0);

	vector<double> vals = ab.cast<vector<double>>();

	// py::finalize_interpreter();


	return make_tuple(vals[0], vals[1]);
	// return make_tuple(1.0, 1.0);
}


void umap::UMAP::prepare_for_fitting(umap::Matrix& X) 
{

	// TODO X = check_array(X)
	// What does check_array do?

	if( this->a == -1.0 || this->b == -1.0 ) {
		tie(this->_a, this->_b) = umap::find_ab_params(this->spread, this->min_dist);
	} else {
		this->_a = this->a;
		this->_b = this->b;
	}

	// TODO: isinstance 
	// dinamically check if the initial positions are:
	// spectral, random, or  a dimensional embedding

	// TODO: validate parameterers

	// TODO: check for unique data

	vector<int> index(X.size());
	vector<int> inverse(X.size());


	iota(index.begin(), index.end(), 0);
	iota(inverse.begin(), inverse.end(), 0);


	if( X.size() <= this->n_neighbors ) {

		// TODO: address when n == 1

		wcout << "n_neighbors is larger than the dataset size; truncating to X.shape[0] -1" << endl;

		this->_n_neighbors = X.size()-1;

	} else {

		this->_n_neighbors = this->n_neighbors;

	}

	// TODO: check if sorting the indices is needed

	// TODO: check random_state

	if( this->verbose )
		cout << "Constructing fuzzy simplical set" << endl;

	if( this->metric == "precomputed" && this->_sparse_data ) {


		if( this->verbose )
			cout << "Computing KNNs for sparse precomputed distancess..." << endl;

		// TODO: implement diagonal distance
		// if( !this->check_diagonal_distance() )
		// 	throw runtime_error("Non-zero distances from samples to themselves");

		this->_knn_indices = vector<vector<int>>(X.size(), vector<int>(this->n_neighbors, 0));
		this->_knn_dists = vector<vector<double>>(X.size(), vector<double>(this->n_neighbors, 0.0));

		for( int row_id = 0; row_id < X.size(); ++row_id ) {
			vector<double> row_data = X.sparse_matrix[row_id].data;
			vector<int> row_indices = X.sparse_matrix[row_id].indices;


			if( row_data.size() < this->n_neighbors ) {
				cout << "row_id: " << row_id << ", " << row_data.size() << " " << this->n_neighbors << endl;

				// vector<int> temp(X.shape(1), 0);
				// for( int i = 0; i < row_indices.size(); ++i )
				// 	temp[row_indices[i]] = 1;
				// for( int i = 0; i < X.shape(1); ++i ) {

				// 	if( ! )

				// }

				throw runtime_error("Some rows contain fewer than n_neighbors distances");
			}

			vector<int> sorted_indices = utils::argsort(row_data);
			vector<int> row_nn_data_indices(sorted_indices.begin(), sorted_indices.begin() + this->n_neighbors);

			this->_knn_indices[row_id] = utils::arrange_by_indices<int>(row_indices, row_nn_data_indices);
			this->_knn_dists[row_id] = utils::arrange_by_indices<double>(row_data, row_nn_data_indices);

		}
			
		// cout << "knn indices: " << endl;
		// for( int i = 0; i < 3; ++i ) {
		// 	for( int j = 0; j < this->_knn_indices[i].size(); ++j )
		// 		cout << this->_knn_indices[i][j] << " ";
		// 	cout << endl;
		// }

		// cout << "knn dists: " << endl;
		// for( int i = 0; i < 3; ++i ) {
		// 	for( int j = 0; j < this->_knn_dists[i].size(); ++j )
		// 		printf("%.5f ", this->_knn_dists[i][j] );
		// 	cout << endl;
		// }
		// cout << endl;

		if( this->verbose )
			cout << "Computing fuzzy simplicial set..." << endl;

		X.sparse_matrix = utils::arrange_by_indices<utils::SparseData>(X.sparse_matrix, index);
		// TODO: Will I be using the this->_metric_kwds?	
		tie(this->graph_, this->_sigmas, this->_rhos) = umap::fuzzy_simplicial_set(
			X,
			this->n_neighbors,
			this->random_state,
			"precomputed",
		//	this->_metric_kwds,
			this->_knn_indices,
			this->_knn_dists,
			this->angular_rp_forest,
			this->set_op_mix_ratio,
			this->local_connectivity,
			true,
			this->verbose,
			this);

	} else if( X.size() < 100 && !this->force_approximation_algorithm ) {

		if( this->verbose )
			cout << "Small matrix. Computing pairwise distances." << endl;

		// TODO: check for erros when computing pairwise distance

		vector<vector<double>> dmat = umap::pairwise_distances(X);
		this->pairwise_distance = umap::Matrix(dmat);

		tie(this->graph_, this->_sigmas, this->_rhos) = umap::fuzzy_simplicial_set(
			this->pairwise_distance,
			this->_n_neighbors,
			random_state,
			"precomputed",
			//this->_metric_kwds,
			this->_knn_indices,
			this->_knn_dists,
			this->angular_rp_forest,
			this->set_op_mix_ratio,
			this->local_connectivity,
			true,
			this->verbose,
			this);


		// TODO: Do I have to compute the nearest neighbors?
	} else {

		if( this->verbose )
			cout << "Normal case, computing nearest neighbors and then fuzzy simplicial set" << endl;
		// TODO: does nndescent have an implementation in c++?
		// if so, try to accomodate distance functions
		string nn_metric = this->metric;

		tie(this->_knn_indices, this->_knn_dists) = umap::nearest_neighbors(
			X,
			this->_n_neighbors,
			nn_metric,
			//this->_metric_kwds,
			this->angular_rp_forest,
			random_state,
			this->knn_args,
			verbose=this->verbose);


		// cout << "knn indices: " << endl;
		// for( int i = 0; i < 3; ++i ) {
		// 	for( int j = 0; j < this->_knn_indices[i].size(); ++j )
		// 		cout << this->_knn_indices[i][j] << " ";
		// 	cout << endl;
		// }

		// cout << "knn dists: " << endl;
		// for( int i = 0; i < 3; ++i ) {
		// 	for( int j = 0; j < this->_knn_dists[i].size(); ++j )
		// 		printf("%.1f ", this->_knn_dists[i][j] );
		// 	cout << endl;
		// }

		// cout << endl;

		// return; 


		tie(this->graph_, this->_sigmas, this->_rhos) = umap::fuzzy_simplicial_set(
			X,
			this->n_neighbors,
			random_state,
			nn_metric,
			//this->_metric_kwds,
			this->_knn_indices,
			this->_knn_dists,
			this->angular_rp_forest,
			this->set_op_mix_ratio,
			this->local_connectivity,
			true,
			this->verbose,
			this);

	}

}

void umap::UMAP::fit(py::array_t<double> X) 
{


	// TODO conver X to Matrix
	// this->prepare_for_fitting(X);

	// TODO: Do I have to put the 'index' and 'inverse'?
	// TODO: Check the other parameters...
	// this->embedding_ = umap::simplicial_set_embedding(
	// 	this->_raw_data,
	// 	this->graph_,
	// 	this->n_components,
	// 	this->_initial_alpha,
	// 	this->_a,
	// 	this->_b,
	// 	this->repulsion_strength,
	// 	this->negative_sample_rate,
	// 	this->n_epochs,
	// 	this->init,
	// 	this->random_state,
	// 	// this->_init_distance_func,
	// 	//this->_metric_kwds,
	// 	// this->_output_distance_func,
	// 	// this->_output_metric_kwds,
	// 	true, // this->output_metric in ('euclidean', 'l2')
	// 	this->random_state == -1,
	// 	this->verbose
	// 	);
}

void umap::UMAP::fit_hierarchy_sparse(const Eigen::SparseMatrix<double, Eigen::RowMajor>& X)
{

	vector<int> rows, cols;
	vector<double> vals;
	
	tie(rows, cols, vals) = utils::to_row_format(X);

	vector<utils::SparseData> sparse_matrix(X.rows(), utils::SparseData());


	for( int i = 0; i < rows.size(); ++i ) {
		sparse_matrix[rows[i]].push(cols[i], vals[i]);
	}

	this->dataset = umap::Matrix(sparse_matrix);
	this->_sparse_data = true;

	this->prepare_for_fitting(this->dataset);
}


void umap::UMAP::fit_hierarchy_sparse(const vector<utils::SparseData>& X)
{

	this->dataset = umap::Matrix(X);
	this->_sparse_data = true;

	this->prepare_for_fitting(this->dataset);
}


// void umap::UMAP::fit_hierarchy_dense(py::array_t<double> X)
void umap::UMAP::fit_hierarchy_dense(vector<vector<double>> X)
{
	this->dataset = umap::Matrix(X);
	this->_sparse_data = false;

	this->prepare_for_fitting(this->dataset);
}

void umap::UMAP::fit_hierarchy(const umap::Matrix& X) 
{
	if( X.is_sparse() )
		this->fit_hierarchy_sparse(X.sparse_matrix);
	else
		this->fit_hierarchy_dense(X.dense_matrix);
}

vector<vector<double>> umap::UMAP::fit_transform(py::array_t<double> X)
{


	this->fit(X);
	return this->embedding_;
}
