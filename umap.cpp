#include "umap.h"
#include "utils.h"

namespace py = pybind11;
using namespace std;

void umap::UMAP::optimize_euclidean_epoch(vector<vector<float>>& head_embedding, vector<vector<float>>& tail_embedding,
										   const vector<int>& head, const vector<int>& tail, int n_vertices, 
										   const vector<float>& epochs_per_sample, float a, float b, vector<long>& rng_state, 
										   float gamma, int dim, bool move_other, float alpha, vector<float>& epochs_per_negative_sample,
										   vector<float>& epoch_of_next_negative_sample, vector<float>& epoch_of_next_sample, int n)
{
	#pragma omp parallel for
	for( int i = 0; i < epochs_per_sample.size(); ++i ) {

		if( epoch_of_next_sample[i] <= n ) {


			int j = head[i];
			int k = tail[i];

			vector<float>* current = &head_embedding[j];
			vector<float>* other = &tail_embedding[k];
			

			float dist_squared = utils::rdist((*current), (*other));

			float grad_coeff = 0.0;

			if( dist_squared > 0.0 ) {
				grad_coeff = -2.0 * a * b * pow(dist_squared, b-1.0);
				grad_coeff /= a * pow(dist_squared, b) + 1.0;

			}

			for( int d = 0; d < dim; ++d ) {

				float grad_d = utils::clip(grad_coeff * ((*current)[d] - (*other)[d]));
				(*current)[d] += (grad_d * alpha);
				if( move_other )
					(*other)[d] += (-grad_d * alpha);
			}


			epoch_of_next_sample[i] += epochs_per_sample[i];

			int n_neg_samples = (int) ((n-epoch_of_next_negative_sample[i])/epochs_per_negative_sample[i]);

			for( int p = 0; p < n_neg_samples; ++p ) {
				int k = utils::tau_rand_int(rng_state) % n_vertices;

				other = &tail_embedding[k];

				dist_squared = utils::rdist(*current, *other);

				if( dist_squared > 0.0 ) {
					grad_coeff = 2.0 * gamma * b;
					grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1.0);
				} else if( j == k ) {
					continue;
				} else 
					grad_coeff = 0.0;

				for( int d = 0; d < dim; ++d ) {
					float grad_d = 0.0;
					if( grad_coeff > 0.0 )
						grad_d = utils::clip(grad_coeff * ((*current)[d] - (*other)[d]));
					else
						grad_d = 4.0;
					(*current)[d] += (grad_d * alpha);
				}

			}



			epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i]);
		}
	}
}

vector<vector<float>> umap::UMAP::optimize_layout_euclidean(vector<vector<float>>& head_embedding, vector<vector<float>>& tail_embedding,
										   const vector<int>& head, const vector<int>& tail, int n_epochs, int n_vertices, 
										   const vector<float>& epochs_per_sample, vector<long>& rng_state)
{
	float a = this->_a;
	float b = this->_b;
	float gamma = this->repulsion_strength;
	float initial_alpha = this->_initial_alpha;
	float negative_sample_rate = this->negative_sample_rate;


	int dim = head_embedding[0].size();
	bool move_other = head_embedding.size() == tail_embedding.size();
	float alpha = initial_alpha;

	// cout << "a: " << a << endl;
	// cout << "b: " << b << endl;
	// cout << "gamma: " << gamma << endl;
	// cout << "initial_alpha: " << initial_alpha << endl;
	// cout << "negative_sample_rate: " << negative_sample_rate << endl;
	// cout << "dim: " << dim << endl;
	// cout << "move_other: " << move_other << endl;
	// cout << "alpha: " << alpha << endl;




	vector<float> epochs_per_negative_sample(epochs_per_sample.size(), 0.0);
	transform(epochs_per_sample.begin(), epochs_per_sample.end(), epochs_per_negative_sample.begin(), 
		[negative_sample_rate](float a) {
			return a/negative_sample_rate;
		});

	// cout << "epochs_per_negative_sample" << endl;
	// for( int i = 0; i < 10; ++i )
	// 	printf("%.4f ", epochs_per_negative_sample[i]);
	// cout << endl;

	vector<float> epoch_of_next_negative_sample(epochs_per_negative_sample.begin(), epochs_per_negative_sample.end());
	// cout << "epoch_of_next_negative_sample" << endl;
	// for( int i = 0; i < 10; ++i )
	// 	printf("%.4f ", epoch_of_next_negative_sample[i]);
	// cout << endl;

	vector<float> epoch_of_next_sample(epochs_per_sample.begin(), epochs_per_sample.end());
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

		alpha = initial_alpha * (1.0 - ((float)epoch/(float)n_epochs));


		if( this->verbose && epoch % (int)(n_epochs/10) == 0)
			printf("\tcompleted %d / %d epochs\n", epoch, n_epochs);

	}
	printf("\tcompleted %d epochs\n", n_epochs);


	return head_embedding;

}

vector<float> umap::UMAP::make_epochs_per_sample(const vector<float>& weights, int n_epochs)
{

	vector<float> result(weights.size(), -1.0);
	vector<float> n_samples(weights.size(), 0.0);

	float max_weight = *max_element(weights.begin(), weights.end());

	transform(weights.begin(), weights.end(), n_samples.begin(), [n_epochs, max_weight](float weight){ 
		return n_epochs*(weight/max_weight); 
	});

	transform(result.begin(), result.end(), n_samples.begin(), result.begin(), [n_epochs](float r, float s) {
		if( s > 0.0 )
			return (float)n_epochs / s;
		else
			return r;
	});


	return result;
}

vector<vector<float>> umap::UMAP::multi_component_layout(const umap::Matrix& data, int n_components, 
														 vector<int>& component_labels, int dim)
{
	
}

vector<vector<float>> umap::UMAP::multi_component_layout(const umap::Matrix& data, 
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& graph, int n_components, 
	vector<int>& component_labels, int dim)
{

	vector<vector<float>>(graph.rows(), vector<float>(dim, 0.0));

	if( n_components > 2*dim ) {
		this->component_layout(data, n_components, component_labels, dim);
	} else {
		cout << "error: n_components <= 2*dim" << endl;
		throw new runtime_error("n_components <= 2*dim");
	}




}

vector<vector<float>> umap::UMAP::spectral_layout(const umap::Matrix& data, 
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& graph, int dim)
{

	int n_samples = graph.rows();

	cout << "spectral 1" << endl;


	py::module csgraph = py::module::import("scipy.sparse.csgraph");
	py::object connected_components = csgraph.attr("connected_components")(graph);

	cout << "spectral 2" << endl;
	int n_components = connected_components.attr("__getitem__")(0).cast<int>();
	vector<int> labels = connected_components.attr("__getitem__")(1).cast<vector<int>>();
cout << "spectral 3" << endl;
	// cout << "n_components: " << n_components << endl;
	// cout << "labels[:10]" << endl;
	// for( int i = 0; i < 10; ++i )
	// 	cout << labels[i] << " ";
	// cout << endl;

	if( n_components > 1) {
		return this->multi_component_layout(data, graph, n_components, labels, dim);
	}
	cout << "spectral 4" << endl;
	Eigen::VectorXf result = graph * Eigen::VectorXf::Ones(graph.cols());
	cout << "spectral 5" << endl;
	vector<float> diag_data(&result[0], result.data() + result.size());
	cout << "spectral 6" << endl;
	vector<float> temp(diag_data.size(), 0.0);
cout << "spectral 7" << endl;
	for( int i = 0; i < temp.size(); ++i )
		temp[i] = 1.0/sqrt(diag_data[i]);
	cout << "spectral 8" << endl;
	// cout << "diag_data[:10]" << endl;
	// for( int i = 0; i < 10; ++i )
	// 	cout << diag_data[i] << " ";
	// cout << endl;


	py::module scipy_sparse = py::module::import("scipy.sparse");
	cout << "spectral 9" << endl;
	py::object Iobj = scipy_sparse.attr("identity")(graph.rows(), py::arg("format") = "csr");
	cout << "spectral 10" << endl;
	py::object Dobj = scipy_sparse.attr("spdiags")(py::cast(temp), 0, graph.rows(), graph.rows(), py::arg("format") = "csr");
	cout << "spectral 11" << endl;
	Eigen::SparseMatrix<float, Eigen::RowMajor> I = Iobj.cast<Eigen::SparseMatrix<float, Eigen::RowMajor>>();
	cout << "spectral 12" << endl;
	Eigen::SparseMatrix<float, Eigen::RowMajor> D = Dobj.cast<Eigen::SparseMatrix<float, Eigen::RowMajor>>();
cout << "spectral 13" << endl;
	Eigen::SparseMatrix<float, Eigen::RowMajor> L = I-D*graph*D;
cout << "spectral 14" << endl;


	// for( int i = 0; i < 20; ++i ) {
	// 	auto row = L.row(i);
	// 	cout  << i << ": ";
	// 	cout << row.head(20);
	// 	cout << endl;
	// }


	int k = dim+1;
	int num_lanczos_vectors = max(2*k+1, (int)sqrt(graph.rows()));
cout << "spectral 15" << endl;
	// cout << "k: " << k << endl;
	// cout << "num_lanczos_vectors: " << num_lanczos_vectors << endl;


	try {
		py::module scipy_sparse_linalg = py::module::import("scipy.sparse.linalg");	
		py::object eigen;

		cout << "spectral 16" << endl;
		if( L.rows() < 2000000 ) {

			eigen = scipy_sparse_linalg.attr("eigsh")(L, k, //nullptr, nullptr, 
				py::arg("which") ="SM", py::arg("v0") = py::cast(vector<int>(L.rows(), 1.0)), 
				py::arg("ncv") = num_lanczos_vectors, py::arg("tol") = 1e-4, py::arg("maxiter") = graph.rows()*5);

			cout << "spectral 17" << endl;
		} else {
		cout << "spectral 18" << endl;
			throw new runtime_error("L.rows() >= 2000000. Not implemented yet.");

		}

cout << "spectral 19" << endl;
		py::object eigenval = eigen.attr("__getitem__")(0);
		py::object eigenvec = eigen.attr("__getitem__")(1);
cout << "spectral 20" << endl;
		vector<float> eigenvalues = eigenval.cast<vector<float>>();
		vector<vector<float>> eigenvectors = eigenvec.cast<vector<vector<float>>>();
		// cout << "eigenvalues" << endl;
		// for( int i = 0; i < 10; ++i )
		// 	printf("%.4f ", eigenvalues[i]);
		// cout << endl << endl;

		// cout << "eigenvectors" << endl;
		// for( int i = 0; i < 10; ++i )
		// {
		// 	for(int j = 0; j < eigenvectors[i].size(); ++j )
		// 		printf("%.4f ", eigenvectors[i][j]);
		// 	cout << endl;
		// }
cout << "spectral 21" << endl;
		vector<int> order_all = utils::argsort(eigenvalues);
		vector<int> order(order_all.begin()+1, order_all.begin()+k);
cout << "spectral 22" << endl;
		vector<vector<float>> spectral_embedding(eigenvectors.size());
cout << "spectral 23" << endl;
		// printf("embedding: \n");
		float max_value = -1.0;
		for( int i = 0; i < eigenvectors.size(); ++i ) {
			spectral_embedding[i] = utils::arrange_by_indices(eigenvectors[i], order);

			// if(  i < 10 )
			// 	printf("%.4f %.4f\n", spectral_embedding[i][0], spectral_embedding[i][1]);

			max_value = max(max_value, 
				abs(*max_element(spectral_embedding[i].begin(), spectral_embedding[i].end(), [](float a, float b) { return abs(a) < abs(b);})));
		}
cout << "spectral 24" << endl;
		// printf("\nmax_value: %.4f\n", max_value);

		py::module scipy_random = py::module::import("numpy.random");
		py::object randomState = scipy_random.attr("RandomState")(this->random_state);
		vector<int> size = {graph.rows(), n_components};
		py::object noiseObj = randomState.attr("normal")(py::arg("scale")=0.0001, py::arg("size")=size);
cout << "spectral 25" << endl;
		vector<vector<float>> noise = noiseObj.cast<vector<vector<float>>>();
cout << "spectral 26" << endl;
		float expansion = 10.0/max_value;
cout << "spectral 27" << endl;
		for( int i = 0; i < spectral_embedding.size(); ++i ) {
			transform(spectral_embedding[i].begin(), spectral_embedding[i].end(), 
				spectral_embedding[i].begin(), [expansion](float &c){ return c*expansion; });
		}
cout << "spectral 28" << endl;
		for( int i = 0; i < spectral_embedding.size(); ++i ) {
			transform(spectral_embedding[i].begin(), spectral_embedding[i].end(), 
				noise[i].begin(), spectral_embedding[i].begin(), plus<float>());
		}
cout << "spectral 29" << endl;
		// printf("\n\n\nFINAL EMBEDDING:\n\n");

		// for( int i = 0; i < 10; ++i ) {
		// 	if(  i < 10 )
		// 		printf("%.4f %.4f\n", spectral_embedding[i][0], spectral_embedding[i][1]);
		// }






		return spectral_embedding;

cout << "spectral 30" << endl;

	} catch(...) {
		// TODO: which exception scipy make?


		// TODO: implement random...
		// py::module random_state = py::module::import()

		cout << "spectral 1" << endl;
		throw new runtime_error("Error when computing Spectral Layout");
	}


}

vector<vector<float>> umap::pairwise_distances(umap::Matrix& X, std::string metric)
{

  int n = X.shape(0);
  int d = X.shape(1);


  vector<vector<float>> pd(n, vector<float>(n, 0.0));


  // TODO: add possibility for other distance functions
  for( int i = 0; i < n; ++i ) {
    for( int j = i+1; j < n; ++j ) {

      float distance = 0;

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

tuple<vector<int>, vector<int>, vector<float>> umap::compute_membership_strenghts(
	vector<vector<int>>& knn_indices, vector<vector<float>>& knn_dists, 
	vector<float>& sigmas, vector<float>& rhos) 
{

	int n_samples = knn_indices.size();
	int n_neighbors = knn_indices[0].size();
	int size = n_samples*n_neighbors;

	vector<int> rows(size, 0);
	vector<int> cols(size, 0);
	vector<float> vals(size, 0.0);


	for( int i = 0; i < n_samples; ++i )
	{
		for( int j = 0; j < n_neighbors; ++j )
		{

			if( knn_indices[i][j] == -1 )
				continue;
			float val = 0.0;
			if( knn_indices[i][j] == i )
				val = 0.0;
			else if( knn_dists[i][j]-rhos[i] <= 0.0 || sigmas[i] == 0.0 )
				val = 1.0;
			else 
				val = exp(-((knn_dists[i][j] - rhos[i])/ (sigmas[i])));

			rows[i * n_neighbors + j] = i;
            cols[i * n_neighbors + j] = knn_indices[i][j];
            vals[i * n_neighbors + j] = val;
		}
	}

	return make_tuple(rows, cols, vals);

}

tuple<vector<float>, vector<float>> umap::smooth_knn_dist(vector<vector<float>>& distances,
	float k, int n_iter, float local_connectivity, float bandwidth)
{

	float target = log2(k) * bandwidth;
	vector<float> rho(distances.size(), 0);
	vector<float> result(distances.size(), 0);

	float mean_distances = 0.0;

	for (int i = 0; i < distances.size(); ++i)
	{
		mean_distances += accumulate( distances[i].begin(), distances[i].end(), 0.0);
	}
	mean_distances /= (distances.size()*distances[0].size());

	// cout << "target: " << target << endl;
	// cout << "mean_distances: " << mean_distances << endl;

	for( int i = 0; i < distances.size(); ++i ) {

		float lo = 0.0;
		float hi = numeric_limits<float>::max();
		float mid = 1.0;

		vector<float> ith_distances = distances[i];
		vector<float> non_zero_dists;
		copy_if(ith_distances.begin(), ith_distances.end(), back_inserter(non_zero_dists), [](float d){ return d > 0.0; });

		if( non_zero_dists.size() >= local_connectivity ) {

			int index = (int) floor(local_connectivity);
			float interpolation = local_connectivity - (float)index;


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

			float psum = 0.0;

			for( int j = 1;  j < distances[0].size(); ++j ) {

				float d = distances[i][j] - rho[i];
				psum += (d > 0.0 ? exp(-(d/mid)) : 1.0);
			}

			if( fabs(psum-target) < umap::SMOOTH_K_TOLERANCE )
				break;

			if( psum > target ) {

				hi = mid;
				mid = (lo+hi)/2.0;
			} else {
				lo = mid;
				if( hi == numeric_limits<float>::max() )
					mid *= 2;
				else
					mid = (lo+hi)/2.0;
			}
		}
		result[i] = mid;

		if( rho[i] > 0.0 ) {
			float mean_ith_distances = accumulate(ith_distances.begin(), ith_distances.end(), 0)/ith_distances.size();
			if( result[i] < umap::MIN_K_DIST_SCALE*mean_ith_distances )
				result[i] = umap::MIN_K_DIST_SCALE*mean_ith_distances;

		} else {
			if( result[i] < umap::MIN_K_DIST_SCALE*mean_distances )
				result[i] = umap::MIN_K_DIST_SCALE*mean_distances;
		}




	}

	return make_tuple(result, rho);
}

tuple<vector<vector<int>>, vector<vector<float>>> umap::nearest_neighbors(umap::Matrix& X,
	int n_neighbors, string metric, bool angular, float random_state, map<string, string> knn_args, bool verbose)
{

	using clock = chrono::system_clock;
	using sec = chrono::duration<double>;
	if( verbose )
		cout << "Finding nearest neighbors" << endl;

	vector<vector<int>> knn_indices;
	vector<vector<float>> knn_dists;

	if( metric == "precomputed" ) {

		knn_indices = vector<vector<int>>(X.size(), vector<int>(n_neighbors, 0));
		knn_dists = vector<vector<float>>(X.size(), vector<float>(n_neighbors, 0.0));

		// #pragma omp parallel for default(shared)		
		for( int i = 0; i < knn_indices.size(); ++i )
		{
			vector<float> row_data = X.dense_matrix[i];
			vector<int> sorted_indices = utils::argsort(row_data);
			vector<int> row_nn_data_indices(sorted_indices.begin(), sorted_indices.begin()+n_neighbors);

			knn_indices[i] = row_nn_data_indices;
			knn_dists[i] = utils::arrange_by_indices<float>(row_data, row_nn_data_indices);

		}

		
	} else {
		string algorithm = knn_args["knn_algorithm"];

		if( algorithm == "FAISS_Flat") {

			py::module faiss = py::module::import("faiss");		
			
			py::object gpu_index_flat = faiss.attr("index_cpu_to_gpu")(faiss.attr("StandardGpuResources")(), 0,faiss.attr("IndexFlatL2")(X.shape(1)));
			py::array_t<float> data = py::cast(X.dense_matrix);
			gpu_index_flat.attr("add")(data);
			
			py::function search = gpu_index_flat.attr("search");
			py::object knn = search(data, n_neighbors);

			py::object knn_dists_ = knn.attr("__getitem__")(0);
			py::object knn_indices_ = knn.attr("__getitem__")(1);

			knn_dists = knn_dists_.cast<vector<vector<float>>>();
			knn_indices = knn_indices_.cast<vector<vector<int>>>();

		} else if( algorithm == "FAISS_IVFFlat" ) {

			int nlist = stoi(knn_args["nlist"]);
			int nprobes = stoi(knn_args["nprobes"]);

			py::module faiss = py::module::import("faiss");
			py::object quantizer = faiss.attr("IndexFlatL2")(X.shape(1));
			py::object index_ivf = faiss.attr("IndexIVFFlat")(quantizer, X.shape(1), nlist, faiss.attr("METRIC_L2"));
			py::object gpu_index_ivf = faiss.attr("index_cpu_to_gpu")(faiss.attr("StandardGpuResources")(), 0, index_ivf);

			py::array_t<float> data = py::cast(X.dense_matrix);

			gpu_index_ivf.attr("train")(data);
			gpu_index_ivf.attr("add")(data);

			gpu_index_ivf.attr("nprobe") = nprobes;

			py::function search = gpu_index_ivf.attr("search");
			py::object knn = search(data, n_neighbors);

			py::object knn_dists_ = knn.attr("__getitem__")(0);
			py::object knn_indices_ = knn.attr("__getitem__")(1);

			knn_dists = knn_dists_.cast<vector<vector<float>>>();
			knn_indices = knn_indices_.cast<vector<vector<int>>>();

		} else if( algorithm == "NNDescent" ) {
			// cout << "L: " << knn_args["L"] << endl;
			// cout << "iter: " << knn_args["iter"] << endl;
			// cout << "S: " << knn_args["S"] << endl;
			// cout << "R: " << knn_args["R"] << endl;
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

			auto before = clock::now();
			float* data = X.data();
			sec duration = clock::now() - before;
			cout << "Constructing array time: " << duration.count() << endl;
			index.Build(X.shape(0), data, params);

			delete data;

			knn_indices = vector<vector<int>>(X.size(), vector<int>(n_neighbors, 0));
			knn_dists = vector<vector<float>>(X.size(), vector<float>(n_neighbors, 0.0));

			for( int i = 0; i < X.shape(0); ++i ) 
			{

				knn_dists[i][0] = 0.0;
				knn_indices[i][0] = i;

				for( int j = 0; j < K; ++j ) {
					knn_dists[i][j+1] = index.graph_[i].pool[j].distance;
					knn_indices[i][j+1] = index.graph_[i].pool[j].id;

				}	
			}



		} else if( algorithm == "KDTree_NNDescent" ) {


			unsigned nTrees = (unsigned) stoi(knn_args["nTrees"]);
			unsigned mLevel = (unsigned) stoi(knn_args["mLevel"]);
			unsigned L = (unsigned) stoi(knn_args["L"]);
			unsigned iter = (unsigned) stoi(knn_args["iter"]);
			unsigned S = (unsigned) stoi(knn_args["S"]);
			unsigned R = (unsigned) stoi(knn_args["R"]);
			unsigned K = (unsigned) n_neighbors;



			float* data = X.data();
			unsigned ndims = (unsigned) X.shape(1);
			unsigned nsamples = (unsigned) X.shape(0);


			// cout << "ndims: " << ndims << endl;
			// cout << "nsamples: " << nsamples << endl;

			// cout << "nTrees: " << knn_args["nTrees"] << endl;
			// cout << "mLevel: " << knn_args["mLevel"] << endl;

			// cout << "L: " << knn_args["L"] << endl;
			// cout << "iter: " << knn_args["iter"] << endl;
			// cout << "S: " << knn_args["S"] << endl;
			// cout << "R: " << knn_args["R"] << endl;



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
			knn_dists = vector<vector<float>>(X.size(), vector<float>(n_neighbors, 0.0));

			#pragma omp parallel for
			for( int i = 0; i < nsamples; ++i ) 
			{

				knn_dists[i][0] = 0.0;
				knn_indices[i][0] = i;
				for( int j = 0; j < K-1; ++j ) {
					knn_dists[i][j+1] = index_nndescent.graph_[i].pool[j].distance;
					knn_indices[i][j+1] = index_nndescent.graph_[i].pool[j].id;

				}	
				// cout << endl;

			}
			free(data_aligned);
		}

	}

	return make_tuple(knn_indices, knn_dists);
}


tuple<Eigen::SparseMatrix<float, Eigen::RowMajor>, vector<float>, vector<float>> umap::fuzzy_simplicial_set(
	umap::Matrix& X, int n_neighbors, float random_state, string metric, 
	vector<vector<int>>& knn_indices, vector<vector<float>>& knn_dists,
	bool angular, float set_op_mix_ratio, float local_connectivity,
	bool apply_set_operations, bool verbose, umap::UMAP* obj)
{



	if( knn_indices.size() == 0 || knn_dists.size() == 0 ) {
		tie(knn_indices, knn_dists) = umap::nearest_neighbors(X, n_neighbors, metric, angular, random_state, obj->knn_args, verbose);
	} 



	// cout << "knn_indices:" << endl;
	// for( int i = 0; i < 3; ++i ) {
	// 	for( int j = 0; j < knn_indices[i].size(); ++j )
	// 		cout << knn_indices[i][j] << " ";

	// 	cout << endl;
	// }

	// cout << "\nknn_dists:" << endl;
	// for( int i = 0; i < 3; ++i ) {
	// 	for( int j = 0; j < knn_dists[i].size(); ++j )
	// 		cout << knn_dists[i][j] << " ";

	// 	cout << endl;
	// }


	vector<float> sigmas, rhos;

	tie(sigmas, rhos) = umap::smooth_knn_dist(knn_dists, (float) n_neighbors, 64, local_connectivity);

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
	vector<float> vals;

	tie(rows, cols, vals) = umap::compute_membership_strenghts(knn_indices, knn_dists, sigmas, rhos);

	if( obj ) {

		obj->rows = rows;
		obj->cols = cols;
		obj->vals = vals;
	}

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
	Eigen::SparseMatrix<float, Eigen::RowMajor> result(X.size(), X.size());
	result.reserve(Eigen::VectorXi::Constant(X.size(), 2*n_neighbors)); // TODO: verificar se é assim (ou com int)
	// result.reserve(rows*2*n_neighbors);
	for( int i = 0; i < vals.size(); ++i )
		result.insert(rows[i], cols[i]) = vals[i];
	result.makeCompressed();

	if( apply_set_operations ) {

		Eigen::SparseMatrix<float, Eigen::RowMajor> transpose = result.transpose();

		Eigen::SparseMatrix<float, Eigen::RowMajor> prod_matrix = result * transpose;

		result = 0.5 * (result + transpose);

		// result = (set_op_mix_ratio * (result + transpose - prod_matrix) + (1.0 - set_op_mix_ratio)*prod_matrix);


        // # result = (
        // #     set_op_mix_ratio * (result + transpose - prod_matrix)
        // #     + (1.0 - set_op_mix_ratio) * prod_matrix
        // # )
	}


	// for( int i = 0; i < 20; ++i ) {
	// 	auto row = result.row(i);
	// 	cout  << i << ": ";
	// 	cout << row.head(20);
	// 	cout << endl;
	// }




	return make_tuple(result, sigmas, rhos);

}

tuple<float, float> umap::find_ab_params(float spread, float min_dist) 
{

	// py::initialize_interpreter();

	// py::scoped_interpreter guard;

	py::module np = py::module::import("numpy");
	py::object random = np.attr("random");
	py::module scipy = py::module::import("scipy.optimize");

	py::module curve_module = py::module::import("CurveModule");

	vector<float> xv = utils::linspace<float>(0, spread*3, 300);
	vector<float> yv = vector<float>(xv.size(), 0.0);


	for( int i = 0; i < yv.size(); ++i )
	{
		if( xv[i] < min_dist )
		{
			yv[i] = 1.0;
		} else {
			yv[i] = exp(-(xv[i]-min_dist)/spread);
		}
	}

	py::array_t<float> pyXV = py::cast(xv);
	py::array_t<float> pyYV = py::cast(yv);

	py::function pyCurve = curve_module.attr("fitting_curve");

	py::function curve_fit = scipy.attr("curve_fit");

	py::object ret = curve_fit(pyCurve, pyXV, pyYV);

	py::object ab = ret.attr("__getitem__")(0);

	vector<float> vals = ab.cast<vector<float>>();

	// py::finalize_interpreter();


	return make_tuple(vals[0], vals[1]);
}


void umap::UMAP::prepare_for_fitting(umap::Matrix& X) 
{

	// TODO X = check_array(X)
	// What does check_array do?

	if( this->a == -1.0 or this->b == -1.0 ) {
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
		this->_knn_dists = vector<vector<float>>(X.size(), vector<float>(this->n_neighbors, 0.0));

		for( int row_id = 0; row_id < X.size(); ++row_id ) {
			vector<float> row_data = X.sparse_matrix[row_id].data;
			vector<int> row_indices = X.sparse_matrix[row_id].indices;


			if( row_data.size() < this->n_neighbors ) {
				cout << "row_id: " << row_id << ", " << row_data.size() << " " << this->n_neighbors << endl;
				throw runtime_error("Some rows contain fewer than n_neighbors distances");
			}

			vector<int> sorted_indices = utils::argsort(row_data);
			vector<int> row_nn_data_indices(sorted_indices.begin(), sorted_indices.begin() + this->n_neighbors);

			this->_knn_indices[row_id] = utils::arrange_by_indices<int>(row_indices, row_nn_data_indices);
			this->_knn_dists[row_id] = utils::arrange_by_indices<float>(row_data, row_nn_data_indices);

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

		vector<vector<float>> dmat = umap::pairwise_distances(X);
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

void umap::UMAP::fit(py::array_t<float> X) 
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

void umap::UMAP::fit_hierarchy_sparse(const Eigen::SparseMatrix<float, Eigen::RowMajor>& X)
{

	vector<int> rows, cols;
	vector<float> vals;
	
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


// void umap::UMAP::fit_hierarchy_dense(py::array_t<float> X)
void umap::UMAP::fit_hierarchy_dense(vector<vector<float>> X)
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

vector<vector<float>> umap::UMAP::fit_transform(py::array_t<float> X)
{


	this->fit(X);
	return this->embedding_;
}
