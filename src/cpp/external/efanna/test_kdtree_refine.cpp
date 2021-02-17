#include "index_graph.h"
#include "index_random.h"
#include "index_kdtree.h"
#include "util.h"


#include <vector>
#include <cmath>
#include <random>

using namespace efanna2e;
using namespace std;


int main(int argc, char** argv)
{


	if( argc != 8) {
		std::cout<< argv[0] <<"nTrees mLevel K L iter S R"<<std::endl; exit(-1);
	}

	unsigned nTrees = (unsigned) atoi(argv[1]);
	unsigned mLevel = (unsigned) atoi(argv[2]);
	unsigned K = (unsigned) atoi(argv[3]);
	unsigned L = (unsigned) atoi(argv[4]);
	unsigned iter = (unsigned) atoi(argv[5]);
	unsigned S = (unsigned) atoi(argv[6]);
	unsigned R = (unsigned) atoi(argv[7]);

	unsigned ndims = 784;
	unsigned nsamples = 70000;

	default_random_engine generator;
	normal_distribution<float> distribution(0.0, 1.0);

	float* data = new float[ndims*nsamples*sizeof(float)];
	cout << "adding data " << endl;
	for( size_t i = 0; i < nsamples; ++i ) {

	for( size_t j = 0; j < ndims; ++j )
	  *(data + i*ndims + j) = distribution(generator);
	}
	cout << "finished " << endl;

	float* data_aligned = efanna2e::data_align(data, nsamples, ndims);
	efanna2e::IndexKDtree index_kdtree(ndims, nsamples, efanna2e::L2, nullptr);

	efanna2e::Parameters params_kdtree;
	params_kdtree.Set<unsigned>("K", K);
	params_kdtree.Set<unsigned>("nTrees", nTrees);
	params_kdtree.Set<unsigned>("mLevel", mLevel);


	auto s = chrono::high_resolution_clock::now();
	index_kdtree.Build(nsamples, data_aligned, params_kdtree);
	auto e = chrono::high_resolution_clock::now();
	chrono::duration<double> diff = e-s;
	cout << "KDTree Time cost: " << diff.count() << endl;


	efanna2e::IndexRandom init_index(ndims, nsamples);
	efanna2e::IndexGraph index_nndescent(ndims, nsamples, efanna2e::L2, (efanna2e::Index*)(&init_index));

	index_nndescent.final_graph_ = index_kdtree.final_graph_;
	// index_nndescent.Load(init_graph_filename);


	efanna2e::Parameters params_nndescent;
	params_nndescent.Set<unsigned>("K", K);
	params_nndescent.Set<unsigned>("L", L);
	params_nndescent.Set<unsigned>("iter", iter);
	params_nndescent.Set<unsigned>("S", S);
	params_nndescent.Set<unsigned>("R", R);

	s = chrono::high_resolution_clock::now();
	index_nndescent.RefineGraph(data_aligned, params_nndescent);
	e = chrono::high_resolution_clock::now();

	diff = e-s;
	cout << "NNDescent refinement cost: " << diff.count() << endl;







	return 0;


}