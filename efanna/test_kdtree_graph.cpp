//
// Created by 付聪 on 2017/6/21.
//

#include "index_kdtree.h"
#include "index_random.h"
#include "util.h"

#include <vector>
#include <cmath>
#include <random>

using namespace efanna2e;

using namespace std;


void load_data(char* filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
    in.seekg(4,std::ios::cur);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
}

int main(int argc, char** argv){
  if(argc!=6){std::cout<< argv[0] <<" data_file nTrees mLevel K saving_graph"<<std::endl; exit(-1);}
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  char* graph_filename = argv[5];
  unsigned nTrees = (unsigned)atoi(argv[2]);
  unsigned mLevel = (unsigned)atoi(argv[3]);
  unsigned K = (unsigned)atoi(argv[4]);



  int ndims = 764;
  int nsamples = 70000;

  default_random_engine generator;
  normal_distribution<float> distribution(0.0, 1.0);

  float* data = new float[ndims*nsamples*sizeof(float)];
  cout << "adding data " << endl;
  for( size_t i = 0; i < nsamples; ++i ) {

    for( size_t j = 0; j < ndims; ++j )
      *(data + i*ndims + j) = distribution(generator);
  }
  cout << "finished " << endl;
  points_num = nsamples;
  dim = ndims;



  data = efanna2e::data_align(data, points_num, dim);//one must align the data before build
  efanna2e::IndexKDtree index(dim, points_num, efanna2e::L2, nullptr);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("nTrees", nTrees);
  paras.Set<unsigned>("mLevel", mLevel);


  auto s = std::chrono::high_resolution_clock::now();

  index.Build(points_num, data, paras);

  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e-s;
  std::cout <<"Time cost: "<< diff.count() << "\n";


  index.Save(graph_filename);

  return 0;
}
