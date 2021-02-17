//
// Created by 付聪 on 2017/6/21.
//

#include "index_graph.h"
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

float distance(float* a, float* b, int size)
{
  float sum = 0.0;

  for( int i = 0; i < size; ++i )
    sum += (a[i]-b[i])*(a[i]-b[i]);

  return sqrt(sum);

}

int main(int argc, char** argv){
  if(argc!=8){std::cout<< argv[0] <<" data_file save_graph K L iter S R"<<std::endl; exit(-1);}
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  char* graph_filename = argv[2];
  unsigned K = (unsigned)atoi(argv[3]);
  unsigned L = (unsigned)atoi(argv[4]);
  unsigned iter = (unsigned)atoi(argv[5]);
  unsigned S = (unsigned)atoi(argv[6]);
  unsigned R = (unsigned)atoi(argv[7]);
  //data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
  ;




  int ndims = 784;
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

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));

  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K); // the number of neighbors to construct the neighbor graph
  paras.Set<unsigned>("L", L); // how many neighbors will I check for refinement?
  paras.Set<unsigned>("iter", iter); // the number of iterations
  paras.Set<unsigned>("S", S); // how many numbers of points in the leaf node; candidate pool size
  paras.Set<unsigned>("R", R); 



  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e-s;
  std::cout <<"Time cost: "<< diff.count() << "\n";

  

  for( int i = 0; i < 5; ++i) {

    std::cout << i << ">> " << distance(&data[i*dim + i], &data[i*dim + i], dim) << std::endl;

    for(int j = 0; j< index.final_graph_[i].size(); ++j) {
      int id = index.final_graph_[i][j];
      std::cout << "(" << index.graph_[i].pool[j].id << ") " << index.graph_[i].pool[j].distance<< "; ";
    }

    std::cout << std::endl;
    std::cout << std::endl;

  }

  index.Save(graph_filename);

  return 0;
}
