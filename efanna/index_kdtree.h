//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_INDEX_KDTREE_H
#define EFANNA2E_INDEX_KDTREE_H

#include <cstddef>
#include <string>
#include <vector>
#include <set>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"


namespace efanna2e {


struct Node
{
	  int DivDim;
	  float DivVal;
	  size_t StartIdx, EndIdx;
	  unsigned treeid;
	  Node* Lchild, * Rchild;

	  ~Node() {
		  if (Lchild!=NULL) Lchild->~Node();
		  if (Rchild!=NULL) Rchild->~Node();
	  }

};

struct Candidate {
    size_t row_id;
    float distance;
    Candidate(const size_t row_id, const float distance): row_id(row_id), distance(distance) { }

    bool operator >(const Candidate& rhs) const {
        if (this->distance == rhs.distance) {
            return this->row_id > rhs.row_id;
        }
        return this->distance > rhs.distance;
    }
    bool operator <(const Candidate& rhs) const {
        if (this->distance == rhs.distance) {
            return this->row_id < rhs.row_id;
        }
        return this->distance < rhs.distance;
    }
};




class IndexKDtree : public Index {
 public:
  explicit IndexKDtree(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexKDtree();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;

 typedef std::vector<nhood> KNNGraph;
  typedef std::vector<std::vector<unsigned > > CompactGraph;
  typedef std::set<Candidate, std::greater<Candidate> > CandidateHeap;

  Index *initializer_;
  KNNGraph graph_;
  CompactGraph final_graph_;
  std::vector<CandidateHeap> knn_graph;


 protected:
 
  enum
  {
	  /**
	   * To improve efficiency, only SAMPLE_NUM random values are used to
	   * compute the mean and variance at each level when building a tree.
	   * A value of 100 seems to perform as well as using all values.
	   */
	  SAMPLE_NUM = 100,
	  /**
	   * Top random dimensions to consider
	   *
	   * When creating random trees, the dimension on which to subdivide is
	   * selected at random from among the top RAND_DIM dimensions with the
	   * highest variance.  A value of 5 works well.
	   */
	  RAND_DIM=5
  };

  std::vector<Node*> tree_roots_;
  std::vector< std::pair<Node*,size_t> > mlNodeList;
  std::vector<std::vector<unsigned>> LeafLists;
  omp_lock_t rootlock;
  bool error_flag;

  int ml;   //merge_level
  int max_deepth;
  size_t TNS=10; //tree node size
  unsigned K; //KNN Graph

 private:

  void meanSplit(std::mt19937& rng, unsigned* indices, unsigned count, unsigned& index, unsigned& cutdim, float& cutval);
  void planeSplit(unsigned* indices, unsigned count, unsigned cutdim, float cutval, unsigned& lim1, unsigned& lim2);
  int selectDivision(std::mt19937& rng, float* v);
  void getMergeLevelNodeList(Node* node, size_t treeid, int deepth);
  Node* SearchToLeaf(Node* node, size_t id);
  void mergeSubGraphs(size_t treeid, Node* node);
  void DFSbuild(Node* node, std::mt19937& rng, unsigned* indices, unsigned count, unsigned offset);
  void DFStest(unsigned level, unsigned dim, Node* node);
};

}

#endif //EFANNA2E_INDEX_KDTREE_H
