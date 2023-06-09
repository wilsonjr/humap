//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#include "index_kdtree.h"
#include "exceptions.h"
#include "parameters.h"


namespace efanna2e {
#define _CONTROL_NUM 100
IndexKDtree::IndexKDtree(const size_t dimension, const size_t n, Metric m, Index *initializer)
: Index(dimension, n, m),
  initializer_{initializer} {
	  max_deepth = 0x0fffffff;
	  error_flag = false;
  }

  IndexKDtree::~IndexKDtree() {}

  void IndexKDtree::meanSplit(std::mt19937& rng, unsigned* indices, unsigned count, unsigned& index, unsigned& cutdim, float& cutval){
	  float* mean_ = new float[dimension_];
	  float* var_ = new float[dimension_];
	  memset(mean_,0,dimension_*sizeof(float));
	  memset(var_,0,dimension_*sizeof(float));

	  /* Compute mean values.  Only the first SAMPLE_NUM values need to be
        sampled to get a good estimate.
	   */
	  unsigned cnt = std::min((unsigned)SAMPLE_NUM+1, count);
	  for (unsigned j = 0; j < cnt; ++j) {
		  const float* v = data_ + indices[j] * dimension_;
		  for (size_t k=0; k<dimension_; ++k) {
			  mean_[k] += v[k];
		  }
	  }
	  float div_factor = float(1)/cnt;
	  for (size_t k=0; k<dimension_; ++k) {
		  mean_[k] *= div_factor;
	  }

	  /* Compute variances (no need to divide by count). */

	  for (unsigned j = 0; j < cnt; ++j) {
		  const float* v = data_ + indices[j] * dimension_;
		  for (size_t k=0; k<dimension_; ++k) {
			  float dist = v[k] - mean_[k];
			  var_[k] += dist * dist;
		  }
	  }

	  /* Select one of the highest variance indices at random. */
	  cutdim = selectDivision(rng, var_);

	  cutval = mean_[cutdim];

	  unsigned lim1, lim2;

	  planeSplit(indices, count, cutdim, cutval, lim1, lim2);
	  //cut the subtree using the id which best balances the tree
	  if (lim1>count/2) index = lim1;
	  else if (lim2<count/2) index = lim2;
	  else index = count/2;

	  /* If either list is empty, it means that all remaining features
	   * are identical. Split in the middle to maintain a balanced tree.
	   */
	  if ((lim1==count)||(lim2==0)) index = count/2;
	  delete[] mean_;
	  delete[] var_;
  }

  void IndexKDtree::planeSplit(unsigned* indices, unsigned count, unsigned cutdim, float cutval, unsigned& lim1, unsigned& lim2){
	  /* Move vector indices for left subtree to front of list. */
	  int left = 0;
	  int right = count-1;
	  for (;; ) {
		  const float* vl = data_ + indices[left] * dimension_;
		  const float* vr = data_ + indices[right] * dimension_;
		  while (left<=right && vl[cutdim]<cutval){
			  ++left;
			  vl = data_ + indices[left] * dimension_;
		  }
		  while (left<=right && vr[cutdim]>=cutval){
			  --right;
			  vr = data_ + indices[right] * dimension_;
		  }
		  if (left>right) break;
		  std::swap(indices[left], indices[right]); ++left; --right;
	  }
	  lim1 = left;//lim1 is the id of the leftmost point <= cutval
	  right = count-1;
	  for (;; ) {
		  const float* vl = data_ + indices[left] * dimension_;
		  const float* vr = data_ + indices[right] * dimension_;
		  while (left<=right && vl[cutdim]<=cutval){
			  ++left;
			  vl = data_ + indices[left] * dimension_;
		  }
		  while (left<=right && vr[cutdim]>cutval){
			  --right;
			  vr = data_ + indices[right] * dimension_;
		  }
		  if (left>right) break;
		  std::swap(indices[left], indices[right]); ++left; --right;
	  }
	  lim2 = left;//lim2 is the id of the leftmost point >cutval
  }
  int IndexKDtree::selectDivision(std::mt19937& rng, float* v){
	  int num = 0;
	  size_t topind[RAND_DIM];

	  //Create a list of the indices of the top RAND_DIM values.
	  for (size_t i = 0; i < dimension_; ++i) {
		  if ((num < RAND_DIM)||(v[i] > v[topind[num-1]])) {
			  // Put this element at end of topind.
			  if (num < RAND_DIM) {
				  topind[num++] = i;            // Add to list.
			  }
			  else {
				  topind[num-1] = i;         // Replace last element.
			  }
			  // Bubble end value down to right location by repeated swapping. sort the varience in decrease order
			  int j = num - 1;
			  while (j > 0  &&  v[topind[j]] > v[topind[j-1]]) {
				  std::swap(topind[j], topind[j-1]);
				  --j;
			  }
		  }
	  }
	  // Select a random integer in range [0,num-1], and return that index.
	  int rnd = rng()%num;
	  return (int)topind[rnd];
  }

  void IndexKDtree::DFSbuild(Node* node, std::mt19937& rng, unsigned* indices, unsigned count, unsigned offset){
	  //omp_set_lock(&rootlock);
	  //std::cout<<node->treeid<<":"<<offset<<":"<<count<<std::endl;
	  //omp_unset_lock(&rootlock);

	  if(count <= TNS){
		  node->DivDim = -1;
		  node->Lchild = NULL;
		  node->Rchild = NULL;
		  node->StartIdx = offset;
		  node->EndIdx = offset + count;
		  //add points

	  }else{
		  unsigned idx;
		  unsigned cutdim;
		  float cutval;
		  meanSplit(rng, indices, count, idx, cutdim, cutval);
		  node->DivDim = cutdim;
		  node->DivVal = cutval;
		  node->StartIdx = offset;
		  node->EndIdx = offset + count;
		  Node* nodeL = new Node(); Node* nodeR = new Node();
		  node->Lchild = nodeL;
		  nodeL->treeid = node->treeid;
		  DFSbuild(nodeL, rng, indices, idx, offset);
		  node->Rchild = nodeR;
		  nodeR->treeid = node->treeid;
		  DFSbuild(nodeR, rng, indices+idx, count-idx, offset+idx);
	  }
  }

	void IndexKDtree::DFStest(unsigned level, unsigned dim, Node* node){
		if(node->Lchild !=NULL){
			DFStest(++level, node->DivDim, node->Lchild);
			//if(level > 15)
			if(node->Lchild->Lchild ==NULL){
				std::vector<unsigned>& tmp = LeafLists[node->treeid];
				for(unsigned i = node->Rchild->StartIdx; i < node->Rchild->EndIdx; i++){
					const float* tmpfea =data_ + tmp[i] * dimension_+ node->DivDim;
				}
			}
		}
		else if(node->Rchild !=NULL){
			DFStest(++level, node->DivDim, node->Rchild);
		}
		else{
			std::vector<unsigned>& tmp = LeafLists[node->treeid];
			for(unsigned i = node->StartIdx; i < node->EndIdx; i++){
				const float* tmpfea =data_ + tmp[i] * dimension_+ dim;
			}
		}
	}


  void IndexKDtree::getMergeLevelNodeList(Node* node, size_t treeid, int deepth){
	  if(node->Lchild != NULL && node->Rchild != NULL && deepth < ml){
		  deepth++;
		  getMergeLevelNodeList(node->Lchild, treeid, deepth);
		  getMergeLevelNodeList(node->Rchild, treeid, deepth);
	  }else if(deepth == ml){
		  mlNodeList.push_back(std::make_pair(node,treeid));
	  }else{
		  error_flag = true;
		  if(deepth < max_deepth)max_deepth = deepth;
	  }
  }

  Node* IndexKDtree::SearchToLeaf(Node* node, size_t id){
	  if(node->Lchild != NULL && node->Rchild !=NULL){
		  const float* v = data_ + id * dimension_;
		  if(v[node->DivDim] < node->DivVal)
			  return SearchToLeaf(node->Lchild, id);
		  else
			  return SearchToLeaf(node->Rchild, id);
	  }
	  else
		  return node;
  }


  void IndexKDtree::mergeSubGraphs(size_t treeid, Node* node){

	  if(node->Lchild != NULL && node->Rchild != NULL){
		  mergeSubGraphs(treeid, node->Lchild);
		  mergeSubGraphs(treeid, node->Rchild);

		  size_t numL = node->Lchild->EndIdx - node->Lchild->StartIdx;
		  size_t numR = node->Rchild->EndIdx - node->Rchild->StartIdx;
		  size_t start,end;
		  Node * root;
		  if(numL < numR){
			  root = node->Rchild;
			  start = node->Lchild->StartIdx;
			  end = node->Lchild->EndIdx;
		  }else{
			  root = node->Lchild;
			  start = node->Rchild->StartIdx;
			  end = node->Rchild->EndIdx;
		  }

		  for(;start < end; start++){

			  size_t feature_id = LeafLists[treeid][start];

			  Node* leaf = SearchToLeaf(root, feature_id);
			  for(size_t i = leaf->StartIdx; i < leaf->EndIdx; i++){
				  size_t tmpfea = LeafLists[treeid][i];
				  float dist = distance_->compare(data_ + tmpfea * dimension_, data_ + feature_id * dimension_, dimension_);

				  {LockGuard guard(graph_[tmpfea].lock);
				  if(knn_graph[tmpfea].size() < K || dist < knn_graph[tmpfea].begin()->distance){
					  Candidate c1(feature_id, dist);
					  knn_graph[tmpfea].insert(c1);
					  if(knn_graph[tmpfea].size() > K)
						  knn_graph[tmpfea].erase(knn_graph[tmpfea].begin());
				  }
				  }

				  {LockGuard guard(graph_[feature_id].lock);
				  if(knn_graph[feature_id].size() < K || dist < knn_graph[feature_id].begin()->distance){
					  Candidate c1(tmpfea, dist);
					  knn_graph[feature_id].insert(c1);
					  if(knn_graph[feature_id].size() > K)
						  knn_graph[feature_id].erase(knn_graph[feature_id].begin());

				  }
				  }
			  }
		  }
	  }
  }



  void IndexKDtree::Build(size_t n, const float *data, const Parameters &parameters) {

	  data_ = data;
	  //assert(initializer_->HasBuilt());


	  //initial
	  unsigned N = n;
	  unsigned seed = 1998;

	  graph_.resize(N);
	  knn_graph.resize(N);

	  /*std::cout<<"TNS "<< TNS <<std::endl;
	  std::cout<<"N "<< N <<std::endl;

	  for (unsigned j = 0; j < 5; ++j) {
		  const float* v = data_ + j * dimension_;
		  for (size_t k=0; k<10; ++k) {
			  std::cout<<v[k] << " ";
		  }
		  std::cout<<std::endl;;
	  }*/

	  //build tree
	  unsigned TreeNum = parameters.Get<unsigned>("nTrees");
	  unsigned TreeNumBuild = parameters.Get<unsigned>("nTrees");
	  ml = parameters.Get<unsigned>("mLevel");
	  K = parameters.Get<unsigned>("K");

	  //std::cout<<"ml "<< ml <<std::endl;
	  //std::cout<<"K "<< K <<std::endl;
	  //std::cout<<"dimension "<< dimension_ <<std::endl;

	  std::vector<int> indices(N);
	  LeafLists.resize(TreeNum);
	  std::vector<Node*> ActiveSet;
	  std::vector<Node*> NewSet;
	  for(unsigned i = 0; i < (unsigned)TreeNum; i++){
		  Node* node = new Node;
		  node->DivDim = -1;
		  node->Lchild = NULL;
		  node->Rchild = NULL;
		  node->StartIdx = 0;
		  node->EndIdx = N;
		  node->treeid = i;
		  tree_roots_.push_back(node);
		  ActiveSet.push_back(node);
	  }

#pragma omp parallel for
	  for(int i = 0; i < N; i++)indices[i] = i;
#pragma omp parallel for
	  for(int i = 0; i < (unsigned)TreeNum; i++){
		  std::vector<unsigned>& myids = LeafLists[i];
		  myids.resize(N);
		  std::copy(indices.begin(), indices.end(),myids.begin());
		  // std::random_shuffle(myids.begin(), myids.end());
		  std::random_device rd;
	      std::mt19937 g(rd());
	      std::shuffle(myids.begin(), myids.end(), g);
	  }
	  omp_init_lock(&rootlock);
	  while(!ActiveSet.empty() && ActiveSet.size() < 1100){
#pragma omp parallel for
		  for(int i = 0; i < ActiveSet.size(); i++){
			  Node* node = ActiveSet[i];
			  unsigned mid;
			  unsigned cutdim;
			  float cutval;
			  std::mt19937 rng(seed ^ omp_get_thread_num());
			  std::vector<unsigned>& myids = LeafLists[node->treeid];

			  meanSplit(rng, &myids[0]+node->StartIdx, node->EndIdx - node->StartIdx, mid, cutdim, cutval);

			  node->DivDim = cutdim;
			  node->DivVal = cutval;
			  //node->StartIdx = offset;
			  //node->EndIdx = offset + count;
			  Node* nodeL = new Node(); Node* nodeR = new Node();
			  nodeR->treeid = nodeL->treeid = node->treeid;
			  nodeL->StartIdx = node->StartIdx;
			  nodeL->EndIdx = node->StartIdx+mid;
			  nodeR->StartIdx = nodeL->EndIdx;
			  nodeR->EndIdx = node->EndIdx;
			  node->Lchild = nodeL;
			  node->Rchild = nodeR;
			  omp_set_lock(&rootlock);
			  if(mid>K)NewSet.push_back(nodeL);
			  if(nodeR->EndIdx - nodeR->StartIdx > K)NewSet.push_back(nodeR);
			  omp_unset_lock(&rootlock);
		  }
		  ActiveSet.resize(NewSet.size());
		  std::copy(NewSet.begin(), NewSet.end(),ActiveSet.begin());
		  NewSet.clear();
	  }

#pragma omp parallel for
	  for(int i = 0; i < ActiveSet.size(); i++){
		  Node* node = ActiveSet[i];
		  //omp_set_lock(&rootlock);
		  //std::cout<<i<<":"<<node->EndIdx-node->StartIdx<<std::endl;
		  //omp_unset_lock(&rootlock);
		  std::mt19937 rng(seed ^ omp_get_thread_num());
		  std::vector<unsigned>& myids = LeafLists[node->treeid];
		  DFSbuild(node, rng, &myids[0]+node->StartIdx, node->EndIdx-node->StartIdx, node->StartIdx);
	  }
	  //DFStest(0,0,tree_roots_[0]);
	  for(size_t i = 0; i < (unsigned)TreeNumBuild; i++){
		  getMergeLevelNodeList(tree_roots_[i], i ,0);
	  }

	  if(error_flag){
		  ;
		  //std::cout << "merge level deeper than tree, max merge deepth is " << max_deepth-1<<std::endl;
	  }

#pragma omp parallel for
	  for(int i = 0; i < mlNodeList.size(); i++){
		  mergeSubGraphs(mlNodeList[i].second, mlNodeList[i].first);
	  }



	  final_graph_.reserve(nd_);
	  std::mt19937 rng(seed ^ omp_get_thread_num());
	  std::set<unsigned> result;
	  for (unsigned i = 0; i < nd_; i++) {
		  std::vector<unsigned> tmp;
		  typename CandidateHeap::reverse_iterator it = knn_graph[i].rbegin();
		  for(;it!= knn_graph[i].rend();it++ ){
			  tmp.push_back(it->row_id);
		  }
		  if(tmp.size() < K){
			  //std::cout << "node "<< i << " only has "<< tmp.size() <<" neighbors!" << std::endl;
			  result.clear();
			  size_t vlen = tmp.size();
			  for(size_t j=0; j<vlen;j++){
				  result.insert(tmp[j]);
			  }
			  while(result.size() < K){
				  unsigned id = rng() % N;
				  result.insert(id);
			  }
			  tmp.clear();
			  std::set<unsigned>::iterator it;
			  for(it=result.begin();it!=result.end();it++){
				  tmp.push_back(*it);
			  }
			  //std::copy(result.begin(),result.end(),tmp.begin());
		  }
		  tmp.reserve(K);
		  final_graph_.push_back(tmp);
	  }
	  // std::vector<nhood>().swap(graph_);
	  has_built = true;
  }


  void IndexKDtree::Save(const char *filename) {
	  std::ofstream out(filename, std::ios::binary | std::ios::out);
	  assert(final_graph_.size() == nd_);
	  unsigned GK = (unsigned) final_graph_[0].size();
	  for (unsigned i = 0; i < nd_; i++) {
		  out.write((char *) &GK, sizeof(unsigned));
		  out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
	  }
	  out.close();
  }

  void IndexKDtree::Load(const char *filename){
  }

  void IndexKDtree::Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) {
  }


}
