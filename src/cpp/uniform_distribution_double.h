#ifndef UNIF_DIST_DOUBLE_H
#define UNIF_DIST_DOUBLE_H


#include <random>
#include <tuple>
#include <iostream>

namespace uni_double {

template<class DoubleType = double>
class my_uniform_double_distribution {
public:
  // types
  typedef DoubleType result_type;
  typedef std::pair<double, double> param_type;

  // constructors and reset functions
  explicit my_uniform_double_distribution(DoubleType a = 0, DoubleType b = std::numeric_limits<DoubleType>::max());
  explicit my_uniform_double_distribution(const param_type& parm);
  explicit my_uniform_double_distribution() {}
  void reset();

  // generating functions
  template<class URNG>
    result_type operator()(URNG& g);
  template<class URNG>
    result_type operator()(URNG& g, const param_type& parm);

  // property functions
  result_type a() const;
  result_type b() const;
  param_type param() const;
  void param(const param_type& parm);
  result_type min() const;
  result_type max() const;

private:
  typedef typename std::make_unsigned<DoubleType>::type diff_type;

  DoubleType lower;
  DoubleType upper;
};

template<class DoubleType>
my_uniform_double_distribution<DoubleType>::my_uniform_double_distribution(DoubleType a, DoubleType b) {
  param({a, b});
}

template<class DoubleType>
my_uniform_double_distribution<DoubleType>::my_uniform_double_distribution(const param_type& parm) {
  param(parm);
}

template<class DoubleType>
void my_uniform_double_distribution<DoubleType>::reset() {}

template<class DoubleType>
template<class URNG>
auto my_uniform_double_distribution<DoubleType>::operator()(URNG& g) -> result_type {
  return operator()(g, param());
}

template<class DoubleType>
template<class URNG>
auto my_uniform_double_distribution<DoubleType>::operator()(URNG& g, const param_type& parm) -> result_type {
  diff_type diff = (diff_type)parm.second - (diff_type)parm.first + 1;
  if (diff == 0) // If the +1 overflows we are using the full range, just return g()
    return g();

  diff_type badDistLimit = std::numeric_limits<diff_type>::max() / diff;
  do {
    diff_type generatedRand = g();

    if (generatedRand / diff < badDistLimit)
      return (DoubleType)((generatedRand % diff) + (diff_type)parm.first);
  } while (true);
}

template<class DoubleType>
auto my_uniform_double_distribution<DoubleType>::a() const -> result_type {
  return lower;
}

template<class DoubleType>
auto my_uniform_double_distribution<DoubleType>::b() const -> result_type {
  return upper;
}

template<class DoubleType>
auto my_uniform_double_distribution<DoubleType>::param() const -> param_type {
  return {lower, upper};
}

template<class DoubleType>
void my_uniform_double_distribution<DoubleType>::param(const param_type& parm) {
  std::tie(lower, upper) = parm;
  if (upper < lower)
    throw std::exception();
}

template<class DoubleType>
auto my_uniform_double_distribution<DoubleType>::min() const -> result_type {
  return lower;
}

template<class DoubleType>
auto my_uniform_double_distribution<DoubleType>::max() const -> result_type {
  return upper;
}

}
#endif 