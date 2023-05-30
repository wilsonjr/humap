#ifndef UNIF_DIST_float_H
#define UNIF_DIST_float_H


#include <random>
#include <tuple>
#include <iostream>

namespace uni_float {

template<class floatType = float>
class my_uniform_float_distribution {
public:
  // types
  typedef floatType result_type;
  typedef std::pair<float, float> param_type;

  // constructors and reset functions
  explicit my_uniform_float_distribution(floatType a = 0, floatType b = std::numeric_limits<floatType>::max());
  explicit my_uniform_float_distribution(const param_type& parm);
  explicit my_uniform_float_distribution() {}
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
  typedef typename std::make_unsigned<floatType>::type diff_type;

  floatType lower;
  floatType upper;
};

template<class floatType>
my_uniform_float_distribution<floatType>::my_uniform_float_distribution(floatType a, floatType b) {
  param({a, b});
}

template<class floatType>
my_uniform_float_distribution<floatType>::my_uniform_float_distribution(const param_type& parm) {
  param(parm);
}

template<class floatType>
void my_uniform_float_distribution<floatType>::reset() {}

template<class floatType>
template<class URNG>
auto my_uniform_float_distribution<floatType>::operator()(URNG& g) -> result_type {
  return operator()(g, param());
}

template<class floatType>
template<class URNG>
auto my_uniform_float_distribution<floatType>::operator()(URNG& g, const param_type& parm) -> result_type {
  diff_type diff = (diff_type)parm.second - (diff_type)parm.first + 1;
  if (diff == 0) // If the +1 overflows we are using the full range, just return g()
    return g();

  diff_type badDistLimit = std::numeric_limits<diff_type>::max() / diff;
  do {
    diff_type generatedRand = g();

    if (generatedRand / diff < badDistLimit)
      return (floatType)((generatedRand % diff) + (diff_type)parm.first);
  } while (true);
}

template<class floatType>
auto my_uniform_float_distribution<floatType>::a() const -> result_type {
  return lower;
}

template<class floatType>
auto my_uniform_float_distribution<floatType>::b() const -> result_type {
  return upper;
}

template<class floatType>
auto my_uniform_float_distribution<floatType>::param() const -> param_type {
  return {lower, upper};
}

template<class floatType>
void my_uniform_float_distribution<floatType>::param(const param_type& parm) {
  std::tie(lower, upper) = parm;
  if (upper < lower)
    throw std::exception();
}

template<class floatType>
auto my_uniform_float_distribution<floatType>::min() const -> result_type {
  return lower;
}

template<class floatType>
auto my_uniform_float_distribution<floatType>::max() const -> result_type {
  return upper;
}

}
#endif 