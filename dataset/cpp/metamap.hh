#pragma once

#include <tuple>
#include <utility>

namespace li {

namespace internal {
struct reduce_add_type {
  template <typename A, typename... B> constexpr auto operator()(A&& a, B&&... b) {
    auto result = a;
    using expand_variadic_pack = int[];
    (void)expand_variadic_pack{0, ((result += b), 0)...};
    return result;
  }
};
#ifdef _WIN32
__declspec(selectany) reduce_add_type reduce_add;
#else
reduce_add_type reduce_add [[gnu::weak]];
#endif

} // namespace internal

template <typename... Ms> struct metamap;

template <typename F, typename... M> constexpr decltype(auto) find_first(metamap<M...>&& map, F fun);

template <typename... Ms> struct metamap;

template <typename M1, typename... Ms> struct metamap<M1, Ms...> : public M1, public Ms... {
  typedef metamap<M1, Ms...> self;
  // Constructors.
  inline metamap() = default;
  inline metamap(self&&) = default;
  inline metamap(const self&) = default;
  self& operator=(const self&) = default;
  self& operator=(self&&) = default;

  // metamap(self& other)
  //  : metamap(const_cast<const self&>(other)) {}

  template <typename M>
  using get_value_type = typename M::_iod_value_type;

  constexpr inline metamap(get_value_type<M1>&& m1, get_value_type<Ms>&&... members) : M1{m1}, Ms{std::forward<get_value_type<Ms>>(members)}... {}
  constexpr inline metamap(M1&& m1, Ms&&... members) : M1(m1), Ms(std::forward<Ms>(members))... {}
  constexpr inline metamap(const M1& m1, const Ms&... members) : M1(m1), Ms((members))... {}

  // Assignemnt ?

  // Retrive a value.
  template <typename K> constexpr decltype(auto) operator[](K k) { return symbol_member_access(*this, k); }

  template <typename K> constexpr decltype(auto) operator[](K k) const {
    return symbol_member_access(*this, k);
  }
};

template <> struct metamap<> {
  typedef metamap<> self;
  // Constructors.
  constexpr inline metamap() = default;
  // inline metamap(self&&) = default;
  constexpr inline metamap(const self&) = default;
  // self& operator=(const self&) = default;

  // metamap(self& other)
  //  : metamap(const_cast<const self&>(other)) {}

  // Assignemnt ?

  // Retrive a value.
  template <typename K> constexpr decltype(auto) operator[](K k) { return symbol_member_access(*this, k); }

  template <typename K> constexpr decltype(auto) operator[](K k) const {
    return symbol_member_access(*this, k);
  }
};

template <typename... Ms> constexpr auto size(metamap<Ms...>) { return sizeof...(Ms); }

template <typename M> struct metamap_size_t {};
template <typename... Ms> struct metamap_size_t<metamap<Ms...>> {
  enum { value = sizeof...(Ms) };
};
template <typename M> constexpr int metamap_size() {
  return metamap_size_t<std::decay_t<M>>::value;
}

template <typename... Ks> constexpr decltype(auto) metamap_values(const metamap<Ks...>& map) {
  return std::forward_as_tuple(map[typename Ks::_iod_symbol_type()]...);
}
template <typename... Ks> constexpr decltype(auto) metamap_values(metamap<Ks...>& map) {
  return std::forward_as_tuple(map[typename Ks::_iod_symbol_type()]...);
}

template <typename... Ks> constexpr decltype(auto) metamap_keys(const metamap<Ks...>& map) {
  return std::make_tuple(typename Ks::_iod_symbol_type()...);
}

template <typename K, typename M> constexpr auto has_key(M&& map, K k) {
  return decltype(has_member(map, k)){};
}

template <typename M, typename K> constexpr auto has_key(K k) {
  return decltype(has_member(std::declval<M>(), std::declval<K>())){};
}

template <typename M, typename K> constexpr auto has_key() {
  return decltype(has_member(std::declval<M>(), std::declval<K>())){};
}

template <typename K, typename M, typename O> constexpr auto get_or(M&& map, K k, O default_) {
  if constexpr (has_key<M, decltype(k)>()) {
    return map[k];
  } else
    return default_;
}

template <typename X> struct is_metamap {
  enum { value = false };
};
template <typename... M> struct is_metamap<metamap<M...>> {
  enum { value = true };
};

} // namespace li

#include <li/metamap/algorithms/cat.hh>
#include <li/metamap/algorithms/intersection.hh>
#include <li/metamap/algorithms/map_reduce.hh>
#include <li/metamap/algorithms/substract.hh>
#include <li/metamap/algorithms/tuple_utils.hh>
#include <li/metamap/algorithms/forward_tuple_as_metamap.hh>
#include <li/metamap/make.hh>
