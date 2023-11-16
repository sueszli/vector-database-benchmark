// Author: Matthieu Garrigues matthieu.garrigues@gmail.com
//
// Single header version the lithium_json library.
// https://github.com/matt-42/lithium
//
// This file is generated do not edit it.

#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#if defined(_MSC_VER)
#include <ciso646>
#endif // _MSC_VER


#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_JSON_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_JSON_HH


#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_DECODER_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_DECODER_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_DECODE_STRINGSTREAM_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_DECODE_STRINGSTREAM_HH

#if defined(_MSC_VER)
#endif


namespace li {

using std::string_view;

namespace internal {
template <typename I> void parse_uint(I* val_, const char* str, const char** end) {
  I& val = *val_;
  val = 0;
  int i = 0;
  while (i < 40) {
    char c = *str;
    if (c < '0' or c > '9')
      break;
    val = val * 10 + c - '0';
    str++;
    i++;
  }
  if (end)
    *end = str;
}

template <typename I> void parse_int(I* val, const char* str, const char** end) {
  bool neg = false;

  if (str[0] == '-') {
    neg = true;
    str++;
  }
  parse_uint(val, str, end);
  if constexpr (!std::is_same<I, bool>::value) {
    if (neg)
      *val = -(*val);
  }
}

inline unsigned long long pow10(unsigned int e) {
  unsigned long long pows[] = {1,
                               10,
                               100,
                               1000,
                               10000,
                               100000,
                               1000000,
                               10000000,
                               100000000,
                               1000000000,
                               10000000000,
                               100000000000,
                               1000000000000,
                               10000000000000,
                               100000000000000,
                               1000000000000000,
                               10000000000000000,
                               100000000000000000};

  if (e < 18)
    return pows[e];
  else
    return 0;
}

template <typename F> void parse_float(F* f, const char* str, const char** end) {
  // 1.234e-10
  // [sign][int][decimal_part][exp]

  const char* it = str;
  int integer_part;
  parse_int(&integer_part, it, &it);
  int sign = integer_part >= 0 ? 1 : -1;
  *f = integer_part;
  if (*it == '.') {
    it++;
    unsigned long long decimal_part;
    const char* dec_end;
    parse_uint(&decimal_part, it, &dec_end);

    if (dec_end > it)
      *f += (F(decimal_part) / pow10(dec_end - it)) * sign;

    it = dec_end;
  }

  if (*it == 'e' || *it == 'E') {
    it++;
    bool neg = false;
    if (*it == '-') {
      neg = true;
      it++;
    }

    unsigned int exp = 0;
    parse_uint(&exp, it, &it);
    if (neg)
      *f = *f / pow10(exp);
    else
      *f = *f * pow10(exp);
  }

  if (end)
    *end = it;
}

} // namespace internal

class decode_stringstream {
public:
  inline decode_stringstream(std::string_view buffer_)
      : cur(buffer_.data()), bad_(false), buffer(buffer_) {}

  inline bool eof() const { return cur > &buffer.back(); }
  inline const char peek() const { return *cur; }
  inline const char get() { return *(cur++); }
  inline int bad() const { return bad_; }
  inline int good() const { return !bad_ && !eof(); }

  template <typename O, typename F> void copy_until(O& output, F until) {
    const char* start = cur;
    const char* end = cur;
    const char* buffer_end = buffer.data() + buffer.size();
    while (until(*end) && end < buffer_end)
      end++;

    output.append(std::string_view(start, end - start));
    cur = end;
  }

  template <typename T> void operator>>(T& value) {
    eat_spaces();
    if constexpr (std::is_floating_point<T>::value) {
      // Decode floating point.
      eat_spaces();
      const char* end = nullptr;
      internal::parse_float(&value, cur, &end);
      if (end == cur)
        bad_ = true;
      cur = end;
    } else if constexpr (std::is_integral<T>::value) {
      // Decode integer.
      const char* end = nullptr;
      internal::parse_int(&value, cur, &end);
      if (end == cur)
        bad_ = true;
      cur = end;
    } else if constexpr (std::is_same<T, std::string>::value) {
      // Decode UTF8 string.
      json_to_utf8(*this, value);
    } else if constexpr (std::is_same<T, string_view>::value) {
      // Decoding to stringview does not decode utf8.

      if (get() != '"') {
        bad_ = true;
        return;
      }

      const char* start = cur;
      bool escaped = false;

      while (peek() != '\0' and (escaped or peek() != '"')) {
        int nb = 0;
        while (peek() == '\\')
          nb++;

        escaped = nb % 2;
        cur++;
      }
      const char* end = cur;
      value = string_view(start, end - start);

      if (get() != '"') {
        bad_ = true;
        return;
      }
    }
  }

private:
  inline void eat_spaces() {
    while (peek() < 33)
      ++cur;
  }

  int bad_;
  const char* cur;
  std::string_view buffer; //
};

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_DECODE_STRINGSTREAM_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_ERROR_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_ERROR_HH


namespace li {

enum json_error_code { JSON_OK = 0, JSON_KO = 1 };

struct json_error {
  json_error& operator=(const json_error&) = default;
  operator bool() { return code != 0; }
  bool good() { return code == 0; }
  bool bad() { return code != 0; }
  int code;
  std::string what;
};

inline int make_json_error(const char* what) { return 1; }
inline int json_no_error() { return 0; }

static int json_ok = json_no_error();

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_ERROR_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_UNICODE_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_UNICODE_HH



#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_METAMAP_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_METAMAP_HH


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

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_CAT_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_CAT_HH

namespace li {

template <typename... T, typename... U>
constexpr inline decltype(auto) cat(const metamap<T...>& a, const metamap<U...>& b) {
  return metamap<T..., U...>(*static_cast<const T*>(&a)..., *static_cast<const U*>(&b)...);
}

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_CAT_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_INTERSECTION_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_INTERSECTION_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_MAKE_METAMAP_SKIP_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_MAKE_METAMAP_SKIP_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_MAKE_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_MAKE_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_SYMBOL_AST_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_SYMBOL_AST_HH


namespace li {

template <typename E> struct Exp {};

template <typename E> struct array_subscriptable;

template <typename E> struct callable;

template <typename E> struct assignable;

template <typename E> struct array_subscriptable;

template <typename M, typename... A>
struct function_call_exp : public array_subscriptable<function_call_exp<M, A...>>,
                           public callable<function_call_exp<M, A...>>,
                           public assignable<function_call_exp<M, A...>>,
                           public Exp<function_call_exp<M, A...>> {
  using assignable<function_call_exp<M, A...>>::operator=;

  function_call_exp(const M& m, A&&... a) : method(m), args(std::forward<A>(a)...) {}

  M method;
  std::tuple<A...> args;
};

template <typename O, typename M>
struct array_subscript_exp : public array_subscriptable<array_subscript_exp<O, M>>,
                             public callable<array_subscript_exp<O, M>>,
                             public assignable<array_subscript_exp<O, M>>,
                             public Exp<array_subscript_exp<O, M>> {
  using assignable<array_subscript_exp<O, M>>::operator=;

  array_subscript_exp(const O& o, const M& m) : object(o), member(m) {}

  O object;
  M member;
};

template <typename L, typename R> struct assign_exp : public Exp<assign_exp<L, R>> {
  typedef L left_t;
  typedef R right_t;

  // template <typename V>
  // assign_exp(L l, V&& r) : left(l), right(std::forward<V>(r)) {}
  // template <typename V>
  inline assign_exp(L l, R r) : left(l), right(r) {}
  // template <typename V>
  // inline assign_exp(L l, const V& r) : left(l), right(r) {}

  L left;
  R right;
};

template <typename E> struct array_subscriptable {
public:
  // Member accessor
  template <typename S> constexpr auto operator[](S&& s) const {
    return array_subscript_exp<E, S>(*static_cast<const E*>(this), std::forward<S>(s));
  }
};

template <typename E> struct callable {
public:
  // Direct call.
  template <typename... A> constexpr auto operator()(A&&... args) const {
    return function_call_exp<E, A...>(*static_cast<const E*>(this), std::forward<A>(args)...);
  }
};

template <typename E> struct assignable {
public:
  template <typename L> auto operator=(L&& l) const {
    return assign_exp<E, L>(static_cast<const E&>(*this), std::forward<L>(l));
  }

  template <typename L> auto operator=(L&& l) {
    return assign_exp<E, L>(static_cast<E&>(*this), std::forward<L>(l));
  }

  template <typename T> auto operator=(const std::initializer_list<T>& l) const {
    return assign_exp<E, std::vector<T>>(static_cast<const E&>(*this), std::vector<T>(l));
  }
};

#define iod_query_declare_binary_op(OP, NAME)                                                      \
  template <typename A, typename B>                                                                \
  struct NAME##_exp : public assignable<NAME##_exp<A, B>>, public Exp<NAME##_exp<A, B>> {          \
    using assignable<NAME##_exp<A, B>>::operator=;                                                 \
    NAME##_exp() {}                                                                                \
    NAME##_exp(A&& a, B&& b) : lhs(std::forward<A>(a)), rhs(std::forward<B>(b)) {}                 \
    typedef A lhs_type;                                                                            \
    typedef B rhs_type;                                                                            \
    lhs_type lhs;                                                                                  \
    rhs_type rhs;                                                                                  \
  };                                                                                               \
  template <typename A, typename B>                                                                \
  inline std::enable_if_t<std::is_base_of<Exp<A>, A>::value || std::is_base_of<Exp<B>, B>::value,  \
                          NAME##_exp<A, B>>                                                        \
  operator OP(const A& b, const B& a) {                                                            \
    return NAME##_exp<std::decay_t<A>, std::decay_t<B>>{b, a};                                     \
  }

iod_query_declare_binary_op(+, plus);
iod_query_declare_binary_op(-, minus);
iod_query_declare_binary_op(*, mult);
iod_query_declare_binary_op(/, div);
iod_query_declare_binary_op(<<, shiftl);
iod_query_declare_binary_op(>>, shiftr);
iod_query_declare_binary_op(<, inf);
iod_query_declare_binary_op(<=, inf_eq);
iod_query_declare_binary_op(>, sup);
iod_query_declare_binary_op(>=, sup_eq);
iod_query_declare_binary_op(==, eq);
iod_query_declare_binary_op(!=, neq);
iod_query_declare_binary_op(&, logical_and);
iod_query_declare_binary_op (^, logical_xor);
iod_query_declare_binary_op(|, logical_or);
iod_query_declare_binary_op(&&, and);
iod_query_declare_binary_op(||, or);

#undef iod_query_declare_binary_op

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_SYMBOL_AST_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_SYMBOL_SYMBOL_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_SYMBOL_SYMBOL_HH


namespace li {

template <typename S>
class symbol : public assignable<S>,
               public array_subscriptable<S>,
               public callable<S>,
               public Exp<S> {};
} // namespace li

#ifdef LI_SYMBOL
#undef LI_SYMBOL
#endif

#define LI_SYMBOL(NAME)                                                                            \
  namespace s {                                                                                    \
  struct NAME##_t : li::symbol<NAME##_t> {                                                         \
                                                                                                   \
    using assignable<NAME##_t>::operator=;                                                         \
                                                                                                   \
    inline constexpr bool operator==(NAME##_t) { return true; }                                    \
    template <typename T> inline constexpr bool operator==(T) { return false; }                    \
                                                                                                   \
    template <typename V> struct variable_t {                                                      \
      typedef NAME##_t _iod_symbol_type;                                                           \
      typedef V _iod_value_type;                                                                   \
      V NAME;                                                                                      \
    };                                                                                             \
                                                                                                   \
    template <typename T, typename... A>                                                           \
    static inline decltype(auto) symbol_method_call(T&& o, A... args) {                            \
      return o.NAME(args...);                                                                      \
    }                                                                                              \
    template <typename T, typename... A> static inline auto& symbol_member_access(T&& o) {         \
      return o.NAME;                                                                               \
    }                                                                                              \
    template <typename T>                                                                          \
    static constexpr auto has_getter(int)                                                          \
        -> decltype(std::declval<T>().NAME(), std::true_type{}) {                                  \
      return {};                                                                                   \
    }                                                                                              \
    template <typename T> static constexpr auto has_getter(long) { return std::false_type{}; }     \
    template <typename T>                                                                          \
    static constexpr auto has_member(int) -> decltype(std::declval<T>().NAME, std::true_type{}) {  \
      return {};                                                                                   \
    }                                                                                              \
    template <typename T> static constexpr auto has_member(long) { return std::false_type{}; }     \
                                                                                                   \
    static inline auto symbol_string() { return #NAME; }                                           \
    static inline auto json_key_string() { return "\"" #NAME "\":"; }                              \
  };                                                                                               \
  static constexpr NAME##_t NAME;                                                                  \
  }

namespace li {

template <typename S> inline decltype(auto) make_variable(S s, char const v[]) {
  typedef typename S::template variable_t<const char*> ret;
  return ret{v};
}

template <typename V, typename S> inline decltype(auto) make_variable(S s, V v) {
  typedef typename S::template variable_t<std::remove_const_t<std::remove_reference_t<V>>> ret;
  return ret{v};
}

template <typename K, typename V> inline decltype(auto) make_variable_reference(K s, V&& v) {
  typedef typename K::template variable_t<V> ret;
  return ret{v};
}

template <typename T, typename S, typename... A>
static inline decltype(auto) symbol_method_call(T&& o, S, A... args) {
  return S::symbol_method_call(o, std::forward<A>(args)...);
}

template <typename T, typename S> static inline decltype(auto) symbol_member_access(T&& o, S) {
  return S::symbol_member_access(o);
}

template <typename T, typename S> constexpr auto has_member(T&& o, S) {
  return S::template has_member<T>(0);
}
template <typename T, typename S> constexpr auto has_member() {
  return S::template has_member<T>(0);
}

template <typename T, typename S> constexpr auto has_getter(T&& o, S) {
  return decltype(S::template has_getter<T>(0)){};
}
template <typename T, typename S> constexpr auto has_getter() {
  return decltype(S::template has_getter<T>(0)){};
}

template <typename S, typename T> struct CANNOT_FIND_REQUESTED_MEMBER_IN_TYPE {};

template <typename T, typename S> decltype(auto) symbol_member_or_getter_access(T&& o, S) {
  if constexpr (has_getter<T, S>()) {
    return symbol_method_call(o, S{});
  } else if constexpr (has_member<T, S>()) {
    return symbol_member_access(o, S{});
  } else {
    return CANNOT_FIND_REQUESTED_MEMBER_IN_TYPE<S, T>::error;
  }
}

template <typename S> auto symbol_string(symbol<S> v) { return S::symbol_string(); }

template <typename V> auto symbol_string(V v, typename V::_iod_symbol_type* = 0) {
  return V::_iod_symbol_type::symbol_string();
}
} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_SYMBOL_SYMBOL_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_CALLABLE_TRAITS_TYPELIST_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_CALLABLE_TRAITS_TYPELIST_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_TUPLE_UTILS_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_TUPLE_UTILS_HH


namespace li {

template <typename... E, typename F> constexpr void apply_each(F&& f, E&&... e) {
  (void)std::initializer_list<int>{((void)f(std::forward<E>(e)), 0)...};
}

template <typename... E, typename F, typename R>
constexpr auto tuple_map_reduce_impl(F&& f, R&& reduce, E&&... e) {
  return reduce(f(std::forward<E>(e))...);
}

template <typename T, typename F> constexpr void tuple_map(T&& t, F&& f) {
  return std::apply([&](auto&&... e) { apply_each(f, std::forward<decltype(e)>(e)...); },
                    std::forward<T>(t));
}

template <typename T, typename F> constexpr auto tuple_reduce(T&& t, F&& f) {
  return std::apply(std::forward<F>(f), std::forward<T>(t));
}

template <typename T, typename F, typename R>
decltype(auto) tuple_map_reduce(T&& m, F map, R reduce) {
  auto fun = [&](auto... e) { return tuple_map_reduce_impl(map, reduce, e...); };
  return std::apply(fun, m);
}

template <typename F> constexpr inline std::tuple<> tuple_filter_impl() { return std::make_tuple(); }

template <typename F, typename... M, typename M1> constexpr auto tuple_filter_impl(M1 m1, M... m) {
  if constexpr (std::is_same<M1, F>::value)
    return tuple_filter_impl<F>(m...);
  else
    return std::tuple_cat(std::make_tuple(m1), tuple_filter_impl<F>(m...));
}

template <typename F, typename... M> constexpr auto tuple_filter(const std::tuple<M...>& m) {

  auto fun = [](auto... e) { return tuple_filter_impl<F>(e...); };
  return std::apply(fun, m);
}

} // namespace li
#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_TUPLE_UTILS_HH


namespace li {

template <typename... T> struct typelist {};

template <typename... T1, typename... T2>
constexpr auto typelist_cat(typelist<T1...> t1, typelist<T2...> t2) {
  return typelist<T1..., T2...>();
}

template <typename T> struct typelist_to_tuple {};

template <typename... T> struct typelist_to_tuple<typelist<T...>> {
  typedef std::tuple<T...> type;
};

template <typename T> struct tuple_to_typelist {};

template <typename... T> struct tuple_to_typelist<std::tuple<T...>> {
  typedef typelist<T...> type;
};

template <typename T> using typelist_to_tuple_t = typename typelist_to_tuple<T>::type;
template <typename T> using tuple_to_typelist_t = typename tuple_to_typelist<T>::type;

template <typename T, typename U> struct typelist_embeds : public std::false_type {};

template <typename... T, typename U>
struct typelist_embeds<typelist<T...>, U>
    : public std::integral_constant<bool, count_first_falses(std::is_same<T, U>::value...) !=
                                              sizeof...(T)> {};

template <typename T, typename E> struct typelist_embeds_any_ref_of : public std::false_type {};

template <typename U, typename... T>
struct typelist_embeds_any_ref_of<typelist<T...>, U>
    : public typelist_embeds<typelist<std::decay_t<T>...>, std::decay_t<U>> {};

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_CALLABLE_TRAITS_TYPELIST_HH


namespace li {

template <typename... Ms> struct metamap;

namespace internal {

template <typename S, typename V> constexpr decltype(auto) exp_to_variable_ref(const assign_exp<S, V>& e) {
  return make_variable_reference(S{}, e.right);
}

template <typename S, typename V> constexpr decltype(auto) exp_to_variable(const assign_exp<S, V>& e) {
  typedef std::remove_const_t<std::remove_reference_t<V>> vtype;
  return make_variable(S{}, e.right);
}

template <typename S> decltype(auto) constexpr exp_to_variable(const symbol<S>& e) {
  return exp_to_variable(S() = int());
}

template <typename... T> constexpr inline decltype(auto) make_metamap_helper(T&&... args) {
  return metamap<T...>(std::forward<T>(args)...);
}

} // namespace internal

// Store copies of values in the map
template <typename... T> constexpr inline decltype(auto) mmm(T&&... args) {
  // Copy values.
  return internal::make_metamap_helper(internal::exp_to_variable(std::forward<T>(args))...);
}

// Store references of values in the map
template <typename... T> constexpr inline decltype(auto) make_metamap_reference(T&&... args) {
  // Keep references.
  return internal::make_metamap_helper(internal::exp_to_variable_ref(std::forward<T>(args))...);
}

template <typename... Ks> constexpr decltype(auto) metamap_clone(const metamap<Ks...>& map) {
  return mmm((typename Ks::_iod_symbol_type() = map[typename Ks::_iod_symbol_type()])...);
}

namespace internal {
  
  template <typename... V>
  auto make_metamap_type(typelist<V...> variables) {
    return mmm(V(*(typename V::left_t*)0, *(typename V::right_t*)0)...);
  };

  template <typename T1, typename T2, typename... V, typename... T>
  auto make_metamap_type(typelist<V...> variables, T1, T2, T... args) {
    return make_metamap_type(typelist<V..., assign_exp<T1, T2>>{},
              args...);
  };
}

// Helper to make a metamap type:
//  metamap_t<s::name_t, string, s::age_t, int>
//  instead of decltype(mmm(s::name = string(), s::age = int()));
template <typename... T>
using metamap_t = decltype(internal::make_metamap_type(typelist<>{}, std::declval<T>()...));


} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_MAKE_HH


namespace li {

struct skip {};
static struct {

  template <typename... M, typename... T>
  constexpr inline decltype(auto) run(metamap<M...> map, skip, T&&... args) const {
    return run(map, std::forward<T>(args)...);
  }

  template <typename T1, typename... M, typename... T>
  constexpr inline decltype(auto) run(metamap<M...> map, T1&& a, T&&... args) const {
    return run(
        cat(map, internal::make_metamap_helper(internal::exp_to_variable(std::forward<T1>(a)))),
        std::forward<T>(args)...);
  }

  template <typename... M> constexpr inline decltype(auto) run(metamap<M...> map) const { return map; }

  template <typename... T> constexpr inline decltype(auto) operator()(T&&... args) const {
    // Copy values.
    return run(metamap<>{}, std::forward<T>(args)...);
  }

} make_metamap_skip;

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_MAKE_METAMAP_SKIP_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_MAP_REDUCE_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_MAP_REDUCE_HH


namespace li {

// Map a function(key, value) on all kv pair
template <typename... M, typename F> constexpr void map(const metamap<M...>& m, F fun) {
  auto apply = [&](auto key) -> decltype(auto) { return fun(key, m[key]); };

  apply_each(apply, typename M::_iod_symbol_type{}...);
}

// Map a function(key, value) on all kv pair. Ensure that the calling order
// is kept.
// template <typename O, typename F>
// void map_sequential2(F fun, O& obj)
// {}
// template <typename O, typename M1, typename... M, typename F>
// void map_sequential2(F fun, O& obj, M1 m1, M... ms)
// {
//   auto apply = [&] (auto key) -> decltype(auto)
//     {
//       return fun(key, obj[key]);
//     };

//   apply(m1);
//   map_sequential2(fun, obj, ms...);
// }
// template <typename... M, typename F>
// void map_sequential(const metamap<M...>& m, F fun)
// {
//   auto apply = [&] (auto key) -> decltype(auto)
//     {
//       return fun(key, m[key]);
//     };

//   map_sequential2(fun, m, typename M::_iod_symbol_type{}...);
// }

// Map a function(key, value) on all kv pair (non const).
template <typename... M, typename F> constexpr void map(metamap<M...>& m, F fun) {
  auto apply = [&](auto key) -> decltype(auto) { return fun(key, m[key]); };

  apply_each(apply, typename M::_iod_symbol_type{}...);
}

template <typename... E, typename F, typename R> constexpr auto apply_each2(F&& f, R&& r, E&&... e) {
  return r(f(std::forward<E>(e))...);
  //(void)std::initializer_list<int>{
  //  ((void)f(std::forward<E>(e)), 0)...};
}

// Map a function(key, value) on all kv pair an reduce
// all the results value with the reduce(r1, r2, ...) function.
template <typename... M, typename F, typename R>
constexpr decltype(auto) map_reduce(const metamap<M...>& m, F map, R reduce) {
  auto apply = [&](auto key) -> decltype(auto) {
    // return map(key, std::forward<decltype(m[key])>(m[key]));
    return map(key, m[key]);
  };

  return apply_each2(apply, reduce, typename M::_iod_symbol_type{}...);
  // return reduce(apply(typename M::_iod_symbol_type{})...);
}

// Map a function(key, value) on all kv pair an reduce
// all the results value with the reduce(r1, r2, ...) function.
template <typename... M, typename R> constexpr decltype(auto) reduce(const metamap<M...>& m, R reduce) {
  return reduce(m[typename M::_iod_symbol_type{}]...);
}

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_MAP_REDUCE_HH



namespace li {

template <typename... T, typename... U>
constexpr inline decltype(auto) intersection(const metamap<T...>& a, const metamap<U...>& b) {
  return map_reduce(a,
                    [&](auto k, auto&& v) -> decltype(auto) {
                      if constexpr (has_key<metamap<U...>, decltype(k)>()) {
                        return k = std::forward<decltype(v)>(v);
                      } else
                        return skip{};
                    },
                    make_metamap_skip);
}

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_INTERSECTION_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_SUBSTRACT_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_SUBSTRACT_HH


namespace li {

template <typename... T, typename... U>
constexpr inline auto substract(const metamap<T...>& a, const metamap<U...>& b) {
  return map_reduce(a,
                    [&](auto k, auto&& v) {
                      if constexpr (!has_key<metamap<U...>, decltype(k)>()) {
                        return k = std::forward<decltype(v)>(v);
                      } else
                        return skip{};
                    },
                    make_metamap_skip);
}

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_SUBSTRACT_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_FORWARD_TUPLE_AS_METAMAP_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_FORWARD_TUPLE_AS_METAMAP_HH


namespace li {

namespace internal {

template <typename... S, typename... V, typename T, T... I>
constexpr auto forward_tuple_as_metamap_impl(std::tuple<S...> keys, std::tuple<V...>& values, std::integer_sequence<T, I...>) {
  return make_metamap_reference((std::get<I>(keys) = std::get<I>(values))...);
}
template <typename... S, typename... V, typename T, T... I>
constexpr auto forward_tuple_as_metamap_impl(std::tuple<S...> keys, const std::tuple<V...>& values, std::integer_sequence<T, I...>) {
  return make_metamap_reference((std::get<I>(keys) = std::get<I>(values))...);
}

} // namespace internal

template <typename... S, typename... V>
constexpr auto forward_tuple_as_metamap(std::tuple<S...> keys, std::tuple<V...>& values) {
  return internal::forward_tuple_as_metamap_impl(keys, values, std::make_index_sequence<sizeof...(V)>{});
}
template <typename... S, typename... V>
constexpr auto forward_tuple_as_metamap(std::tuple<S...> keys, const std::tuple<V...>& values) {
  return internal::forward_tuple_as_metamap_impl(keys, values, std::make_index_sequence<sizeof...(V)>{});
}

} // namespace li
#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_ALGORITHMS_FORWARD_TUPLE_AS_METAMAP_HH


#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_METAMAP_METAMAP_HH


#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_SYMBOLS_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_SYMBOLS_HH

#ifndef LI_SYMBOL_append
#define LI_SYMBOL_append
    LI_SYMBOL(append)
#endif

#ifndef LI_SYMBOL_generator
#define LI_SYMBOL_generator
    LI_SYMBOL(generator)
#endif

#ifndef LI_SYMBOL_json_key
#define LI_SYMBOL_json_key
    LI_SYMBOL(json_key)
#endif

#ifndef LI_SYMBOL_name
#define LI_SYMBOL_name
    LI_SYMBOL(name)
#endif

#ifndef LI_SYMBOL_size
#define LI_SYMBOL_size
    LI_SYMBOL(size)
#endif

#ifndef LI_SYMBOL_type
#define LI_SYMBOL_type
    LI_SYMBOL(type)
#endif


#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_SYMBOLS_HH


namespace li {

template <typename O> inline decltype(auto) wrap_json_output_stream(O&& s) {
  return mmm(s::append = [&s](auto c) { s << c; });
}

inline decltype(auto) wrap_json_output_stream(std::ostringstream& s) {
  return mmm(s::append = [&s](auto c) { s << c; });
}

inline decltype(auto) wrap_json_output_stream(std::string& s) {
  return mmm(s::append = [&s](auto c) {
    using C = std::remove_reference_t<decltype(c)>;
    if constexpr(std::is_same_v<C, char> || std::is_same_v<C, unsigned char> || std::is_same_v<C, int>)
      s.append(1, c); 
    else
      s.append(c);
  });
}

inline decltype(auto) wrap_json_input_stream(std::istringstream& s) { return s; }
inline decltype(auto) wrap_json_input_stream(std::stringstream& s) { return s; }
inline decltype(auto) wrap_json_input_stream(decode_stringstream& s) { return s; }
inline decltype(auto) wrap_json_input_stream(const std::string& s) { return decode_stringstream(s); }
inline decltype(auto) wrap_json_input_stream(const char* s) {
  return decode_stringstream(s);
}
inline decltype(auto) wrap_json_input_stream(const std::string_view& s) {
  return decode_stringstream(s);
}

namespace unicode_impl {
template <typename S, typename T> auto json_to_utf8(S&& s, T&& o);

template <typename S, typename T> auto utf8_to_json(S&& s, T&& o);
} // namespace unicode_impl

template <typename I, typename O> auto json_to_utf8(I&& i, O&& o) {
  return unicode_impl::json_to_utf8(wrap_json_input_stream(std::forward<I>(i)),
                                    wrap_json_output_stream(std::forward<O>(o)));
}

template <typename I, typename O> auto utf8_to_json(I&& i, O&& o) {
  return unicode_impl::utf8_to_json(wrap_json_input_stream(std::forward<I>(i)),
                                    wrap_json_output_stream(std::forward<O>(o)));
}

enum json_encodings { UTF32BE, UTF32LE, UTF16BE, UTF16LE, UTF8 };

// Detection of encoding depending on the pattern of the
// first fourth characters.
inline auto detect_encoding(char a, char b, char c, char d) {
  // 00 00 00 xx  UTF-32BE
  // xx 00 00 00  UTF-32LE
  // 00 xx 00 xx  UTF-16BE
  // xx 00 xx 00  UTF-16LE
  // xx xx xx xx  UTF-8

  if (a == 0 and b == 0)
    return UTF32BE;
  else if (c == 0 and d == 0)
    return UTF32LE;
  else if (a == 0)
    return UTF16BE;
  else if (b == 0)
    return UTF16LE;
  else
    return UTF8;
}

// The JSON RFC escapes character codepoints prefixed with a \uXXXX (7-11 bits codepoints)
// or \uXXXX\uXXXX (20 bits codepoints)

// uft8 string have 4 kinds of character representation encoding the codepoint of the character.

// 1 byte : 0xxxxxxx  -> 7 bits codepoint ASCII chars from 0x00 to 0x7F
// 2 bytes: 110xxxxx 10xxxxxx -> 11 bits codepoint
// 3 bytes: 1110xxxx 10xxxxxx 10xxxxxx -> 11 bits codepoint

// 1 and 3 bytes representation are escaped as \uXXXX with X a char in the 0-9A-F range. It
// is possible since the codepoint is less than 16 bits.

// the 4 bytes representation uses the UTF-16 surrogate pair (high and low surrogate).

// The high surrogate is in the 0xD800..0xDBFF range (HR) and
// the low surrogate is in the 0xDC00..0xDFFF range (LR).

// to encode a given 20bits codepoint c to the surrogate pair.
//  - substract 0x10000 to c
//  - separate the result in a high (first 10 bits) and low (last 10bits) surrogate.
//  - Add 0xD800 to the high surrogate
//  - Add 0xDC00 to the low surrogate
//  - the 32 bits code is (high << 16) + low.

// and to json-escape the high-low(H-L) surrogates representation (16+16bits):
//  - Check that H and L are respectively in the HR and LR ranges.
//  - add to H-L 0x0001_0000 - 0xD800_DC00 to get the 20bits codepoint c.
//  - Encode the codepoint in a string of \uXXXX\uYYYY with X and Y the respective hex digits
//    of the high and low sequence of 10 bits.

// In addition to utf8, JSON escape characters ( " \ / ) with a backslash and translate
// \n \r \t \b \r in their matching two characters string, for example '\n' to  '\' followed by 'n'.

namespace unicode_impl {
template <typename S, typename T> auto json_to_utf8(S&& s, T&& o) {
  // Convert a JSON string into an UTF-8 string.
  if (s.get() != '"')
    return JSON_KO; // make_json_error("json_to_utf8: JSON strings should start with a double
                    // quote.");

  while (true) {
    // Copy until we find the escaping backslash or the end of the string (double quote).
    while (s.peek() != EOF and s.peek() != '"' and s.peek() != '\\')
      o.append(s.get());

    // If eof found before the end of the string, return an error.
    if (s.eof())
      return JSON_KO; // make_json_error("json_to_utf8: Unexpected end of string when parsing a
                      // string.");

    // If end of json string, return
    if (s.peek() == '"') {
      break;
      return JSON_OK;
    }

    // Get the '\'.
    assert(s.peek() == '\\');
    s.get();

    switch (s.get()) {
      // List of escaped char from http://www.json.org/
    default:
      return JSON_KO; // make_json_error("json_to_utf8: Bad JSON escaped character.");
    case '"':
      o.append('"');
      break;
    case '\\':
      o.append('\\');
      break;
    case '/':
      o.append('/');
      break;
    case 'n':
      o.append('\n');
      break;
    case 'r':
      o.append('\r');
      break;
    case 't':
      o.append('\t');
      break;
    case 'b':
      o.append('\b');
      break;
    case 'f':
      o.append('\f');
      break;
    case 'u':
      char a, b, c, d;

      a = s.get();
      b = s.get();
      c = s.get();
      d = s.get();

      if (s.eof())
        return JSON_KO; // make_json_error("json_to_utf8: Unexpected end of string when decoding an
                        // utf8 character");

      auto decode_hex_c = [](char c) {
        if (c >= '0' and c <= '9')
          return c - '0';
        else
          return (10 + std::toupper(c) - 'A');
      };

      uint16_t x = (decode_hex_c(a) << 12) + (decode_hex_c(b) << 8) + (decode_hex_c(c) << 4) +
                   decode_hex_c(d);

      // If x in the  0xD800..0xDBFF range -> decode a surrogate pair \uXXXX\uYYYY -> 20 bits
      // codepoint.
      if (x >= 0xD800 and x <= 0xDBFF) {
        if (s.get() != '\\' or s.get() != 'u')
          return JSON_KO; // make_json_error("json_to_utf8: Missing low surrogate.");

        uint16_t y = (decode_hex_c(s.get()) << 12) + (decode_hex_c(s.get()) << 8) +
                     (decode_hex_c(s.get()) << 4) + decode_hex_c(s.get());

        if (s.eof())
          return JSON_KO; // make_json_error("json_to_utf8: Unexpected end of string when decoding
                          // an utf8 character");

        x -= 0xD800;
        y -= 0xDC00;

        int cp = (x << 10) + y + 0x10000;

        o.append(0b11110000 | (cp >> 18));
        o.append(0b10000000 | ((cp & 0x3F000) >> 12));
        o.append(0b10000000 | ((cp & 0x00FC0) >> 6));
        o.append(0b10000000 | (cp & 0x003F));

      }
      // else encode the codepoint with the 1-2, or 3 bytes utf8 representation.
      else {
        if (x <= 0x007F) // 7bits codepoints, ASCII 0xxxxxxx.
        {
          o.append(uint8_t(x));
        } else if (x >= 0x0080 and x <= 0x07FF) // 11bits codepoint -> 110xxxxx 10xxxxxx
        {
          o.append(0b11000000 | (x >> 6));
          o.append(0b10000000 | (x & 0x003F));
        } else if (x >= 0x0800 and x <= 0xFFFF) // 16bits codepoint -> 1110xxxx 10xxxxxx 10xxxxxx
        {
          o.append(0b11100000 | (x >> 12));
          o.append(0b10000000 | ((x & 0x0FC0) >> 6));
          o.append(0b10000000 | (x & 0x003F));
        } else
          return JSON_KO; // make_json_error("json_to_utf8: Bad UTF8 codepoint.");
      }
      break;
    }
  }

  if (s.get() != '"')
    return JSON_KO; // make_json_error("JSON strings must end with a double quote.");

  return JSON_OK; // json_no_error();
}

template <typename S, typename T> auto utf8_to_json(S&& s, T&& o) {
  o.append('"');

  auto encode_16bits = [&](uint16_t b) {
    const char lt[] = "0123456789ABCDEF";
    o.append(lt[b >> 12]);
    o.append(lt[(b & 0x0F00) >> 8]);
    o.append(lt[(b & 0x00F0) >> 4]);
    o.append(lt[b & 0x000F]);
  };

  while (!s.eof()) {
    // 7-bits codepoint
    if constexpr(std::is_same_v<std::remove_reference_t<S>, decode_stringstream>)
      s.copy_until(o, [] (char c) { return c > '"' && c <= '~' && c != '\\'; });

    while (s.good() and s.peek() <= 0x7F and s.peek() != EOF) {
      switch (s.peek()) {
      case '"':
        o.append('\\');
        o.append('"');
        break;
      case '\\':
        o.append('\\');
        o.append('\\');
        break;
        // case '/': o.append('/'); break; Do not escape /
      case '\n':
        o.append('\\');
        o.append('n');
        break;
      case '\r':
        o.append('\\');
        o.append('r');
        break;
      case '\t':
        o.append('\\');
        o.append('t');
        break;
      case '\b':
        o.append('\\');
        o.append('b');
        break;
      case '\f':
        o.append('\\');
        o.append('f');
        break;
      default:
        o.append(s.peek());
      }
      s.get();
    }

    if (s.eof())
      break;

    // uft8 prefix \u.
    o.append('\\');
    o.append('u');

    uint8_t c1 = s.get();
    uint8_t c2 = s.get();
    {
      // extract codepoints.
      if (c1 < 0b11100000) // 11bits - 2 char: 110xxxxx	10xxxxxx
      {
        uint16_t cp = ((c1 & 0b00011111) << 6) + (c2 & 0b00111111);
        if (cp >= 0x0080 and cp <= 0x07FF)
          encode_16bits(cp);
        else
          return JSON_KO;         // make_json_error("utf8_to_json: Bad UTF8 codepoint.");
      } else if (c1 < 0b11110000) // 16 bits - 3 char: 1110xxxx	10xxxxxx	10xxxxxx
      {
        uint16_t cp = ((c1 & 0b00001111) << 12) + ((c2 & 0b00111111) << 6) + (s.get() & 0b00111111);

        if (cp >= 0x0800 and cp <= 0xFFFF)
          encode_16bits(cp);
        else
          return JSON_KO; // make_json_error("utf8_to_json: Bad UTF8 codepoint.");
      } else              // 21 bits - 4 chars: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
      {
        int cp = ((c1 & 0b00000111) << 18) + ((c2 & 0b00111111) << 12) +
                 ((s.get() & 0b00111111) << 6) + (s.get() & 0b00111111);

        cp -= 0x10000;

        uint16_t H = (cp >> 10) + 0xD800;
        uint16_t L = (cp & 0x03FF) + 0xDC00;

        // check if we are in the right range.
        // The high surrogate is in the 0xD800..0xDBFF range (HR) and
        // the low surrogate is in the 0xDC00..0xDFFF range (LR).
        assert(H >= 0xD800 and H <= 0xDBFF and L >= 0xDC00 and L <= 0xDFFF);

        encode_16bits(H);
        o.append('\\');
        o.append('u');
        encode_16bits(L);
      }
    }
  }
  o.append('"');
  return JSON_OK;
}
} // namespace unicode_impl
} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_UNICODE_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_UTILS_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_UTILS_HH



namespace li {

template <typename T> struct json_object_base;

template <typename T> struct json_object_;
template <typename T> struct json_vector_;
template <typename T, unsigned N> struct json_static_array_;
template <typename V> struct json_value_;
template <typename V> struct json_map_;
template <typename... T> struct json_tuple_;
template <typename... T> struct json_variant_;
struct json_key;

namespace impl {
template <typename S, typename... A>
auto make_json_object_member(const function_call_exp<S, A...>& e);
template <typename S> auto make_json_object_member(const li::symbol<S>&);

template <typename S, typename T> auto make_json_object_member(const assign_exp<S, T>& e) {
  return cat(make_json_object_member(e.left), mmm(s::type = e.right));
}

template <typename S> auto make_json_object_member(const li::symbol<S>&) {
  return mmm(s::name = S{});
}

template <typename V> auto to_json_schema(V v);
template <typename... M> auto to_json_schema(const metamap<M...>& m);
template <typename V> auto to_json_schema(const std::vector<V>& arr);
template <typename... V> auto to_json_schema(const std::tuple<V...>& arr);
template <typename... V> auto to_json_schema(const std::variant<V...>& v);
template <typename K, typename V> auto to_json_schema(const std::unordered_map<K, V>& arr);
template <typename K, typename V> auto to_json_schema(const std::map<K, V>& arr);
template <typename... M> auto to_json_schema(const metamap<M...>& m);
template <typename V, unsigned N> auto to_json_schema(const V (&v)[N]);
template <unsigned N> auto to_json_schema(const char (&v)[N]);

template <typename V> auto to_json_schema(V v) {
  if constexpr (std::is_pointer_v<V> and !std::is_same_v<const char*, V> and !std::is_array_v<V>)
    return to_json_schema(*v);
  else
   return json_value_<V>{}; 
}

template <typename V> auto to_json_schema(const std::vector<V>& arr) {
  auto elt = to_json_schema(V{});
  return json_vector_<decltype(elt)>{elt};
}
template <unsigned N> auto to_json_schema(const char (&v)[N]) {
  return json_value_<const char[N]>{};
}

template <typename... V> auto to_json_schema(const std::tuple<V...>& arr) {
  return json_tuple_<decltype(to_json_schema(V{}))...>(to_json_schema(V{})...);
}
template <typename... V> auto to_json_schema(const std::variant<V...>& v) {
  return json_variant_<decltype(to_json_schema(V{}))...>(to_json_schema(V{})...);
}
template <typename V, unsigned N> auto to_json_schema(const V (&v)[N]) {
  auto elt = to_json_schema(V{});
  return json_static_array_<decltype(elt), N>(elt);
}
template <typename K, typename V> auto to_json_schema(const std::unordered_map<K, V>& arr) {
  return json_map_<decltype(to_json_schema(V{}))>(to_json_schema(V{}));
}
template <typename K, typename V> auto to_json_schema(const std::map<K, V>& arr) {
  return json_map_<decltype(to_json_schema(V{}))>(to_json_schema(V{}));
}

template <typename... M> auto to_json_schema(const metamap<M...>& m) {
  auto tuple_maker = [](auto&&... t) { return std::make_tuple(std::forward<decltype(t)>(t)...); };

  auto entities = map_reduce(
      m, [](auto k, auto v) { return mmm(s::name = k, s::type = to_json_schema(v)); }, tuple_maker);

  return json_object_<decltype(entities)>(entities);
}


template <typename... E> auto json_object_to_metamap(const json_object_<std::tuple<E...>>& s) {
  auto make_mmm = [](auto... elt) { return mmm((elt.name = elt.type)...); };
  return std::apply(make_mmm, s.entities);
}

template <typename S, typename... A>
auto make_json_object_member(const function_call_exp<S, A...>& e) {
  auto res = mmm(s::name = e.method, s::json_key = symbol_string(e.method));

  auto parse = [&](auto a) {
    if constexpr (std::is_same<decltype(a), json_key>::value) {
      res.json_key = a.key;
    }
  };

  ::li::tuple_map(e.args, parse);
  return res;
}

} // namespace impl

template <typename T> struct json_object_;

template <typename O> struct json_vector_;

template <typename T> struct is_json_vector : std::false_type {};
template <typename T> struct is_json_vector<json_vector_<T>> : std::true_type {};

template <typename E> constexpr auto json_is_vector(json_vector_<E>) -> std::true_type {
  return {};
}
template <typename E> constexpr auto json_is_vector(E) -> std::false_type { return {}; }
template <typename E, unsigned N> constexpr auto json_is_static_array(json_static_array_<E, N>) -> std::true_type {
  return {};
}
template <typename E> constexpr auto json_is_static_array(E) -> std::false_type { return {}; }

template <typename... E> constexpr auto json_is_tuple(json_tuple_<E...>) -> std::true_type {
  return {};
}
template <typename E> constexpr auto json_is_tuple(E) -> std::false_type { return {}; }

template <typename... E> constexpr auto json_is_variant(json_variant_<E...>) -> std::true_type {
  return {};
}
template <typename E> constexpr auto json_is_variant(E) -> std::false_type { return {}; }

template <typename E> constexpr auto json_is_object(json_object_<E>) -> std::true_type {
  return {};
}
template <typename E> constexpr auto json_is_object(json_map_<E>) -> std::true_type {
  return {};
}
template <typename E> constexpr auto json_is_object(E) -> std::false_type { return {}; }


template <typename E> constexpr auto json_is_value(json_object_<E>) -> std::false_type {
  return {};
}
template <typename... E> constexpr auto json_is_value(json_variant_<E...>) -> std::false_type {
  return {};
}
template <typename E> constexpr auto json_is_value(json_vector_<E>) -> std::false_type {
  return {};
}
template <typename... E> constexpr auto json_is_value(json_tuple_<E...>) -> std::false_type {
  return {};
}
template <typename E> constexpr auto json_is_value(json_map_<E>) -> std::false_type {
  return {};
}
template <typename E> constexpr auto json_is_value(E) -> std::true_type { return {}; }

template <typename T> constexpr auto is_std_optional(std::optional<T>) -> std::true_type;
template <typename T> constexpr auto is_std_optional(T) -> std::false_type;

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_UTILS_HH


namespace li {

namespace impl {

template <typename S> struct json_parser {
  inline json_parser(S&& s) : ss(s) {}
  inline json_parser(S& s) : ss(s) {}

  inline decltype(auto) peek() { return ss.peek(); }
  inline decltype(auto) get() { return ss.get(); }

  inline void skip_one() { ss.get(); }

  inline bool eof() { return ss.eof(); }
  inline json_error_code eat(char c, bool skip_spaces = true) {
    if (skip_spaces)
      eat_spaces();

    char g = ss.get();
    if (g != c)
      return make_json_error("Unexpected char. Got '", char(g), "' expected ", c);
    return JSON_OK;
  }

  inline json_error_code eat(const char* str, bool skip_spaces = true) {
    if (skip_spaces)
      eat_spaces();

    const char* str_it = str;
    while (*str_it) {
      char g = ss.get();
      if (g != *str_it)
        return make_json_error("Unexpected char. Got '", char(g), "' expected '", *str_it,
                               "' when parsing string ", str);
      str_it++;
    }
    return JSON_OK;
  }

  json_error_code eat_json_key(char* buffer, int buffer_size, int& key_size) {
    if (auto err = eat('"'))
      return err;
    key_size = 0;
    while (!eof() and peek() != '"' and key_size < (buffer_size - 1))
      buffer[key_size++] = get();
    buffer[key_size] = 0;
    if (auto err = eat('"', false))
      return err;
    return JSON_OK;
  }

  template <typename... T> inline json_error_code make_json_error(T&&... t) {
    if (!error_stream)
      error_stream = std::make_unique<std::ostringstream>();
    *error_stream << "json error: ";
    auto add = [this](auto w) { *error_stream << w; };
    apply_each(add, t...);
    return JSON_KO;
  }
  inline void eat_spaces() {
    while (ss.peek() >= 0 and ss.peek() < 33)
      ss.get();
  }

  template <typename X> struct JSON_INVALID_TYPE;

  // Integers and floating points.
  template <typename T> json_error_code fill(T& t) {

    if constexpr (std::is_floating_point<T>::value or std::is_integral<T>::value or
                  std::is_same<T, std::string_view>::value) {
      ss >> t;
      if (ss.bad())
        return make_json_error("Ill-formated value.");
      return JSON_OK;
    } else
      // The JSON decoder only parses floating-point, integral and string types.
      return JSON_INVALID_TYPE<T>::error;
  }

  // Strings
  inline json_error_code fill(std::string& str) {
    eat_spaces();
    str.clear();
    return json_to_utf8(ss, str);
  }

  template <typename T> inline json_error_code fill(std::optional<T>& opt) {
    opt.emplace();
    return fill(opt.value());
  }
  template <typename T, std::size_t N> inline json_error_code fill(T (&opt)[N]) {
    if (auto err = eat('['))
      return err;
    for (int i = 0; i < N; i++) {
      if (i > 0)
        if (auto err = eat(','))
          return err;
      fill(opt[i]);
    }

    if (auto err = eat(']'))
      return err;
    return JSON_OK;
  }

  S& ss;
  std::unique_ptr<std::ostringstream> error_stream = nullptr;
};

template <typename P, typename O, typename S> json_error_code json_decode2(P& p, O& obj, S) {
  auto err = p.fill(obj);
  if (err)
    return err;
  else
    return JSON_OK;
}

template <typename S>
struct json_vector_element_type { typedef json_value_<int> type; };
template <typename S>
struct json_vector_element_type<json_vector_<S>> { typedef std::remove_reference_t<S> type; };

template <typename P, typename O, typename S>
json_error_code json_decode2(P& p, std::vector<O>& obj, S schema) {
  obj.clear();
  bool first = true;
  auto err = p.eat('[');
  if (err)
    return err;

  p.eat_spaces();
  while (p.peek() != ']') {
    if (!first) {
      if ((err = p.eat(',')))
        return err;
    }
    first = false;

    obj.resize(obj.size() + 1);
    auto element_schema = [&schema] () {
      if constexpr (is_json_vector<S>::value) return schema.schema;
      else return json_value_<O>();
    }();

    if ((err = json_decode2(p, obj.back(), element_schema)))
      return err;
    p.eat_spaces();
  }

  if ((err = p.eat(']')))
    return err;
  else
    return JSON_OK;
}
// template <typename P, typename O, typename S>
// json_error_code json_decode2(P& p, std::vector<O>& obj, S schema) {
//   obj.clear();
//   bool first = true;
//   auto err = p.eat('[');
//   if (err)
//     return err;

//   p.eat_spaces();
//   while (p.peek() != ']') {
//     if (!first) {
//       if ((err = p.eat(',')))
//         return err;
//     }
//     first = false;

//     obj.resize(obj.size() + 1);
//     if ((err = json_decode2(p, obj.back(), p)))
//       return err;
//     p.eat_spaces();
//   }

//   if ((err = p.eat(']')))
//     return err;
//   else
//     return JSON_OK;
// }

template <typename F, typename... E, typename... T, std::size_t... I>
inline void json_decode_tuple_elements(F& decode_fun, std::tuple<T...>& tu,
                                       const std::tuple<E...>& schema, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{((void)decode_fun(std::get<I>(tu), std::get<I>(schema)), 0)...};
}

template <typename P, typename... O, typename... S>
json_error_code json_decode2(P& p, std::tuple<O...>& tu, json_tuple_<S...> schema) {
  bool first = true;
  auto err = p.eat('[');
  if (err)
    return err;

  auto decode_one_element = [&first, &p, &err](auto& value, auto value_schema) {
    if (!first) {
      if ((err = p.eat(',')))
        return err;
    }
    first = false;
    if ((err = json_decode2(p, value, value_schema)))
      return err;
    p.eat_spaces();
    return JSON_OK;
  };

  json_decode_tuple_elements(decode_one_element, tu, schema.elements,
                             std::make_index_sequence<sizeof...(O)>{});

  if ((err = p.eat(']')))
    return err;
  else
    return JSON_OK;
}

template <typename F, typename... E, typename... T, std::size_t... I>
inline void json_decode_variant_elements(F& decode_fun, std::variant<T...>& tu,
                                         const std::tuple<E...>& schema,
                                         std::index_sequence<I...>) {
  (void)std::initializer_list<int>{
      ((void)decode_fun([] { return T(); }, std::get<I>(schema)), 0)...};
}

template <typename P, typename... O, typename... S>
json_error_code json_decode2(P& p, std::variant<O...>& tu, json_variant_<S...> schema) {
  if (auto err = p.eat('{'))
    return err;
  if (auto err = p.eat("\"idx\""))
    return err;
  if (auto err = p.eat(':'))
    return err;

  int idx = 0;
  p.fill(idx);
  if (auto err = p.eat(','))
    return err;
  if (auto err = p.eat("\"value\""))
    return err;
  if (auto err = p.eat(':'))
    return err;

  int decode_idx = -1;
  json_error_code error = JSON_OK;

  auto decode_one_element = [idx, &p, &tu, &error, &decode_idx](auto get_value, auto value_schema) {
    decode_idx++;
    if (idx == decode_idx) {
      auto val = get_value();
      if (auto err = json_decode2(p, val, value_schema))
        error = err;
      else {
        tu = std::variant<O...>{val};
      }
    }
  };
  json_decode_variant_elements(decode_one_element, tu, schema.elements,
                               std::make_index_sequence<sizeof...(O)>{});

  if (error)
    return error;

  if (auto err = p.eat('}'))
    return err;
  else
    return JSON_OK;
}

template <typename P, typename O, typename V>
json_error_code json_decode2(json_parser<P>& p, O& obj, json_map_<V> schema) {
  if (auto err = p.eat('{'))
    return err;

  p.eat_spaces();

  using mapped_type = typename O::mapped_type;
  while (true) {
    // Parse key:
    char key[50];
    int key_size = 0;
    if (auto err = p.eat_json_key(key, sizeof(key), key_size))
      return err;

    std::string_view key_str(key, key_size);

    if (auto err = p.eat(':'))
      return err;

    // Parse value.
    mapped_type& map_value = obj[std::string(key_str)];
    if (auto err = json_decode2(p, map_value, V{}))
      return err;

    p.eat_spaces();
    if (p.peek() == ',')
      p.get();
    else
      break;
  }

  if (auto err = p.eat('}'))
    return err;

  return JSON_OK;
}

template <typename P, typename O, typename S>
json_error_code json_decode2(P& p, O& obj, json_object_<S> schema) {
  json_error_code err;
  if ((err = p.eat('{')))
    return err;

  struct attr_info {
    bool filled;
    bool required;
    const char* name;
    int name_len;
    std::function<json_error_code(P&)> parse_value;
  };
  constexpr int n_members = int(std::tuple_size<decltype(schema.schema)>());
  attr_info A[n_members];
  int i = 0;
  auto prepare = [&](auto m) {
    A[i].filled = false;
    A[i].required = true;
    A[i].name = symbol_string(m.name);
    A[i].name_len = int(strlen(symbol_string(m.name)));

    if constexpr (has_key(m, s::json_key)) {
      A[i].name = m.json_key;
      A[i].name_len = strlen(m.json_key);
    }

    if constexpr (decltype(is_std_optional(symbol_member_or_getter_access(obj, m.name))){}) {
      A[i].required = false;
    }

    A[i].parse_value = [m, &obj](P& p) {
      using V = decltype(symbol_member_or_getter_access(obj, m.name));
      using VS = decltype(get_or(m, s::type, json_value_<V>{}));

      // if constexpr (decltype(json_is_value(VS{})){}) {
      //   if (auto err = p.fill(symbol_member_or_getter_access(obj, m.name)))
      //     return err;
      //   else
      //     return JSON_OK;
      // } else {
        if (auto err = json_decode2(p, symbol_member_or_getter_access(obj, m.name), get_or(m, s::type, obj)))
          return err;
        else
          return JSON_OK;
      // }
    };

    i++;
  };

  std::apply([&](auto... m) { apply_each(prepare, m...); }, schema.schema);

  while (p.peek() != '}') {

    bool found = false;
    if ((err = p.eat('"')))
      return err;
    char symbol[50 + 1];
    int symbol_size = 0;
    while (!p.eof() and p.peek() != '"' and symbol_size < 50)
      symbol[symbol_size++] = p.get();
    symbol[symbol_size] = 0;
    if ((err = p.eat('"', false)))
      return err;

    for (int i = 0; i < n_members; i++) {
      int len = A[i].name_len;
      if (len == symbol_size && !strncmp(symbol, A[i].name, len)) {
        if ((err = p.eat(':')))
          return err;
        if (A[i].filled)
          return p.make_json_error("Duplicate json key: ", A[i].name);

        if ((err = A[i].parse_value(p)))
          return err;
        A[i].filled = true;
        found = true;
        break;
      }
    }

    if (!found)
      return p.make_json_error("Unknown json key: ", symbol);
    p.eat_spaces();
    if (p.peek() == ',') {
      if ((err = p.eat(',')))
        return err;
    }
  }
  if ((err = p.eat('}')))
    return err;

  for (int i = 0; i < n_members; i++) {
    if (A[i].required and !A[i].filled)
      return p.make_json_error("Missing json key ", A[i].name);
  }
  return JSON_OK;
}

template <typename C, typename O, typename S> json_error json_decode(C& input, O& obj, S schema) {
  auto stream = decode_stringstream(input);
  json_parser<decode_stringstream> p(stream);
  if (json_decode2(p, obj, schema))
    return json_error{1, p.error_stream ? p.error_stream->str() : "Json error"};
  else
    return json_error{0};
}

} // namespace impl

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_DECODER_HH

#ifndef LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_ENCODER_HH
#define LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_ENCODER_HH


namespace li {

using std::string_view;

template <typename... T> struct json_tuple_;
template <typename T> struct json_object_;

namespace impl {

// Json encoder.
// =============================================

template <typename C, typename O, typename E>
inline std::enable_if_t<!std::is_pointer<O>::value, void>
json_encode(C& ss, O obj, const json_object_<E>& schema);
template <typename C, typename... E, typename... T>
inline void json_encode(C& ss, const std::tuple<T...>& tu, const json_tuple_<E...>& schema);
template <typename T, typename C, typename E>
inline void json_encode(C& ss, const T& value, const E& schema);
template <typename T, typename C, typename E>
inline void json_encode(C& ss, const std::vector<T>& array, const json_vector_<E>& schema);
template <typename C, typename O, typename S>
inline void json_encode(C& ss, O* obj, const S& schema);
template <typename C, typename O, typename S>
inline void json_encode(C& ss, const O* obj, const S& schema);

template <typename T, typename C> inline void json_encode_value(C& ss, const T& t) { 
  ss << t; 
  }

template <typename C> inline void json_encode_value(C& ss, const char* s) {
  // ss << s;
  utf8_to_json(s, ss);
}

template <typename C> inline void json_encode_value(C& ss, const string_view& s) {
  // ss << s;
  utf8_to_json(s, ss);
}

template <typename C> inline void json_encode_value(C& ss, const std::string& s) {
  // ss << s;
  utf8_to_json(s, ss);
}

template <typename C> inline void json_encode_value(C& ss, const char& c) {
  ss << int(c);
}
template <typename C> inline void json_encode_value(C& ss, const uint8_t& c) {
  ss << int(c);
}

// template <typename C, unsigned N, typename T> inline void json_encode_value(C& ss, T (&s)[N]) {
//   // ss << s;
//   std::cout << "ARRAY! " << N << std::endl;

//  }

template <typename C, typename... T> inline void json_encode_value(C& ss, const metamap<T...>& s) {
  json_encode(ss, s, to_json_schema(s));
}

template <typename T, typename C> inline void json_encode_value(C& ss, const std::optional<T>& t) {
  if (t.has_value())
    json_encode_value(ss, t.value());
}

template <typename F, typename... E, typename... T, std::size_t... I>
inline void json_encode_variant_value(F& encode_fun, const std::variant<T...>& tu,
                                      const std::tuple<E...>& schema, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{
      ((void)encode_fun([&tu] { return std::get<I>(tu); }, std::get<I>(schema)), 0)...};
}

template <typename C, typename... T, typename... E>
inline void json_encode(C& ss, const std::variant<T...>& t, const json_variant_<E...>& schema) {
  ss << "{\"idx\":" << t.index() << ",\"value\":";
  int idx = -1;
  auto encode_one_element = [&t, &ss, &idx](auto get_value, auto value_schema) {
    idx++;
    if (idx == t.index()) {
      json_encode(ss, get_value(), value_schema);
    }
  };

  json_encode_variant_value(encode_one_element, t, schema.elements,
                            std::make_index_sequence<sizeof...(T)>{});

  ss << '}';
}

template <typename T, typename C, typename E>
inline void json_encode(C& ss, const T& value, const E& schema) {
  json_encode_value(ss, value);
}

template <typename T, typename C, typename E>
inline void json_encode(C& ss, const std::vector<T>& array, const json_vector_<E>& schema) {
  ss << '[';
  for (const auto& t : array) {
      json_encode(ss, t, schema.schema);
    if (&t != &array.back())
      ss << ',';
  }
  ss << ']';
}
template <typename T, typename C, typename E>
inline void json_encode(C& ss, const std::vector<T>& array, const E&) {
  ss << '[';
  for (const auto& t : array) {
      json_encode(ss, t, ss);
    if (&t != &array.back())
      ss << ',';
  }
  ss << ']';
}

template <typename E, typename C, typename G>
inline void json_encode(C& ss,
                        const metamap<typename s::size_t::variable_t<int>,
                                      typename s::generator_t::variable_t<G>>& generator,
                        const json_vector_<E>& schema) {
  ss << '[';
  for (int i = 0; i < generator.size; i++) {
    json_encode(ss, generator.generator(), schema.schema);

    if (i != generator.size - 1)
      ss << ',';
  }
  ss << ']';
}

template <typename V, typename C, typename M>
inline void json_encode(C& ss, const M& map, const json_map_<V>& schema) {
  ss << '{';
  bool first = true;
  for (const auto& pair : map) {
    if (!first)
      ss << ',';

    json_encode_value(ss, pair.first);
    ss << ':';

    json_encode(ss, pair.second, schema.mapped_schema);

    first = false;
  }

  ss << '}';
}

template <typename F, typename... E, typename... T, std::size_t... I>
inline void json_encode_tuple_elements(F& encode_fun, const std::tuple<T...>& tu,
                                       const std::tuple<E...>& schema, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{((void)encode_fun(std::get<I>(tu), std::get<I>(schema)), 0)...};
}

template <typename C, typename... E, typename... T>
inline void json_encode(C& ss, const std::tuple<T...>& tu, const json_tuple_<E...>& schema) {
  ss << '[';
  bool first = true;
  auto encode_one_element = [&first, &ss](auto value, auto value_schema) {
    if (!first)
      ss << ',';
    first = false;
    json_encode(ss, value, value_schema);
  };

  json_encode_tuple_elements(encode_one_element, tu, schema.elements,
                             std::make_index_sequence<sizeof...(T)>{});
  ss << ']';
}

template <unsigned N, typename O, typename C>
inline void json_encode(C& ss, O (&t)[N], C&) {
  ss << '[';
  for (int i = 0; i < N; i++) {
    if (i > 0)
      ss << ',';
    json_encode(ss, t[i], ss);
  }
  ss << ']';
}

template <typename T, unsigned N, typename O, typename C>
inline void json_encode(C& ss, O* t, const json_static_array_<T, N>& schema) {
  ss << '[';
  for (int i = 0; i < N; i++) {
    if (i > 0)
      ss << ',';
    json_encode(ss, t[i], schema.element_schema);
  }
  ss << ']';
}

template <typename C, typename O, typename E>
inline std::enable_if_t<!std::is_pointer<O>::value, void>
json_encode(C& ss, O obj, const json_object_<E>& schema) {
  ss << '{';
  bool first = true;

  auto encode_one_entity = [&](auto e) {
    if constexpr (decltype(is_std_optional(symbol_member_or_getter_access(obj, e.name))){}) {
      if (!symbol_member_or_getter_access(obj, e.name).has_value())
        return;
    }

    if (!first) {
      ss << ',';
    }
    first = false;
    if constexpr (has_key(e, s::json_key)) {
      json_encode_value(ss, e.json_key);
      ss << ':';
    } else
      ss << e.name.json_key_string();

    const auto& var_to_encode = symbol_member_or_getter_access(obj, e.name);
    if constexpr (has_key(e, s::type)) {
      json_encode(ss, var_to_encode, e.type);
    }
    else {
      json_encode(ss, var_to_encode, ss);
    }
  };

  tuple_map(schema.schema, encode_one_entity);
  ss << '}';
}

template <typename C, typename O, typename S>
inline void json_encode(C& ss, O* obj, const S& schema) {
  if constexpr (std::is_same_v<char, O>)
    return json_encode_value(ss, obj);
  json_encode(ss, *obj, schema);
}

template <typename C, typename O, typename S>
inline void json_encode(C& ss, const O* obj, const S& schema) {
  // Special case for pointers.
  // string: const char* -> json_encode_value
  if constexpr (std::is_same_v<char, O>)
    return json_encode_value(ss, obj);
  // other pointers, dereference encode(*v);
  json_encode(ss, *obj, schema);
}

} // namespace impl

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_ENCODER_HH


namespace li {

template <typename T> struct is_json_vector;

template <typename T> struct json_object_base {

public:
  inline auto downcast() const { return static_cast<const T*>(this); }

  template <typename C, typename O> void encode(C& output, O&& obj) const {
    return impl::json_encode(output, std::forward<O>(obj), *downcast());
  }

  template <typename C, typename... M> void encode(C& output, const metamap<M...>& obj) const {
    return impl::json_encode(output, obj, *downcast());
  }

  template <typename O> std::string encode(O obj) const {
    std::ostringstream ss;
    impl::json_encode(ss, std::forward<O>(obj), *downcast());
    return ss.str();
  }

  template <typename... M> std::string encode(const metamap<M...>& obj) const {
    std::ostringstream ss;
    impl::json_encode(ss, obj, *downcast());
    return ss.str();
  }

  template <typename C, typename G> void encode_generator(C& output, int size, G& generator) const {
    static_assert(is_json_vector<T>::value, "encode_generator is only supported on json vector");
      return this->encode(output, mmm(s::size = size, s::generator = generator));
  }
  template <typename G> std::string encode_generator(int size, G& generator) const {
    static_assert(is_json_vector<T>::value, "encode_generator is only supported on json vector");
      return this->encode(mmm(s::size = size, s::generator = generator));
  }


  template <typename C, typename O> json_error decode(C& input, O& obj) const {
    return impl::json_decode(input, obj, *downcast());
  }

  template <typename C, typename... M> auto decode(C& input) const {
    auto map = impl::json_object_to_metamap(*downcast());
    impl::json_decode(input, map, *downcast());
    return map;
  }
};

template <typename T> struct json_object_ : public json_object_base<json_object_<T>> {
  json_object_() = default;
  json_object_(const T& s) : schema(s) {}
  T schema;
};

template <typename... S> auto json_object(S&&... s) {
  auto members = std::make_tuple(impl::make_json_object_member(std::forward<S>(s))...);
  return json_object_<decltype(members)>{members};
}

template <typename V> struct json_value_ : public json_object_base<json_value_<V>> {
  json_value_() = default;
};

template <typename V> auto json_value(V&& v) { return json_value_<V>{}; }

template <typename T> struct json_vector_ : public json_object_base<json_vector_<T>> {
  enum { is_json_vector = true };
  json_vector_() = default;
  json_vector_(const T& s) : schema(s) {}
  T schema;
};

template <typename... S> auto json_object_vector(S&&... s) {
  auto obj = json_object(std::forward<S>(s)...);
  return json_vector_<decltype(obj)>{obj};
}
template <typename E> auto json_vector(E&& element) {
  return json_vector_<decltype(element)>{element};
}

template <typename T, unsigned N> struct json_static_array_ : public json_object_base<json_static_array_<T, N>> {
  enum { is_json_static_array = true };
  json_static_array_() = default;
  json_static_array_(const T& s) : element_schema(s) {}
  T element_schema;
};

template <unsigned N, typename... S> auto json_static_array(S&&... s) {
  auto obj = json_object(std::forward<S>(s)...);
  return json_static_array_<decltype(obj), N>{obj};
}

template <typename... T> struct json_tuple_ : public json_object_base<json_tuple_<T...>> {
  json_tuple_() = default;
  json_tuple_(const T&... s) : elements(s...) {}
  std::tuple<T...> elements;
};

template <typename... S> auto json_tuple(S&&... s) { return json_tuple_<S...>{s...}; }

struct json_key {
  inline json_key(const char* c) : key(c) {}
  const char* key;
};

template <typename V> struct json_map_ : public json_object_base<json_map_<V>> {
  json_map_() = default;
  json_map_(const V& s) : mapped_schema(s) {}
  V mapped_schema;
};

template <typename V> auto json_map() { return json_map_<V>(); }

template <typename... T> struct json_variant_ : public json_object_base<json_variant_<T...>> {
  json_variant_() = default;
  json_variant_(const T&... s) : elements(s...) {}
  std::tuple<T...> elements;
};

template <typename... S> auto json_variant(S&&... s) { return json_variant_<S...>(s...); }

template <typename C, typename M> decltype(auto) json_decode(C& input, M& obj) {
  return impl::to_json_schema(obj).decode(input, obj);
}

template <typename C, typename M> decltype(auto) json_encode(C& output, const M& obj) {
  impl::to_json_schema(obj).encode(output, obj);
}

template <typename M> auto json_encode(M&& obj) {
  return impl::to_json_schema(obj).encode(std::forward<M>(obj));
}

template <typename C, typename F> decltype(auto) json_encode_generator(C& output, int N, F generator) {
  auto elt = impl::to_json_schema(decltype(generator()){});
  json_vector_<decltype(elt)>(elt).encode_generator(output, N, generator);
}
template <typename F> decltype(auto) json_encode_generator(int N, F generator) {
  auto elt = impl::to_json_schema(decltype(generator()){});
  return json_vector_<decltype(elt)>(elt).encode_generator(N, generator);
}

template <typename A, typename B, typename... C>
auto json_encode(const assign_exp<A, B>& exp, C... c) {
  auto obj = mmm(exp, c...);
  return impl::to_json_schema(obj).encode(obj);
}

} // namespace li

#endif // LITHIUM_SINGLE_HEADER_GUARD_LI_JSON_JSON_HH

