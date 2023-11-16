#pragma once

#include <li/json/symbols.hh>
#include <li/json/unicode.hh>
#include <li/json/utils.hh>
#include <li/metamap/metamap.hh>
#include <li/symbol/symbol.hh>
#include <string_view>
#include <tuple>
#include <variant>

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
