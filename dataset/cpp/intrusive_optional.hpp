#pragma once

#include <optional>
#include <type_traits>
#include <utility>

#include "appfat.h"

/// An optional that uses a field of the stored class and some value to store nullopt.
#define DEFINE_INTRUSIVE_OPTIONAL_IMPL(OPTIONAL_CLASS, VALUE_CLASS, FIELD, NULL_VALUE, CONSTEXPR) \
public:                                                                                           \
	CONSTEXPR OPTIONAL_CLASS() = default;                                                         \
                                                                                                  \
	template <class U = VALUE_CLASS>                                                              \
	CONSTEXPR OPTIONAL_CLASS(VALUE_CLASS &&value)                                                 \
	    : value_(std::forward<U>(value))                                                          \
	{                                                                                             \
	}                                                                                             \
                                                                                                  \
	CONSTEXPR OPTIONAL_CLASS(std::nullopt_t)                                                      \
	    : OPTIONAL_CLASS()                                                                        \
	{                                                                                             \
	}                                                                                             \
                                                                                                  \
	template <typename... Args>                                                                   \
	CONSTEXPR VALUE_CLASS &emplace(Args &&...args)                                                \
	{                                                                                             \
		value_ = VALUE_CLASS(std::forward<Args>(args)...);                                        \
		return value_;                                                                            \
	}                                                                                             \
                                                                                                  \
	template <class U = VALUE_CLASS>                                                              \
	CONSTEXPR OPTIONAL_CLASS &operator=(VALUE_CLASS &&value)                                      \
	{                                                                                             \
		value_ = std::forward<U>(value);                                                          \
		return *this;                                                                             \
	}                                                                                             \
                                                                                                  \
	CONSTEXPR OPTIONAL_CLASS &operator=(std::nullopt_t)                                           \
	{                                                                                             \
		value_ = VALUE_CLASS {};                                                                  \
		return *this;                                                                             \
	}                                                                                             \
                                                                                                  \
	CONSTEXPR const VALUE_CLASS &operator*() const                                                \
	{                                                                                             \
		assert(value_.FIELD != NULL_VALUE);                                                       \
		return value_;                                                                            \
	}                                                                                             \
                                                                                                  \
	CONSTEXPR VALUE_CLASS &operator*()                                                            \
	{                                                                                             \
		assert(value_.FIELD != NULL_VALUE);                                                       \
		return value_;                                                                            \
	}                                                                                             \
                                                                                                  \
	CONSTEXPR const VALUE_CLASS *operator->() const                                               \
	{                                                                                             \
		assert(value_.FIELD != NULL_VALUE);                                                       \
		return &value_;                                                                           \
	}                                                                                             \
                                                                                                  \
	CONSTEXPR VALUE_CLASS *operator->()                                                           \
	{                                                                                             \
		assert(value_.FIELD != NULL_VALUE);                                                       \
		return &value_;                                                                           \
	}                                                                                             \
                                                                                                  \
	CONSTEXPR operator bool() const                                                               \
	{                                                                                             \
		return value_.FIELD != NULL_VALUE;                                                        \
	}                                                                                             \
                                                                                                  \
private:                                                                                          \
	VALUE_CLASS value_;

#define DEFINE_CONSTEXPR_INTRUSIVE_OPTIONAL(OPTIONAL_CLASS, VALUE_CLASS, FIELD, NULL_VALUE) \
	DEFINE_INTRUSIVE_OPTIONAL_IMPL(OPTIONAL_CLASS, VALUE_CLASS, FIELD, NULL_VALUE, constexpr)

#define DEFINE_INTRUSIVE_OPTIONAL(OPTIONAL_CLASS, VALUE_CLASS, FIELD, NULL_VALUE) \
	DEFINE_INTRUSIVE_OPTIONAL_IMPL(OPTIONAL_CLASS, VALUE_CLASS, FIELD, NULL_VALUE, )
