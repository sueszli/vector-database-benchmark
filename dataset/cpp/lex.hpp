#pragma once

#include <string_view>

#include <boost/variant.hpp>

#include <jank/result.hpp>
#include <jank/option.hpp>

namespace jank::read::lex
{
  enum class token_kind
  {
    open_paren,
    close_paren,
    open_square_bracket,
    close_square_bracket,
    open_curly_bracket,
    close_curly_bracket,
    single_quote,
    /* Has string data. */
    comment,
    nil,
    /* Has bool data. */
    boolean,
    /* Has string data. */
    symbol,
    /* Has string data. */
    keyword,
    /* Has int data. */
    integer,
    /* Has double data. */
    real,
    /* Has string data. */
    string,
    eof,
  };

  struct token
  {
    token(token_kind const k);
    token(size_t const p, token_kind const k);
    token(size_t const p, token_kind const k, native_integer const);
    token(size_t const p, token_kind const k, native_real const);
    token(size_t const p, token_kind const k, native_string_view const);
    token(size_t const p, token_kind const k, bool const);

    token(size_t const p, size_t const s, token_kind const k);
    token(size_t const p, size_t const s, token_kind const k, native_integer const);
    token(size_t const p, size_t const s, token_kind const k, native_real const);
    token(size_t const p, size_t const s, token_kind const k, native_string_view const);
    token(size_t const p, size_t const s, token_kind const k, bool const);

    bool operator ==(token const &rhs) const;
    bool operator !=(token const &rhs) const;

    struct no_data
    {
      bool operator ==(no_data const &rhs) const;
      bool operator !=(no_data const &rhs) const;
    };

    /* For some values, when comparing tokens, the position doesn't matter.
     * For example, EOF tokens. In these cases, this cardinal value is used. */
    static constexpr size_t ignore_pos{ std::numeric_limits<size_t>::max() };
    size_t pos{ ignore_pos };
    size_t size{ 1 };
    token_kind kind;
    boost::variant
    <
      no_data,
      native_integer,
      native_real,
      native_string_view,
      native_bool
    > data;
  };
  std::ostream& operator <<(std::ostream &os, token const &t);
  std::ostream& operator <<(std::ostream &os, token::no_data const &t);
}

namespace jank::read
{
  struct error
  {
    error(size_t const s, native_string const &m);
    error(size_t const s, size_t const e, native_string const &m);
    error(native_string const &m);

    bool operator ==(error const &rhs) const;
    bool operator !=(error const &rhs) const;

    size_t start{};
    size_t end{};
    native_string message;
  };
  std::ostream& operator <<(std::ostream &os, error const &e);
}

namespace jank::read::lex
{
  struct processor
  {
    struct iterator
    {
      using iterator_category = std::input_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = result<token, error>;
      using pointer = value_type*;
      using reference = value_type&;

      value_type const& operator *() const;
      value_type const* operator ->() const;
      iterator& operator ++();
      bool operator ==(iterator const &rhs) const;
      bool operator !=(iterator const &rhs) const;

      option<value_type> latest;
      processor &p;
    };

    processor(native_string_view const &f);

    result<token, error> next();
    option<char> peek() const;
    option<error> check_whitespace(bool const found_space);

    iterator begin();
    iterator end();

    size_t pos{};
    /* Whether or not the previous token requires a space after it. */
    bool require_space{};
    native_string_view file;
  };
}
