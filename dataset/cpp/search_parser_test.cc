// Copyright 2023, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "base/gtest.h"
#include "base/logging.h"
#include "core/search/base.h"
#include "core/search/query_driver.h"
#include "core/search/search.h"

namespace dfly::search {

using namespace std;

class SearchParserTest : public ::testing::Test {
 protected:
  SearchParserTest() {
    query_driver_.scanner()->set_debug(1);
  }

  void SetInput(const std::string& str) {
    query_driver_.SetInput(str);
  }

  Parser::symbol_type Lex() {
    return query_driver_.Lex();
  }

  int Parse(const std::string& str) {
    query_driver_.ResetScanner();
    query_driver_.SetInput(str);

    return Parser(&query_driver_)();
  }

  void SetParams(const QueryParams* params) {
    query_driver_.SetParams(params);
  }

  QueryDriver query_driver_;
};

// tokens are not assignable, so we can not reuse them. This macros reduce the boilerplate.
#define NEXT_EQ(tok_enum, type, val)                    \
  {                                                     \
    auto tok = Lex();                                   \
    ASSERT_EQ(tok.type_get(), Parser::token::tok_enum); \
    EXPECT_EQ(val, tok.value.as<type>());               \
  }

#define NEXT_TOK(tok_enum)                              \
  {                                                     \
    auto tok = Lex();                                   \
    ASSERT_EQ(tok.type_get(), Parser::token::tok_enum); \
  }
#define NEXT_ERROR()                          \
  {                                           \
    bool caught = false;                      \
    try {                                     \
      auto tok = Lex();                       \
    } catch (const Parser::syntax_error& e) { \
      caught = true;                          \
    }                                         \
    ASSERT_TRUE(caught);                      \
  }

TEST_F(SearchParserTest, Scanner) {
  SetInput("ab cd");
  // 3.5.1 does not have name() method.
  // EXPECT_STREQ("term", tok.name());

  NEXT_EQ(TOK_TERM, string, "ab");
  NEXT_EQ(TOK_TERM, string, "cd");
  NEXT_TOK(TOK_YYEOF);

  SetInput("(5a 6) ");

  NEXT_TOK(TOK_LPAREN);
  NEXT_EQ(TOK_TERM, string, "5a");
  NEXT_EQ(TOK_UINT32, uint32_t, 6);
  NEXT_TOK(TOK_RPAREN);

  SetInput(R"( "hello\"world" )");
  NEXT_EQ(TOK_TERM, string, R"(hello"world)");

  SetInput("@field:hello");
  NEXT_EQ(TOK_FIELD, string, "@field");
  NEXT_TOK(TOK_COLON);
  NEXT_EQ(TOK_TERM, string, "hello");

  SetInput("@field:{ tag }");
  NEXT_EQ(TOK_FIELD, string, "@field");
  NEXT_TOK(TOK_COLON);
  NEXT_TOK(TOK_LCURLBR);
  NEXT_EQ(TOK_TERM, string, "tag");
  NEXT_TOK(TOK_RCURLBR);

  SetInput("почтальон Печкин");
  NEXT_EQ(TOK_TERM, string, "почтальон");
  NEXT_EQ(TOK_TERM, string, "Печкин");

  double d;
  absl::SimpleAtod("33.3", &d);
  SetInput("33.3");
  NEXT_EQ(TOK_DOUBLE, double, d);
}

TEST_F(SearchParserTest, Parse) {
  EXPECT_EQ(0, Parse(" foo bar (baz) "));
  EXPECT_EQ(0, Parse(" -(foo) @foo:bar @ss:[1 2]"));
  EXPECT_EQ(0, Parse("@foo:{ tag1 | tag2 }"));

  EXPECT_EQ(1, Parse(" -(foo "));
  EXPECT_EQ(1, Parse(" foo:bar "));
  EXPECT_EQ(1, Parse(" @foo:@bar "));
  EXPECT_EQ(1, Parse(" @foo: "));
}

TEST_F(SearchParserTest, ParseParams) {
  QueryParams params;
  params["k"] = "10";
  params["name"] = "alex";
  SetParams(&params);

  SetInput("$name $k");
  NEXT_EQ(TOK_TERM, string, "alex");
  NEXT_EQ(TOK_UINT32, uint32_t, 10);
}

TEST_F(SearchParserTest, Quotes) {
  SetInput(" \"fir  st\"  'sec@o@nd' \":third:\" 'four\\\"th' ");
  NEXT_EQ(TOK_TERM, string, "fir  st");
  NEXT_EQ(TOK_TERM, string, "sec@o@nd");
  NEXT_EQ(TOK_TERM, string, ":third:");
  NEXT_EQ(TOK_TERM, string, "four\"th");
}

TEST_F(SearchParserTest, Numeric) {
  SetInput("11 123123123123 '22'");
  NEXT_EQ(TOK_UINT32, uint32_t, 11);
  NEXT_EQ(TOK_DOUBLE, double, 123123123123.0);
  NEXT_EQ(TOK_TERM, string, "22");
}

}  // namespace dfly::search
