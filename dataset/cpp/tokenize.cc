/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License ("the license") as
 * published by the Free Software Foundation, either version 3 of the License,
 * or any later version.
 *
 * In accordance with Section 7(e) of the license, the licensing of the Program
 * under the license does not imply a trademark license. Therefore any rights,
 * title and interest in our trademarks remain entirely with us.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the license for more details.
 *
 * You can be released from the requirements of the license by purchasing a
 * commercial license. Buying such a license is mandatory as soon as you develop
 * commercial activities involving this program without disclosing the source
 * code of your own applications
 */

#include <stdlib.h>
#include <assert.h>
#include "tokenize.h"

namespace csql {

void tokenizeQuery(
    const char** cur,
    const char* end,
    std::vector<Token>* token_list) {
  char quote_char = 0;
  Token::kTokenType string_type = Token::T_STRING;

next:

  /* skip whitespace */
  while ((
      **cur == ' ' ||
      **cur == '\t' ||
      **cur == '\n' ||
      **cur == '\r')
      && *cur < end) {
    (*cur)++;
  }

  if (*cur >= end) {
    return;
  }

  /* single character tokens */
  switch (**cur) {
    case ';': {
      token_list->emplace_back(Token::T_SEMICOLON);
      (*cur)++;
      goto next;
    }

    case ',': {
      token_list->emplace_back(Token::T_COMMA);
      (*cur)++;
      goto next;
    }

    case '.': {
      token_list->emplace_back(Token::T_DOT);
      (*cur)++;
      goto next;
    }

    case '(': {
      token_list->emplace_back(Token::T_LPAREN);
      (*cur)++;
      goto next;
    }

    case ')': {
      token_list->emplace_back(Token::T_RPAREN);
      (*cur)++;
      goto next;
    }

    /* numeric literals */
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': {
      const char* begin = *cur;
      for (; (**cur >= '0' && **cur <= '9') || **cur == '.'; (*cur)++);
      token_list->emplace_back(Token::T_NUMERIC, begin, *cur - begin);
      goto next;
    }

    case '`':
      string_type = Token::T_IDENTIFIER;
      /* fallthrough */

    case '"':
    case '\'':
      quote_char = **cur;
      (*cur)++;
      break;

    default:
      break;
  }

  /* multi char tokens */
  const char* begin = *cur;
  size_t len = 0;

  /* quoted strings */
  if (quote_char) {
    std::string str;

    bool escaped = false;
    bool eof = false;
    for (; !eof && *cur < end; (*cur)++) {
      auto chr = **cur;
      switch (chr) {

        case '"':
        case '\'':
        case '`':
          if (escaped || quote_char != chr) {
            str += chr;
            break;
          } else {
            eof = true;
            continue;
          }

        case '\\':
          if (escaped) {
            str += "\\";
            break;
          } else {
            escaped = true;
            continue;
          }

        default:
          str += chr;
          break;

      }

      escaped = false;
    }

    token_list->emplace_back(string_type, str);
    quote_char = 0;
    string_type = Token::T_STRING;
    goto next;
  }

  /* operators */
  if (**cur == '=') {
    token_list->emplace_back(Token::T_EQUAL);
    (*cur)++;
    goto next;
  }

  if (**cur == '+') {
    token_list->emplace_back(Token::T_PLUS);
    (*cur)++;
    goto next;
  }

  if (**cur == '-') {
    if (*cur + 1 < end && (*cur)[1] == '-') {
      for (; *cur < end && **cur != '\n'; (*cur)++);
      goto next;
    }

    token_list->emplace_back(Token::T_MINUS);
    (*cur)++;
    goto next;
  }

  if (**cur == '*') {
    token_list->emplace_back(Token::T_ASTERISK);
    (*cur)++;
    goto next;
  }

  if (**cur == '!') {
    if (*cur + 1 < end && (*cur)[1] == '=') {
      token_list->emplace_back(Token::T_NEQUAL);
      (*cur) += 2;
      goto next;
    }

    token_list->emplace_back(Token::T_BANG);
    (*cur)++;
    goto next;
  }

  if (**cur == '/') {
    token_list->emplace_back(Token::T_SLASH);
    (*cur)++;
    goto next;
  }

  if (**cur == '^') {
    token_list->emplace_back(Token::T_CIRCUMFLEX);
    (*cur)++;
    goto next;
  }

  if (**cur == '~') {
    token_list->emplace_back(Token::T_TILDE);
    (*cur)++;
    goto next;
  }

  if (**cur == '%') {
    token_list->emplace_back(Token::T_PERCENT);
    (*cur)++;
    goto next;
  }

  if (**cur == '&') {
    token_list->emplace_back(Token::T_AMPERSAND);
    (*cur)++;
    goto next;
  }

  if (**cur == '|') {
    token_list->emplace_back(Token::T_PIPE);
    (*cur)++;
    goto next;
  }

  if (**cur == '<') {
    if (*cur + 1 < end && (*cur)[1] == '=') {
      token_list->emplace_back(Token::T_LTE);
      (*cur) += 2;
      goto next;
    }

    token_list->emplace_back(Token::T_LT);
    (*cur)++;
    goto next;
  }

  if (**cur == '>') {
    if (*cur + 1 < end && (*cur)[1] == '=') {
      token_list->emplace_back(Token::T_GTE);
      (*cur) += 2;
      goto next;
    }

    token_list->emplace_back(Token::T_GT);
    (*cur)++;
    goto next;
  }

  /* identifiers */
  while (
      **cur != ' ' &&
      **cur != '\t' &&
      **cur != '\n' &&
      **cur != '\r' &&
      **cur != ',' &&
      **cur != '.' &&
      **cur != ';' &&
      **cur != '(' &&
      **cur != ')' &&
      **cur != '"' &&
      **cur != '\'' &&
      **cur != '`' &&
      **cur != '=' &&
      **cur != '+' &&
      **cur != '-' &&
      **cur != '*' &&
      **cur != '!' &&
      **cur != '/' &&
      **cur != '^' &&
      **cur != '~' &&
      **cur != '%' &&
      **cur != '&' &&
      **cur != '|' &&
      **cur != '<' &&
      **cur != '>' &&
      *cur < end) {
    len++;
    (*cur)++;

    if (len > 2) {
      continue;
    }

    Token token(Token::T_IDENTIFIER, begin, len);

  }

  // FIXPAUL this should be a hashmap/trie lookup!
  Token token(Token::T_IDENTIFIER, begin, len);

  if (token == "AS") {
    token_list->emplace_back(Token::T_AS);
    goto next;
  }

  if (token == "ASC") {
    token_list->emplace_back(Token::T_ASC);
    goto next;
  }

  if (token == "DESC") {
    token_list->emplace_back(Token::T_DESC);
    goto next;
  }

  if (token == "NOT") {
    token_list->emplace_back(Token::T_NOT);
    goto next;
  }

  if (token == "NULL") {
    token_list->emplace_back(Token::T_NULL);
    goto next;
  }

  if (token == "TRUE") {
    token_list->emplace_back(Token::T_TRUE);
    goto next;
  }

  if (token == "FALSE") {
    token_list->emplace_back(Token::T_FALSE);
    goto next;
  }

  if (token == "SELECT") {
    token_list->emplace_back(Token::T_SELECT);
    goto next;
  }

  if (token == "FROM") {
    token_list->emplace_back(Token::T_FROM);
    goto next;
  }

  if (token == "WHERE") {
    token_list->emplace_back(Token::T_WHERE);
    goto next;
  }

  if (token == "GROUP") {
    token_list->emplace_back(Token::T_GROUP);
    goto next;
  }

  if (token == "ORDER") {
    token_list->emplace_back(Token::T_ORDER);
    goto next;
  }

  if (token == "BY") {
    token_list->emplace_back(Token::T_BY);
    goto next;
  }

  if (token == "HAVING") {
    token_list->emplace_back(Token::T_HAVING);
    goto next;
  }

  if (token == "AND") {
    token_list->emplace_back(Token::T_AND);
    goto next;
  }

  if (token == "OR") {
    token_list->emplace_back(Token::T_OR);
    goto next;
  }

  if (token == "LIMIT") {
    token_list->emplace_back(Token::T_LIMIT);
    goto next;
  }

  if (token == "OFFSET") {
    token_list->emplace_back(Token::T_OFFSET);
    goto next;
  }

  if (token == "CREATE") {
    token_list->emplace_back(Token::T_CREATE);
    goto next;
  }

  if (token == "WITH") {
    token_list->emplace_back(Token::T_WITH);
    goto next;
  }

  if (token == "LIKE") {
    token_list->emplace_back(Token::T_LIKE);
    goto next;
  }

  if (token == "REGEX" || token == "REGEXP") {
    token_list->emplace_back(Token::T_REGEX);
    goto next;
  }

  if (token == "BEGIN") {
    token_list->emplace_back(Token::T_BEGIN);
    goto next;
  }

  if (token == "WITHIN") {
    token_list->emplace_back(Token::T_WITHIN);
    goto next;
  }

  if (token == "RECORD") {
    token_list->emplace_back(Token::T_RECORD);
    goto next;
  }

  if (token == "MOD") {
    token_list->emplace_back(Token::T_MOD);
    goto next;
  }

  if (token == "DRAW") {
    token_list->emplace_back(Token::T_DRAW);
    goto next;
  }

  if (token == "TOP") {
    token_list->emplace_back(Token::T_TOP);
    goto next;
  }

  if (token == "RIGHT") {
    token_list->emplace_back(Token::T_RIGHT);
    goto next;
  }

  if (token == "BOTTOM") {
    token_list->emplace_back(Token::T_BOTTOM);
    goto next;
  }

  if (token == "LEFT") {
    token_list->emplace_back(Token::T_LEFT);
    goto next;
  }

  if (token == "IMPORT") {
    token_list->emplace_back(Token::T_IMPORT);
    goto next;
  }

  if (token == "TABLE") {
    token_list->emplace_back(Token::T_TABLE);
    goto next;
  }

  if (token == "TABLES") {
    token_list->emplace_back(Token::T_TABLES);
    goto next;
  }

  if (token == "DATABASE") {
    token_list->emplace_back(Token::T_DATABASE);
    goto next;
  }

  if (token == "USE") {
    token_list->emplace_back(Token::T_USE);
    goto next;
  }

  if (token == "AXIS") {
    token_list->emplace_back(Token::T_AXIS);
    goto next;
  }

  if (token == "BARCHART") {
    token_list->emplace_back(Token::T_BARCHART);
    goto next;
  }

  if (token == "LINECHART") {
    token_list->emplace_back(Token::T_LINECHART);
    goto next;
  }

  if (token == "AREACHART") {
    token_list->emplace_back(Token::T_AREACHART);
    goto next;
  }

  if (token == "POINTCHART") {
    token_list->emplace_back(Token::T_POINTCHART);
    goto next;
  }

  if (token == "HEATMAP") {
    token_list->emplace_back(Token::T_HEATMAP);
    goto next;
  }

  if (token == "HISTOGRAM") {
    token_list->emplace_back(Token::T_HISTOGRAM);
    goto next;
  }

  if (token == "ORIENTATION") {
    token_list->emplace_back(Token::T_ORIENTATION);
    goto next;
  }

  if (token == "HORIZONTAL") {
    token_list->emplace_back(Token::T_HORIZONTAL);
    goto next;
  }

  if (token == "VERTICAL") {
    token_list->emplace_back(Token::T_VERTICAL);
    goto next;
  }

  if (token == "STACKED") {
    token_list->emplace_back(Token::T_STACKED);
    goto next;
  }

  if (token == "ON") {
    token_list->emplace_back(Token::T_ON);
    goto next;
  }

  if (token == "OFF") {
    token_list->emplace_back(Token::T_OFF);
    goto next;
  }

  if (token == "SHOW") {
    token_list->emplace_back(Token::T_SHOW);
    goto next;
  }

  if (token == "DESCRIBE") {
    token_list->emplace_back(Token::T_DESCRIBE);
    goto next;
  }

  if (token == "EXPLAIN") {
    token_list->emplace_back(Token::T_EXPLAIN);
    goto next;
  }

  if (token == "PARTITIONS") {
    token_list->emplace_back(Token::T_PARTITIONS);
    goto next;
  }

  if (token == "CLUSTER") {
    token_list->emplace_back(Token::T_CLUSTER);
    goto next;
  }

  if (token == "SERVERS") {
    token_list->emplace_back(Token::T_SERVERS);
    goto next;
  }

  if (token == "PRIMARY") {
    token_list->emplace_back(Token::T_PRIMARY);
    goto next;
  }

  if (token == "PARTITION") {
    token_list->emplace_back(Token::T_PARTITION);
    goto next;
  }

  if (token == "KEY") {
    token_list->emplace_back(Token::T_KEY);
    goto next;
  }

  if (token == "JOIN") {
    token_list->emplace_back(Token::T_JOIN);
    goto next;
  }

  if (token == "CROSS") {
    token_list->emplace_back(Token::T_CROSS);
    goto next;
  }

  if (token == "NATURAL") {
    token_list->emplace_back(Token::T_NATURAL);
    goto next;
  }

  if (token == "INNER") {
    token_list->emplace_back(Token::T_INNER);
    goto next;
  }

  if (token == "OUTER") {
    token_list->emplace_back(Token::T_OUTER);
    goto next;
  }

  if (token == "USING") {
    token_list->emplace_back(Token::T_USING);
    goto next;
  }

  if (token == "REPEATED") {
    token_list->emplace_back(Token::T_REPEATED);
    goto next;
  }

  if (token == "PRIMARY") {
    token_list->emplace_back(Token::T_PRIMARY);
    goto next;
  }

  if (token == "KEY") {
    token_list->emplace_back(Token::T_KEY);
    goto next;
  }

  if (token == "INSERT") {
    token_list->emplace_back(Token::T_INSERT);
    goto next;
  }

  if (token == "INTO") {
    token_list->emplace_back(Token::T_INTO);
    goto next;
  }

  if (token == "VALUES") {
    token_list->emplace_back(Token::T_VALUES);
    goto next;
  }

  if (token == "JSON") {
    token_list->emplace_back(Token::T_JSON);
    goto next;
  }

  if (token == "ALTER") {
    token_list->emplace_back(Token::T_ALTER);
    goto next;
  }

  if (token == "ADD") {
    token_list->emplace_back(Token::T_ADD);
    goto next;
  }

  if (token == "DROP") {
    token_list->emplace_back(Token::T_DROP);
    goto next;
  }

  if (token == "COLUMN") {
    token_list->emplace_back(Token::T_COLUMN);
    goto next;
  }

  if (token == "SET") {
    token_list->emplace_back(Token::T_SET);
    goto next;
  }

  if (token == "PROPERTY") {
    token_list->emplace_back(Token::T_PROPERTY);
    goto next;
  }

  if (token == "XDOMAIN") {
    token_list->emplace_back(Token::T_XDOMAIN);
    goto next;
  }

  if (token == "YDOMAIN") {
    token_list->emplace_back(Token::T_YDOMAIN);
    goto next;
  }

  if (token == "ZDOMAIN") {
    token_list->emplace_back(Token::T_ZDOMAIN);
    goto next;
  }

  if (token == "LOGARITHMIC") {
    token_list->emplace_back(Token::T_LOGARITHMIC);
    goto next;
  }

  if (token == "INVERT") {
    token_list->emplace_back(Token::T_INVERT);
    goto next;
  }

  if (token == "TITLE") {
    token_list->emplace_back(Token::T_TITLE);
    goto next;
  }

  if (token == "SUBTITLE") {
    token_list->emplace_back(Token::T_SUBTITLE);
    goto next;
  }

  if (token == "GRID") {
    token_list->emplace_back(Token::T_GRID);
    goto next;
  }

  if (token == "LABELS") {
    token_list->emplace_back(Token::T_LABELS);
    goto next;
  }

  if (token == "TICKS") {
    token_list->emplace_back(Token::T_TICKS);
    goto next;
  }

  if (token == "INSIDE") {
    token_list->emplace_back(Token::T_INSIDE);
    goto next;
  }

  if (token == "OUTSIDE") {
    token_list->emplace_back(Token::T_OUTSIDE);
    goto next;
  }

  if (token == "ROTATE") {
    token_list->emplace_back(Token::T_ROTATE);
    goto next;
  }

  if (token == "LEGEND") {
    token_list->emplace_back(Token::T_LEGEND);
    goto next;
  }

  if (token == "OVER") {
    token_list->emplace_back(Token::T_OVER);
    goto next;
  }

  if (token == "TIMEWINDOW") {
    token_list->emplace_back(Token::T_TIMEWINDOW);
    goto next;
  }

  if (token == "<<") {
    token_list->emplace_back(Token::T_LSHIFT);
    goto next;
  }

  if (token == ">>") {
    token_list->emplace_back(Token::T_RSHIFT);
    goto next;
  }

  token_list->push_back(token);
  goto next; // poor mans tail recursion optimization
}

void tokenizeQuery(
    const std::string& query,
    std::vector<Token>* token_list) {
  const char* str = query.c_str();
  tokenizeQuery(&str, str + query.size(), token_list);
}

}
