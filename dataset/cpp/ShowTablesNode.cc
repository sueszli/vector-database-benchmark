/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *   - Laura Schlimmer <laura@eventql.io>
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
#include "eventql/eventql.h"
#include <eventql/sql/qtree/ShowTablesNode.h>
#include <eventql/sql/statements/show_tables.h>

namespace csql {

Vector<RefPtr<QueryTreeNode>> ShowTablesNode::inputTables() const {
  return Vector<RefPtr<QueryTreeNode>>{};
}

RefPtr<QueryTreeNode> ShowTablesNode::deepCopy() const {
  return new ShowTablesNode();
}

Vector<String> ShowTablesNode::getResultColumns() const {
  std::vector<std::string> list;
  for (const auto& e : ShowTablesExpression::kOutputColumns) {
    list.emplace_back(e.first);
  }
  return list;
}

Vector<QualifiedColumn> ShowTablesNode::getAvailableColumns() const {
  Vector<QualifiedColumn> cols;

  for (const auto& e : ShowTablesExpression::kOutputColumns) {
    cols.emplace_back(e.first, e.first, e.second);
  }

  return cols;
}

size_t ShowTablesNode::getComputedColumnIndex(
    const String& column_name,
    bool allow_add /* = false */) {
  return -1; // FIXME
}

size_t ShowTablesNode::getNumComputedColumns() const {
  return ShowTablesExpression::kOutputColumns.size();
}

SType ShowTablesNode::getColumnType(size_t idx) const {
  assert(idx < ShowTablesExpression::kOutputColumns.size());
  return ShowTablesExpression::kOutputColumns[idx].second;
}

String ShowTablesNode::toString() const {
  return "(show-tables)";
}

void ShowTablesNode::encode(
    QueryTreeCoder* coder,
    const ShowTablesNode& node,
    OutputStream* os) {}

RefPtr<QueryTreeNode> ShowTablesNode::decode (
    QueryTreeCoder* coder,
    InputStream* is) {
  return new ShowTablesNode();
}


} // namespace csql
