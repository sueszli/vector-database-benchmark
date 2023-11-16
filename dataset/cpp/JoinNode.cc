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
#include <eventql/sql/qtree/JoinNode.h>
#include <eventql/sql/qtree/ColumnReferenceNode.h>
#include <eventql/sql/qtree/QueryTreeUtil.h>

namespace csql {

JoinNode::JoinNode(
    JoinType join_type,
    RefPtr<QueryTreeNode> base_table,
    RefPtr<QueryTreeNode> joined_table) :
    join_type_(join_type),
    base_table_(base_table),
    joined_table_(joined_table) {
  addChild(&base_table_);
  addChild(&joined_table_);
}

JoinNode::JoinNode(
    JoinType join_type,
    RefPtr<QueryTreeNode> base_table,
    RefPtr<QueryTreeNode> joined_table,
    Vector<RefPtr<SelectListNode>> select_list,
    Option<RefPtr<ValueExpressionNode>> where_expr,
    Option<RefPtr<ValueExpressionNode>> join_cond) :
    join_type_(join_type),
    base_table_(base_table),
    joined_table_(joined_table),
    select_list_(select_list),
    where_expr_(where_expr),
    join_cond_(join_cond) {
  for (const auto& sl : select_list_) {
    column_names_.emplace_back(sl->columnName());
  }

  addChild(&base_table_);
  addChild(&joined_table_);
}

JoinNode::JoinNode(
    const JoinNode& other) :
    join_type_(other.join_type_),
    base_table_(other.base_table_->deepCopy()),
    joined_table_(other.joined_table_->deepCopy()),
    input_map_(other.input_map_),
    column_names_(other.column_names_),
    join_cond_(other.join_cond_) {
  for (const auto& e : other.select_list_) {
    select_list_.emplace_back(e->deepCopyAs<SelectListNode>());
  }

  if (!other.where_expr_.isEmpty()) {
    where_expr_ = Some(
        other.where_expr_.get()->deepCopyAs<ValueExpressionNode>());
  }

  addChild(&base_table_);
  addChild(&joined_table_);
}

JoinType JoinNode::joinType() const {
  return join_type_;
}

void JoinNode::setJoinType(JoinType type) {
  join_type_ = type;
}

RefPtr<QueryTreeNode> JoinNode::baseTable() const {
  return base_table_;
}

RefPtr<QueryTreeNode> JoinNode::joinedTable() const {
  return joined_table_;
}

Vector<RefPtr<SelectListNode>> JoinNode::selectList() const {
  return select_list_;
}

void JoinNode::addSelectList(RefPtr<SelectListNode> sl) {
  column_names_.emplace_back(sl->columnName());
  select_list_.emplace_back(sl);
}

Vector<String> JoinNode::getResultColumns() const {
  return column_names_;
}

Vector<QualifiedColumn> JoinNode::getAvailableColumns() const {
  Vector<QualifiedColumn> cols;

  for (const auto& c :
      base_table_.asInstanceOf<TableExpressionNode>()->getAvailableColumns()) {
    cols.emplace_back(c);
  }

  for (const auto& c :
      joined_table_.asInstanceOf<TableExpressionNode>()->getAvailableColumns()) {
    cols.emplace_back(c);
  }

  return cols;
}

size_t JoinNode::getComputedColumnIndex(
    const String& column_name,
    bool allow_add /* = false */) {
  for (size_t i = 0; i < column_names_.size(); ++i) {
    if (column_names_[i] == column_name) {
      return i;
    }
  }

  auto input_idx = getInputColumnIndex(column_name, allow_add);
  if (input_idx != size_t(-1)) {
    auto slnode = new SelectListNode(
        new ColumnReferenceNode(input_idx, getInputColumnType(input_idx)));
    slnode->setAlias(column_name);
    select_list_.emplace_back(slnode);
    return select_list_.size() - 1;
  }

  return -1;
}

size_t JoinNode::getNumComputedColumns() const {
  return select_list_.size();
}

SType JoinNode::getColumnType(size_t idx) const {
  assert(idx < select_list_.size());
  return select_list_[idx]->expression()->getReturnType();
}

size_t JoinNode::getInputColumnIndex(
    const String& column_name,
    bool allow_add /* = false */) {
  for (size_t i = 0; i < input_map_.size(); ++i) {
    if (input_map_[i].column == column_name) {
      return i;
    }
  }

  auto base_table_idx = base_table_
      .asInstanceOf<TableExpressionNode>()
      ->getComputedColumnIndex(column_name, allow_add);

  auto joined_table_idx = joined_table_
      .asInstanceOf<TableExpressionNode>()
      ->getComputedColumnIndex(column_name, allow_add);

  if (base_table_idx != size_t(-1) && joined_table_idx != size_t(-1)) {
    RAISEF(
        kRuntimeError,
        "ambiguous column reference: '$0'",
        column_name);
  }

  if (base_table_idx != size_t(-1)) {
    input_map_.emplace_back(InputColumnRef{
      .column = column_name,
      .table_idx = 0,
      .column_idx = base_table_idx,
      .type = base_table_.asInstanceOf<TableExpressionNode>()->getColumnType(base_table_idx)
    });

    return input_map_.size() - 1;
  }

  if (joined_table_idx != size_t(-1)) {
    input_map_.emplace_back(InputColumnRef{
      .column = column_name,
      .table_idx = 1,
      .column_idx = joined_table_idx,
      .type = joined_table_.asInstanceOf<TableExpressionNode>()->getColumnType(joined_table_idx)
    });

    return input_map_.size() - 1;
  }

  return -1;
}

SType JoinNode::getInputColumnType(size_t idx) const {
  if (idx >= input_map_.size()) {
    RAISEF(
        kRuntimeError,
        "invalid column index: '$0'",
        idx);
  }

  switch (input_map_[idx].table_idx) {
    case 0:
      return base_table_.asInstanceOf<TableExpressionNode>()->getColumnType(
          input_map_[idx].column_idx);
    case 1:
      return joined_table_.asInstanceOf<TableExpressionNode>()->getColumnType(
          input_map_[idx].column_idx);
    default:
      return SType::NIL;
  }
}

std::pair<size_t, SType> JoinNode::getInputColumnInfo(
    const String& column_name,
    bool allow_add) {
  auto idx = getInputColumnIndex(column_name, allow_add);
  if (idx == size_t(-1)) {
    return std::make_pair(idx, SType::NIL);
  } else {
    return std::make_pair(idx, getInputColumnType(idx));
  }
}


Option<RefPtr<ValueExpressionNode>> JoinNode::whereExpression() const {
  return where_expr_;
}

void JoinNode::setWhereExpression(RefPtr<ValueExpressionNode> expr) {
  where_expr_ = Some(expr);
}

Option<RefPtr<ValueExpressionNode>> JoinNode::joinCondition() const {
  return join_cond_;
}

void JoinNode::setJoinCondition(RefPtr<ValueExpressionNode> expr) {
  join_cond_ = Some(expr);
}

RefPtr<QueryTreeNode> JoinNode::deepCopy() const {
  return new JoinNode(*this);
}

const Vector<JoinNode::InputColumnRef>& JoinNode::inputColumnMap() const {
  return input_map_;
}

String JoinNode::toString() const {
  auto str = StringUtil::format(
      "(join (base-table $0) (joined-table $0) (select-list",
      base_table_->toString(),
      joined_table_->toString());

  for (const auto& e : select_list_) {
    str += " " + e->toString();
  }
  str += ")";

  if (!where_expr_.isEmpty()) {
    str += StringUtil::format(" (where $0)", where_expr_.get()->toString());
  }

  if (!join_cond_.isEmpty()) {
    str += StringUtil::format(" (join-cond $0)", join_cond_.get()->toString());
  }

  str += ")";
  return str;
};

void JoinNode::encode(
    QueryTreeCoder* coder,
    const JoinNode& node,
    OutputStream* os) {
  os->appendUInt8((uint8_t) node.join_type_);

  os->appendVarUInt(node.select_list_.size());
  for (const auto& e : node.select_list_) {
    coder->encode(e.get(), os);
  }

  uint8_t flags = 0;
  if (!node.where_expr_.isEmpty()) {
    flags |= kHasWhereExprFlag;
  }
  if (!node.join_cond_.isEmpty()) {
    flags |= kHasJoinExprFlag;
  }

  os->appendUInt8(flags);

  if (!node.where_expr_.isEmpty()) {
    coder->encode(node.where_expr_.get().get(), os);
  }

  if (!node.join_cond_.isEmpty()) {
    coder->encode(node.join_cond_.get().get(), os);
  }

  coder->encode(node.base_table_.get(), os);
  coder->encode(node.joined_table_.get(), os);
}

RefPtr<QueryTreeNode> JoinNode::decode (
    QueryTreeCoder* coder,
    InputStream* is) {
  auto join_type = (JoinType) is->readUInt8();

  Vector<RefPtr<SelectListNode>> select_list;
  auto select_list_size = is->readVarUInt();
  for (size_t i = 0; i < select_list_size; ++i) {
    select_list.emplace_back(coder->decode(is).asInstanceOf<SelectListNode>());
  }

  Option<RefPtr<ValueExpressionNode>> where_expr;
  Option<RefPtr<ValueExpressionNode>> join_cond;
  auto flags = is->readUInt8();

  if ((flags & kHasWhereExprFlag) > 0) {
    where_expr = coder->decode(is).asInstanceOf<ValueExpressionNode>();
  }

  if ((flags & kHasJoinExprFlag) > 0) {
    join_cond = coder->decode(is).asInstanceOf<ValueExpressionNode>();
  }

  auto base_tbl = coder->decode(is);
  auto joined_tbl = coder->decode(is);

  return new JoinNode(
      join_type,
      base_tbl,
      joined_tbl,
      select_list,
      where_expr,
      join_cond);
}

} // namespace csql

