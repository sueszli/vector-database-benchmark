#pragma once

#include <memory>
#include <string>
#include <vector>

#include "abstract_rule.hpp"

namespace hyrise {

class AbstractLQPNode;
class PredicateNode;

/**
 * This optimizer rule finds chains of adjacent PredicateNodes and sorts them based on the expected cardinality.
 * By that predicates with a low selectivity are executed first to (hopefully) reduce the size of intermediate results.
 *
 * Note:
 * For now this rule only finds adjacent PredicateNodes, meaning that if there is another node, e.g. a ProjectionNode,
 * between two
 * chains of PredicateNodes we won't order all of them, but only each chain separately.
 * A potential optimization would be to ignore certain intermediate nodes, such as ProjectionNode or SortNode, but
 * respect
 * others, such as JoinNode or UnionNode.
 */
class PredicateReorderingRule : public AbstractRule {
 public:
  std::string name() const override;

 protected:
  void _apply_to_plan_without_subqueries(const std::shared_ptr<AbstractLQPNode>& lqp_root) const override;

 private:
  void _reorder_predicates(const std::vector<std::shared_ptr<AbstractLQPNode>>& predicates) const;
};

}  // namespace hyrise
