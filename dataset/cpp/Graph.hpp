#pragma once

#ifndef RAZ_GRAPH_HPP
#define RAZ_GRAPH_HPP

#include <memory>
#include <vector>

namespace Raz {

template <typename NodeT>
class Graph;

/// GraphNode class, representing a base node in a Graph. Must be inherited to be used by a Graph.
/// \tparam T Graph node's specific type. Must be the derived class itself.
template <typename T>
class GraphNode {
  friend Graph<T>;

public:
  GraphNode(const GraphNode&) = delete;
  GraphNode(GraphNode&&) noexcept = default;

  const std::vector<T*>& getParents() const noexcept { return m_parents; }
  std::size_t getParentCount() const noexcept { return m_parents.size(); }
  const T& getParent(std::size_t index) const noexcept;
  T& getParent(std::size_t index) noexcept;
  const std::vector<T*>& getChildren() const noexcept { return m_children; }
  std::size_t getChildCount() const noexcept { return m_children.size(); }
  const T& getChild(std::size_t index) const noexcept;
  T& getChild(std::size_t index) noexcept;
  /// Checks if the current node is a root, that is, a node without any parent.
  /// \return True if it is a root node, false otherwise.
  bool isRoot() const noexcept { return m_parents.empty(); }
  /// Checks if the current node is a leaf, that is, a node without any child.
  /// \return True if it is a leaf node, false otherwise.
  bool isLeaf() const noexcept { return m_children.empty(); }
  /// Checks if the current node is isolated, that is, a node which is both a root & a leaf (without any parent or child).
  /// \return True if it is an isolated node, false otherwise.
  bool isIsolated() const noexcept { return isRoot() && isLeaf(); }

  /// Links the given nodes as parents of the current one.
  /// \tparam OtherNodesTs Types of the other nodes to link.
  /// \param node First node to be linked.
  /// \param otherNodes Other nodes to be linked.
  template<typename... OtherNodesTs>
  void addParents(GraphNode& node, OtherNodesTs&&... otherNodes);
  /// Unlinks the given nodes as parents of the current one; this also removes the current one from the nodes' children.
  /// \tparam OtherNodesTs Types of the other nodes to unlink.
  /// \param node First node to be unlinked.
  /// \param otherNodes Other nodes to be unlinked.
  template<typename... OtherNodesTs>
  void removeParents(GraphNode& node, OtherNodesTs&&... otherNodes);
  /// Links the given nodes as children of the current one.
  /// \tparam OtherNodesTs Types of the other nodes to link.
  /// \param node First node to be linked.
  /// \param otherNodes Other nodes to be linked.
  template<typename... OtherNodesTs>
  void addChildren(GraphNode& node, OtherNodesTs&&... otherNodes);
  /// Unlinks the given nodes as children of the current one; this also removes the current one from the nodes' parents.
  /// \tparam OtherNodesTs Types of the other nodes to unlink.
  /// \param node First node to be unlinked.
  /// \param otherNodes Other nodes to be unlinked.
  template<typename... OtherNodesTs>
  void removeChildren(GraphNode& node, OtherNodesTs&&... otherNodes);

  GraphNode& operator=(const GraphNode&) = delete;
  GraphNode& operator=(GraphNode&&) noexcept = default;

  virtual ~GraphNode() = default;

protected:
  GraphNode() = default;

  /// Unlinks the given node only as a parent of the current one; the given node's children are left untouched.
  /// \param node Node to be unlinked.
  void unlinkParent(const GraphNode& node);
  /// Unlinks the given node only as a child of the current one; the given node's parents are left untouched.
  /// \param node Node to be unlinked.
  void unlinkChild(const GraphNode& node);

  std::vector<T*> m_parents {};
  std::vector<T*> m_children {};
};

/// Graph class, representing a [directed graph](https://en.wikipedia.org/wiki/Directed_graph).
/// This class is implemented as an [adjacency list](https://en.wikipedia.org/wiki/Adjacency_list).
/// \tparam NodeT Type of the graph's nodes. Must be a type derived from GraphNode.
template <typename NodeT>
class Graph {
  static_assert(std::is_base_of_v<GraphNode<NodeT>, NodeT>, "Error: The graph node type must be derived from GraphNode.");

  using NodePtr = std::unique_ptr<NodeT>;

public:
  /// Default constructor.
  Graph() = default;
  /// Creates a Graph while preallocating the given amount of nodes.
  /// \param nodeCount Amount of nodes to reserve.
  explicit Graph(std::size_t nodeCount) { m_nodes.reserve(nodeCount); }
  Graph(const Graph&) = delete;
  Graph(Graph&&) noexcept = default;

  std::size_t getNodeCount() const noexcept { return m_nodes.size(); }
  const NodeT& getNode(std::size_t index) const noexcept;
  NodeT& getNode(std::size_t index) noexcept;

  /// Adds a node into the graph.
  /// \tparam Args Types of the arguments to be forwarded to the node's constructor.
  /// \param args Arguments to be forwarded to the node's constructor.
  /// \return Reference to the newly added node.
  template <typename... Args>
  NodeT& addNode(Args&&... args);
  /// Removes a node from the graph, after unlinking it from all its parents & children.
  /// \param node Node to be removed.
  void removeNode(NodeT& node);

  Graph& operator=(const Graph&) = delete;
  Graph& operator=(Graph&&) noexcept = default;

protected:
  std::vector<NodePtr> m_nodes {};
};

} // namespace Raz

#include "RaZ/Data/Graph.inl"

#endif // RAZ_GRAPH_HPP
