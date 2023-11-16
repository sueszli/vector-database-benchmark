#include "QtGraphEdge.h"

#include <QCursor>
#include <QGraphicsItemGroup>
#include <QGraphicsSceneEvent>

#include "Edge.h"
#include "GraphFocusHandler.h"
#include "GraphViewStyle.h"
#include "MessageActivateEdge.h"
#include "MessageActivateTrailEdge.h"
#include "MessageFocusIn.h"
#include "MessageFocusOut.h"
#include "MessageGraphNodeBundleSplit.h"
#include "MessageGraphNodeHide.h"
#include "MessageTooltipHide.h"
#include "MessageTooltipShow.h"
#include "QtGraphNode.h"
#include "QtLineItemAngled.h"
#include "QtLineItemBezier.h"
#include "QtLineItemStraight.h"
#include "TokenComponentInheritanceChain.h"
#include "TokenComponentIsAmbiguous.h"
#include "utility.h"

QtGraphEdge* QtGraphEdge::s_focusedEdge = nullptr;
QtGraphEdge* QtGraphEdge::s_focusedBezierEdge = nullptr;

void QtGraphEdge::unfocusBezierEdge()
{
	if (s_focusedBezierEdge)
	{
		s_focusedBezierEdge->coFocusOut();
		s_focusedBezierEdge = nullptr;
	}
}

void QtGraphEdge::clearFocusedEdges()
{
	s_focusedEdge = nullptr;
	s_focusedBezierEdge = nullptr;
}

QtGraphEdge::QtGraphEdge(
	GraphFocusHandler* focusHandler,
	QtGraphNode* owner,
	QtGraphNode* target,
	const Edge* data,
	size_t weight,
	bool isActive,
	bool isInteractive,
	bool horizontal,
	TokenComponentBundledEdges::Direction direction)
	: m_focusHandler(focusHandler)
	, m_data(data)
	, m_owner(owner)
	, m_target(target)
	, m_isActive(isActive)
	, m_isHorizontal(horizontal)
	, m_weight(weight)
	, m_direction(direction)
	, m_isInteractive(isInteractive)
{
	this->setCursor(Qt::PointingHandCursor);

	if (m_direction == TokenComponentBundledEdges::DIRECTION_BACKWARD)
	{
		QtGraphNode* temp = m_owner;
		m_owner = m_target;
		m_target = temp;
	}
}

QtGraphEdge::~QtGraphEdge() {}

const Edge* QtGraphEdge::getData() const
{
	return m_data;
}

QtGraphNode* QtGraphEdge::getOwner()
{
	return m_owner;
}

QtGraphNode* QtGraphEdge::getTarget()
{
	return m_target;
}

Id QtGraphEdge::getTokenId() const
{
	if (getData())
	{
		return getData()->getId();
	}

	return 0;
}

void QtGraphEdge::updateLine()
{
	const QtGraphNode* owner = m_owner;
	const QtGraphNode* target = m_target;

	Edge::EdgeType type = (getData() ? getData()->getType() : Edge::EDGE_BUNDLED_EDGES);
	GraphViewStyle::EdgeStyle style = GraphViewStyle::getStyleForEdgeType(
		type, m_isActive | m_isCoFocused, m_isFocused, m_isTrailEdge, isAmbiguous());

	Vec4i ownerRect = owner->getBoundingRect();
	Vec4i targetRect = target->getBoundingRect();

	Vec4i ownerParentRect;
	Vec4i targetParentRect;

	const QtGraphNode* ownerParent = owner->getLastParent();
	const QtGraphNode* targetParent = target->getLastParent();

	const QtGraphNode* ownerNonGroupParent = owner->getLastNonGroupParent();
	const QtGraphNode* targetNonGroupParent = target->getLastNonGroupParent();

	if (ownerParent == targetParent && ownerParent->isGroupNode())
	{
		ownerParentRect = ownerNonGroupParent->getBoundingRect();
		targetParentRect = targetNonGroupParent->getBoundingRect();
	}
	else
	{
		ownerParentRect = ownerParent->getBoundingRect();
		targetParentRect = targetParent->getBoundingRect();
	}

	QtLineItemBase::Route route = m_isHorizontal ? QtLineItemBase::ROUTE_HORIZONTAL
												 : QtLineItemBase::ROUTE_VERTICAL;

	if (m_useBezier)
	{
		for (QGraphicsItem* item: childItems())
		{
			item->hide();
			item->setParentItem(nullptr);
		}

		style.originOffset.y() = 0;
		style.targetOffset.y() = 0;

		for (const Vec4i& rect: m_path)
		{
			QtLineItemBezier* bezier = new QtLineItemBezier(this);
			bezier->updateLine(ownerRect, rect, ownerParentRect, rect, style, m_weight, false);
			bezier->setRoute(route);

			QtLineItemStraight* line = new QtLineItemStraight(this);
			if (route == QtLineItemBase::ROUTE_HORIZONTAL)
			{
				line->updateLine(
					Vec2i(rect.x(), (rect.y() + rect.w()) / 2),
					Vec2i(rect.z(), (rect.y() + rect.w()) / 2),
					style);
			}
			else
			{
				line->updateLine(
					Vec2i((rect.x() + rect.z()) / 2, rect.y()),
					Vec2i((rect.x() + rect.z()) / 2, rect.w()),
					style);
			}

			ownerRect = rect;
			ownerParentRect = rect;
		}

		bool showArrow = m_direction != TokenComponentBundledEdges::DIRECTION_NONE;

		QtLineItemBezier* bezier = new QtLineItemBezier(this);
		m_child = bezier;
		bezier->updateLine(
			ownerRect, targetRect, ownerParentRect, targetParentRect, style, m_weight, showArrow);
		bezier->setRoute(route);

		if (ownerNonGroupParent == targetNonGroupParent)
		{
			if (ownerRect.y() < target->getBoundingRect().y())
			{
				bezier->setOnBack(true);
			}
			else
			{
				bezier->setOnFront(true);
			}
		}
	}
	else
	{
		const Vec2i* ownerColumnSize;
		const Vec2i* targetColumnSize;

		if (ownerParent != targetParent)
		{
			ownerColumnSize = &ownerParent->getColumnSize();
			targetColumnSize = &targetParent->getColumnSize();
		}
		else
		{
			ownerColumnSize = &ownerNonGroupParent->getColumnSize();
			targetColumnSize = &targetNonGroupParent->getColumnSize();
		}

		ownerParentRect.z = std::max(ownerParentRect.x + ownerColumnSize->x, ownerParentRect.z());
		targetParentRect.z = std::max(targetParentRect.x + targetColumnSize->x, targetParentRect.z());

		if (!m_child)
		{
			m_child = new QtLineItemAngled(this);
		}

		QtLineItemAngled* child = dynamic_cast<QtLineItemAngled*>(m_child);

		if (owner->getIsActive() && ownerNonGroupParent == targetNonGroupParent)
		{
			child->setOnBack(true);
		}

		if (target->getIsActive())
		{
			child->setEarlyBend(true);

			if (ownerNonGroupParent == targetNonGroupParent ||
				(type == Edge::EDGE_OVERRIDE &&
				 targetParentRect.z() + style.targetOffset.x + style.originOffset.x >
					 ownerParentRect.x()))
			{
				child->setOnFront(true);
			}
			else
			{
				child->setOnFront(false);
			}
		}

		// increase z-value for inheritance and specialization edges between members
		if ((type == Edge::EDGE_INHERITANCE || type == Edge::EDGE_TEMPLATE_SPECIALIZATION) &&
			(owner != ownerNonGroupParent || target != targetNonGroupParent))
		{
			if (style.zValue < 0)
			{
				style.zValue += 5;
			}
		}
		else if (
			type == Edge::EDGE_INHERITANCE ||
			(type == Edge::EDGE_TEMPLATE_SPECIALIZATION && owner == ownerNonGroupParent &&
			 target == targetNonGroupParent))
		{
			route = QtLineItemBase::ROUTE_VERTICAL;

			if (target->hasActiveChild())
			{
				child->setEarlyBend(true);
			}
		}
		else if (type != Edge::EDGE_BUNDLED_EDGES || owner != ownerNonGroupParent || target != targetNonGroupParent)
		{
			route = QtLineItemBase::ROUTE_HORIZONTAL;
		}

		child->setRoute(route);

		bool showArrow = true;
		if (type == Edge::EDGE_BUNDLED_EDGES)
		{
			showArrow = m_direction != TokenComponentBundledEdges::DIRECTION_NONE;
		}

		if (getData())
		{
			TokenComponentInheritanceChain* componentInheritance =
				getData()->getComponent<TokenComponentInheritanceChain>();
			if (componentInheritance && componentInheritance->inheritanceEdgeIds.size() > 1)
			{
				style.dashed = true;
			}
		}

		child->updateLine(
			ownerRect, targetRect, ownerParentRect, targetParentRect, style, m_weight, showArrow);
	}

	this->setZValue(style.zValue);
}

bool QtGraphEdge::getIsActive() const
{
	return m_isActive;
}

void QtGraphEdge::setIsActive(bool isActive)
{
	if (m_isActive != isActive)
	{
		m_isActive = isActive;
		updateLine();
	}
}

void QtGraphEdge::setIsFocused(bool isFocused)
{
	if (m_isFocused != isFocused)
	{
		m_isFocused = isFocused;
		updateLine();
	}
}

void QtGraphEdge::setIsCoFocused(bool isCoFocused)
{
	if (m_isCoFocused != isCoFocused)
	{
		m_isCoFocused = isCoFocused;
		updateLine();
	}
}

bool QtGraphEdge::isFocusable() const
{
	return m_isInteractive;
}

void QtGraphEdge::onClick()
{
	if (isExpandable())
	{
		QtGraphNode* node =
			(m_direction == TokenComponentBundledEdges::DIRECTION_BACKWARD ? m_owner : m_target);
		if (m_owner->isGroupNode())
		{
			node = m_owner;
		}
		else if (m_target->isGroupNode())
		{
			node = m_target;
		}

		MessageGraphNodeBundleSplit(node->getTokenId()).dispatch();
	}
	else if (isTrailEdge())
	{
		MessageActivateTrailEdge(
			{getData()->getId()},
			getData()->getType(),
			getData()->getFrom()->getNameHierarchy(),
			getData()->getTo()->getNameHierarchy())
			.dispatch();
	}
	else
	{
		TokenComponentInheritanceChain* componentInheritance =
			getData()->getComponent<TokenComponentInheritanceChain>();

		MessageActivateEdge msg(
			getData()->getId(),
			componentInheritance ? Edge::EDGE_BUNDLED_EDGES : getData()->getType(),
			getData()->getFrom()->getNameHierarchy(),
			getData()->getTo()->getNameHierarchy());

		if (getData()->getType() == Edge::EDGE_BUNDLED_EDGES)
		{
			msg.bundledEdgesIds = utility::toVector<Id>(
				getData()->getComponent<TokenComponentBundledEdges>()->getBundledEdgesIds());
		}
		else if (componentInheritance)
		{
			msg.bundledEdgesIds = componentInheritance->inheritanceEdgeIds;
		}

		msg.dispatch();
	}
}

void QtGraphEdge::onHide()
{
	Id tokenId = getTokenId();

	if (tokenId)
	{
		MessageGraphNodeHide(tokenId).dispatch();
	}
}

void QtGraphEdge::coFocusIn()
{
	if (!m_isCoFocused)
	{
		m_isCoFocused = true;
		updateLine();

		if (s_focusedEdge == this)
		{
			Edge::EdgeType type = (getData() ? getData()->getType() : Edge::EDGE_BUNDLED_EDGES);

			TooltipInfo info;
			info.title = Edge::getReadableTypeString(type);

			if (isAmbiguous())
			{
				info.title = L"ambiguous " + info.title;
			}

			if (type == Edge::EDGE_BUNDLED_EDGES &&
				m_direction == TokenComponentBundledEdges::DIRECTION_NONE)
			{
				info.title = L"bidirectional " + info.title;
			}

			if (type == Edge::EDGE_BUNDLED_EDGES)
			{
				info.count = static_cast<int>(m_weight);
				info.countText = "edge";
			}
			info.offset = Vec2i(10, 20);

			if (type == Edge::EDGE_INHERITANCE && getData())
			{
				TokenComponentInheritanceChain* componentInheritance =
					getData()->getComponent<TokenComponentInheritanceChain>();
				if (componentInheritance && componentInheritance->inheritanceEdgeIds.size() > 1)
				{
					info.title = L"multi-level " + info.title;
				}
			}

			MessageTooltipShow(info, TOOLTIP_ORIGIN_GRAPH).dispatch();
		}
	}
}

void QtGraphEdge::coFocusOut()
{
	if (m_isCoFocused)
	{
		m_isCoFocused = false;
		updateLine();

		if (s_focusedEdge == this)
		{
			MessageTooltipHide().dispatch();
		}
	}
}

void QtGraphEdge::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	m_mousePos = Vec2i(
		static_cast<int>(event->scenePos().x()), static_cast<int>(event->scenePos().y()));
	m_mouseMoved = false;
}

void QtGraphEdge::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	Vec2i mousePos = Vec2i(
		static_cast<int>(event->scenePos().x()), static_cast<int>(event->scenePos().y()));

	if ((mousePos - m_mousePos).getLength() > 1.0f)
	{
		m_mouseMoved = true;
	}
}

void QtGraphEdge::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	if (!m_mouseMoved && m_isInteractive && event->button() == Qt::LeftButton)
	{
		if (event->modifiers() & Qt::AltModifier)
		{
			this->onHide();
		}
		else
		{
			this->onClick();
		}
	}
}

void QtGraphEdge::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
	m_focusHandler->focusEdge(this);

	if (m_useBezier)
	{
		if (s_focusedBezierEdge && s_focusedBezierEdge != this)
		{
			s_focusedBezierEdge->coFocusOut();
		}

		s_focusedBezierEdge = this;
	}

	s_focusedEdge = this;

	if (getData() && !m_useBezier)
	{
		MessageFocusIn({getData()->getId()}, TOOLTIP_ORIGIN_GRAPH).dispatch();
	}
	else
	{
		coFocusIn();
	}
}

void QtGraphEdge::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
	m_focusHandler->defocusEdge(this);

	s_focusedBezierEdge = nullptr;

	if (getData() && !m_useBezier)
	{
		MessageFocusOut({getData()->getId()}).dispatch();
	}
	else
	{
		coFocusOut();
	}

	s_focusedEdge = nullptr;
}

void QtGraphEdge::setDirection(TokenComponentBundledEdges::Direction direction)
{
	if (m_direction != direction)
	{
		m_direction = direction;
		updateLine();
	}
}

bool QtGraphEdge::isHorizontal() const
{
	return m_isHorizontal;
}

bool QtGraphEdge::isExpandable() const
{
	return !getData() || m_owner->isGroupNode() || m_target->isGroupNode();
}

bool QtGraphEdge::isTrailEdge() const
{
	return m_isTrailEdge;
}

void QtGraphEdge::setIsTrailEdge(const std::vector<Vec4i>& path, bool horizontal)
{
	m_path = path;
	m_isTrailEdge = true;
	m_useBezier = true;
	m_isHorizontal = horizontal;
}

bool QtGraphEdge::isBezierEdge() const
{
	return m_useBezier;
}

void QtGraphEdge::setUseBezier(bool useBezier)
{
	m_useBezier = useBezier;
	m_isHorizontal = true;
}

void QtGraphEdge::clearPath()
{
	m_path.clear();
}

bool QtGraphEdge::isAmbiguous() const
{
	return m_data && m_data->getComponent<TokenComponentIsAmbiguous>();
}

QRectF QtGraphEdge::getBoundingRect() const
{
	if (m_child)
	{
		return m_child->sceneBoundingRect();
	}

	return QRectF();
}
