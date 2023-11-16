// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "grouping.h"
#include "hitsaggregationresult.h"
#include <vespa/searchlib/expression/stringresultnode.h>
#include <vespa/searchlib/expression/enumresultnode.h>
#include <vespa/searchlib/expression/resultvector.h>
#include <vespa/searchlib/expression/attributenode.h>
#include <vespa/searchlib/expression/current_index_setup.h>
#include <vespa/searchlib/expression/documentaccessornode.h>
#include <vespa/searchlib/attribute/stringbase.h>
#include <vespa/vespalib/objects/serializer.hpp>
#include <vespa/vespalib/objects/deserializer.hpp>
#include <vespa/searchlib/common/idocumentmetastore.h>

#include <vespa/log/log.h>
LOG_SETUP(".searchlib.aggregation.grouping");

using namespace search::expression;
using vespalib::Serializer;
using vespalib::Deserializer;

namespace search::aggregation {

namespace {

void
selectGroups(const vespalib::ObjectPredicate &p, vespalib::ObjectOperation &op,
             Group &group, uint32_t first, uint32_t last, uint32_t curr)
{
    if (curr > last) {
        return;
    }
    if (curr >= first) {
        group.select(p, op);
    }
    Group::GroupList list = group.groups();
    for (uint32_t i(0), m(group.getChildrenSize()); i < m; ++i) {
        selectGroups(p, op, *list[i], first, last, curr + 1);
    }
}

using search::aggregation::Grouping;
using search::aggregation::GroupingLevel;
using search::aggregation::Group;
using search::expression::ExpressionTree;
using search::expression::ExpressionNode;
using search::expression::AttributeNode;
using search::expression::EnumResultNode;
using search::expression::EnumResultNodeVector;
using search::expression::StringResultNode;
using search::expression::ResultNode;
using search::StringAttribute;

class EnumConverter : public vespalib::ObjectOperation, public vespalib::ObjectPredicate
{
private:
    Grouping &_grouping;
    uint32_t _level;
public:
    EnumConverter(Grouping & g, uint32_t level) : _grouping(g), _level(level) { }
    void execute(vespalib::Identifiable &obj) override {
        Group &group = static_cast<Group &>(obj);
        uint32_t tmplevel = _level;
        if (group.hasId()) {
            if (group.getId().inherits(EnumResultNode::classId)) {
                const EnumResultNode & er = static_cast<const EnumResultNode &>(group.getId());
                const Grouping::GroupingLevelList &gll = _grouping.getLevels();
                const GroupingLevel & gl = gll[_level];
                const ExpressionNode * en = gl.getExpression().getRoot();
                const AttributeNode & an = static_cast<const AttributeNode &>(*en);
                StringResultNode srn(an.getAttribute()->getStringFromEnum(er.getEnum()));
                group.setId(srn);
            }
            tmplevel++;
        }
        EnumConverter enumConverter(_grouping, tmplevel);
        Group::GroupList list = group.groups();
        for (uint32_t i(0), m(group.getChildrenSize()); i < m; ++i) {
            list[i]->select(enumConverter, enumConverter);
        }
    }
    bool check(const vespalib::Identifiable &obj) const override { return obj.inherits(Group::classId); }
};

class GlobalIdConverter : public vespalib::ObjectOperation, public vespalib::ObjectPredicate
{
private:
    const IDocumentMetaStore &_metaStore;
public:
    GlobalIdConverter(const IDocumentMetaStore &metaStore) : _metaStore(metaStore) {}
    void execute(vespalib::Identifiable & obj) override {
        FS4Hit & hit = static_cast<FS4Hit &>(obj);
        document::GlobalId gid;
        _metaStore.getGid(hit.getDocId(), gid);
        hit.setGlobalId(gid);
        LOG(debug, "GlobalIdConverter: lid(%u) -> gid(%s)", hit.getDocId(), hit.getGlobalId().toString().c_str());
    }
    bool check(const vespalib::Identifiable & obj) const override {
        return obj.inherits(FS4Hit::classId);
    }
};

// extend to also handle document access nodes when that time comes (streaming)
struct ResolveCurrentIndex : vespalib::ObjectOperation, vespalib::ObjectPredicate {
    const CurrentIndexSetup &setup;
    ResolveCurrentIndex(const CurrentIndexSetup &setup_in) noexcept : setup(setup_in) {}
    void execute(vespalib::Identifiable &obj) override {
        auto &attr = static_cast<AttributeNode &>(obj);
        if (attr.getCurrentIndex() == nullptr) {
            attr.setCurrentIndex(setup.resolve(attr.getAttributeName()));
        }
    }
    bool check(const vespalib::Identifiable &obj) const override {
        return obj.inherits(AttributeNode::classId);
    }
};

} // namespace search::aggregation::<unnamed>

IMPLEMENT_IDENTIFIABLE_NS2(search, aggregation, Grouping, vespalib::Identifiable);

Grouping::Grouping() noexcept
    : _id(0),
      _valid(true),
      _all(false),
      _topN(-1),
      _firstLevel(0),
      _lastLevel(0),
      _levels(),
      _root()
{ }

Grouping::Grouping(const Grouping &) = default;
Grouping & Grouping::operator = (const Grouping &) = default;
Grouping::~Grouping() = default;

void
Grouping::selectMembers(const vespalib::ObjectPredicate &predicate,
                        vespalib::ObjectOperation &operation)
{
    for (GroupingLevel & level : _levels) {
        level.select(predicate, operation);
    }
    selectGroups(predicate, operation, _root, _firstLevel, _lastLevel, 0);
}

void
Grouping::prune(const Grouping & b)
{
    _root.prune(b._root, b._lastLevel, 0);
}

void
Grouping::mergePartial(const Grouping & b)
{
    _root.mergePartial(_levels, _firstLevel, _lastLevel, 0, b._root);
}


void
Grouping::merge(Grouping & b)
{
    _root.merge(_levels, _firstLevel, 0, b._root);
}

void
Grouping::postMerge()
{
    _root.postMerge(_levels, _firstLevel, 0);
}

void
Grouping::preAggregate(bool isOrdered)
{
    for (size_t i(0), m(_levels.size()); i < m; i++) {
        _levels[i].prepare(this, i, isOrdered);
    }
    _root.preAggregate();
}

void
Grouping::aggregate(DocId from, DocId to)
{
    preAggregate(false);
    if (to > from) {
        for(DocId i(from), m(i + getMaxN(to-from)); i < m; i++) {
            aggregate(i, 0.0);
        }
    }
    postProcess();
}

void
Grouping::postProcess()
{
    postAggregate();
    postMerge();
    bool hasEnums(false);
    for (size_t i(0), m(_levels.size()); !hasEnums && (i < m); i++) {
        const GroupingLevel & l = _levels[i];
        const ResultNode & id(*l.getExpression().getResult());
        hasEnums = id.inherits(EnumResultNode::classId) ||
                   id.inherits(EnumResultNodeVector::classId);
        const Group & g(l.getGroupPrototype());
        for (size_t j(0), n(g.getAggrSize()); !hasEnums && (j < n); j++) {
            const ResultNode & r(*g.getAggregationResult(j).getResult());
            hasEnums = r.inherits(EnumResultNode::classId) ||
                       r.inherits(EnumResultNodeVector::classId);
        }
    }
    if (hasEnums) {
        EnumConverter enumConverter(*this, 0);
        _root.select(enumConverter, enumConverter);
    }
    sortById();
}

void
Grouping::aggregate(const RankedHit * rankedHit, unsigned int len)
{
    bool isOrdered(! needResort());
    preAggregate(isOrdered);
    HitsAggregationResult::SetOrdered pred;
    select(pred, pred);
    for(unsigned int i(0), m(getMaxN(len)); i < m; i++) {
        aggregate(rankedHit[i].getDocId(), rankedHit[i].getRank());
    }
    postProcess();
}

void
Grouping::aggregate(DocId docId, HitRank rank)
{
    _root.aggregate(*this, 0, docId, rank);
}

void
Grouping::aggregate(const document::Document & doc, HitRank rank)
{
    _root.aggregate(*this, 0, doc, rank);
}

void
Grouping::convertToGlobalId(const search::IDocumentMetaStore &metaStore)
{
    GlobalIdConverter conv(metaStore);
    select(conv, conv);
}

void
Grouping::postAggregate()
{
    _root.postAggregate();
}

void
Grouping::sortById()
{
    _root.sortById();
}

void
Grouping::configureStaticStuff(const ConfigureStaticParams & params)
{
    if (params._enableNestedMultivalueGrouping) {
        CurrentIndexSetup setup;
        ResolveCurrentIndex resolver(setup);
        size_t end = std::min(size_t(_lastLevel + 1), _levels.size());
        for (size_t i = _firstLevel; i < end; ++i) {
            _levels[i].wire_current_index(setup, resolver, resolver);
        }
    }

    if (params._attrCtx != nullptr) {
        AttributeNode::Configure confAttr(*params._attrCtx);
        select(confAttr, confAttr);
    }

    if (params._docType != nullptr) {
        DocumentAccessorNode::Configure confDoc(*params._docType);
        select(confDoc, confDoc);
    }
    ExpressionTree::Configure treeConf;
    select(treeConf, treeConf);

    AggregationResult::Configure aggrConf;
    select(aggrConf, aggrConf);
}

void
Grouping::cleanupAttributeReferences()
{
    AttributeNode::CleanupAttributeReferences cleanupAttr;
    select(cleanupAttr, cleanupAttr);
}

void
Grouping::cleanTemporary()
{
    for (GroupingLevel & level : _levels) {
        if (level.getExpression().getRoot()->inherits(FunctionNode::classId)) {
            static_cast<FunctionNode &>(*level.getExpression().getRoot()).reset();
        }
    }
}

bool
Grouping::needResort() const
{
    bool resort(_root.needResort());
    for (auto it(_levels.begin()), mt(_levels.end()); !resort && (it != mt); ++it) {
        resort = it->needResort();
    }
    return (resort && getTopN() <= 0);
}


Serializer &
Grouping::onSerialize(Serializer & os) const
{
    LOG(spam, "Grouping = %s", asString().c_str());
    return os << _id << _valid << _all << _topN << _firstLevel << _lastLevel << _levels << _root;
}

Deserializer &
Grouping::onDeserialize(Deserializer & is)
{
    is >> _id >> _valid >> _all >> _topN >> _firstLevel >> _lastLevel >> _levels >> _root;
    LOG(spam, "Grouping = %s", asString().c_str());
    return is;
}

void
Grouping::visitMembers(vespalib::ObjectVisitor &visitor) const
{
    visit(visitor, "id",         _id);
    visit(visitor, "valid",      _valid);
    visit(visitor, "all",        _all);
    visit(visitor, "topN",       _topN);
    visit(visitor, "firstLevel", _firstLevel);
    visit(visitor, "lastLevel",  _lastLevel);
    visit(visitor, "levels",     _levels);
    visit(visitor, "root",       _root);
}

}

// this function was added by ../../forcelink.sh
void forcelink_file_searchlib_aggregation_grouping() {}
