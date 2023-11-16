// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "fieldsearchspec.h"
#include <vespa/vespalib/stllike/asciistream.h>
#include <vespa/vsm/searcher/boolfieldsearcher.h>
#include <vespa/vsm/searcher/floatfieldsearcher.h>
#include <vespa/vsm/searcher/futf8strchrfieldsearcher.h>
#include <vespa/vsm/searcher/geo_pos_field_searcher.h>
#include <vespa/vsm/searcher/intfieldsearcher.h>
#include <vespa/vsm/searcher/nearest_neighbor_field_searcher.h>
#include <vespa/vsm/searcher/utf8exactstringfieldsearcher.h>
#include <vespa/vsm/searcher/utf8flexiblestringfieldsearcher.h>
#include <vespa/vsm/searcher/utf8strchrfieldsearcher.h>
#include <vespa/vsm/searcher/utf8substringsearcher.h>
#include <vespa/vsm/searcher/utf8suffixstringfieldsearcher.h>
#include <regex>

#include <vespa/log/log.h>
LOG_SETUP(".vsm.fieldsearchspec");

#define DEBUGMASK 0x01

using search::streaming::ConstQueryTermList;
using search::streaming::Query;
using search::streaming::QueryTerm;

namespace vsm {

namespace {

void setMatchType(FieldSearcherContainer & searcher, vespalib::stringref arg1) {
    if (arg1 == "prefix") {
        searcher->setMatchType(FieldSearcher::PREFIX);
    } else if (arg1 == "substring") {
        searcher->setMatchType(FieldSearcher::SUBSTRING);
    } else if (arg1 == "suffix") {
        searcher->setMatchType(FieldSearcher::SUFFIX);
    } else if (arg1 == "exact") {
        searcher->setMatchType(FieldSearcher::EXACT);
    } else if (arg1 == "word") {
        searcher->setMatchType(FieldSearcher::EXACT);
    }
}

}

FieldSearchSpec::FieldSearchSpec() :
    _id(0),
    _name(),
    _maxLength(0x100000),
    _searcher(),
    _searchMethod(VsmfieldsConfig::Fieldspec::Searchmethod::NONE),
    _arg1(),
    _reconfigured(false)
{
}
FieldSearchSpec::~FieldSearchSpec() = default;

FieldSearchSpec::FieldSearchSpec(FieldSearchSpec&& rhs) noexcept = default;
FieldSearchSpec& FieldSearchSpec::operator=(FieldSearchSpec&& rhs) noexcept = default;

FieldSearchSpec::FieldSearchSpec(const FieldIdT & fid, const vespalib::string & fname,
                                 VsmfieldsConfig::Fieldspec::Searchmethod searchDef,
                                 const vespalib::string & arg1, size_t maxLength_) :
    _id(fid),
    _name(fname),
    _maxLength(maxLength_),
    _searcher(),
    _searchMethod(searchDef),
    _arg1(arg1),
    _reconfigured(false)
{
    switch(searchDef) {
    default:
        LOG(warning, "Unknown searchdef = %d. Defaulting to AUTOUTF8", static_cast<int>(searchDef));
        [[fallthrough]];
    case VsmfieldsConfig::Fieldspec::Searchmethod::AUTOUTF8:
    case VsmfieldsConfig::Fieldspec::Searchmethod::NONE:
    case VsmfieldsConfig::Fieldspec::Searchmethod::SSE2UTF8:
    case VsmfieldsConfig::Fieldspec::Searchmethod::UTF8:
        if (arg1 == "substring") {
            _searcher = std::make_unique<UTF8SubStringFieldSearcher>(fid);
        } else if (arg1 == "suffix") {
            _searcher = std::make_unique<UTF8SuffixStringFieldSearcher>(fid);
        } else if (arg1 == "exact") {
            _searcher = std::make_unique<UTF8ExactStringFieldSearcher>(fid);
        } else if (arg1 == "word") {
            _searcher = std::make_unique<UTF8ExactStringFieldSearcher>(fid);
        } else if (searchDef == VsmfieldsConfig::Fieldspec::Searchmethod::UTF8) {
            _searcher = std::make_unique<UTF8StrChrFieldSearcher>(fid);
        } else {
            _searcher = std::make_unique<FUTF8StrChrFieldSearcher>(fid);
        }
        break;
    case VsmfieldsConfig::Fieldspec::Searchmethod::BOOL:
        _searcher = std::make_unique<BoolFieldSearcher>(fid);
        break;
    case VsmfieldsConfig::Fieldspec::Searchmethod::INT8:
    case VsmfieldsConfig::Fieldspec::Searchmethod::INT16:
    case VsmfieldsConfig::Fieldspec::Searchmethod::INT32:
    case VsmfieldsConfig::Fieldspec::Searchmethod::INT64:
        _searcher = std::make_unique<IntFieldSearcher>(fid);
        break;
    case VsmfieldsConfig::Fieldspec::Searchmethod::FLOAT:
        _searcher = std::make_unique<FloatFieldSearcher>(fid);
        break;
    case VsmfieldsConfig::Fieldspec::Searchmethod::DOUBLE:
        _searcher = std::make_unique<DoubleFieldSearcher>(fid);
        break;
    case VsmfieldsConfig::Fieldspec::Searchmethod::GEOPOS:
        _searcher = std::make_unique<GeoPosFieldSearcher>(fid);
        break;
    case VsmfieldsConfig::Fieldspec::Searchmethod::NEAREST_NEIGHBOR:
        auto dm = NearestNeighborFieldSearcher::distance_metric_from_string(arg1);
        _searcher = std::make_unique<NearestNeighborFieldSearcher>(fid, dm);
        break;
    }
    if (_searcher) {
        setMatchType(_searcher, arg1);
        _searcher->maxFieldLength(maxLength());
    }
}

void
FieldSearchSpec::reconfig(const QueryTerm & term)
{
    if (_reconfigured) {
        return;
    }
    switch (_searchMethod) {
    case VsmfieldsConfig::Fieldspec::Searchmethod::NONE:
    case VsmfieldsConfig::Fieldspec::Searchmethod::AUTOUTF8:
    case VsmfieldsConfig::Fieldspec::Searchmethod::UTF8:
    case VsmfieldsConfig::Fieldspec::Searchmethod::SSE2UTF8:
        if ((term.isSubstring() && _arg1 != "substring") ||
            (term.isSuffix() && _arg1 != "suffix") ||
            (term.isExactstring() && _arg1 != "exact") ||
            (term.isPrefix() && _arg1 == "suffix"))
        {
            _searcher = std::make_unique<UTF8FlexibleStringFieldSearcher>(id());
            // preserve the basic match property of the searcher
            setMatchType(_searcher, _arg1);
            LOG(debug, "Reconfigured to use UTF8FlexibleStringFieldSearcher (%s) for field '%s' with id '%d'",
                _searcher->prefix() ? "prefix" : "regular", name().c_str(), id());
            _reconfigured = true;
        }
        break;
    default:
        break;
    }
}

vespalib::asciistream & operator <<(vespalib::asciistream & os, const FieldSearchSpec & f)
{
    os << f._id << ' ' << f._name << ' ';
    if ( ! f._searcher) {
        os << " No searcher defined.\n";
    }
    return os;
}

FieldSearchSpecMap::FieldSearchSpecMap() = default;

FieldSearchSpecMap::~FieldSearchSpecMap() = default;

namespace {
    const std::string _G_empty("");
    const std::string _G_value(".value");
    const std::regex _G_map1("\\{[a-zA-Z0-9]+\\}");
    const std::regex _G_map2("\\{\".*\"\\}");
    const std::regex _G_array("\\[[0-9]+\\]");
}

vespalib::string FieldSearchSpecMap::stripNonFields(const vespalib::string & rawIndex)
{
    if ((rawIndex.find('[') != vespalib::string::npos) || (rawIndex.find('{') != vespalib::string::npos)) {
        std::string index = std::regex_replace(std::string(rawIndex), _G_map1, _G_value);
        index = std::regex_replace(index, _G_map2, _G_value);
        index = std::regex_replace(index, _G_array, _G_empty);
        return index;
    }
    return rawIndex;
}

bool FieldSearchSpecMap::buildFieldsInQuery(const Query & query, StringFieldIdTMap & fieldsInQuery) const
{
    bool retval(true);
    ConstQueryTermList qtl;
    query.getLeafs(qtl);

    for (const auto & term : qtl) {
        for (const auto & dtm : documentTypeMap()) {
            const IndexFieldMapT & fim = dtm.second;
            vespalib::string rawIndex(term->index());
            vespalib::string index(stripNonFields(rawIndex));
            auto fIt = fim.find(index);
            if (fIt != fim.end()) {
                for(FieldIdT fid : fIt->second) {
                    const FieldSearchSpec & spec = specMap().find(fid)->second;
                    LOG(debug, "buildFieldsInQuery = rawIndex='%s', index='%s'", rawIndex.c_str(), index.c_str());
                    if ((rawIndex != index) && (spec.name().find(index) == 0)) {
                        vespalib::string modIndex(rawIndex);
                        modIndex.append(spec.name().substr(index.size()));
                        fieldsInQuery.add(modIndex, spec.id());
                    } else {
                        fieldsInQuery.add(spec.name(),spec.id());
                    }
                }
            } else {
                LOG(warning, "No valid indexes registered for index %s", term->index().c_str());
                retval = false;
            }
        }
    }
    return retval;
}

void FieldSearchSpecMap::buildFromConfig(const std::vector<vespalib::string> & otherFieldsNeeded)
{
    for(size_t i(0), m(otherFieldsNeeded.size()); i < m; i++) {
        LOG(debug, "otherFieldsNeeded[%zd] = '%s'", i, otherFieldsNeeded[i].c_str());
        _nameIdMap.add(otherFieldsNeeded[i]);
    }
}

namespace {

FieldIdTList
buildFieldSet(const VsmfieldsConfig::Documenttype::Index & ci, const FieldSearchSpecMapT & specMap,
              const VsmfieldsConfig::Documenttype::IndexVector & indexes)
{
    LOG(spam, "Index %s with %zd fields", ci.name.c_str(), ci.field.size());
    FieldIdTList ifm;
    for (const VsmfieldsConfig::Documenttype::Index::Field & cf : ci.field) {
        LOG(spam, "Parsing field %s", cf.name.c_str());
        auto foundIndex = std::find_if(indexes.begin(), indexes.end(),
                                       [&cf](const auto & v) { return v.name == cf.name;});
        if ((foundIndex != indexes.end()) && (cf.name != ci.name)) {
            FieldIdTList sub = buildFieldSet(*foundIndex, specMap, indexes);
            ifm.insert(ifm.end(), sub.begin(), sub.end());
        } else {
            auto foundField = std::find_if(specMap.begin(), specMap.end(),
                                           [&cf](const auto & v) { return v.second.name() == cf.name;} );
            if (foundField != specMap.end()) {
                ifm.push_back(foundField->second.id());
            } else {
                LOG(warning, "Field %s not defined. Ignoring....", cf.name.c_str());
            }
        }
    }
    return ifm;
}

}

bool FieldSearchSpecMap::buildFromConfig(const VsmfieldsHandle & conf)
{
    bool retval(true);
    LOG(spam, "Parsing %zd fields", conf->fieldspec.size());
    for(const VsmfieldsConfig::Fieldspec & cfs : conf->fieldspec) {
        LOG(spam, "Parsing %s", cfs.name.c_str());
        FieldIdT fieldId = specMap().size();
        FieldSearchSpec fss(fieldId, cfs.name, cfs.searchmethod, cfs.arg1.c_str(), cfs.maxlength);
        _specMap[fieldId] = std::move(fss);
        _nameIdMap.add(cfs.name, fieldId);
        LOG(spam, "M in %d = %s", fieldId, cfs.name.c_str());
    }

    LOG(spam, "Parsing %zd document types", conf->documenttype.size());
    for(const VsmfieldsConfig::Documenttype & di : conf->documenttype) {
        IndexFieldMapT indexMapp;
        LOG(spam, "Parsing document type %s with %zd indexes", di.name.c_str(), di.index.size());
        for(const VsmfieldsConfig::Documenttype::Index & ci : di.index) {
            indexMapp[ci.name] = buildFieldSet(ci, specMap(), di.index);
        }
        _documentTypeMap[di.name] = indexMapp;
    }
    return retval;
}

void
FieldSearchSpecMap::reconfigFromQuery(const Query & query)
{
    ConstQueryTermList qtl;
    query.getLeafs(qtl);

    for (const auto & termA : qtl) {
        for (const auto & ifm : documentTypeMap()) {
            auto itc = ifm.second.find(termA->index());
            if (itc != ifm.second.end()) {
                for (FieldIdT fid : itc->second) {
                    FieldSearchSpec & spec = _specMap.find(fid)->second;
                    spec.reconfig(*termA);
                }
            }
        }
    }
}

bool lesserField(const FieldSearcherContainer & a, const FieldSearcherContainer & b)
{
    return a->field() < b->field();
}

void FieldSearchSpecMap::buildSearcherMap(const StringFieldIdTMapT & fieldsInQuery, FieldIdTSearcherMap & fieldSearcherMap)
{
    fieldSearcherMap.clear();
    for (const auto & entry : fieldsInQuery) {
        FieldIdT fId = entry.second;
        const FieldSearchSpec & spec = specMap().find(fId)->second;
        fieldSearcherMap.emplace_back(spec.searcher().duplicate());
    }
    std::sort(fieldSearcherMap.begin(), fieldSearcherMap.end(), lesserField);
}

search::attribute::DistanceMetric
FieldSearchSpecMap::get_distance_metric(const vespalib::string& name) const
{
    auto dm = search::attribute::DistanceMetric::Euclidean;
    auto fid = _nameIdMap.fieldNo(name);
    if (fid == vsm::StringFieldIdTMap::npos) {
        return dm;
    }
    auto itr = _specMap.find(fid);
    if (itr == _specMap.end()) {
        return dm;
    }
    if (!itr->second.uses_nearest_neighbor_search_method()) {
        return dm;
    }
    return vsm::NearestNeighborFieldSearcher::distance_metric_from_string(itr->second.get_arg1());
}

vespalib::asciistream & operator <<(vespalib::asciistream & os, const FieldSearchSpecMap & df)
{
    os << "DocumentTypeMap = \n";
    for (const auto & dtm : df.documentTypeMap()) {
        os << "DocType = " << dtm.first << "\n";
        os << "IndexMap = \n";
        for (const auto &index : dtm.second) {
            os << index.first << ": ";
            for (FieldIdT fid : index.second) {
                os << fid << ' ';
            }
            os << '\n';
        }
    }
    os << "SpecMap = \n";
    for (const auto & entry : df.specMap()) {
        os << entry.first << " = " << entry.second << '\n';
    }
    os << "NameIdMap = \n" << df.nameIdMap();
    return os;
}

}
