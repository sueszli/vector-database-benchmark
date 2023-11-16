// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "routingtable.h"
#include "hop.h"
#include "routingtablespec.h"

namespace mbus {

RoutingTable::HopIterator::HopIterator(const std::map<string, HopBlueprint> &hops) :
    _pos(hops.begin()),
    _end(hops.end())
{ }

RoutingTable::RouteIterator::RouteIterator(const std::map<string, Route> &routes) :
    _pos(routes.begin()),
    _end(routes.end())
{ }

RoutingTable::RoutingTable(const RoutingTableSpec &spec) :
    _name(spec.getProtocol()),
    _hops(),
    _routes()
{
    for (uint32_t i = 0; i < spec.getNumHops(); ++i) {
        const HopSpec& hopSpec = spec.getHop(i);
        _hops.emplace(hopSpec.getName(), HopBlueprint(hopSpec));
    }
    for (uint32_t i = 0; i < spec.getNumRoutes(); ++i) {
        Route route;
        const RouteSpec &routeSpec = spec.getRoute(i);
        for (uint32_t j = 0; j < routeSpec.getNumHops(); ++j) {
            route.addHop(Hop(routeSpec.getHop(j)));
        }
        _routes.emplace(routeSpec.getName(), std::move(route));
    }
}

bool
RoutingTable::hasHop(const string &name) const
{
    return _hops.find(name) != _hops.end();
}

const HopBlueprint *
RoutingTable::getHop(const string &name) const
{
    auto it = _hops.find(name);
    return it != _hops.end() ? &(it->second) : nullptr;
}

bool
RoutingTable::hasRoute(const string &name) const
{
    return _routes.find(name) != _routes.end();
}

const Route *
RoutingTable::getRoute(const string &name) const
{
    auto it = _routes.find(name);
    return it != _routes.end() ? &(it->second) : nullptr;
}

} // namespace mbus
