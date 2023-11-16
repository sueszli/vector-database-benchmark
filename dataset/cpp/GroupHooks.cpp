/*
 * Copyright (C) 2010 - 2016 Eluna Lua Engine <http://emudevs.com/>
 * This program is free software licensed under GPL version 3
 * Please see the included DOCS/LICENSE.md for more information
 */

#include "Hooks.h"
#include "HookHelpers.h"
#include "LuaEngine.h"
#include "BindingMap.h"
#include "ElunaTemplate.h"

using namespace Hooks;

#define START_HOOK(EVENT) \
    if (!IsEnabled())\
        return;\
    auto key = EventKey<GroupEvents>(EVENT);\
    if (!GroupEventBindings->HasBindingsFor(key))\
        return;\
    LOCK_ELUNA

#define START_HOOK_WITH_RETVAL(EVENT, RETVAL) \
    if (!IsEnabled())\
        return RETVAL;\
    auto key = EventKey<GroupEvents>(EVENT);\
    if (!GroupEventBindings->HasBindingsFor(key))\
        return RETVAL;\
    LOCK_ELUNA

void Eluna::OnAddMember(Group* group, ObjectGuid guid)
{
    START_HOOK(GROUP_EVENT_ON_MEMBER_ADD);
    Push(group);
    Push(guid);
    CallAllFunctions(GroupEventBindings, key);
}

void Eluna::OnInviteMember(Group* group, ObjectGuid guid)
{
    START_HOOK(GROUP_EVENT_ON_MEMBER_INVITE);
    Push(group);
    Push(guid);
    CallAllFunctions(GroupEventBindings, key);
}

void Eluna::OnRemoveMember(Group* group, ObjectGuid guid, uint8 method)
{
    START_HOOK(GROUP_EVENT_ON_MEMBER_REMOVE);
    Push(group);
    Push(guid);
    Push(method);
    CallAllFunctions(GroupEventBindings, key);
}

void Eluna::OnChangeLeader(Group* group, ObjectGuid newLeaderGuid, ObjectGuid oldLeaderGuid)
{
    START_HOOK(GROUP_EVENT_ON_LEADER_CHANGE);
    Push(group);
    Push(newLeaderGuid);
    Push(oldLeaderGuid);
    CallAllFunctions(GroupEventBindings, key);
}

void Eluna::OnDisband(Group* group)
{
    START_HOOK(GROUP_EVENT_ON_DISBAND);
    Push(group);
    CallAllFunctions(GroupEventBindings, key);
}

#if defined (TRINITY) && defined (CATA)
void Eluna::OnCreate(Group* group, ObjectGuid leaderGuid, GroupFlags groupType)
{
    START_HOOK(GROUP_EVENT_ON_CREATE);
    Push(group);
    Push(leaderGuid);
    Push(groupType);
    CallAllFunctions(GroupEventBindings, key);
}
#else
void Eluna::OnCreate(Group* group, ObjectGuid leaderGuid, GroupType groupType)
{
    START_HOOK(GROUP_EVENT_ON_CREATE);
    Push(group);
    Push(leaderGuid);
    Push(groupType);
    CallAllFunctions(GroupEventBindings, key);
}
#endif

bool Eluna::OnMemberAccept(Group* group, Player* player)
{
    START_HOOK_WITH_RETVAL(GROUP_EVENT_ON_MEMBER_ACCEPT, true);
    Push(group);
    Push(player);
    return CallAllFunctionsBool(GroupEventBindings, key, true);
}
