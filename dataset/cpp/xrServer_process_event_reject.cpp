#include "stdafx.h"
#include "xrserver.h"
#include "xrserver_objects.h"

bool xrServer::Process_event_reject(NET_Packet& P, const ClientID sender, const u32 time, const u16 id_parent, const u16 id_entity, bool send_message)
{
    // Parse message
    CSE_Abstract* e_parent = game->get_entity_from_eid(id_parent);
    CSE_Abstract* e_entity = game->get_entity_from_eid(id_entity);

#ifdef DEBUG
    Msg("sv reject. id_parent %s id_entity %s [%d]", ent_name_safe(id_parent).c_str(), ent_name_safe(id_entity).c_str(), Device.dwFrame);
#endif
    //	R_ASSERT			(e_parent && e_entity);
    if (!e_parent)
    {
        Msg("xrServer::Process_event_reject: no parent object! ID %d", id_parent);
        return false;
    }
    if (!e_entity)
    {
        Msg("xrServer::Process_event_reject: no entity object! ID %d", id_entity);
        return false;
    }
    game->OnDetach(id_parent, id_entity);

    if (0xffff == e_entity->ID_Parent)
    {
#ifdef DEBUG
        Msg("~ ERROR: can't detach independant object. entity[%s:%d], parent[%s:%d], section[%s]", e_entity->name_replace(), id_entity, e_parent->name_replace(), id_parent,
            *e_entity->s_name);
#endif

        return false;
    }
    auto& C = e_parent->children;
    auto __c = std::find(C.begin(), C.end(), id_entity);
    if (__c == C.end())
    {
        Msg("! ERROR: SV: can't find children [%d] of parent [%d]", id_entity, e_parent);
        return false;
    }

    if (0xffff == e_entity->ID_Parent)
    {
        Msg("! ERROR: can't detach independant object. entity[%s][%d], parent[%s][%d], section[%s]", e_entity->name_replace(), id_entity, e_parent->name_replace(), id_parent,
            e_entity->s_name.c_str());
        return false;
    }

    // Rebuild parentness
    if (e_entity->ID_Parent != id_parent)
    {
        Msg("! ERROR: e_entity->ID_Parent = [%d]  parent = [%d][%s]  entity_id = [%d]  frame = [%d]", e_entity->ID_Parent, id_parent, e_parent->name_replace(), id_entity,
            Device.dwFrame);
        // it can't be !!!
    }
    e_entity->ID_Parent = 0xffff;

    C.erase(__c);

    // Signal to everyone (including sender)
    if (send_message)
    {
        DWORD MODE = net_flags(TRUE, TRUE, FALSE, TRUE);
        SendBroadcast(BroadcastCID, P, MODE);
    }

    return true;
}
