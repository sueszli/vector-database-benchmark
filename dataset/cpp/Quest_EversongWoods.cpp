/*
 * ArcScripts for ArcEmu MMORPG Server
 * Copyright (C) 2009 ArcEmu Team <http://www.arcemu.org/>
 * Copyright (C) 2008-2009 Sun++ Team <http://www.sunscripting.com/>
 * Copyright (C) 2009 WhyScripts Team <http://www.whydb.org/>
 * Copyright (C) 2005-2007 Ascent Team <http://www.ascentcommunity.com/>
 * Copyright (C) 2007-2008 Moon++ Team <http://sunplusplus.info//>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Setup.h"

static LocationExtra ProspectorAnvilwardWaypoints[] =
{
	{ 9294.834f, -6681.092f, 22.428f, 1.284f, 0 },
	{ 9297.834f, -6671.092f, 22.387f, 0.793f, 0 },
	{ 9310.375f, -6658.936f, 22.43f,  2.046f, 0 },
	{ 9306.596f, -6650.905f, 25.222f, 2.666f, 0 },
	{ 9299.666f, -6648.099f, 28.39f,  3.468f, 0 },
	{ 9292.345f, -6650.509f, 30.908f, 4.249f, 0 },
	{ 9289.426f, -6657.825f, 31.829f, 6.154f, 0 },
	{ 9294.095f, -6658.863f, 34.482f, 6.063f, 0 },
};

class ProspectorAnvilwardGossip : public GossipScript
{
	public:
		void GossipHello(Object* pObject, Player* Plr);
		void GossipSelectOption(Object* pObject, Player* Plr, uint32 Id, uint32 IntId, const char* EnteredCode);
		void GossipEnd(Object* pObject, Player* Plr) { Plr->CloseGossip(); }
};

void ProspectorAnvilwardGossip::GossipHello(Object* pObject, Player* Plr)
{
	GossipMenu* Menu;
	objmgr.CreateGossipMenuForPlayer(&Menu, pObject->GetGUID(), 2, Plr);

	QuestLogEntry* qLogEntry = Plr->GetQuestLogForEntry(8483);
	if(qLogEntry != NULL)
	{
		Menu->AddItem(0, "Show me...", 1);
	}

	Menu->SendTo(Plr);
}

void ProspectorAnvilwardGossip::GossipSelectOption(Object* pObject, Player* Plr, uint32 Id, uint32 IntId, const char* EnteredCode)
{
	if(!pObject->IsCreature())
		return;
	Creature* _unit = TO< Creature* >(pObject);
	switch(IntId)
	{
		case 1:
			{
				QuestLogEntry* qLogEntry = Plr->GetQuestLogForEntry(8483);
				if(qLogEntry != NULL)
				{
					_unit->SendChatMessage(CHAT_MSG_MONSTER_SAY, LANG_UNIVERSAL, "Follow me!");
					_unit->GetAIInterface()->setMoveType( MOVEMENTTYPE_FORWARDTHANSTOP );
					_unit->GetAIInterface()->StopMovement( 0 );
					_unit->GetAIInterface()->setWaypointToMove( 0 );
				}
				else
				{
					_unit->SendChatMessage(CHAT_MSG_MONSTER_SAY, LANG_UNIVERSAL, "I have nothing for you. Go away!");
				}
				GossipEnd(pObject, Plr);
			}
			break;
	}
}

class ProspectorAnvilwardAI : public CreatureAIScript
{
	public:
		ADD_CREATURE_FACTORY_FUNCTION(ProspectorAnvilwardAI);
		ProspectorAnvilwardAI(Creature* c) : CreatureAIScript(c)
		{
		}

		void OnLoad()
		{
			_unit->SetFaction(35);
			_unit->GetAIInterface()->setMoveType(MOVEMENTTYPE_DONTMOVEWP);

			WayPointMap *waypoints = new WayPointMap;
			
			_unit->setCustomWayPoints( waypoints );
			_unit->GetAIInterface()->SetWaypointMap(_unit->getCustomWayPoints());
			
			WayPoint* wp = new WayPoint;
			wp->id = 1;
			wp->x = _unit->GetSpawnX();
			wp->y = _unit->GetSpawnY();
			wp->z = _unit->GetSpawnZ() + 2.05f;
			wp->o = _unit->GetSpawnO();
			wp->flags = 256;
			wp->backwardskinid = wp->forwardskinid = _unit->GetDisplayId();
			wp->backwardemoteid = wp->forwardemoteid = 0;
			wp->backwardemoteoneshot = wp->forwardemoteoneshot = false;
			wp->waittime = 0;
			waypoints->push_back(wp);
			
			for(uint32 i = 0; i < sizeof(ProspectorAnvilwardWaypoints) / sizeof(LocationExtra); i++)
			{
				wp = new WayPoint;
				wp->id = i + 2;
				wp->x = ProspectorAnvilwardWaypoints[i].x;
				wp->y = ProspectorAnvilwardWaypoints[i].y;
				wp->z = ProspectorAnvilwardWaypoints[i].z;
				wp->o = ProspectorAnvilwardWaypoints[i].o;
				wp->flags = 256;
				wp->backwardskinid = wp->forwardskinid = _unit->GetDisplayId();
				wp->backwardemoteid = wp->forwardemoteid = 0;
				wp->backwardemoteoneshot = wp->forwardemoteoneshot = false;
				wp->waittime = 0;
				waypoints->push_back(wp);
			}
		}

		void OnReachWP(uint32 iWaypointId, bool bForwards)
		{
			if(iWaypointId == sizeof(ProspectorAnvilwardWaypoints) / sizeof(LocationExtra) && bForwards)
			{
				_unit->SetFaction(14);
			}
		}

		void OnDied(Unit* mKiller)
		{
			_unit->GetAIInterface()->setMoveType(MOVEMENTTYPE_DONTMOVEWP);
			_unit->GetAIInterface()->SetWaypointMap(NULL);
		}
};



void SetupEversongWoods(ScriptMgr* mgr)
{
	mgr->register_gossip_script(15420, new ProspectorAnvilwardGossip);
	mgr->register_creature_script(15420, &ProspectorAnvilwardAI::Create);
}
