/*
	C-Dogs SDL
	A port of the legendary (and fun) action/arcade cdogs.
	Copyright (c) 2013-2016, 2019-2020, 2023 Cong Xu
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	Redistributions of source code must retain the above copyright notice, this
	list of conditions and the following disclaimer.
	Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation
	and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
	ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
	LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
	CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
	SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
	INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
	ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
	POSSIBILITY OF SUCH DAMAGE.
*/
#include "editor_ui_objectives.h"

#include <cdogs/draw/draw.h>
#include <cdogs/draw/draw_actor.h>
#include <cdogs/font.h>

#include "destroy_objective_dialog.h"
#include "editor_ui.h"
#include "editor_ui_common.h"
#include "pickup_objective_dialog.h"

typedef struct
{
	Campaign *Campaign;
	int MissionObjectiveIndex;
} MissionObjectiveData;
static char *MissionGetObjectiveDescription(UIObject *o, void *data)
{
	MissionObjectiveData *mData = data;
	Mission *m = CampaignGetCurrentMission(mData->Campaign);
	if (!m)
	{
		return NULL;
	}
	int i = mData->MissionObjectiveIndex;
	if ((int)m->Objectives.size <= i)
	{
		if (i == 0)
		{
			// first objective and mission has no objectives
			o->u.Textbox.IsEditable = false;
			return "-- mission objectives --";
		}
		return NULL;
	}
	o->u.Textbox.IsEditable = true;
	return ((const Objective *)CArrayGet(&m->Objectives, i))->Description;
}
static void MissionCheckObjectiveDescription(UIObject *o, void *data)
{
	MissionObjectiveData *mData = data;
	Mission *m = CampaignGetCurrentMission(mData->Campaign);
	if (!m)
	{
		o->IsVisible = false;
		return;
	}
	int i = mData->MissionObjectiveIndex;
	if ((int)m->Objectives.size <= i)
	{
		if (i == 0)
		{
			// first objective and mission has no objectives
			o->IsVisible = true;
			return;
		}
		o->IsVisible = false;
		return;
	}
	o->IsVisible = true;
}
static char **MissionGetObjectiveDescriptionSrc(void *data)
{
	MissionObjectiveData *mData = data;
	Mission *m = CampaignGetCurrentMission(mData->Campaign);
	if (!m)
	{
		return NULL;
	}
	int i = mData->MissionObjectiveIndex;
	if ((int)m->Objectives.size <= i)
	{
		return NULL;
	}
	return &((Objective *)CArrayGet(&m->Objectives, i))->Description;
}
static EditorResult MissionChangeObjectiveDescription(void *data, int d)
{
	// Dummy change func to return result file changed
	UNUSED(data);
	UNUSED(d);
	return EDITOR_RESULT_CHANGED;
}
typedef struct
{
	Campaign *co;
	int index;
} MissionIndexData;
static const char *MissionGetObjectiveStr(UIObject *o, void *vData)
{
	UNUSED(o);
	MissionIndexData *data = vData;
	if (!CampaignGetCurrentMission(data->co))
		return NULL;
	if ((int)CampaignGetCurrentMission(data->co)->Objectives.size <=
		data->index)
		return NULL;
	return ObjectiveTypeStr(
		((const Objective *)CArrayGet(
			 &CampaignGetCurrentMission(data->co)->Objectives, data->index))
			->Type);
}
static Objective *GetMissionObjective(const Mission *m, const int idx)
{
	return CArrayGet(&m->Objectives, idx);
}
static const char *MissionGetObjectiveRequired(UIObject *o, void *vData)
{
	static char s[128];
	UNUSED(o);
	MissionIndexData *data = vData;
	if (!CampaignGetCurrentMission(data->co))
		return NULL;
	if ((int)CampaignGetCurrentMission(data->co)->Objectives.size <=
		data->index)
	{
		return NULL;
	}
	sprintf(
		s, "%d",
		GetMissionObjective(CampaignGetCurrentMission(data->co), data->index)
			->Required);
	return s;
}
static const char *MissionGetObjectiveTotal(UIObject *o, void *vData)
{
	static char s[128];
	UNUSED(o);
	MissionIndexData *data = vData;
	if (!CampaignGetCurrentMission(data->co))
		return NULL;
	if ((int)CampaignGetCurrentMission(data->co)->Objectives.size <=
		data->index)
	{
		return NULL;
	}
	sprintf(
		s, "out of %d",
		GetMissionObjective(CampaignGetCurrentMission(data->co), data->index)
			->Count);
	return s;
}
static const char *MissionGetObjectiveFlags(UIObject *o, void *vData)
{
	int flags;
	static char s[128];
	UNUSED(o);
	MissionIndexData *data = vData;
	if (!CampaignGetCurrentMission(data->co))
		return NULL;
	if ((int)CampaignGetCurrentMission(data->co)->Objectives.size <=
		data->index)
	{
		return NULL;
	}
	flags =
		GetMissionObjective(CampaignGetCurrentMission(data->co), data->index)
			->Flags;
	if (!flags)
	{
		return "(normal)";
	}
	sprintf(
		s, "%s %s %s %s %s", (flags & OBJECTIVE_HIDDEN) ? "hidden" : "",
		(flags & OBJECTIVE_POSKNOWN) ? "pos.known" : "",
		(flags & OBJECTIVE_HIACCESS) ? "access" : "",
		(flags & OBJECTIVE_UNKNOWNCOUNT) ? "no-count" : "",
		(flags & OBJECTIVE_NOACCESS) ? "no-access" : "");
	return s;
}

typedef struct
{
	Campaign *C;
	int ObjectiveIdx;
	ObjectiveType Type;
} ObjectiveChangeTypeData;
static void MissionResetObjectiveIndex(Objective *o);
static EditorResult ObjectiveChangeType(void *vData, int d)
{
	UNUSED(d);
	ObjectiveChangeTypeData *data = vData;
	Objective *o = GetMissionObjective(
		CampaignGetCurrentMission(data->C), data->ObjectiveIdx);
	if (o->Type == data->Type)
	{
		return EDITOR_RESULT_NONE;
	}
	o->Type = data->Type;
	MissionResetObjectiveIndex(o);
	return EDITOR_RESULT_CHANGED;
}
static void MissionResetObjectiveIndex(Objective *o)
{
	switch (o->Type)
	{
	case OBJECTIVE_COLLECT:
		ObjectiveSetPickup(o, IntScorePickupClass(0));
		break;
	case OBJECTIVE_DESTROY: {
		const char **destructibleName =
			CArrayGet(&gMapObjects.Destructibles, 0);
		ObjectiveSetDestroy(o, StrMapObject(*destructibleName));
	}
		break;
	default:
		o->u.Index = 0;
		break;
	}
}
static EditorResult MissionChangeObjectiveRequired(void *vData, int d)
{
	MissionIndexData *data = vData;
	Objective *o =
		GetMissionObjective(CampaignGetCurrentMission(data->co), data->index);
	o->Required = CLAMP_OPPOSITE(o->Required + d, 0, MIN(1000, o->Count));
	return EDITOR_RESULT_CHANGED;
}
static EditorResult MissionChangeObjectiveTotal(void *vData, int d)
{
	MissionIndexData *data = vData;
	const Mission *m = CampaignGetCurrentMission(data->co);
	Objective *o = GetMissionObjective(m, data->index);
	if (gEventHandlers.keyboard.modState & KMOD_SHIFT)
	{
		d *= 10;
	}
	o->Count = CLAMP_OPPOSITE(o->Count + d, o->Required, 1000);
	// Don't let the total reduce to less than static ones we've placed
	if (m->Type == MAPTYPE_STATIC)
	{
		CA_FOREACH(const ObjectivePositions, op, m->u.Static.Objectives)
		if (op->Index == data->index)
		{
			o->Count = MAX(o->Count, (int)op->PositionIndices.size);
			break;
		}
		CA_FOREACH_END()
	}
	return EDITOR_RESULT_CHANGED;
}
static EditorResult MissionChangeObjectiveFlags(void *vData, int d)
{
	MissionIndexData *data = vData;
	Objective *o =
		GetMissionObjective(CampaignGetCurrentMission(data->co), data->index);
	// Max is combination of all flags, i.e. largest flag doubled less one
	o->Flags = CLAMP_OPPOSITE(o->Flags + d, 0, OBJECTIVE_NOACCESS * 2 - 1);
	return EDITOR_RESULT_CHANGED;
}

static UIObject *CreateObjectiveObjs(
	struct vec2i pos, Campaign *co, const int idx);
void CreateObjectivesObjs(Campaign *co, UIObject *c, struct vec2i pos)
{
	const int th = FontH();
	struct vec2i objectivesPos = svec2i(0, 7 * th);
	UIObject *o =
		UIObjectCreate(UITYPE_TEXTBOX, 0, svec2i_zero(), svec2i(300, th));
	o->Flags = UI_SELECT_ONLY;

	for (int i = 0; i < OBJECTIVE_MAX_OLD; i++)
	{
		UIObject *o2 = UIObjectCopy(o);
		o2->Id = YC_OBJECTIVES + i;
		o2->Type = UITYPE_TEXTBOX;
		o2->u.Textbox.TextLinkFunc = MissionGetObjectiveDescription;
		o2->u.Textbox.TextSourceFunc = MissionGetObjectiveDescriptionSrc;
		o2->IsDynamicData = true;
		o2->ChangeFunc = MissionChangeObjectiveDescription;
		CMALLOC(o2->Data, sizeof(MissionObjectiveData));
		((MissionObjectiveData *)o2->Data)->Campaign = co;
		((MissionObjectiveData *)o2->Data)->MissionObjectiveIndex = i;
		CSTRDUP(o2->u.Textbox.Hint, "(Objective description)");
		o2->Pos = pos;
		CSTRDUP(
			o2->Tooltip, "Insert/" KMOD_CMD_NAME "+i, Delete/" KMOD_CMD_NAME
						 "+d: add/remove objective");
		o2->CheckVisible = MissionCheckObjectiveDescription;
		UIObjectAddChild(o2, CreateObjectiveObjs(objectivesPos, co, i));
		UIObjectAddChild(c, o2);
		pos.y += th;
	}
	UIObjectDestroy(o);
}
static void CreateObjectiveItemObjs(
	UIObject *c, const struct vec2i pos, Campaign *co, const int idx);
static UIObject *CreateObjectiveObjs(
	struct vec2i pos, Campaign *co, const int idx)
{
	const int th = FontH();
	UIObject *c;
	UIObject *o;
	UIObject *o2;
	c = UIObjectCreate(UITYPE_NONE, 0, svec2i_zero(), svec2i_zero());
	c->Flags = UI_ENABLED_WHEN_PARENT_HIGHLIGHTED_ONLY;

	o = UIObjectCreate(UITYPE_LABEL, 0, svec2i_zero(), svec2i_zero());
	o->Flags = UI_LEAVE_YC;

	pos.y -= idx * th;
	// Drop-down menu for objective type
	o2 = UIObjectCopy(o);
	o2->Size = svec2i(35, th);
	o2->u.LabelFunc = MissionGetObjectiveStr;
	CMALLOC(o2->Data, sizeof(MissionIndexData));
	o2->IsDynamicData = true;
	((MissionIndexData *)o2->Data)->co = co;
	((MissionIndexData *)o2->Data)->index = idx;
	o2->Pos = pos;
	CSTRDUP(o2->Tooltip, "Objective type");
	UIObject *oObjType =
		UIObjectCreate(UITYPE_CONTEXT_MENU, 0, svec2i_zero(), svec2i_zero());
	for (int i = 0; i < (int)OBJECTIVE_MAX; i++)
	{
		UIObject *oObjTypeChild =
			UIObjectCreate(UITYPE_LABEL, 0, svec2i_zero(), svec2i(50, th));
		oObjTypeChild->Pos.y = i * th;
		UIObjectSetDynamicLabel(
			oObjTypeChild, ObjectiveTypeStr((ObjectiveType)i));
		oObjTypeChild->IsDynamicData = true;
		CMALLOC(oObjTypeChild->Data, sizeof(ObjectiveChangeTypeData));
		ObjectiveChangeTypeData *octd = oObjTypeChild->Data;
		octd->C = co;
		octd->ObjectiveIdx = idx;
		octd->Type = (ObjectiveType)i;
		oObjTypeChild->ChangeFunc = ObjectiveChangeType;
		UIObjectAddChild(oObjType, oObjTypeChild);
	}
	UIObjectAddChild(o2, oObjType);
	UIObjectAddChild(c, o2);

	pos.x += 40;
	// Choose objective object/item
	CreateObjectiveItemObjs(c, pos, co, idx);

	pos.x += 30;
	o2 = UIObjectCopy(o);
	o2->u.LabelFunc = MissionGetObjectiveRequired;
	o2->ChangeFunc = MissionChangeObjectiveRequired;
	CMALLOC(o2->Data, sizeof(MissionIndexData));
	o2->IsDynamicData = 1;
	((MissionIndexData *)o2->Data)->co = co;
	((MissionIndexData *)o2->Data)->index = idx;
	o2->Pos = pos;
	o2->Size = svec2i(20, th);
	CSTRDUP(o2->Tooltip, "0: optional objective");
	UIObjectAddChild(c, o2);
	pos.x += 20;
	o2 = UIObjectCopy(o);
	o2->u.LabelFunc = MissionGetObjectiveTotal;
	o2->ChangeFunc = MissionChangeObjectiveTotal;
	CMALLOC(o2->Data, sizeof(MissionIndexData));
	o2->IsDynamicData = 1;
	((MissionIndexData *)o2->Data)->co = co;
	((MissionIndexData *)o2->Data)->index = idx;
	o2->Pos = pos;
	o2->Size = svec2i(40, th);
	UIObjectAddChild(c, o2);
	pos.x += 45;
	o2 = UIObjectCopy(o);
	o2->u.LabelFunc = MissionGetObjectiveFlags;
	o2->ChangeFunc = MissionChangeObjectiveFlags;
	CMALLOC(o2->Data, sizeof(MissionIndexData));
	o2->IsDynamicData = 1;
	((MissionIndexData *)o2->Data)->co = co;
	((MissionIndexData *)o2->Data)->index = idx;
	o2->Pos = pos;
	o2->Size = svec2i(100, th);
	CSTRDUP(
		o2->Tooltip, "hidden: not shown on map\n"
					 "pos.known: always shown on map\n"
					 "access: in locked room\n"
					 "no-count: don't show completed count\n"
					 "no-access: not in locked rooms");
	UIObjectAddChild(c, o2);

	UIObjectDestroy(o);
	return c;
}

static void MissionDrawKillObjective(
	UIObject *o, GraphicsDevice *g, struct vec2i pos, void *vData);
static void MissionDrawCollectObjectives(
	UIObject *o, GraphicsDevice *g, struct vec2i pos, void *vData);
static void MissionDrawDestroyObjectives(
	UIObject *o, GraphicsDevice *g, struct vec2i pos, void *vData);
static void MissionDrawRescueObjective(
	UIObject *o, GraphicsDevice *g, struct vec2i pos, void *vData);
static EditorResult MissionChangeCollectObjectivePickups(void *vData, int d);
static EditorResult MissionChangeDestroyObjectives(void *vData, int d);
static EditorResult MissionChangeRescueObjectiveIndex(void *vData, int d);
static void MissionCheckObjectiveIsKill(UIObject *o, void *vData);
static void MissionCheckObjectiveIsCollect(UIObject *o, void *vData);
static void MissionCheckObjectiveIsDestroy(UIObject *o, void *vData);
static void MissionCheckObjectiveIsRescue(UIObject *o, void *vData);
static void CreateObjectiveItemObjs(
	UIObject *c, const struct vec2i pos, Campaign *co, const int idx)
{
	// TODO: context menu
	UIObject *o = UIObjectCreate(UITYPE_CUSTOM, 0, pos, svec2i(30, 20));
	o->Flags = UI_LEAVE_YC;
	UIObject *o2;

	o2 = UIObjectCopy(o);
	o2->u.CustomDrawFunc = MissionDrawKillObjective;
	o2->IsDynamicData = true;
	CMALLOC(o2->Data, sizeof(MissionIndexData));
	((MissionIndexData *)o2->Data)->co = co;
	((MissionIndexData *)o2->Data)->index = idx;
	o2->CheckVisible = MissionCheckObjectiveIsKill;
	CSTRDUP(o2->Tooltip, "Kill mission objective characters");
	UIObjectAddChild(c, o2);

	o2 = UIObjectCopy(o);
	o2->u.CustomDrawFunc = MissionDrawCollectObjectives;
	o2->ChangeFunc = MissionChangeCollectObjectivePickups;
	o2->IsDynamicData = true;
	CMALLOC(o2->Data, sizeof(MissionIndexData));
	((MissionIndexData *)o2->Data)->co = co;
	((MissionIndexData *)o2->Data)->index = idx;
	o2->CheckVisible = MissionCheckObjectiveIsCollect;
	CSTRDUP(o2->Tooltip, "Choose item(s) to collect...");
	UIObjectAddChild(c, o2);

	o2 = UIObjectCopy(o);
	o2->u.CustomDrawFunc = MissionDrawDestroyObjectives;
	o2->ChangeFunc = MissionChangeDestroyObjectives;
	o2->IsDynamicData = true;
	CMALLOC(o2->Data, sizeof(MissionIndexData));
	((MissionIndexData *)o2->Data)->co = co;
	((MissionIndexData *)o2->Data)->index = idx;
	o2->CheckVisible = MissionCheckObjectiveIsDestroy;
	CSTRDUP(o2->Tooltip, "Choose object(s) to destroy");
	UIObjectAddChild(c, o2);

	o2 = UIObjectCopy(o);
	o2->u.CustomDrawFunc = MissionDrawRescueObjective;
	o2->ChangeFunc = MissionChangeRescueObjectiveIndex;
	o2->IsDynamicData = true;
	CMALLOC(o2->Data, sizeof(MissionIndexData));
	o2->IsDynamicData = true;
	((MissionIndexData *)o2->Data)->co = co;
	((MissionIndexData *)o2->Data)->index = idx;
	o2->CheckVisible = MissionCheckObjectiveIsRescue;
	UIObjectAddChild(c, o2);
}

static void MissionDrawKillObjective(
	UIObject *o, GraphicsDevice *g, struct vec2i pos, void *vData)
{
	MissionIndexData *data = vData;
	UNUSED(g);
	const Mission *m = CampaignGetCurrentMission(data->co);
	if (m == NULL)
		return;
	if ((int)m->Objectives.size <= data->index)
		return;
	// TODO: only one kill and rescue objective allowed
	CharacterStore *store = &data->co->Setting.characters;
	if (store->specialIds.size > 0)
	{
		const Character *c = CArrayGet(
			&store->OtherChars, CharacterStoreGetSpecialId(store, 0));
		const struct vec2i drawPos = svec2i_add(
			svec2i_add(pos, o->Pos), svec2i_scale_divide(o->Size, 2));
		DrawCharacterSimple(c, drawPos, DIRECTION_DOWN, false, true, c->Gun);
	}
}
static void MissionDrawCollectObjectives(
	UIObject *o, GraphicsDevice *g, struct vec2i pos, void *vData)
{
	MissionIndexData *data = vData;
	UNUSED(g);
	const Mission *m = CampaignGetCurrentMission(data->co);
	if (m == NULL)
		return;
	if ((int)m->Objectives.size <= data->index)
		return;
	const Objective *obj = CArrayGet(&m->Objectives, data->index);
	CA_FOREACH(const PickupClass *, pc, obj->u.Pickups)
	const Pic *p = CPicGetPic(&(*pc)->Pic, 0);
	const struct vec2i drawPos =
		svec2i_add(svec2i_add(pos, o->Pos), svec2i_scale_divide(o->Size, 2));
	PicRender(
		p, g->gameWindow.renderer,
		svec2i_subtract(drawPos, svec2i_scale_divide(p->size, 2)), colorWhite,
		0, svec2_one(), SDL_FLIP_NONE, Rect2iZero());
	pos = svec2i_add(pos, svec2i(4, 2));
	CA_FOREACH_END()
}
static void MissionDrawDestroyObjectives(
	UIObject *o, GraphicsDevice *g, struct vec2i pos, void *vData)
{
	MissionIndexData *data = vData;
	UNUSED(g);
	const Mission *m = CampaignGetCurrentMission(data->co);
	if (m == NULL)
		return;
	if ((int)m->Objectives.size <= data->index)
		return;
	const Objective *obj = CArrayGet(&m->Objectives, data->index);
	CA_FOREACH(const MapObject *, mo, obj->u.MapObjects)
	struct vec2i offset;
	const Pic *p = MapObjectGetPic(*mo, &offset);
	const struct vec2i drawPos =
		svec2i_add(svec2i_add(pos, o->Pos), svec2i_scale_divide(o->Size, 2));
	PicRender(
		p, g->gameWindow.renderer,
		svec2i_subtract(drawPos, svec2i_scale_divide(p->size, 2)), colorWhite,
		0, svec2_one(), SDL_FLIP_NONE, Rect2iZero());
	pos = svec2i_add(pos, svec2i(4, 2));
	CA_FOREACH_END()
}
static void MissionDrawRescueObjective(
	UIObject *o, GraphicsDevice *g, struct vec2i pos, void *vData)
{
	MissionIndexData *data = vData;
	UNUSED(g);
	const Mission *m = CampaignGetCurrentMission(data->co);
	if (m == NULL)
		return;
	if ((int)m->Objectives.size <= data->index)
		return;
	// TODO: only one rescue objective allowed
	CharacterStore *store = &data->co->Setting.characters;
	if (store->prisonerIds.size > 0)
	{
		const Character *c = CArrayGet(
			&store->OtherChars, CharacterStoreGetPrisonerId(store, 0));
		const struct vec2i drawPos = svec2i_add(
			svec2i_add(pos, o->Pos), svec2i_scale_divide(o->Size, 2));
		DrawCharacterSimple(c, drawPos, DIRECTION_DOWN, false, true, c->Gun);
	}
}

static EditorResult MissionChangeCollectObjectivePickups(void *vData, int d)
{
	UNUSED(d);
	MissionIndexData *data = vData;
	Objective *o =
		GetMissionObjective(CampaignGetCurrentMission(data->co), data->index);
	return PickupObjectiveDialog(&gPicManager, &gEventHandlers, o);
}
static EditorResult MissionChangeDestroyObjectives(void *vData, int d)
{
	UNUSED(d);
	MissionIndexData *data = vData;
	Objective *o =
		GetMissionObjective(CampaignGetCurrentMission(data->co), data->index);
	return DestroyObjectiveDialog(&gPicManager, &gEventHandlers, o);
}
static EditorResult MissionChangeRescueObjectiveIndex(void *vData, int d)
{
	MissionIndexData *data = vData;
	Objective *o =
		GetMissionObjective(CampaignGetCurrentMission(data->co), data->index);
	int idx = o->u.Index;
	const int limit = (int)(data->co->Setting.characters.OtherChars.size - 1);
	idx = CLAMP_OPPOSITE(idx + d, 0, limit);
	o->u.Index = idx;
	return EDITOR_RESULT_CHANGED;
}

static void MissionCheckObjectiveIsKill(UIObject *o, void *vData)
{
	MissionIndexData *data = vData;
	const Mission *m = CampaignGetCurrentMission(data->co);
	if (m == NULL)
		return;
	if ((int)m->Objectives.size <= data->index)
		return;
	const Objective *obj = CArrayGet(&m->Objectives, data->index);
	o->IsVisible = obj->Type == OBJECTIVE_KILL;
}
static void MissionCheckObjectiveIsCollect(UIObject *o, void *vData)
{
	MissionIndexData *data = vData;
	const Mission *m = CampaignGetCurrentMission(data->co);
	if (m == NULL)
		return;
	if ((int)m->Objectives.size <= data->index)
		return;
	const Objective *obj = CArrayGet(&m->Objectives, data->index);
	o->IsVisible = obj->Type == OBJECTIVE_COLLECT;
}
static void MissionCheckObjectiveIsDestroy(UIObject *o, void *vData)
{
	MissionIndexData *data = vData;
	const Mission *m = CampaignGetCurrentMission(data->co);
	if (m == NULL)
		return;
	if ((int)m->Objectives.size <= data->index)
		return;
	const Objective *obj = CArrayGet(&m->Objectives, data->index);
	o->IsVisible = obj->Type == OBJECTIVE_DESTROY;
}
static void MissionCheckObjectiveIsRescue(UIObject *o, void *vData)
{
	MissionIndexData *data = vData;
	const Mission *m = CampaignGetCurrentMission(data->co);
	if (m == NULL)
		return;
	if ((int)m->Objectives.size <= data->index)
		return;
	const Objective *obj = CArrayGet(&m->Objectives, data->index);
	o->IsVisible = obj->Type == OBJECTIVE_RESCUE;
}
