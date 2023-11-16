#include "PrecompiledHeader.h"

#include "ActionElement.h"
#include "MTGCardInstance.h"
#include "Targetable.h"
#include "TargetChooser.h"

ActionElement::ActionElement(int id) :
    JGuiObject(id)
{
    activity = Inactive;
    modal = 0;
    waitingForAnswer = 0;
    currentPhase = MTG_PHASE_INVALID;
    newPhase = MTG_PHASE_INVALID;
    tc = NULL;
}

ActionElement::ActionElement(const ActionElement& a): JGuiObject(a)
{
    activity = a.activity;
    tc = a.tc ? a.tc->clone() : NULL;
    currentPhase = a.currentPhase;
    newPhase = a.newPhase;
    modal = a.modal;
    waitingForAnswer = a.waitingForAnswer;
}

ActionElement::~ActionElement()
{
    SAFE_DELETE(tc);
}

int ActionElement::isReactingToTargetClick(Targetable * object)
{
    if (MTGCardInstance * cObject = dynamic_cast<MTGCardInstance *>(object))
        return isReactingToClick(cObject);
    return 0;
}

int ActionElement::reactToTargetClick(Targetable * object)
{
    if (MTGCardInstance * cObject = dynamic_cast<MTGCardInstance *>(object))
        return reactToClick(cObject);
    return 0;
}
