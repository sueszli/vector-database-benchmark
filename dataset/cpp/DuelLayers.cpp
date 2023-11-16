#include "PrecompiledHeader.h"

#include "MTGRules.h"
#include "CardSelector.h"
#include "GuiCombat.h"
#include "GuiBackground.h"
#include "GuiFrame.h"
#include "GuiPhaseBar.h"
#include "GuiAvatars.h"
#include "GuiHand.h"
#include "GuiPlay.h"
#include "GuiMana.h"
#include "Trash.h"
#include "DuelLayers.h"

void DuelLayers::CheckUserInput(int isAI)
{
    JButton key;
    int x, y;
    JGE* jge = observer->getInput();
    if(!jge) return;

    while ((key = jge->ReadButton()) || jge->GetLeftClickCoordinates(x, y))
    {
        if ((!isAI) && ((0 != key) ||  jge->GetLeftClickCoordinates(x, y)))
        {
            if (stack->CheckUserInput(key)) {
                jge->LeftClickedProcessed();
                break;
            }
            if (combat->CheckUserInput(key)) {
                jge->LeftClickedProcessed();
                break;
            }
            if (avatars->CheckUserInput(key)) {
                jge->LeftClickedProcessed();
                break; //avatars need to check their input before action (CTRL_CROSS)
            }
            if (action->CheckUserInput(key)) {
                jge->LeftClickedProcessed();
                break;
            }
            if (hand->CheckUserInput(key)) {
                jge->LeftClickedProcessed();
                break;
            }
            if (mCardSelector->CheckUserInput(key)) {
                jge->LeftClickedProcessed();
                break;
            }
        }
        jge->LeftClickedProcessed();
    }
}

void DuelLayers::Update(float dt, Player * currentPlayer)
{
    for (int i = 0; i < nbitems; ++i)
        objects[i]->Update(dt);

    int isAI = currentPlayer->isAI() || currentPlayer != getObserver()->players[mPlayerViewIndex]; // Fix for 2 players hand.
    if (isAI && !currentPlayer->getObserver()->isLoading())
        currentPlayer->Act(dt);

    CheckUserInput(isAI);
}

ActionStack * DuelLayers::stackLayer()
{
    return stack;
}

GuiCombat * DuelLayers::combatLayer()
{
    return combat;
}

ActionLayer * DuelLayers::actionLayer()
{
    return action;
}

GuiAvatars * DuelLayers::GetAvatars()
{
    return avatars;
}

DuelLayers::DuelLayers(GameObserver* go, int playerViewIndex) :
    nbitems(0), mPlayerViewIndex(playerViewIndex)
{
    observer = go;
    observer->mLayers = this;
    mCardSelector = NEW CardSelector(go, this);
    //1 Action Layer
    action = NEW ActionLayer(go);
    action->Add(phaseHandler = NEW MTGGamePhase(go, action->getMaxId())); //Phases handler
    action->Add(NEW OtherAbilitiesEventReceiver(go, -1)); //autohand, etc... handler
    //Other display elements
    action->Add(NEW HUDDisplay(go, -1));

    Add(NEW GuiMana(20, 20, getRenderedPlayerOpponent()));
    Add(NEW GuiMana(440, 20, getRenderedPlayer()));
    Add(stack = NEW ActionStack(go));
    Add(combat = NEW GuiCombat(go));
    Add(action);
    Add(mCardSelector);
    Add(hand = NEW GuiHandSelf(go, getRenderedPlayer()->game->hand));
    Add(avatars = NEW GuiAvatars(this));
    Add(NEW GuiHandOpponent(go, getRenderedPlayerOpponent()->game->hand));
    Add(NEW GuiPlay(this));
    Add(NEW GuiPhaseBar(this));
    Add(NEW GuiFrame(go));
    Add(NEW GuiBackground(go));
}

DuelLayers::~DuelLayers()
{
    int _nbitems = nbitems;
    nbitems = 0;
    for (int i = 0; i < _nbitems; ++i)
    {
        if (objects[i] != mCardSelector)
        {
            SAFE_DELETE(objects[i]);
            objects[i] = NULL;
        }
    }

    for (size_t i = 0; i < waiters.size(); ++i)
        delete (waiters[i]);
    observer->mTrash->cleanup();

    SAFE_DELETE(mCardSelector);
}

void DuelLayers::Add(GuiLayer * layer)
{
    objects.push_back(layer);
    nbitems++;
}

void DuelLayers::Remove()
{
    --nbitems;
}

void DuelLayers::Render()
{
    bool focusMakesItThrough = true;
    for (int i = 0; i < nbitems; ++i)
    {
        objects[i]->hasFocus = focusMakesItThrough;
        if (objects[i]->modal)
            focusMakesItThrough = false;
    }
    for (int i = nbitems - 1; i >= 0; --i)
        objects[i]->Render();
}

int DuelLayers::receiveEvent(WEvent * e)
{

#if 0
#define PRINT_IF(type) { type *foo = dynamic_cast<type*>(e); if (foo) cout << "Is a " #type " " << *foo << endl; }
    cout << "Received event " << e << " ";
    PRINT_IF(WEventZoneChange);
    PRINT_IF(WEventDamage);
    PRINT_IF(WEventPhaseChange);
    PRINT_IF(WEventCardUpdate);
    PRINT_IF(WEventCardTap);
    PRINT_IF(WEventCreatureAttacker);
    PRINT_IF(WEventCreatureBlocker);
    PRINT_IF(WEventCreatureBlockerRank);
    PRINT_IF(WEventCombatStepChange);
    PRINT_IF(WEventEngageMana);
    PRINT_IF(WEventConsumeMana);
    PRINT_IF(WEventEmptyManaPool);
#endif

    int used = 0;
    for (int i = 0; i < nbitems; ++i)
        used |= objects[i]->receiveEventPlus(e);
    if (!used)
    {
        Pos* p;
        if (WEventZoneChange *event = dynamic_cast<WEventZoneChange*>(e))
        {
            MTGCardInstance* card = event->card;
            if (card->view)
                waiters.push_back(p = NEW Pos(*(card->view)));
            else
                waiters.push_back(p = NEW Pos(0, 0, 0, 0, 255));
            const Pos* ref = card->view;
            while (card)
            {
                if (ref == card->view)
                    card->view = p;
                card = card->next;
            }
        }
    }
    for (int i = 0; i < nbitems; ++i)
        objects[i]->receiveEventMinus(e);

    if (WEventPhaseChange *event = dynamic_cast<WEventPhaseChange*>(e))
        if (MTG_PHASE_BEFORE_BEGIN == event->to->id)
            observer->mTrash->cleanup();

    return 1;
}

float DuelLayers::RightBoundary()
{
    return MIN (hand->LeftBoundary(), avatars->LeftBoundarySelf());
}

Player* DuelLayers::getRenderedPlayer()
{
    return observer->players[mPlayerViewIndex]; 
};

Player* DuelLayers::getRenderedPlayerOpponent()
{ 
    return observer->players[mPlayerViewIndex]->opponent(); 
};
