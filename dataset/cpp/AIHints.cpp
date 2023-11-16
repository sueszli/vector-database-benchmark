#include "PrecompiledHeader.h"

#include "AIHints.h"
#include "AIPlayerBaka.h"
#include "utils.h"
#include "AllAbilities.h"

#include <sstream>

AIHint::AIHint(string _line)
{
    string line = _line;
    if (!line.length())
    {
        DebugTrace("AIHINTS: line is empty");
        return;
    }
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    mCondition = line;
    string action = line;
    vector<string> splitAction = parseBetween(action, "sourceid(", ")");
    if (splitAction.size())
    {
        mAction = splitAction[0];
        mSourceId = atoi(splitAction[1].c_str()); 
    }
    else
    {
        mAction = action;
        mSourceId = 0;
    }
    
    vector<string> splitDontAttack = parseBetween(action, "dontattackwith(", ")");
    if(splitDontAttack.size())
    {
        mCombatAttackTip = splitDontAttack[1];
    }

    vector<string> splitAlwaysAttack = parseBetween(action, "alwaysattackwith(", ")");
    if(splitAlwaysAttack.size())
    {
        mCombatAlwaysAttackTip = splitAlwaysAttack[1];
    }

    vector<string> splitDontBlock = parseBetween(action, "dontblockwith(", ")");
    if(splitDontBlock.size())
    {
        mCombatBlockTip = splitDontBlock[1];
    }

    vector<string> splitAlwaysBlock = parseBetween(action, "alwaysblockwith(", ")");
    if(splitAlwaysBlock.size())
    {
        mCombatAlwaysBlockTip = splitAlwaysBlock[1];
    }

    vector<string> splitSetEffgood = parseBetween(action, "good(", ")");
    if(splitSetEffgood.size())
    {
        mCardEffGood = splitSetEffgood[1];
    }

    vector<string> splitSetEffbad = parseBetween(action, "bad(", ")");
    if(splitSetEffbad.size())
    {
        mCardEffBad = splitSetEffbad[1];
    }

    vector<string> splitCastOrder = parseBetween(action, "castpriority(", ")");
    if(splitCastOrder.size())
    {
        castOrder = split(splitCastOrder[1],',');
    }

    if(action.find( "combo ") != string::npos)
    {
        string Combo = action.c_str() + 6;
        combos.push_back(Combo);
    }
    
}

AIHints::AIHints(AIPlayerBaka * player): mPlayer(player)
{
}

void AIHints::add(string line)
{
    hints.push_back(NEW AIHint(line)); 
}

AIHints::~AIHints()
{
    for (size_t i = 0; i < hints.size(); ++i)
        SAFE_DELETE(hints[i]);
    hints.clear();
}

AIHint * AIHints::getByCondition (string condition)
{
    if (!condition.size())
        return NULL;

    for (size_t i = 0; i < hints.size(); ++i)
    {
        if (hints[i]->mCondition.compare(condition) == 0)
            return hints[i];
    }
    return NULL;
}

bool AIHints::HintSaysDontAttack(GameObserver* observer,MTGCardInstance * card)
{
    TargetChooserFactory tfc(observer);
    TargetChooser * hintTc = NULL;
    for(unsigned int i = 0; i < hints.size();i++)
    {
        if (hints[i]->mCombatAttackTip.size())
        {
            hintTc = tfc.createTargetChooser(hints[i]->mCombatAttackTip,card);
            if(hintTc && hintTc->canTarget(card,true))
            {
                SAFE_DELETE(hintTc);
                return true;
            }
            SAFE_DELETE(hintTc);
        }
    }
    return false;
}

bool AIHints::HintSaysAlwaysAttack(GameObserver* observer,MTGCardInstance * card)
{
    TargetChooserFactory tfc(observer);
    TargetChooser * hintTc = NULL;
    for(unsigned int i = 0; i < hints.size();i++)
    {
        if (hints[i]->mCombatAlwaysAttackTip.size())
        {
            hintTc = tfc.createTargetChooser(hints[i]->mCombatAlwaysAttackTip,card);
            if(hintTc && hintTc->canTarget(card,true))
            {
                SAFE_DELETE(hintTc);
                return true;
            }
            SAFE_DELETE(hintTc);
        }
    }
    return false;
}

bool AIHints::HintSaysDontBlock(GameObserver* observer,MTGCardInstance * card)
{
    TargetChooserFactory tfc(observer);
    TargetChooser * hintTc = NULL;
    for(unsigned int i = 0; i < hints.size();i++)
    {
        if (hints[i]->mCombatBlockTip.size())
        {
            hintTc = tfc.createTargetChooser(hints[i]->mCombatBlockTip,card);
            if(hintTc && hintTc->canTarget(card,true))
            {
                SAFE_DELETE(hintTc);
                return true;
            }
            SAFE_DELETE(hintTc);
        }
    }
    return false;
}

bool AIHints::HintSaysAlwaysBlock(GameObserver* observer,MTGCardInstance * card)
{
    TargetChooserFactory tfc(observer);
    TargetChooser * hintTc = NULL;
    for(unsigned int i = 0; i < hints.size();i++)
    {
        if (hints[i]->mCombatAlwaysBlockTip.size())
        {
            hintTc = tfc.createTargetChooser(hints[i]->mCombatAlwaysBlockTip,card);
            if(hintTc && hintTc->canTarget(card,true))
            {
                SAFE_DELETE(hintTc);
                return true;
            }
            SAFE_DELETE(hintTc);
        }
    }
    return false;
}

bool AIHints::HintSaysCardIsGood(GameObserver* observer,MTGCardInstance * card)
{
    TargetChooserFactory tfc(observer);
    TargetChooser * hintTc = NULL;
    for(unsigned int i = 0; i < hints.size();i++)
    {
        if (hints[i]->mCardEffGood.size())
        {
            hintTc = tfc.createTargetChooser(hints[i]->mCardEffGood,card);
            if(hintTc && hintTc->canTarget(card,true))
            {
                SAFE_DELETE(hintTc);
                return true;
            }
            SAFE_DELETE(hintTc);
        }
    }
    return false;
}

bool AIHints::HintSaysCardIsBad(GameObserver* observer,MTGCardInstance * card)
{
    TargetChooserFactory tfc(observer);
    TargetChooser * hintTc = NULL;
    for(unsigned int i = 0; i < hints.size();i++)
    {
        if (hints[i]->mCardEffBad.size())
        {
            hintTc = tfc.createTargetChooser(hints[i]->mCardEffBad,card);
            if(hintTc && hintTc->canTarget(card,true))
            {
                SAFE_DELETE(hintTc);
                return true;
            }
            SAFE_DELETE(hintTc);
        }
    }
    return false;
}

bool AIHints::HintSaysItsForCombo(GameObserver* observer,MTGCardInstance * card)
{
    TargetChooserFactory tfc(observer);
    TargetChooser * hintTc = NULL;
    bool forCombo = false;
    for(unsigned int i = 0; i < hints.size();i++)
    {
        if (hints[i]->combos.size())
        {
            //time to find the parts and condiations of the combo.
            string part = "";
            if(!hints[i]->partOfCombo.size() && hints[i]->combos.size())
            {
                for(unsigned int cPart = 0; cPart < hints[i]->combos.size(); cPart++)
                {
                    //here we disect the different parts of a given combo
                    part = hints[i]->combos[cPart];
                    hints[i]->partOfCombo = split(part,'^');
                    for(int dPart = int(hints[i]->partOfCombo.size()-1);dPart >= 0;dPart--)
                    {
                        vector<string>asTc;
                        asTc = parseBetween(hints[i]->partOfCombo[dPart],"hold(",")");
                        if(asTc.size())
                        {
                            hints[i]->hold.push_back(asTc[1]);
                            asTc.clear();
                        }
                        asTc = parseBetween(hints[i]->partOfCombo[dPart],"until(",")");
                        if(asTc.size())
                        {
                            hints[i]->until.push_back(asTc[1]);
                            asTc.clear();
                        }
                        asTc = parseBetween(hints[i]->partOfCombo[dPart],"restriction{","}");
                        if(asTc.size())
                        {
                            hints[i]->restrict.push_back(asTc[1]);
                            asTc.clear();
                        }
                        asTc = parseBetween(hints[i]->partOfCombo[dPart],"cast(",")");
                        if(asTc.size())
                        {
                            hints[i]->casting.push_back(asTc[1]);
                            vector<string>cht = parseBetween(hints[i]->partOfCombo[dPart],"targeting(",")");
                            if(cht.size())
                            hints[i]->cardTargets[asTc[1]] = cht[1];
                        }
                        asTc = parseBetween(hints[i]->partOfCombo[dPart],"totalmananeeded(",")");
                        if(asTc.size())
                        {
                            hints[i]->manaNeeded = asTc[1];
                            asTc.clear();
                        }
                        if(dPart == 0)
                            break;
                    }
                }
            }//we collect the peices of a combo on first run.
            for(unsigned int hPart = 0; hPart < hints[i]->hold.size(); hPart++)
            {
                hintTc = tfc.createTargetChooser(hints[i]->hold[hPart],card);
                if(hintTc && hintTc->canTarget(card,true))
                {
                    forCombo = true;
                }
                SAFE_DELETE(hintTc);
            }
        }
    }
    return forCombo;//return forCombo that way all hints that are combos are predisected.
}
//if it's not part of a combo or there is more to gather, then return false
bool AIHints::canWeCombo(GameObserver* observer,MTGCardInstance * card,AIPlayerBaka * Ai)
{
    TargetChooserFactory tfc(observer);
    TargetChooser * hintTc = NULL;
    bool gotCombo = false;

    for(unsigned int i = 0; i < hints.size();i++)
    {
        int comboPartsHold = 0;
        int comboPartsUntil = 0;
        int comboPartsRestriction = 0;

        if(gotCombo)
            return gotCombo;//because more than one might be possible at any time.
        if (hints[i]->hold.size())
        {
            for(unsigned int hPart = 0; hPart < hints[i]->hold.size(); hPart++)
            {
                hintTc = tfc.createTargetChooser(hints[i]->hold[hPart],card);
                int TcCheck = hintTc->countValidTargets();
                if(hintTc && TcCheck >= hintTc->maxtargets)
                {
                    comboPartsHold +=1;
                }
                SAFE_DELETE(hintTc);
            }
        }
        if (hints[i]->until.size())
        {
            for(unsigned int hPart = 0; hPart < hints[i]->until.size(); hPart++)
            {
                hintTc = tfc.createTargetChooser(hints[i]->until[hPart],card);
                int TcCheck = hintTc->countValidTargets();
                if(hintTc && TcCheck >= hintTc->maxtargets)
                {
                    comboPartsUntil +=1;
                }
                SAFE_DELETE(hintTc);
            }
        }
        if (hints[i]->restrict.size())
        {
            for(unsigned int hPart = 0; hPart < hints[i]->restrict.size(); hPart++)
            {
                AbilityFactory af(observer);
                int checkCond = af.parseCastRestrictions(card,card->controller(),hints[i]->restrict[hPart]);
                if(checkCond >= 1)
                {
                    comboPartsRestriction +=1;
                }
            }
        }
        if( comboPartsUntil >= int(hints[i]->until.size()) && comboPartsHold >= int(hints[i]->hold.size()) && comboPartsRestriction >= int(hints[i]->restrict.size()) && hints[i]->combos.size() )
        {
            ManaCost * needed = ManaCost::parseManaCost(hints[i]->manaNeeded, NULL, card);
            if(Ai->canPayManaCost(card,needed).size()||!needed->getConvertedCost())
            {
                gotCombo = true;
                Ai->comboHint = hints[i];//set the combo we are doing.
            }
            SAFE_DELETE(needed);
        }
    }
    return gotCombo;
}


vector<string>AIHints::mCastOrder()
{
    for(unsigned int i = 0; i < hints.size();i++)
    {
        if (hints[i]->castOrder.size())
        {
            return hints[i]->castOrder;
        }
    }
    return vector<string>();
}

//return true if a given ability matches a hint's description
//Eventually this will look awfully similar to the parser...any way to merge them somehow ?
bool AIHints::abilityMatches(MTGAbility * ability, AIHint * hint)
{
    string s = hint->mAction;

    MTGAbility * a = AbilityFactory::getCoreAbility(ability);

    //Here we want to check that the card reacting to the MTGAbility is the one mentioned in the hint,
    // to avoid mistaking the MTGAbility with a similar one.
    //Ideally we would find all cards with this ID, and see if the ability reacts to a click on one of these cards.
    // This is the poor man's version, based on the fact that most cards are the source of their own abilities
    if (hint->mSourceId && ((!a->source) || a->source->getMTGId() != hint->mSourceId))
        return false;

    if ( AACounter * counterAbility = dynamic_cast<AACounter *> (a) )
    {
        vector<string> splitCounter = parseBetween(s, "counter(", ")");
        if (!splitCounter.size())
            return false;

        string counterstring = counterAbility->name;
        std::transform(counterstring.begin(), counterstring.end(), counterstring.begin(), ::tolower);
        return (splitCounter[1].compare(counterstring) == 0);
    }

    if ( ATokenCreator * tokenAbility = dynamic_cast<ATokenCreator *> (a) )
    {
        vector<string> splitToken = parseBetween(s, "token(", ")");
        if (!splitToken.size())
            return false;
        return (tokenAbility->tokenId && tokenAbility->tokenId == atoi(splitToken[1].c_str()));
    }

    return false;
}

//Finds all mtgAbility matching the Hint description
// For now we limit findings
vector<MTGAbility *> AIHints::findAbilities(AIHint * hint)
{
    std::vector<MTGAbility *> elems;
    ActionLayer * al = mPlayer->getObserver()->mLayers->actionLayer();

    for (size_t i = 1; i < al->mObjects.size(); i++) //0 is not a mtgability...hackish
    { 
        MTGAbility * a = ((MTGAbility *) al->mObjects[i]);
        if (abilityMatches(a, hint))
            elems.push_back(a);
    }
    return elems;

}

//Finds a mtgAbility matching the Hint description, and returns a valid AIAction matching this mtgability
RankingContainer AIHints::findActions(AIHint * hint)
{
    RankingContainer ranking;

    vector<MTGAbility *> abilities = findAbilities(hint);

    for (size_t i = 0; i < abilities.size(); ++i)
    {
        MTGAbility * a = abilities[i];

        for (int j = 0; j < mPlayer->game->inPlay->nb_cards; j++)
        {
            MTGCardInstance * card = mPlayer->game->inPlay->cards[j];
            if (a->isReactingToClick(card, a->getCost()))
            {
                mPlayer->createAbilityTargets(a, card, ranking); //TODO make that function static?
                break; //For performance... ?
            }
        }
    }

    return ranking;
}

//Returns true if a card with the given MTG ID exists
bool AIHints::findSource(int sourceId)
{
    for (int i = 0; i < mPlayer->game->inPlay->nb_cards; i++)
    {
        MTGCardInstance * c = mPlayer->game->inPlay->cards[i];
        if (c->getMTGId() == sourceId)
            return true;
    }
    return false;
}

string AIHints::constraintsNotFulfilled(AIAction * action, AIHint * hint, ManaCost * potentialMana)
{
    std::stringstream out;

    if (!action)
    {
        if (hint->mCombatAttackTip.size())
        {
            out << "to see if this can attack[" << hint->mCombatAttackTip << "]";
            return out.str();
        }
        if (hint->mCombatBlockTip.size())
        {
            out << "to see if this can block[" << hint->mCombatBlockTip << "]";
            return out.str();
        }
        if (hint->mSourceId && !findSource(hint->mSourceId))
        {
            out << "needcardinplay[" << hint->mSourceId << "]";
            return out.str();
        }
        out << "needability[" << hint->mAction << "]";
        return out.str();      
    }

    MTGAbility * a = action->ability;
    if (!a)
        return "not supported";

    MTGCardInstance * card = action->click;
    if (!card)
        return "not supported";

    //dummy test: would the ability work if we were sure to fulfill its mana requirements?
    if (!a->isReactingToClick(card, a->getCost()))
    {
        DebugTrace("This shouldn't happen, this AIAction doesn't seem like a good choice");
        return "not supported";
    }
    
    if (!a->isReactingToClick(card, potentialMana))
    {
        //Not enough Mana, try to find which mana we should get in priority
        ManaCost * diff = potentialMana->Diff(a->getCost());
        for (int i = 0; i < Constants::NB_Colors; i++)
        {
            if(diff->getCost(i) < 0)
            {
                out << "needmana[" <<  Constants::MTGColorChars[i] << "]";
                if (Constants::MTGColorChars[i] == 'r')
                    DebugTrace("Got it");
                SAFE_DELETE(diff);
                return out.str();
            }

        }

        //TODO, handle more cases where the cost cannot be paid
        return "not supported, can't afford cost for some reason";
    }

    //No problem found, we believe this is a good action to perform
    return "";

}

AIAction * AIHints::findAbilityRecursive(AIHint * hint, ManaCost * potentialMana)
{
        RankingContainer ranking = findActions(hint);

        AIAction * a = NULL;
        if (ranking.size())
        {
            a = NEW AIAction(ranking.begin()->first);
        }

        string s = constraintsNotFulfilled(a, hint, potentialMana);
        if (hint->mCombatAttackTip.size() || hint->mCombatBlockTip.size() || hint->castOrder.size())
            return NULL;
        if (s.size())
        {
            SAFE_DELETE(a);
            AIHint * nextHint = getByCondition(s);
            DebugTrace("**I Need " << s << ", this can be provided by " << (nextHint ? nextHint->mAction : "NULL") << "\n\n");
            if (nextHint && nextHint != hint)
                return findAbilityRecursive(nextHint, potentialMana);
            return NULL;
        }

        return a;

}

AIAction * AIHints::suggestAbility(ManaCost * potentialMana)
{
    for (size_t i = 0; i < hints.size(); ++i)
    {
        //Don't suggest abilities that require a condition, for now
        if (hints[i]->mCondition.size())
            continue;

        AIAction * a = findAbilityRecursive(hints[i], potentialMana);
        if (a)
        {  
            DebugTrace("**I Decided that the best to fulfill " << hints[i]->mAction << " is to play " << a->ability->getMenuText() << "\n\n");
            return a;
        }

    }
    return NULL;
}
