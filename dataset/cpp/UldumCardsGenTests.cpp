// Copyright (c) 2017-2023 Chris Ohk

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Utils/CardSetHeaders.hpp>

// ------------------------------------------ SPELL - DRUID
// [ULD_131] Untapped Potential - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> End 4 turns with any unspent Mana.
//       <b>Reward:</b> Ossirian Tear.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 4
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 53499
// --------------------------------------------------------
TEST_CASE("[Druid : Spell] - ULD_131 : Untapped Potential")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(1);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());
    const auto curSecret = curPlayer->GetSecretZone();

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Untapped Potential"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Druid of the Claw"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Starfall"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Malygos"));

    auto quest = dynamic_cast<Spell*>(card1);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 0);
    CHECK_EQ(quest->GetQuestProgressTotal(), 4);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(quest->GetQuestProgress(), 0);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(quest->GetQuestProgress(), 1);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(quest->GetQuestProgress(), 2);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(quest->GetQuestProgress(), 3);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK(!curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 4);
    CHECK_EQ(curPlayer->GetHero()->heroPower->card->id, "ULD_131p");

    game.Process(opPlayer, PlayCardTask::Minion(card4));

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->GetAttack(), 5);
    CHECK_EQ(curField[0]->GetHealth(), 6);
    CHECK_EQ(curField[0]->HasRush(), true);
    CHECK_EQ(curField[0]->HasTaunt(), true);

    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);

    game.Process(curPlayer, PlayCardTask::SpellTarget(card3, card4));
    CHECK_EQ(opField[0]->GetHealth(), 5);
}

// ----------------------------------------- MINION - DRUID
// [ULD_133] Crystal Merchant - COST:2 [ATK:1/HP:4]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: If you have any unspent Mana at the end of your turn,
//       draw a card.
// --------------------------------------------------------
TEST_CASE("[Druid : Minion] - ULD_133 : Crystal Merchant")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& opHand = *(opPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Crystal Merchant"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Pyroblast"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 5);
    CHECK_EQ(opHand.GetCount(), 5);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(curHand.GetCount(), 6);
    CHECK_EQ(opHand.GetCount(), 6);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(curHand.GetCount(), 7);
    CHECK_EQ(opHand.GetCount(), 6);

    game.Process(curPlayer,
                 PlayCardTask::SpellTarget(card2, opPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetRemainingMana(), 0);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(curHand.GetCount(), 6);
    CHECK_EQ(opHand.GetCount(), 7);
}

// ------------------------------------------ SPELL - DRUID
// [ULD_134] BEEEES!!! - COST:3 [ATK:1/HP:4]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: Choose a minion. Summon four 1/1 Bees that attack it.
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// - REQ_MINION_TARGET = 0
// - REQ_NUM_MINION_SLOTS = 1
// --------------------------------------------------------
TEST_CASE("[Druid : Spell] - ULD_134 : BEEEES!!!")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::DRUID;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Frostwolf Grunt"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("River Crocolisk"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("BEEEES!!!"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card3, card2));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(opField.GetCount(), 1);
    CHECK_EQ(opField[0]->card->name, "Bee");
}

// ------------------------------------------ SPELL - DRUID
// [ULD_135] Hidden Oasis - COST:6
// - Set: Uldum, Rarity: Rare
// - Spell School: Nature
// --------------------------------------------------------
// Text: <b>Choose One</b> - Summon a 6/6 Ancient with <b>Taunt</b>;
//       or Restore 12 Health.
// --------------------------------------------------------
// GameTag:
// - CHOOSE_ONE = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// --------------------------------------------------------
// RefTag:
// - TAUNT = 1
// --------------------------------------------------------
TEST_CASE("[Druid : Spell] - ULD_135 : Hidden Oasis")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);
    curPlayer->GetHero()->SetDamage(15);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Hidden Oasis"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Hidden Oasis"));

    game.Process(curPlayer, PlayCardTask::Spell(card1, 1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->card->name, "Vir'naal Ancient");

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer,
                 PlayCardTask::SpellTarget(card2, curPlayer->GetHero(), 2));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 27);
}

// ------------------------------------------ SPELL - DRUID
// [ULD_136] Worthy Expedition - COST:1
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Discover</b> a <b>Choose One</b> card.
// --------------------------------------------------------
// GameTag:
// - DISCOVER = 1
// - USE_DISCOVER_VISUALS = 1
// --------------------------------------------------------
TEST_CASE("[Druid : Spell] - ULD_136 : Worthy Expedition")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Worthy Expedition"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curPlayer->choice);

    const auto cards = TestUtils::GetChoiceCards(game);
    for (auto& card : cards)
    {
        CHECK_EQ(card->HasGameTag(GameTag::CHOOSE_ONE), true);
    }
}

// ----------------------------------------- MINION - DRUID
// [ULD_137] Garden Gnome - COST:4 [ATK:2/HP:3]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If you're holding a spell that
//       costs (5) or more, summon two 2/2 Treants.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Druid : Minion] - ULD_137 : Garden Gnome")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Garden Gnome"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Garden Gnome"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Starfire"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[0]->card->name, "Treant");
    CHECK_EQ(curField[1]->card->name, "Garden Gnome");
    CHECK_EQ(curField[2]->card->name, "Treant");

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer,
                 PlayCardTask::SpellTarget(card3, opPlayer->GetHero()));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 4);
    CHECK_EQ(curField[3]->card->name, "Garden Gnome");
}

// ----------------------------------------- MINION - DRUID
// [ULD_138] Anubisath Defender - COST:5 [ATK:3/HP:5]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Taunt</b>. Costs (0) if you've cast a spell that
//       costs (5) or more this turn.
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// --------------------------------------------------------
TEST_CASE("[Druid : Minion] - ULD_138 : Anubisath Defender")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Anubisath Defender"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Starfire"));

    CHECK_EQ(card1->GetCost(), 5);

    game.Process(curPlayer,
                 PlayCardTask::SpellTarget(card2, opPlayer->GetHero()));
    CHECK_EQ(card1->GetCost(), 0);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(card1->GetCost(), 5);
}

// ----------------------------------------- MINION - DRUID
// [ULD_139] Elise the Enlightened - COST:5 [ATK:5/HP:5]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If your deck has no duplicates,
//       duplicate your hand.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Druid : Minion] - ULD_139 : Elise the Enlightened")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Elise the Enlightened"));
    [[maybe_unused]] const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    [[maybe_unused]] const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    [[maybe_unused]] const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Moonfire"));
    [[maybe_unused]] const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wrath"));
    [[maybe_unused]] const auto card6 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Acornbearer"));
    [[maybe_unused]] const auto card7 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Ironbark Protector"));

    CHECK_EQ(curHand.GetCount(), 7);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 10);
    CHECK_EQ(curHand[6]->card->name, "Wisp");
    CHECK_EQ(curHand[7]->card->name, "Wolfrider");
    CHECK_EQ(curHand[8]->card->name, "Moonfire");
    CHECK_EQ(curHand[9]->card->name, "Wrath");
}

// ---------------------------------------- MINION - SHAMAN
// [ULD_158] Sandstorm Elemental - COST:2 [ATK:2/HP:2]
// - Race: Elemental, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Deal 1 damage to all enemy minions.
//       <b>Overload:</b> (1)
// --------------------------------------------------------
// GameTag:
// - OVERLOAD = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Shaman : Minion] - ULD_158 : Sandstorm Elemental")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Silverback Patriarch"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 = Generic::DrawCard(
        opPlayer, Cards::FindCardByName("Sandstorm Elemental"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card4));
    game.Process(opPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 3);
    CHECK_EQ(opField.GetCount(), 2);
    CHECK_EQ(opPlayer->GetRemainingMana(), 5);
    CHECK_EQ(opPlayer->GetOverloadOwed(), 1);
    CHECK_EQ(opPlayer->GetOverloadLocked(), 0);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(opPlayer->GetRemainingMana(), 9);
    CHECK_EQ(opPlayer->GetOverloadOwed(), 0);
    CHECK_EQ(opPlayer->GetOverloadLocked(), 1);
}

// ---------------------------------------- MINION - SHAMAN
// [ULD_169] Mogu Fleshshaper - COST:9 [ATK:3/HP:4]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Rush</b>. Costs (1) less for each minion
//       on the battlefield.
// --------------------------------------------------------
// GameTag:
// - RUSH = 1
// --------------------------------------------------------
TEST_CASE("[Shaman : Minion] - ULD_169 : Mogu Fleshshaper")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mogu Fleshshaper"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));
    const auto card6 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));

    CHECK_EQ(card1->GetCost(), 9);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    game.Process(curPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(card1->GetCost(), 6);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card5));
    game.Process(opPlayer, PlayCardTask::Minion(card6));
    CHECK_EQ(card1->GetCost(), 4);

    game.Process(opPlayer, HeroPowerTask(card2));
    CHECK_EQ(card1->GetCost(), 5);
}

// ---------------------------------------- MINION - SHAMAN
// [ULD_170] Weaponized Wasp - COST:3 [ATK:3/HP:3]
// - Race: Beast, Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If you control a <b>Lackey</b>,
//       deal 3 damage.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_IF_AVAILABLE = 0
// --------------------------------------------------------
// RefTag:
// - MARK_OF_EVIL = 1
// --------------------------------------------------------
TEST_CASE("[Shaman : Minion] - ULD_170 : Weaponized Wasp")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto opHero = opPlayer->GetHero();

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByID("DAL_739"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Weaponized Wasp"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Weaponized Wasp"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Weaponized Wasp"));

    game.Process(curPlayer, PlayCardTask::MinionTarget(card2, opHero));
    CHECK_EQ(opHero->GetHealth(), 30);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card1, card2));
    game.Process(curPlayer, PlayCardTask::MinionTarget(card3, opHero));
    CHECK_EQ(opHero->GetHealth(), 27);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card4, card3));
    CHECK_EQ(curField.GetCount(), 3);
}

// ----------------------------------------- SPELL - SHAMAN
// [ULD_171] Totemic Surge - COST:0
// - Set: Uldum, Rarity: Common
// - Spell School: Nature
// --------------------------------------------------------
// Text: Give your Totems +2 Attack.
// --------------------------------------------------------
TEST_CASE("[Shaman : Spell] - ULD_171 : Totemic Surge")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Totemic Surge"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));

    game.Process(curPlayer, HeroPowerTask());

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, HeroPowerTask());
    int totem1Attack = curField[0]->GetAttack();
    int totem2Attack = curField[1]->GetAttack();

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curField[0]->GetAttack(), totem1Attack + 2);
    CHECK_EQ(curField[1]->GetAttack(), totem2Attack + 2);
    CHECK_EQ(curField[2]->GetAttack(), 3);
}

// ----------------------------------------- SPELL - SHAMAN
// [ULD_172] Plague of Murlocs - COST:3
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: Transform all minions into random Murlocs.
// --------------------------------------------------------
TEST_CASE("[Shaman : Spell] - ULD_172 : Plague of Murlocs")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Plague of Murlocs"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Ironfur Grizzly"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Faerie Dragon"));
    const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card6 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Ironfur Grizzly"));
    const auto card7 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Faerie Dragon"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    game.Process(curPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(curField.GetCount(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card5));
    game.Process(opPlayer, PlayCardTask::Minion(card6));
    game.Process(opPlayer, PlayCardTask::Minion(card7));
    CHECK_EQ(opField.GetCount(), 3);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[0]->IsRace(Race::MURLOC), true);
    CHECK_EQ(curField[1]->IsRace(Race::MURLOC), true);
    CHECK_EQ(curField[2]->IsRace(Race::MURLOC), true);
    CHECK_EQ(opField.GetCount(), 3);
    CHECK_EQ(opField[0]->IsRace(Race::MURLOC), true);
    CHECK_EQ(opField[1]->IsRace(Race::MURLOC), true);
    CHECK_EQ(opField[2]->IsRace(Race::MURLOC), true);
}

// ---------------------------------------- MINION - SHAMAN
// [ULD_173] Vessina - COST:4 [ATK:2/HP:6]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: While you're <b>Overloaded</b>,
//       your other minions have +2 Attack.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - AURA = 1
// --------------------------------------------------------
// RefTag:
// - OVERLOAD = 1
// --------------------------------------------------------
TEST_CASE("[Shaman : Minion] - ULD_173 : Vessina")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Vessina"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Dust Devil"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 2);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField[1]->GetAttack(), 3);

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField[0]->GetAttack(), 2);
    CHECK_EQ(curField[1]->GetAttack(), 5);
    CHECK_EQ(curField[2]->GetAttack(), 5);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curField[0]->GetAttack(), 2);
    CHECK_EQ(curField[1]->GetAttack(), 5);
    CHECK_EQ(curField[2]->GetAttack(), 5);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curField[0]->GetAttack(), 2);
    CHECK_EQ(curField[1]->GetAttack(), 3);
    CHECK_EQ(curField[2]->GetAttack(), 3);
}

// ----------------------------------------- SPELL - SHAMAN
// [ULD_181] Earthquake - COST:7
// - Set: Uldum, Rarity: Rare
// - Spell School: Nature
// --------------------------------------------------------
// Text: Deal 5 damage to all minions,
//       then deal 2 damage to all minions.
// --------------------------------------------------------
TEST_CASE("[Shaman : Spell] - ULD_181 : Earthquake")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Earthquake"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Hench-Clan Hogsteed"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Argent Commander"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curField.GetCount(), 0);
}

// ---------------------------------------- MINION - SHAMAN
// [ULD_276] EVIL Totem - COST:2 [ATK:0/HP:2]
// - Race: Totem, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: At the end of your turn,
//       add a <b>Lackey</b> to your hand.
// --------------------------------------------------------
// GameTag:
// - 1359 = 1
// --------------------------------------------------------
// RefTag:
// - MARK_OF_EVIL = 1
// --------------------------------------------------------
TEST_CASE("[Shaman : Minion] - ULD_276 : EVIL Totem")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("EVIL Totem"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 0);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curHand.GetCount(), 1);
    CHECK_EQ(curHand[0]->card->IsLackey(), true);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curHand.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curHand.GetCount(), 2);
    CHECK_EQ(curHand[1]->card->IsLackey(), true);
}

// ----------------------------------------- SPELL - SHAMAN
// [ULD_291] Corrupt the Waters - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> Play 6 <b>Battlecry</b> cards.
//       <b>Reward:</b> Heart of Vir'naal.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 6
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 54370
// --------------------------------------------------------
// RefTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Shaman : Spell] - ULD_291 : Corrupt the Waters")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);
    opPlayer->GetHero()->SetDamage(0);

    auto curHero = curPlayer->GetHero();
    auto opHero = opPlayer->GetHero();
    auto& curField = *(curPlayer->GetFieldZone());
    const auto curSecret = curPlayer->GetSecretZone();

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Corrupt the Waters"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Elven Archer"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Elven Archer"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Elven Archer"));
    const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Elven Archer"));
    const auto card6 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Elven Archer"));
    const auto card7 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Elven Archer"));
    const auto card8 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Elven Archer"));
    const auto card9 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Razorfen Hunter"));
    const auto card10 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Abusive Sergeant"));
    const auto card11 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Arcane Explosion"));

    auto quest = dynamic_cast<Spell*>(card1);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 0);
    CHECK_EQ(quest->GetQuestProgressTotal(), 6);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card2, opHero));
    CHECK_EQ(quest->GetQuestProgress(), 1);
    CHECK_EQ(opHero->GetHealth(), 29);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card3, opHero));
    CHECK_EQ(quest->GetQuestProgress(), 2);
    CHECK_EQ(opHero->GetHealth(), 28);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card4, opHero));
    CHECK_EQ(quest->GetQuestProgress(), 3);
    CHECK_EQ(opHero->GetHealth(), 27);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card5, opHero));
    CHECK_EQ(quest->GetQuestProgress(), 4);
    CHECK_EQ(opHero->GetHealth(), 26);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card6, opHero));
    CHECK_EQ(quest->GetQuestProgress(), 5);
    CHECK_EQ(opHero->GetHealth(), 25);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card7, opHero));
    CHECK(!curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 6);
    CHECK_EQ(curHero->heroPower->card->id, "ULD_291p");
    CHECK_EQ(opHero->GetHealth(), 24);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Spell(card11));
    CHECK_EQ(curField.GetCount(), 0);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    opHero->SetDamage(0);

    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curPlayer->ExtraBattlecry(), true);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card8, opHero));
    CHECK_EQ(opHero->GetHealth(), 28);

    game.Process(curPlayer, PlayCardTask::Minion(card9));
    CHECK_EQ(curField.GetCount(), 4);
    CHECK_EQ(curField[1]->card->name, "Razorfen Hunter");
    CHECK_EQ(curField[2]->card->name, "Boar");
    CHECK_EQ(curField[3]->card->name, "Boar");

    game.Process(curPlayer, PlayCardTask::MinionTarget(card10, curField[2]));
    CHECK_EQ(curField[2]->GetAttack(), 5);
    CHECK_EQ(curField[3]->GetAttack(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curPlayer->ExtraBattlecry(), false);
    CHECK_EQ(curField[2]->GetAttack(), 1);
}

// ---------------------------------------- WEAPON - SHAMAN
// [ULD_413] Splitting Axe - COST:4 [ATK:3/HP:0]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Summon copies of your Totems.
// --------------------------------------------------------
// GameTag:
// - DURABILITY = 2
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Shaman : Weapon] - ULD_413 : Splitting Axe")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Splitting Axe"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("EVIL Totem"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curField.GetCount(), 3);

    game.Process(curPlayer, PlayCardTask::Weapon(card1));
    CHECK_EQ(curField.GetCount(), 5);
    CHECK_EQ(curField[3]->IsRace(Race::TOTEM), true);
    CHECK_EQ(curField[4]->IsRace(Race::TOTEM), true);
}

// ---------------------------------------- SPELL - WARLOCK
// [ULD_140] Supreme Archaeology - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> Draw 20 cards.
//       <b>Reward:</b> Tome of Origination.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 20
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 53740
// --------------------------------------------------------
TEST_CASE("[Warlock : Spell] - ULD_140 : Supreme Archaeology")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    for (int i = 0; i < 21; ++i)
    {
        config.player1Deck[i] = Cards::FindCardByName("Doomguard");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto questCard = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Supreme Archaeology"));

    game.Process(curPlayer, PlayCardTask::Spell(questCard));

    // Repeat 'TURN_END' twenty times to draw twenty cards.
    for (int i = 0; i < 20; ++i)
    {
        Generic::Draw(curPlayer);
        curHand.Remove(curHand.GetAll()[0]);
    }

    CHECK_EQ(curPlayer->GetHeroPower().card->name, "Tome of Origination");

    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curHand.GetCount(), 5);
    CHECK_EQ(curHand.GetAll()[4]->GetCost(), 0);
}

// ---------------------------------------- SPELL - WARLOCK
// [ULD_160] Sinister Deal - COST:1
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Discover</b> a <b>Lackey</b>.
// --------------------------------------------------------
// GameTag:
// - DISCOVER = 1
// - USE_DISCOVER_VISUALS = 1
// - 1359 = 1
// --------------------------------------------------------
// RefTag:
// - MARK_OF_EVIL = 1
// --------------------------------------------------------
TEST_CASE("[Warlock : Spell] - ULD_160 : Sinister Deal")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::PRIEST;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Sinister Deal"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curPlayer->choice);

    auto cards = TestUtils::GetChoiceCards(game);
    for (auto& card : cards)
    {
        CHECK_EQ(card->GetCardType(), CardType::MINION);
        CHECK_EQ(card->IsLackey(), true);
    }
}

// --------------------------------------- MINION - WARLOCK
// [ULD_161] Neferset Thrasher - COST:3 [ATK:4/HP:5]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: Whenever this attacks, deal 3 damage to your hero.
// --------------------------------------------------------
TEST_CASE("[Warlock : Minion] - ULD_161 : Neferset Thrasher")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::PRIEST;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Neferset Thrasher"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 30);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, AttackTask(card1, opPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 27);
}

// --------------------------------------- MINION - WARLOCK
// [ULD_162] EVIL Recruiter - COST:3 [ATK:3/HP:3]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Destroy a friendly <b>Lackey</b>
//       to summon a 5/5 Demon.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// - 1359 = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_IF_AVAILABLE = 0
// - REQ_FRIENDLY_TARGET = 0
// - REQ_MINION_TARGET = 0
// - REQ_LACKEY_TARGET = 0
// --------------------------------------------------------
// RefTag:
// - MARK_OF_EVIL = 1
// --------------------------------------------------------
TEST_CASE("[Warlock : Minion] - ULD_162 : EVIL Recruiter")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::PRIEST;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("EVIL Recruiter"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByID("DAL_739"));

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card1, card2));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card1, card3));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[0]->GetAttack(), 5);
    CHECK_EQ(curField[0]->GetHealth(), 5);
}

// --------------------------------------- MINION - WARLOCK
// [ULD_163] Expired Merchant - COST:2 [ATK:2/HP:1]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Discard your highest Cost card.
//       <b>Deathrattle:</b> Add 2 copies of it to your hand.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Warlock : Minion] - ULD_163 : Expired Merchant")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Expired Merchant"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Tasty Flyfish"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Frostbolt"));
    [[maybe_unused]] const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Malygos"));
    [[maybe_unused]] const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    [[maybe_unused]] const auto card6 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::SpellTarget(card3, card2));
    CHECK_EQ(dynamic_cast<Minion*>(card4)->GetAttack(), 6);
    CHECK_EQ(dynamic_cast<Minion*>(card4)->GetHealth(), 14);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 2);
    CHECK_EQ(curHand[0]->card->name, "Wolfrider");
    CHECK_EQ(curHand[1]->card->name, "Wisp");

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, HeroPowerTask(card1));
    CHECK_EQ(curHand.GetCount(), 4);
    CHECK_EQ(curHand[2]->card->name, "Malygos");
    CHECK_EQ(dynamic_cast<Minion*>(curHand[2])->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[2])->GetHealth(), 12);
    CHECK_EQ(curHand[3]->card->name, "Malygos");
    CHECK_EQ(dynamic_cast<Minion*>(curHand[3])->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[3])->GetHealth(), 12);
}

// --------------------------------------- MINION - WARLOCK
// [ULD_165] Riftcleaver - COST:6 [ATK:7/HP:5]
// - Race: Demon, Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Destroy a minion.
//       Your hero takes damage equal to its Health.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
TEST_CASE("[Warlock : Minion] - ULD_165 : Riftcleaver")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Riftcleaver"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Malygos"));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(opField[0]->GetHealth(), 12);

    opPlayer->SetUsedMana(0);

    game.Process(opPlayer, HeroPowerTask(card2));
    CHECK_EQ(opField[0]->GetHealth(), 11);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card1, card2));
    CHECK_EQ(opField.GetCount(), 0);
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 19);
}

// ---------------------------------------- SPELL - PALADIN
// [ULD_143] Pharaoh's Blessing - COST:6
// - Faction: Neutral, Set: Uldum, Rarity: Rare
// - Spell School: Holy
// --------------------------------------------------------
// Text: Give a minion +4/+4, <b>Divine Shield</b>,
//       and <b>Taunt</b>.
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
// RefTag:
// - TAUNT = 1
// - DIVINE_SHIELD = 1
// --------------------------------------------------------
TEST_CASE("[PALADIN : Spell] - ULD_143 : Pharaoh's Blessing")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::WARLOCK;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Pharaoh's Blessing"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField[0]->GetGameTag(GameTag::DIVINE_SHIELD), 0);
    CHECK_EQ(curField[0]->GetGameTag(GameTag::TAUNT), 0);
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[0]->GetHealth(), 1);

    game.Process(curPlayer, PlayCardTask::SpellTarget(card1, card2));
    CHECK_EQ(curField[0]->GetGameTag(GameTag::DIVINE_SHIELD), 1);
    CHECK_EQ(curField[0]->GetGameTag(GameTag::TAUNT), 1);
    CHECK_EQ(curField[0]->GetAttack(), 7);
    CHECK_EQ(curField[0]->GetHealth(), 5);
}

// --------------------------------------- MINION - PALADIN
// [ULD_145] Brazen Zealot - COST:1 [ATK:2/HP:1]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: Whenever you summon a minion, gain +1 Attack.
// --------------------------------------------------------
TEST_CASE("[Paladin : Minion] - ULD_144 : Brazen Zealot")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    constexpr int brazenZealotCardCount = 3;
    const std::array<Playable*, brazenZealotCardCount> brazenZealotCards = {
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Brazen Zealot")),
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Brazen Zealot")),
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Brazen Zealot")),
    };
    const auto minionCard =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Commander Rhyssa"));

    const auto curField = curPlayer->GetFieldZone();

    constexpr int defaultAttack = 2;
    for (int i = 0; i < 3; ++i)
    {
        game.Process(curPlayer, PlayCardTask::Minion(brazenZealotCards[i]));
        for (int pos = 0; pos <= i; ++pos)
        {
            CHECK_EQ(defaultAttack + i - pos,
                     curField->GetAll()[pos]->GetAttack());
        }
    }

    game.Process(curPlayer, PlayCardTask::Minion(minionCard));
    CHECK_EQ(defaultAttack + 3, curField->GetAll()[0]->GetAttack());
    CHECK_EQ(defaultAttack + 2, curField->GetAll()[1]->GetAttack());
    CHECK_EQ(defaultAttack + 1, curField->GetAll()[2]->GetAttack());
    CHECK_EQ(4, curField->GetAll()[3]->GetAttack());
}

// ------------------------------------------ SPELL - DRUID
// [ULD_273] Overflow - COST:7
// - Set: Uldum, Rarity: Rare
// - Spell School: Nature
// --------------------------------------------------------
// Text: Restore 5 Health to all characters. Draw 5 cards.
// --------------------------------------------------------
TEST_CASE("[Druid : Spell] - ULD_273 : Overflow")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;
    config.doShuffle = false;

    for (int i = 0; i < 30; ++i)
    {
        config.player1Deck[i] = config.player2Deck[i] =
            Cards::FindCardByName("Magma Rager");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    auto& curHand = *(curPlayer->GetHandZone());
    auto& opHand = *(opPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Injured Tol'vir"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Injured Tol'vir"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fireball"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));
    const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Overflow"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer,
                 PlayCardTask::SpellTarget(card3, curPlayer->GetHero()));
    CHECK_EQ(curField[0]->GetHealth(), 3);
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 24);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    game.Process(opPlayer,
                 PlayCardTask::SpellTarget(card4, opPlayer->GetHero()));
    CHECK_EQ(opField[0]->GetHealth(), 3);
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 24);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card5));
    CHECK_EQ(curField[0]->GetHealth(), 6);
    CHECK_EQ(opField[0]->GetHealth(), 6);
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 29);
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 29);
    CHECK_EQ(curHand.GetCount(), 10);
    CHECK_EQ(opHand.GetCount(), 6);
}

// ----------------------------------------- MINION - DRUID
// [ULD_292] Oasis Surger - COST:5 [ATK:3/HP:3]
// - Race: Elemental, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Rush</b> <b>Choose One -</b> Gain +2/+2;
//       or Summon a copy of this minion.
// --------------------------------------------------------
// GameTag:
// - CHOOSE_ONE = 1
// - RUSH = 1
// --------------------------------------------------------
TEST_CASE("[Druid : Minion] - ULD_292 : Oasis Surger")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Oasis Surger"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Oasis Surger"));

    game.Process(curPlayer, PlayCardTask::Minion(card1, 1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->GetAttack(), 5);
    CHECK_EQ(curField[0]->GetHealth(), 5);

    game.Process(curPlayer, PlayCardTask::Minion(card2, 2));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[1]->GetAttack(), 3);
    CHECK_EQ(curField[1]->GetHealth(), 3);
    CHECK_EQ(curField[2]->GetAttack(), 3);
    CHECK_EQ(curField[2]->GetHealth(), 3);
}

// ---------------------------------------- MINION - HUNTER
// [ULD_151] Ramkahen Wildtamer - COST:3 [ATK:4/HP:3]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Copy a random Beast in your hand.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Minion] - ULD_151 : Ramkahen Wildtamer")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Ramkahen Wildtamer"));
    [[maybe_unused]] const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    [[maybe_unused]] const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("River Crocolisk"));
    [[maybe_unused]] const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Young Dragonhawk"));

    CHECK_EQ(curHand.GetCount(), 4);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 4);
    const bool check = curHand[3]->card->name == "River Crocolisk" ||
                       curHand[3]->card->name == "Young Dragonhawk";
    CHECK_EQ(check, true);
}

// ----------------------------------------- SPELL - HUNTER
// [ULD_152] Pressure Plate - COST:2
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Secret:</b> After your opponent casts a spell,
//       destroy a random enemy minion.
// --------------------------------------------------------
// GameTag:
// - SECRET = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Spell] - ULD_152 : Pressure Plate")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curSecret = *(curPlayer->GetSecretZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Pressure Plate"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curSecret.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer,
                 PlayCardTask::SpellTarget(card2, curPlayer->GetHero()));
    CHECK_EQ(curSecret.GetCount(), 1);
    CHECK_EQ(card1->GetGameTag(GameTag::REVEALED), 0);

    game.Process(opPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(opField.GetCount(), 1);

    game.Process(opPlayer,
                 PlayCardTask::SpellTarget(card3, curPlayer->GetHero()));
    CHECK_EQ(curSecret.GetCount(), 0);
    CHECK_EQ(card1->GetGameTag(GameTag::REVEALED), 1);
    CHECK_EQ(opField.GetCount(), 0);
}

// ---------------------------------------- MINION - HUNTER
// [ULD_154] Hyena Alpha - COST:4 [ATK:3/HP:3]
// - Race: Beast, Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If you control a <b>Secret</b>,
//       summon two 2/2 Hyenas.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// RefTag:
// - SECRET = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Minion] - ULD_154 : Hyena Alpha")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Redemption"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Redemption"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Hyena Alpha"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Hyena Alpha"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(opField.GetCount(), 1);

    game.Process(opPlayer, PlayCardTask::Spell(card2));
    CHECK_EQ(opPlayer->GetSecretZone()->GetCount(), 1);

    game.Process(opPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(opField.GetCount(), 4);
    CHECK_EQ(opField[0]->card->name, "Hyena Alpha");
    CHECK_EQ(opField[2]->card->name, "Hyena Alpha");
    CHECK_EQ(opField[1]->GetAttack(), 2);
    CHECK_EQ(opField[1]->GetHealth(), 2);
    CHECK_EQ(opField[1]->card->name, "Hyena");
    CHECK_EQ(opField[3]->GetAttack(), 2);
    CHECK_EQ(opField[3]->GetHealth(), 2);
    CHECK_EQ(opField[3]->card->name, "Hyena");
}

// ----------------------------------------- SPELL - HUNTER
// [ULD_155] Unseal the Vault - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> Summon 20 minions.
//       <b>Reward:</b> Ramkahen Roar.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 20
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 53925
// --------------------------------------------------------
TEST_CASE("[Hunter : Spell] - ULD_155 : Unseal the Vault")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.skipMulligan = true;
    config.doShuffle = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Unseal the Vault"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curPlayer->GetSecretZone()->quest->card->name, "Unseal the Vault");

    for (int i = 1; i <= 4; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            game.Process(curPlayer,
                         PlayCardTask::Minion(Generic::DrawCard(
                             curPlayer, Cards::FindCardByName("Wisp"))));
        }
        CHECK_EQ(card1->GetGameTag(GameTag::QUEST_PROGRESS), i * 5);
        CHECK_EQ(curField.GetCount(), 5);

        game.Process(curPlayer,
                     PlayCardTask::Spell(Generic::DrawCard(
                         curPlayer, Cards::FindCardByName("Twisting Nether"))));
        CHECK_EQ(curField.GetCount(), 0);

        game.Process(curPlayer, EndTurnTask());
        game.ProcessUntil(Step::MAIN_ACTION);

        game.Process(opPlayer, PlayCardTask::Minion(Generic::DrawCard(
                                   opPlayer, Cards::FindCardByName("Wisp"))));
        CHECK_EQ(card1->GetGameTag(GameTag::QUEST_PROGRESS), i * 5);

        game.Process(opPlayer,
                     PlayCardTask::Spell(Generic::DrawCard(
                         opPlayer, Cards::FindCardByName("Twisting Nether"))));
        CHECK_EQ(opField.GetCount(), 0);

        game.Process(opPlayer, EndTurnTask());
        game.ProcessUntil(Step::MAIN_ACTION);
    }
    CHECK_EQ(curPlayer->GetHeroPower().card->name, "Pharaoh's Warmask");

    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));
    const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[1]->GetAttack(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card4));
    game.Process(opPlayer, PlayCardTask::Minion(card5));

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curField[0]->GetAttack(), 5);
    CHECK_EQ(curField[0]->GetAttack(), 5);
    CHECK_EQ(opField[0]->GetAttack(), 1);
    CHECK_EQ(opField[0]->GetAttack(), 1);
}

// ---------------------------------------- MINION - HUNTER
// [ULD_156] Dinotamer Brann - COST:7 [ATK:2/HP:4]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If your deck has no duplicates,
//       summon King Krush.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Minion] - ULD_156 : Dinotamer Brann")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    for (int i = 0; i < 6; ++i)
    {
        config.player1Deck[i] = Cards::FindCardByName("Malygos");
        config.player2Deck[i] = Cards::FindCardByName("Malygos");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Dinotamer Brann"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Dinotamer Brann"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask(card2, nullptr, 0));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[0]->card->name, "Dinotamer Brann");
    CHECK_EQ(curField[1]->card->name, "King Krush");
    CHECK_EQ(curField[1]->GetAttack(), 8);
    CHECK_EQ(curField[1]->GetHealth(), 8);
    CHECK_EQ(curField[1]->HasCharge(), true);
    CHECK_EQ(curField[2]->card->name, "Dinotamer Brann");
}

// ---------------------------------------- MINION - HUNTER
// [ULD_212] Wild Bloodstinger - COST:6 [ATK:6/HP:9]
// - Race: Beast, Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Summon a minion from
//       your opponent's hand. Attack it.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Minion] - ULD_212 : Wild Bloodstinger")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    for (int i = 0; i < 6; ++i)
    {
        config.player1Deck[i] = Cards::FindCardByName("Malygos");
        config.player2Deck[i] = Cards::FindCardByName("Malygos");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Wild Bloodstinger"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 5);
    CHECK_EQ(opField.GetCount(), 1);
    CHECK_EQ(opField[0]->GetHealth(), 6);
}

// ---------------------------------------- MINION - HUNTER
// [ULD_410] Scarlet Webweaver - COST:6 [ATK:5/HP:5]
// - Race: Beast, Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Reduce the Cost of a random Beast
//       in your hand by (5).
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Minion] - ULD_410 : Scarlet Webweaver")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Scarlet Webweaver"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("King Krush"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Malygos"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(card2->GetCost(), 4);
    CHECK_EQ(card3->GetCost(), 9);
}

// ----------------------------------------- SPELL - HUNTER
// [ULD_429] Hunter's Pack - COST:3
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: Add a random Hunter Beast, <b>Secret</b>,
//       and weapon to your hand.
// --------------------------------------------------------
// RefTag:
// - SECRET = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Spell] - ULD_429 : Hunter's Pack")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Hunter's Pack"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curHand.GetCount(), 3);
    CHECK_EQ(curHand[0]->card->GetCardClass(), CardClass::HUNTER);
    CHECK_EQ(curHand[0]->card->GetCardType(), CardType::MINION);
    CHECK_EQ(curHand[0]->card->GetRace(), Race::BEAST);
    CHECK_EQ(curHand[1]->card->GetCardClass(), CardClass::HUNTER);
    CHECK_EQ(curHand[1]->card->GetCardType(), CardType::SPELL);
    CHECK_EQ(curHand[1]->card->IsSecret(), true);
    CHECK_EQ(curHand[2]->card->GetCardClass(), CardClass::HUNTER);
    CHECK_EQ(curHand[2]->card->GetCardType(), CardType::WEAPON);
}

// ---------------------------------------- WEAPON - HUNTER
// [ULD_430] Desert Spear - COST:3 [ATK:1/HP:0]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: After your hero attacks,
//       summon a 1/1 Locust with <b>Rush</b>.
// --------------------------------------------------------
// GameTag:
// - DURABILITY = 3
// --------------------------------------------------------
// RefTag:
// - RUSH = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Weapon] - ULD_430 : Desert Spear")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Desert Spear"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Stonetusk Boar"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fiery War Axe"));

    game.Process(curPlayer, PlayCardTask::Weapon(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, AttackTask(card2, opPlayer->GetHero()));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer,
                 AttackTask(curPlayer->GetHero(), opPlayer->GetHero()));
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(opField.GetCount(), 0);
    CHECK_EQ(curField[1]->card->name, "Locust");
    CHECK_EQ(curField[1]->GetAttack(), 1);
    CHECK_EQ(curField[1]->GetHealth(), 1);
    CHECK_EQ(curField[1]->HasRush(), true);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Weapon(card3));
    game.Process(opPlayer,
                 AttackTask(opPlayer->GetHero(), curPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 27);
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(opField.GetCount(), 0);
}

// ----------------------------------------- SPELL - HUNTER
// [ULD_713] Swarm of Locusts - COST:6
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: Summon seven 1/1 Locusts with <b>Rush</b>.
// --------------------------------------------------------
// PlayReq:
// - REQ_NUM_MINION_SLOTS = 1
// --------------------------------------------------------
// RefTag:
// - RUSH = 1
// --------------------------------------------------------
TEST_CASE("[Hunter : Spell] - ULD_713 : Swarm of Locusts")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Swarm of Locusts"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    for (int i = 0; i < 7; ++i)
    {
        CHECK_EQ(curField[i]->card->name, "Locust");
        CHECK_EQ(curField[i]->GetAttack(), 1);
        CHECK_EQ(curField[i]->GetHealth(), 1);
        CHECK_EQ(curField[i]->HasRush(), true);
    }
}

// ------------------------------------------- SPELL - MAGE
// [ULD_216] Puzzle Box of Yogg-Saron - COST:10
// - Faction: Neutral, Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: Cast 10 random spells <i>(targets chosen randomly).</i>
// --------------------------------------------------------
TEST_CASE("[Mage : Spell] - ULD_216 : Puzzle Box of Yogg-Saron")
{
    for (int i = 0; i < 100; ++i)
    {
        GameConfig config;
        config.player1Class = CardClass::MAGE;
        config.player2Class = CardClass::PALADIN;
        config.startPlayer = PlayerType::PLAYER1;
        config.doFillDecks = true;
        config.autoRun = false;

        Game game(config);
        game.Start();
        game.ProcessUntil(Step::MAIN_ACTION);

        Player* curPlayer = game.GetCurrentPlayer();
        Player* opPlayer = game.GetOpponentPlayer();
        curPlayer->SetTotalMana(10);
        curPlayer->SetUsedMana(0);
        opPlayer->SetTotalMana(10);
        opPlayer->SetUsedMana(0);

        const auto card1 = Generic::DrawCard(
            curPlayer, Cards::FindCardByName("Puzzle Box of Yogg-Saron"));

        game.Process(curPlayer, PlayCardTask::Spell(card1));
        CHECK_EQ(curPlayer->GetGameTag(GameTag::CAST_RANDOM_SPELLS), 0);
        CHECK_EQ(curPlayer->GetNumSpellsCastThisTurn(), 1);
    }
}

// ------------------------------------------ MINION - MAGE
// [ULD_236] Tortollan Pilgrim - COST:8 [ATK:5/HP:5]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry</b>: <b>Discover</b> a spell
//       in your deck and cast it with random targets.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// - DISCOVER = 1
// - USE_DISCOVER_VISUALS = 1
// --------------------------------------------------------
TEST_CASE("[Mage : Minion] - ULD_236 : Tortollan Pilgrim")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    for (int i = 0; i < 30; ++i)
    {
        config.player1Deck[i] = Cards::FindCardByName("Frostbolt");
        config.player2Deck[i] = Cards::FindCardByName("Frostbolt");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto curHero = curPlayer->GetHero();
    auto opHero = opPlayer->GetHero();
    auto& curDeck = *(curPlayer->GetDeckZone());
    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Tortollan Pilgrim"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curPlayer->choice);
    CHECK_EQ(curDeck.GetCount(), 26);

    auto cards = TestUtils::GetChoiceCards(game);
    CHECK_EQ(cards.size(), 1);

    // NOTE: dbfID of the card 'Frostbolt' is 662
    const int dbfTotal = cards[0]->dbfID;
    CHECK_EQ(dbfTotal, 662);

    TestUtils::ChooseNthChoice(game, 1);
    const int totalHealth =
        curHero->GetHealth() + opHero->GetHealth() + curField[0]->GetHealth();
    CHECK_EQ(totalHealth, 62);
    CHECK_EQ(curDeck.GetCount(), 25);
}

// ------------------------------------------ MINION - MAGE
// [ULD_238] Reno the Relicologist - COST:6 [ATK:4/HP:6]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If your deck has no duplicates,
//       deal 10 damage randomly split among all enemy minions.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Mage : Minion] - ULD_238 : Reno the Relicologist")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    for (int i = 0; i < 7; ++i)
    {
        config.player1Deck[i] = Cards::FindCardByName("Malygos");
        config.player2Deck[i] = Cards::FindCardByName("Malygos");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        opPlayer, Cards::FindCardByName("Reno the Relicologist"));
    const auto card2 = Generic::DrawCard(
        opPlayer, Cards::FindCardByName("Reno the Relicologist"));

    game.Process(curPlayer, PlayCardTask::Minion(curHand[0]));
    CHECK_EQ(curField[0]->GetHealth(), 12);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetHealth(), 12);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(curHand[0]));
    CHECK_EQ(curField[0]->GetHealth(), 12);
    CHECK_EQ(curField[1]->GetHealth(), 12);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 30);
    const int totalHealth = curField[0]->GetHealth() + curField[1]->GetHealth();
    CHECK_EQ(totalHealth, 14);
}

// ------------------------------------------- SPELL - MAGE
// [ULD_239] Flame Ward - COST:3
// - Faction: Neutral, Set: Uldum, Rarity: Common
// - Spell School: Fire
// --------------------------------------------------------
// Text: <b>Secret:</b> After a minion attacks your hero,
//       deal 3 damage to all enemy minions.
// --------------------------------------------------------
// GameTag:
// - SECRET = 1
// --------------------------------------------------------
TEST_CASE("[Mage : Spell] - ULD_239 : Flame Ward")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::HUNTER;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curSecret = *(curPlayer->GetSecretZone());
    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Flame Ward"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Oasis Snapjaw"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Chillwind Yeti"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curSecret.GetCount(), 1);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card3));
    game.Process(opPlayer, PlayCardTask::Minion(card4));
    game.Process(opPlayer, PlayCardTask::Minion(card5));

    game.Process(opPlayer, AttackTask(card3, card2));
    CHECK_EQ(curSecret.GetCount(), 1);
    CHECK_EQ(card1->GetGameTag(GameTag::REVEALED), 0);
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(opField.GetCount(), 2);

    game.Process(opPlayer, AttackTask(card4, curPlayer->GetHero()));
    CHECK_EQ(curSecret.GetCount(), 0);
    CHECK_EQ(card1->GetGameTag(GameTag::REVEALED), 1);
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 27);
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 30);
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(opField.GetCount(), 1);
    CHECK_EQ(opField[0]->GetHealth(), 2);
}

// ------------------------------------------ MINION - MAGE
// [ULD_240] Arcane Flakmage - COST:2 [ATK:3/HP:2]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: After you play a <b>Secret</b>,
//       deal 2 damage to all enemy minions.
// --------------------------------------------------------
// RefTag:
// - SECRET = 1
// --------------------------------------------------------
TEST_CASE("[Mage : Minion] - ULD_240 : Arcane Flakmage")
{
    GameConfig config;
    config.player1Class = CardClass::HUNTER;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opSecret = *(opPlayer->GetSecretZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Oasis Snapjaw"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Arcane Flakmage"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Flame Ward"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(opSecret.GetCount(), 0);

    game.Process(opPlayer, PlayCardTask::Spell(card4));
    CHECK_EQ(opSecret.GetCount(), 1);
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 5);
}

// ------------------------------------------ MINION - MAGE
// [ULD_293] Cloud Prince - COST:5 [ATK:4/HP:4]
// - Race: Elemental, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If you control a <b>Secret</b>,
//       deal 6 damage.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_IF_AVAILABLE_AND_MINIMUM_FRIENDLY_SECRETS = 1
// --------------------------------------------------------
// RefTag:
// - SECRET = 1
// --------------------------------------------------------
TEST_CASE("[Mage : Minion] - ULD_293 : Cloud Prince")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::PALADIN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Cloud Prince"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Cloud Prince"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Flame Ward"));

    game.Process(curPlayer,
                 PlayCardTask::MinionTarget(card1, opPlayer->GetHero()));
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 30);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card3));
    game.Process(curPlayer,
                 PlayCardTask::MinionTarget(card2, opPlayer->GetHero()));
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 24);
}

// --------------------------------------- MINION - WARLOCK
// [ULD_167] Diseased Vulture - COST:4 [ATK:3/HP:5]
// - Race: Beast, Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: After your hero takes damage on your turn,
//       summon a random 3-Cost minion.
// --------------------------------------------------------
TEST_CASE("[Warlock : Minion] - ULD_167 : Diseased Vulture")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Diseased Vulture"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(curField[1]->GetCost(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    const int fieldCount = curField.GetCount();

    game.Process(opPlayer, HeroPowerTask(curPlayer->GetHero()));
    CHECK_EQ(curField.GetCount(), fieldCount);
}

// --------------------------------------- MINION - WARLOCK
// [ULD_168] Dark Pharaoh Tekahn - COST:5 [ATK:4/HP:4]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Battlecry:</b> For the rest of the game,
//       your <b>Lackeys</b> are 4/4.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - BATTLECRY = 1
// --------------------------------------------------------
// RefTag:
// - MARK_OF_EVIL = 1
// --------------------------------------------------------
TEST_CASE("[Warlock : Minion] - ULD_168 : Dark Pharaoh Tekahn")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    for (int i = 0; i < 30; ++i)
    {
        config.player1Deck[i] = Cards::FindCardByID("DAL_739");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Dark Pharaoh Tekahn"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByID("DAL_739"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetAttack(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetHealth(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(card3)->GetAttack(), 3);
    CHECK_EQ(dynamic_cast<Minion*>(card3)->GetHealth(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetAttack(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetHealth(), 1);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetHealth(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(card3)->GetAttack(), 3);
    CHECK_EQ(dynamic_cast<Minion*>(card3)->GetHealth(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetHealth(), 4);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card4, card1));

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curHand.GetCount(), 6);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[5])->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[5])->GetHealth(), 4);
}

// ---------------------------------------- SPELL - WARLOCK
// [ULD_324] Impbalming - COST:4
// - Set: Uldum, Rarity: Rare
// - Spell School: Fel
// --------------------------------------------------------
// Text: Destroy a minion. Shuffle 3 Worthless Imps into your deck.
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
TEST_CASE("[Warlock : Spell] - ULD_324 : Impbalming")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::PALADIN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& curDeck = *(curPlayer->GetDeckZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Impbalming"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 1);
    CHECK_EQ(curDeck.GetCount(), 0);

    game.Process(curPlayer, PlayCardTask::SpellTarget(card2, card1));
    CHECK_EQ(curHand.GetCount(), 0);
    CHECK_EQ(curDeck.GetCount(), 3);
    CHECK_EQ(curDeck[0]->card->name, "Worthless Imp");
}

// ---------------------------------------- SPELL - WARLOCK
// [ULD_717] Plague of Flames - COST:1
// - Set: Uldum, Rarity: Rare
// - Spell School: Fire
// --------------------------------------------------------
// Text: Destroy all your minions.
//       For each one, destroy a random enemy minion.
// --------------------------------------------------------
TEST_CASE("[Warlock : Spell] - ULD_717 : Plague of Flames")
{
    GameConfig config;
    config.player1Class = CardClass::WARLOCK;
    config.player2Class = CardClass::PALADIN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Plague of Flames"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));
    const auto card6 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));
    const auto card7 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));
    const auto card8 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));
    const auto card9 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wisp"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    game.Process(curPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(curField.GetCount(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card5));
    game.Process(opPlayer, PlayCardTask::Minion(card6));
    game.Process(opPlayer, PlayCardTask::Minion(card7));
    game.Process(opPlayer, PlayCardTask::Minion(card8));
    game.Process(opPlayer, PlayCardTask::Minion(card9));
    CHECK_EQ(opField.GetCount(), 5);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curField.GetCount(), 0);
    CHECK_EQ(opField.GetCount(), 2);
}

// ----------------------------------------- MINION - ROGUE
// [ULD_327] Bazaar Mugger - COST:5 [ATK:3/HP:5]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Rush</b> <b>Battlecry:</b> Add a random minion
//       from another class to your hand.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// - RUSH = 1
// --------------------------------------------------------
TEST_CASE("[Rogue : Minion] - ULD_327 : Bazaar Mugger")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::PALADIN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Bazaar Mugger"));
    CHECK_EQ(curHand.GetCount(), 1);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 1);
    CHECK_EQ(curHand[0]->card->GetCardType(), CardType::MINION);
    CHECK_NE(curHand[0]->card->GetCardClass(), CardClass::ROGUE);
}

// ------------------------------------------ MINION - MAGE
// [ULD_329] Dune Sculptor - COST:3 [ATK:3/HP:3]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: After you cast a spell, add a random Mage
//       minion to your hand.
// --------------------------------------------------------
TEST_CASE("[Mage : Minion] - ULD_329 : Dune Sculptor")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::PALADIN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Dune Sculptor"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fireball"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 2);

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curHand.GetCount(), 1);

    game.Process(curPlayer, PlayCardTask::SpellTarget(card2, card3));
    CHECK_EQ(curHand.GetCount(), 1);
    CHECK_EQ(curHand[0]->card->GetCardType(), CardType::MINION);
    CHECK_EQ(curHand[0]->card->GetCardClass(), CardClass::MAGE);
}

// ------------------------------------------- SPELL - MAGE
// [ULD_433] Raid the Sky Temple - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> Cast 10 spells.
//       <b>Reward: </b>Ascendant Scroll.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 10
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 53946
// --------------------------------------------------------
TEST_CASE("[Mage : Spell] - ULD_433 : Raid the Sky Temple")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto curHero = curPlayer->GetHero();
    auto& curHand = *(curPlayer->GetHandZone());
    const auto curSecret = curPlayer->GetSecretZone();

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Raid the Sky Temple"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));
    const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));
    const auto card6 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));

    auto quest = dynamic_cast<Spell*>(card1);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 0);
    CHECK_EQ(quest->GetQuestProgressTotal(), 10);

    game.Process(curPlayer, PlayCardTask::Spell(card2));
    CHECK_EQ(quest->GetQuestProgress(), 1);

    game.Process(curPlayer, PlayCardTask::Spell(card3));
    CHECK_EQ(quest->GetQuestProgress(), 2);

    game.Process(curPlayer, PlayCardTask::Spell(card4));
    CHECK_EQ(quest->GetQuestProgress(), 3);

    game.Process(curPlayer, PlayCardTask::Spell(card5));
    CHECK_EQ(quest->GetQuestProgress(), 4);

    game.Process(curPlayer, PlayCardTask::Spell(card6));
    CHECK_EQ(quest->GetQuestProgress(), 5);

    const auto card7 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));
    const auto card8 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));
    const auto card9 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));
    const auto card10 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));
    const auto card11 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Preparation"));

    game.Process(curPlayer, PlayCardTask::Spell(card7));
    CHECK_EQ(quest->GetQuestProgress(), 6);

    game.Process(curPlayer, PlayCardTask::Spell(card8));
    CHECK_EQ(quest->GetQuestProgress(), 7);

    game.Process(curPlayer, PlayCardTask::Spell(card9));
    CHECK_EQ(quest->GetQuestProgress(), 8);

    game.Process(curPlayer, PlayCardTask::Spell(card10));
    CHECK_EQ(quest->GetQuestProgress(), 9);

    game.Process(curPlayer, PlayCardTask::Spell(card11));
    CHECK(!curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 10);
    CHECK_EQ(curHero->heroPower->card->id, "ULD_433p");
    CHECK_EQ(curHand.GetCount(), 0);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curHand.GetCount(), 1);
    const int originalCost = curHand[0]->card->GetCost();
    const int reducedCost = curHand[0]->GetCost();
    CHECK_LE(originalCost - reducedCost, 2);
}

// ------------------------------------------ MINION - MAGE
// [ULD_435] Naga Sand Witch - COST:5 [ATK:5/HP:5]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Change the Cost of spells
//       in your hand to (5).
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Mage : Minion] - ULD_435 : Naga Sand Witch")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::PALADIN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Naga Sand Witch"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fireball"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Frostbolt"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Pyroblast"));

    CHECK_EQ(card2->GetCost(), 4);
    CHECK_EQ(card3->GetCost(), 2);
    CHECK_EQ(card4->GetCost(), 10);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(card2->GetCost(), 5);
    CHECK_EQ(card3->GetCost(), 5);
    CHECK_EQ(card4->GetCost(), 5);
}

// ------------------------------------------- SPELL - MAGE
// [ULD_726] Ancient Mysteries - COST:2
// - Set: Uldum, Rarity: Common
// - Spell School: Arcane
// --------------------------------------------------------
// Text: Draw a <b>Secret</b> from your deck. It costs (0).
// --------------------------------------------------------
// RefTag:
// - SECRET = 1
// --------------------------------------------------------
TEST_CASE("[Mage : Spell] - ULD_726 : Ancient Mysteries")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::PALADIN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    for (int i = 0; i < 30; i += 2)
    {
        config.player1Deck[i] = Cards::FindCardByName("Flame Ward");
        config.player1Deck[i + 1] = Cards::FindCardByName("Wisp");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& curDeck = *(curPlayer->GetDeckZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Ancient Mysteries"));

    CHECK_EQ(curHand.GetCount(), 5);
    CHECK_EQ(curDeck.GetCount(), 26);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curHand.GetCount(), 5);
    CHECK_EQ(curHand[4]->card->IsSecret(), true);
    CHECK_EQ(curHand[4]->GetCost(), 0);
    CHECK_EQ(curDeck.GetCount(), 25);
}

// --------------------------------------- MINION - PALADIN
// [ULD_207] Ancestral Guardian - COST:4 [ATK:4/HP:2]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Lifesteal</b> <b>Reborn</b>
// --------------------------------------------------------
// GameTag:
// - LIFESTEAL = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Paladin : Minion] - ULD_207 : Ancestral Guardian")
{
    // Do nothing
}

// --------------------------------------- MINION - PALADIN
// [ULD_217] Micro Mummy - COST:2 [ATK:1/HP:2]
// - Race: Mechanical, Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Reborn</b> At the end of your turn, give
//       another random friendly minion +1 Attack.
// --------------------------------------------------------
// GameTag:
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Paladin : Minion] - ULD_217 : Micro Mummy")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Micro Mummy"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Acidic Swamp Ooze"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));

    int totalAttack = curField[1]->GetAttack();
    totalAttack += curField[2]->GetAttack();
    CHECK_EQ(totalAttack, 6);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    totalAttack = curField[1]->GetAttack();
    totalAttack += curField[2]->GetAttack();
    CHECK_EQ(totalAttack, 7);
}

// ---------------------------------------- SPELL - PALADIN
// [ULD_431] Making Mummies - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> Play 5 <b>Reborn</b> minions.
//       <b>Reward:</b> Emperor Wraps.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 5
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 53908
// --------------------------------------------------------
// RefTag:
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Paladin : Spell] - ULD_431 : Making Mummies")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto curHero = curPlayer->GetHero();
    auto& curField = *(curPlayer->GetFieldZone());
    const auto curSecret = curPlayer->GetSecretZone();

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Making Mummies"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Murmy"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Murmy"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Murmy"));
    const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Murmy"));
    const auto card6 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Murmy"));

    auto quest = dynamic_cast<Spell*>(card1);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 0);
    CHECK_EQ(quest->GetQuestProgressTotal(), 5);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(quest->GetQuestProgress(), 1);

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(quest->GetQuestProgress(), 2);

    game.Process(curPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(quest->GetQuestProgress(), 3);

    game.Process(curPlayer, PlayCardTask::Minion(card5));
    CHECK_EQ(quest->GetQuestProgress(), 4);

    game.Process(curPlayer, PlayCardTask::Minion(card6));
    CHECK(!curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 5);
    CHECK_EQ(curHero->heroPower->card->id, "ULD_431p");

    game.Process(curPlayer, HeroPowerTask(card2));
    CHECK_EQ(curField.GetCount(), 6);
    CHECK_EQ(curField[0]->GetAttack(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 1);
    CHECK_EQ(curField[0]->HasReborn(), true);
    CHECK_EQ(curField[1]->GetAttack(), 2);
    CHECK_EQ(curField[1]->GetHealth(), 2);
    CHECK_EQ(curField[1]->HasReborn(), true);
    CHECK_EQ(curField[2]->GetAttack(), 1);
    CHECK_EQ(curField[2]->GetHealth(), 1);
    CHECK_EQ(curField[2]->HasReborn(), true);
}

// --------------------------------------- MINION - PALADIN
// [ULD_438] Salhet's Pride - COST:3 [ATK:3/HP:1]
// - Race: Beast, Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Deathrattle:</b> Draw two 1-Health minions
//       from your deck.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Paladin : Minion] - ULD_438 : Salhet's Pride")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;
    config.doShuffle = false;

    for (int i = 0; i < 30; i += 3)
    {
        config.player1Deck[i] = Cards::FindCardByName("Harvest Golem");
        config.player1Deck[i + 1] = Cards::FindCardByName("Murloc Raider");
        config.player1Deck[i + 2] = Cards::FindCardByName("Goldshire Footman");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Salhet's Pride"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 4);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, HeroPowerTask(card1));
    CHECK_EQ(card1->isDestroyed, true);
    CHECK_EQ(curHand.GetCount(), 6);
    CHECK_EQ(curHand[4]->card->GetCost(), 1);
    CHECK_EQ(curHand[5]->card->GetCost(), 1);
}

// --------------------------------------- MINION - PALADIN
// [ULD_439] Sandwasp Queen - COST:2 [ATK:3/HP:1]
// - Race: Beast, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Add two 2/1 Sandwasps to your hand.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Paladin : Minion] - ULD_439 : Sandwasp Queen")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Sandwasp Queen"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 2);
    CHECK_EQ(curHand[0]->card->name, "Sandwasp");
    CHECK_EQ(curHand[1]->card->name, "Sandwasp");
}

// --------------------------------------- MINION - PALADIN
// [ULD_500] Sir Finley of the Sands - COST:2 [ATK:2/HP:3]
// - Race: Murloc, Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If your deck has no duplicates,
//       <b>Discover</b> an upgraded Hero Power.
// --------------------------------------------------------
// Entourage: HERO_01bp2, HERO_02bp2, HERO_03bp2, HERO_04bp2,
//            HERO_05bp2, HERO_06bp2, HERO_07bp2, HERO_08bp2,
//            HERO_09bp2, HERO_10bp2
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - BATTLECRY = 1
// - DISCOVER = 1
// - USE_DISCOVER_VISUALS = 1
// --------------------------------------------------------
TEST_CASE("[Paladin : Minion] - ULD_500 : Sir Finley of the Sands")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Sir Finley of the Sands"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK(curPlayer->choice);

    auto cards = TestUtils::GetChoiceCards(game);
    for (auto& card : cards)
    {
        CHECK_EQ(card->GetCardType(), CardType::HERO_POWER);
    }

    TestUtils::ChooseNthChoice(game, 1);

    HeroPower* heroPower = curPlayer->GetHero()->heroPower;
    CHECK(heroPower);

    // Warrior
    SUBCASE("Warrior - HERO_01bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_01bp2"));

        game.Process(curPlayer, HeroPowerTask());
        CHECK_EQ(curPlayer->GetHero()->GetArmor(), 4);
    }

    // Shaman
    SUBCASE("Shaman - HERO_02bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_02bp2"));

        game.Process(curPlayer, HeroPowerTask());
        CHECK(curPlayer->choice);

        auto totemCards = TestUtils::GetChoiceCards(game);
        for (auto& card : totemCards)
        {
            CHECK_EQ(card->GetRace(), Race::TOTEM);
        }

        auto& curField = *(curPlayer->GetFieldZone());

        SUBCASE("Shaman - Healing Totem")
        {
            TestUtils::ChooseNthChoice(game, 1);
            CHECK_EQ(curField[1]->card->name, "Healing Totem");
        }

        SUBCASE("Shaman - Searing Totem")
        {
            TestUtils::ChooseNthChoice(game, 2);
            CHECK_EQ(curField[1]->card->name, "Searing Totem");
        }

        SUBCASE("Shaman - Stoneclaw Totem")
        {
            TestUtils::ChooseNthChoice(game, 3);
            CHECK_EQ(curField[1]->card->name, "Stoneclaw Totem");
        }

        SUBCASE("Shaman - Strength Totem")
        {
            TestUtils::ChooseNthChoice(game, 4);
            CHECK_EQ(curField[1]->card->name, "Strength Totem");
        }
    }

    // Rogue
    SUBCASE("Rogue - HERO_03bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_03bp2"));

        game.Process(curPlayer, HeroPowerTask());
        CHECK_EQ(curPlayer->GetHero()->HasWeapon(), true);
        CHECK_EQ(curPlayer->GetHero()->weapon->GetAttack(), 2);
        CHECK_EQ(curPlayer->GetHero()->weapon->GetDurability(), 2);
    }

    // Paladin
    SUBCASE("Paladin - HERO_04bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_04bp2"));

        auto& curField = *(curPlayer->GetFieldZone());

        game.Process(curPlayer, HeroPowerTask());
        CHECK_EQ(curField.GetCount(), 3);
        CHECK_EQ(curField[1]->card->name, "Silver Hand Recruit");
        CHECK_EQ(curField[1]->GetAttack(), 1);
        CHECK_EQ(curField[1]->GetHealth(), 1);
        CHECK_EQ(curField[2]->card->name, "Silver Hand Recruit");
        CHECK_EQ(curField[2]->GetAttack(), 1);
        CHECK_EQ(curField[2]->GetHealth(), 1);
    }

    // Hunter
    SUBCASE("Hunter - HERO_05bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_05bp2"));

        game.Process(curPlayer, HeroPowerTask());
        CHECK_EQ(opPlayer->GetHero()->GetHealth(), 17);
    }

    // Druid
    SUBCASE("Druid - HERO_06bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_06bp2"));

        game.Process(curPlayer, HeroPowerTask());
        CHECK_EQ(curPlayer->GetHero()->GetAttack(), 2);
        CHECK_EQ(curPlayer->GetHero()->GetArmor(), 2);

        game.Process(curPlayer, EndTurnTask());
        game.ProcessUntil(Step::MAIN_ACTION);

        CHECK_EQ(curPlayer->GetHero()->GetAttack(), 0);
        CHECK_EQ(curPlayer->GetHero()->GetArmor(), 2);
    }

    // Warlock
    SUBCASE("Warlock - HERO_07bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_07bp2"));

        game.Process(curPlayer, HeroPowerTask());
        CHECK_EQ(curPlayer->GetHero()->GetHealth(), 15);
    }

    // Mage
    SUBCASE("Mage - HERO_08bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_08bp2"));

        game.Process(curPlayer, HeroPowerTask(opPlayer->GetHero()));
        CHECK_EQ(opPlayer->GetHero()->GetHealth(), 18);
    }

    // Priest
    SUBCASE("Priest - HERO_09bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_09bp2"));

        game.Process(curPlayer, HeroPowerTask(curPlayer->GetHero()));
        CHECK_EQ(curPlayer->GetHero()->GetHealth(), 24);
    }

    // Demon Hunter
    SUBCASE("Demon Hunter - HERO_10bp2")
    {
        TestUtils::ChangeHeroPower(curPlayer,
                                   Cards::FindCardByID("HERO_10bp2"));

        game.Process(curPlayer, HeroPowerTask());
        CHECK_EQ(curPlayer->GetHero()->GetAttack(), 2);

        game.Process(curPlayer, EndTurnTask());
        game.ProcessUntil(Step::MAIN_ACTION);

        CHECK_EQ(curPlayer->GetHero()->GetAttack(), 0);
    }
}

// ---------------------------------------- SPELL - PALADIN
// [ULD_716] Tip the Scales - COST:8
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: Summon 7 Murlocs from your deck.
// --------------------------------------------------------
// PlayReq:
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
TEST_CASE("[Paladin : Spell] - ULD_716 : Tip the Scales")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    for (int i = 0; i < 30; i += 3)
    {
        config.player1Deck[i] = Cards::FindCardByName("Murloc Raider");
        config.player1Deck[i + 1] = Cards::FindCardByName("Bluegill Warrior");
        config.player1Deck[i + 2] = Cards::FindCardByName("Faerie Dragon");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Tip the Scales"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curField.GetCount(), 7);
    for (int i = 0; i < 7; ++i)
    {
        CHECK_EQ(curField[i]->card->GetRace(), Race::MURLOC);
    }
}

// ---------------------------------------- SPELL - PALADIN
// [ULD_728] Subdue - COST:2
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: Set a minion's Attack and Health to 1.
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
TEST_CASE("[Paladin : Spell] - ULD_728 : Subdue")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Subdue"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Subdue"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Murloc Raider"));
    const auto card4 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Shattered Sun Cleric"));
    const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Ironfur Grizzly"));
    const auto card6 = Generic::DrawCard(
        opPlayer, Cards::FindCardByName("Stormwind Champion"));

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    game.Process(curPlayer, PlayCardTask::MinionTarget(card4, card3));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[0]->GetHealth(), 2);

    game.Process(curPlayer, PlayCardTask::SpellTarget(card1, card3));
    CHECK_EQ(curField[0]->GetAttack(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card5));
    game.Process(opPlayer, PlayCardTask::Minion(card6));
    CHECK_EQ(opField[0]->GetAttack(), 4);
    CHECK_EQ(opField[0]->GetHealth(), 4);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::SpellTarget(card2, card5));
    CHECK_EQ(opField[0]->GetAttack(), 2);
    CHECK_EQ(opField[0]->GetHealth(), 2);
}

// ---------------------------------------- MINION - PRIEST
// [ULD_262] High Priest Amet - COST:4 [ATK:2/HP:7]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: Whenever you summon a minion,
//       set its Health equal to this minion's.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// --------------------------------------------------------
TEST_CASE("[Priest : Minion] - ULD_262 : High Priest Amet")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("High Priest Amet"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Shattered Sun Cleric"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Stonetusk Boar"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Stonetusk Boar"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField[1]->GetHealth(), 7);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card2, card1));
    game.Process(curPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(curField[3]->GetHealth(), 8);
}

// ----------------------------------------- SPELL - PRIEST
// [ULD_265] Embalming Ritual - COST:1
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: Give a minion <b>Reborn</b>.
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
// RefTag:
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Priest : Spell] - ULD_265 : Embalming Ritual")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Embalming Ritual"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Stonetusk Boar"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::SpellTarget(card1, card2));
    CHECK_EQ(curField[0]->HasReborn(), true);
}

// ---------------------------------------- MINION - PRIEST
// [ULD_266] Grandmummy - COST:2 [ATK:1/HP:2]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Reborn</b> <b>Deathrattle:</b> Give a random
//       friendly minion +1/+1.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Priest : Minion] - ULD_266 : Grandmummy")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Grandmummy"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Stonetusk Boar"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Frostbolt"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField[0]->GetAttack(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card3, card1));
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(curField[0]->GetAttack(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 1);
    CHECK_EQ(curField[0]->HasReborn(), false);
    CHECK_EQ(curField[1]->GetAttack(), 2);
    CHECK_EQ(curField[1]->GetHealth(), 2);
}

// ---------------------------------------- MINION - PRIEST
// [ULD_268] Psychopomp - COST:4 [ATK:3/HP:1]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Summon a random friendly minion
//       that died this game. Give it <b>Reborn</b>.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// RefTag:
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Priest : Minion] - ULD_268 : Psychopomp")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Psychopomp"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Stonetusk Boar"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, HeroPowerTask(card2));

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(curField[1]->card->name, "Stonetusk Boar");
    CHECK_EQ(curField[1]->GetAttack(), 1);
    CHECK_EQ(curField[1]->GetHealth(), 1);
    CHECK_EQ(curField[1]->HasReborn(), true);
}

// ---------------------------------------- MINION - PRIEST
// [ULD_269] Wretched Reclaimer - COST:3 [ATK:3/HP:3]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Destroy a friendly minion,
//       then return it to life with full Health.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_MINION_TARGET = 0
// - REQ_FRIENDLY_TARGET = 0
// - REQ_TARGET_IF_AVAILABLE = 0
// --------------------------------------------------------
TEST_CASE("[Priest : Minion] - ULD_269 : Wretched Reclaimer")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::HUNTER;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Injured Blademaster"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Shattered Sun Cleric"));
    const auto card3 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Wretched Reclaimer"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Serpent Egg"));
    const auto card5 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Wretched Reclaimer"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::MinionTarget(card2, card1));
    CHECK_EQ(curField[0]->card->name, "Injured Blademaster");
    CHECK_EQ(curField[0]->GetAttack(), 5);
    CHECK_EQ(curField[0]->GetHealth(), 4);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card3, card1));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[0]->card->name, "Injured Blademaster");
    CHECK_EQ(curField[0]->GetAttack(), 4);
    CHECK_EQ(curField[0]->GetHealth(), 7);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(curField.GetCount(), 4);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card5, card4));
    CHECK_EQ(curField.GetCount(), 6);
    CHECK_EQ(curField[3]->card->name, "Serpent Egg");
    CHECK_EQ(curField[4]->card->name, "Sea Serpent");
    CHECK_EQ(curField[4]->GetAttack(), 3);
    CHECK_EQ(curField[4]->GetHealth(), 4);
}

// ---------------------------------------- MINION - PRIEST
// [ULD_270] Sandhoof Waterbearer - COST:5 [ATK:5/HP:5]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: At the end of your turn, restore 5 Health
//       to a damaged friendly character.
// --------------------------------------------------------
TEST_CASE("[Priest : Minion] - ULD_270 : Sandhoof Waterbearer")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::HUNTER;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);
    curPlayer->GetHero()->SetDamage(5);

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Sandhoof Waterbearer"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 25);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 30);
}

// ----------------------------------------- SPELL - PRIEST
// [ULD_272] Holy Ripple - COST:2
// - Set: Uldum, Rarity: Rare
// - Spell School: Holy
// --------------------------------------------------------
// Text: Deal 1 damage to all enemies. Restore 1 Health
//       to all friendly characters.
// --------------------------------------------------------
TEST_CASE("[Priest : Spell] - ULD_272 : Holy Ripple")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::HUNTER;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);
    curPlayer->GetHero()->SetDamage(1);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Holy Ripple"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Injured Blademaster"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Arcane Servant"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card3));

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 30);
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 29);
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(opField[0]->GetHealth(), 2);
}

// ----------------------------------------- SPELL - PRIEST
// [ULD_714] Penance - COST:2
// - Set: Uldum, Rarity: Common
// - Spell School: Holy
// --------------------------------------------------------
// Text: <b>Lifesteal</b> Deal 3 damage to a minion.
// --------------------------------------------------------
// GameTag:
// - LIFESTEAL = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
TEST_CASE("[Priest : Spell] - ULD_714 : Penance")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::HUNTER;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);
    curPlayer->GetHero()->SetDamage(3);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Penance"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Doomsayer"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::SpellTarget(card1, curField[0]));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 30);
    CHECK_EQ(curField[0]->GetHealth(), 4);
}

// ----------------------------------------- SPELL - PRIEST
// [ULD_718] Plague of Death - COST:9
// - Set: Uldum, Rarity: Epic
// - Spell School: Shadow
// --------------------------------------------------------
// Text: <b>Silence</b> and destroy all minions.
// --------------------------------------------------------
// GameTag:
// - SILENCE = 1
// --------------------------------------------------------
TEST_CASE("[Priest : Spell] - ULD_718 : Plague of Death")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::HUNTER;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto& curField = *(curPlayer->GetFieldZone());
    const auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Plague of Death"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Serpent Egg"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Serpent Egg"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card3));

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curField.GetCount(), 0);
    CHECK_EQ(opField.GetCount(), 0);
}

// ----------------------------------------- SPELL - PRIEST
// [ULD_724] Activate the Obelisk - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> Restore 15 Health.
//       <b>Reward:</b> Obelisk's Eye.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 15
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 54750
// --------------------------------------------------------
TEST_CASE("[Priest : Spell] - ULD_724 : Activate the Obelisk")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::PRIEST;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto curHero = curPlayer->GetHero();
    auto& curField = *(curPlayer->GetFieldZone());
    const auto curSecret = curPlayer->GetSecretZone();

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Activate the Obelisk"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Injured Blademaster"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Holy Light"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Holy Light"));

    auto quest = dynamic_cast<Spell*>(card1);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 0);
    CHECK_EQ(quest->GetQuestProgressTotal(), 15);

    game.Process(curPlayer, HeroPowerTask(curHero));
    CHECK_EQ(curHero->GetHealth(), 30);
    CHECK_EQ(quest->GetQuestProgress(), 0);

    curHero->SetDamage(24);

    game.Process(curPlayer, PlayCardTask::Spell(card3));
    CHECK_EQ(curHero->GetHealth(), 14);
    CHECK_EQ(quest->GetQuestProgress(), 8);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, HeroPowerTask(curHero));
    CHECK_EQ(curHero->GetHealth(), 16);
    CHECK_EQ(quest->GetQuestProgress(), 10);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card4));
    CHECK(!curSecret->quest);
    CHECK_EQ(curHero->GetHealth(), 24);
    CHECK_EQ(quest->GetQuestProgress(), 18);
    CHECK_EQ(curHero->heroPower->card->id, "ULD_724p");

    game.Process(curPlayer, HeroPowerTask(curHero));
    CHECK_EQ(curHero->GetHealth(), 27);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField[0]->GetAttack(), 4);
    CHECK_EQ(curField[0]->GetHealth(), 3);

    game.Process(curPlayer, HeroPowerTask(card2));
    CHECK_EQ(curField[0]->GetAttack(), 7);
    CHECK_EQ(curField[0]->GetHealth(), 9);
}

// ----------------------------------------- MINION - ROGUE
// [ULD_186] Pharaoh Cat - COST:1 [ATK:1/HP:2]
// - Race: Beast, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Add a random <b>Reborn</b> minion
//       to your hand.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// RefTag:
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Rogue : Minion] - ULD_186 : Pharaoh Cat")
{
    GameConfig config;
    config.formatType = FormatType::WILD;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Pharaoh Cat"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 1);
    CHECK_EQ(curHand[0]->card->GetCardType(), CardType::MINION);
    CHECK_EQ(curHand[0]->card->HasGameTag(GameTag::REBORN), true);
}

// ----------------------------------------- MINION - ROGUE
// [ULD_231] Whirlkick Master - COST:2 [ATK:1/HP:2]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: Whenever you play a <b>Combo</b> card,
//       add a random <b>Combo</b> card to your hand.
// --------------------------------------------------------
// RefTag:
// - COMBO = 1
// --------------------------------------------------------
TEST_CASE("[Rogue : Minion] - ULD_231 : Whirlkick Master")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& opHand = *(opPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Whirlkick Master"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Whirlkick Master"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Eviscerate"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Eviscerate"));
    const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Backstab"));
    const auto card6 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Eviscerate"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));

    game.Process(curPlayer,
                 PlayCardTask::SpellTarget(card3, opPlayer->GetHero()));
    CHECK_EQ(curHand.GetCount(), 4);

    game.Process(curPlayer, PlayCardTask::SpellTarget(card5, card2));
    CHECK_EQ(curHand.GetCount(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer,
                 PlayCardTask::SpellTarget(card6, curPlayer->GetHero()));
    CHECK_EQ(curHand.GetCount(), 3);
    CHECK_EQ(opHand.GetCount(), 1);  // The Coin

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curHand.GetCount(), 3);
    game.Process(curPlayer,
                 PlayCardTask::SpellTarget(card4, curPlayer->GetHero()));
    CHECK_EQ(curHand.GetCount(), 3);
    CHECK_EQ(curHand[0]->card->HasGameTag(GameTag::COMBO), true);
    CHECK_EQ(curHand[1]->card->HasGameTag(GameTag::COMBO), true);
    CHECK_EQ(curHand[2]->card->HasGameTag(GameTag::COMBO), true);
}

// ----------------------------------------- MINION - ROGUE
// [ULD_280] Sahket Sapper - COST:4 [ATK:4/HP:4]
// - Race: Pirate, Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Deathrattle:</b> Return a random enemy minion
//       to your opponent's hand.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Rogue : Minion] - ULD_280 : Sahket Sapper")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::ROGUE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& opHand = *(opPlayer->GetHandZone());
    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    for (int i = 0; i < 10; i++)
    {
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Sahket Sapper"));
    }
    game.Process(curPlayer, PlayCardTask::Minion(curHand[0]));
    Generic::DrawCard(curPlayer, Cards::FindCardByName("Sahket Sapper"));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    const auto card1 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Sahket Sapper"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Sahket Sapper"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Eviscerate"));

    game.Process(opPlayer, PlayCardTask::Minion(card1));
    game.Process(opPlayer, PlayCardTask::Minion(card2));
    game.Process(opPlayer, PlayCardTask::SpellTarget(card3, card1));

    CHECK_EQ(curField.GetCount(), 0);
    CHECK_EQ(opField.GetCount(), 0);
    CHECK_EQ(curHand.GetCount(), 10);
    CHECK_EQ(opHand.GetCount(), 2);  // 'The Coin' and returned 'Sahket Sapper'
}

// ----------------------------------------- WEAPON - ROGUE
// [ULD_285] Hooked Scimitar - COST:3 [ATK:2/HP:0]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Combo:</b> Gain +2 Attack.
// --------------------------------------------------------
// GameTag:
// - DURABILITY = 2
// - COMBO = 1
// --------------------------------------------------------
TEST_CASE("[Rogue : Weapon] - ULD_285 : Hooked Scimitar")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Hooked Scimitar"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Hooked Scimitar"));

    game.Process(curPlayer, PlayCardTask::Weapon(card1));
    CHECK_EQ(curPlayer->GetHero()->GetAttack(), 2);

    game.Process(curPlayer, PlayCardTask::Weapon(card2));
    CHECK_EQ(curPlayer->GetHero()->GetAttack(), 4);
}

// ------------------------------------------ SPELL - ROGUE
// [ULD_286] Shadow of Death - COST:4
// - Set: Uldum, Rarity: Epic
// - Spell School: Shadow
// --------------------------------------------------------
// Text: Choose a minion. Shuffle 3 'Shadows' into your deck
//       that summon a copy when drawn.
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
TEST_CASE("[Rogue : Spell] - ULD_286 : Shadow of Death")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Dread Raven"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Shadow of Death"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fan of Knives"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));

    CHECK_EQ(curField.GetCount(), 1);
    game.Process(curPlayer, PlayCardTask::SpellTarget(card2, card1));
    CHECK_EQ(curField.GetCount(), 1);
    game.Process(curPlayer, PlayCardTask::Spell(card3));

    CHECK_EQ(curField.GetCount(), 4);
    CHECK_EQ(curField[0]->GetAttack(), 12);
}

// ----------------------------------------- MINION - ROGUE
// [ULD_288] Anka, the Buried - COST:5 [ATK:5/HP:5]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Change each <b>Deathrattle</b>
//       minion in your hand into a 1/1 that costs (1).
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - BATTLECRY = 1
// --------------------------------------------------------
// RefTag:
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Rogue : Minion] - ULD_288 : Anka, the Buried")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Anka, the Buried"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Chromatic Egg"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Waxadred"));
    const auto card4 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Necrium Apothecary"));
    const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Waggle Pick"));
    const auto card6 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Chromatic Egg"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));

    const auto CheckMinion = [](Playable* card, bool isChanged) {
        const auto minion = dynamic_cast<Minion*>(card);
        CHECK_EQ(minion->GetCost() == 1, isChanged);
        CHECK_EQ(minion->GetAttack() == 1, isChanged);
        CHECK_EQ(minion->GetHealth() == 1, isChanged);
    };
    CheckMinion(card2, true);
    CheckMinion(card3, true);
    CheckMinion(card4, false);
    CheckMinion(card6, false);

    const auto weapon = dynamic_cast<Weapon*>(card5);
    CHECK_NE(weapon->GetCost(), 1);
}

// ------------------------------------------ SPELL - ROGUE
// [ULD_326] Bazaar Burglary - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> Add 4 cards from other classes to your hand.
//       <b>Reward: </b>Ancient Blades.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 4
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 54312
// --------------------------------------------------------
TEST_CASE("[Rogue : Spell] - ULD_326 : Bazaar Burglary")
{
    {
        GameConfig config;
        config.player1Class = CardClass::ROGUE;
        config.player2Class = CardClass::WARRIOR;
        config.startPlayer = PlayerType::PLAYER1;
        config.doFillDecks = false;
        config.autoRun = false;

        Game game(config);
        game.Start();
        game.ProcessUntil(Step::MAIN_ACTION);

        Player* curPlayer = game.GetCurrentPlayer();
        Player* opPlayer = game.GetOpponentPlayer();
        curPlayer->SetTotalMana(10);
        curPlayer->SetUsedMana(0);
        opPlayer->SetTotalMana(10);
        opPlayer->SetUsedMana(0);

        auto curHero = curPlayer->GetHero();
        auto opHero = opPlayer->GetHero();
        auto curSecret = curPlayer->GetSecretZone();
        auto& opField = *opPlayer->GetFieldZone();

        const auto card1 = Generic::DrawCard(
            curPlayer, Cards::FindCardByName("Bazaar Burglary"));
        const auto card2 = Generic::DrawCard(
            curPlayer, Cards::FindCardByName("Bazaar Mugger"));
        const auto card3 =
            Generic::DrawCard(curPlayer, Cards::FindCardByName("Witch's Brew"));
        const auto card4 =
            Generic::DrawCard(curPlayer, Cards::FindCardByName("Rapid Fire"));
        const auto card5 = Generic::DrawCard(
            curPlayer, Cards::FindCardByName("Ysera, Unleashed"));
        const auto card6 =
            Generic::DrawCard(curPlayer, Cards::FindCardByName("Shiv"));

        opField.Add(Entity::GetFromCard(
            opPlayer, Cards::FindCardByName("Boulderfist Ogre")));

        auto quest = dynamic_cast<Spell*>(card1);

        game.Process(curPlayer, PlayCardTask::Spell(card1));
        CHECK(curSecret->quest);
        CHECK_EQ(quest->GetQuestProgress(), 0);
        CHECK_EQ(quest->GetQuestProgressTotal(), 4);

        game.Process(curPlayer, PlayCardTask::Minion(card2));
        CHECK_EQ(quest->GetQuestProgress(), 1);

        game.Process(curPlayer, PlayCardTask::SpellTarget(card3, curHero));
        CHECK_EQ(quest->GetQuestProgress(), 2);

        game.Process(curPlayer, PlayCardTask::SpellTarget(card4, opHero));
        CHECK_EQ(quest->GetQuestProgress(), 3);

        curPlayer->SetUsedMana(0);
        game.Process(curPlayer, PlayCardTask::Minion(card5));
        CHECK_EQ(quest->GetQuestProgress(), 3);

        curPlayer->SetUsedMana(0);
        CHECK_EQ(curPlayer->GetFieldZone()->GetCount(), 2);
        game.Process(curPlayer, PlayCardTask::SpellTarget(card6, opHero));
        CHECK_EQ(curPlayer->GetFieldZone()->GetCount(), 7);
        CHECK_EQ(quest->GetQuestProgress(), 4);

        CHECK(!curSecret->quest);
        CHECK_EQ(curHero->heroPower->card->id, "ULD_326p");

        const int curHealth = curHero->GetHealth();
        game.Process(curPlayer, HeroPowerTask());
        game.Process(curPlayer, AttackTask(curHero, opField[0]));
        CHECK_EQ(curHero->GetHealth(), curHealth);
        CHECK_EQ(curHero->GetGameTag(GameTag::IMMUNE), 0);
    }

    {
        GameConfig config;
        config.player1Class = CardClass::ROGUE;
        config.player2Class = CardClass::WARRIOR;
        config.startPlayer = PlayerType::PLAYER1;
        config.doFillDecks = false;
        config.autoRun = false;

        Game game(config);
        game.Start();
        game.ProcessUntil(Step::MAIN_ACTION);

        Player* curPlayer = game.GetCurrentPlayer();
        Player* opPlayer = game.GetOpponentPlayer();
        curPlayer->SetTotalMana(10);
        curPlayer->SetUsedMana(0);
        opPlayer->SetTotalMana(10);
        opPlayer->SetUsedMana(0);

        auto curSecret = curPlayer->GetSecretZone();

        const auto card1 = Generic::DrawCard(
            curPlayer, Cards::FindCardByName("Bazaar Burglary"));
        const auto card2 = Generic::DrawCard(
            curPlayer, Cards::FindCardByID("LOOT_998k"));  // Golden Kobold

        for (int i = 0; i < 8; ++i)
        {
            Generic::DrawCard(curPlayer, Cards::FindCardByID("LOOT_998k"));
        }

        auto quest = dynamic_cast<Spell*>(card1);

        game.Process(curPlayer, PlayCardTask::Spell(card1));
        CHECK(curSecret->quest);
        CHECK_EQ(quest->GetQuestProgress(), 0);
        CHECK_EQ(quest->GetQuestProgressTotal(), 4);

        game.Process(curPlayer, PlayCardTask::Minion(card2));
        CHECK_EQ(quest->GetQuestProgress(), 0);
    }
}

// ------------------------------------------ SPELL - ROGUE
// [ULD_328] Clever Disguise - COST:2
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: Add 2 random spells from another class to your hand.
// --------------------------------------------------------
TEST_CASE("[Rogue : Spell] - ULD_328 : Clever Disguise")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);
    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Clever Disguise"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));

    CHECK_EQ(curHand.GetCount(), 2);
    CHECK(curHand[0]->card->GetCardClass() != CardClass::ROGUE);
    CHECK_EQ(curHand[0]->card->GetCardType(), CardType::SPELL);
    CHECK(curHand[1]->card->GetCardClass() != CardClass::ROGUE);
    CHECK_EQ(curHand[1]->card->GetCardType(), CardType::SPELL);
}

// ------------------------------------------ SPELL - ROGUE
// [ULD_715] Plague of Madness - COST:1
// - Set: Uldum, Rarity: Rare
// - Spell School: Shadow
// --------------------------------------------------------
// Text: Each player equips a 2/2 Knife with <b>Poisonous</b>.
// --------------------------------------------------------
// GameTag:
// - 858 = 2451
// --------------------------------------------------------
// RefTag:
// - POISONOUS = 1
// --------------------------------------------------------
TEST_CASE("[Rogue : Spell] - ULD_715 : Plague of Madness")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Plague of Madness"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Chillwind Yeti"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));

    CHECK_EQ(curPlayer->GetHero()->weapon->GetAttack(), 2);
    CHECK_EQ(curPlayer->GetHero()->weapon->GetDurability(), 2);
    CHECK_EQ(opPlayer->GetHero()->weapon->GetAttack(), 2);
    CHECK_EQ(opPlayer->GetHero()->weapon->GetDurability(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    game.Process(opPlayer,
                 AttackTask(opPlayer->GetHero(), curPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->isDestroyed, false);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, AttackTask(curPlayer->GetHero(), card2));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 24);
    CHECK_EQ(opField.GetCount(), 0);
}

// --------------------------------------- MINION - WARRIOR
// [ULD_195] Frightened Flunky - COST:2 [ATK:2/HP:2]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Taunt</b>
//       <b>Battlecry:</b> <b>Discover</b> a <b>Taunt</b> minion.
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// - BATTLECRY = 1
// - DISCOVER = 1
// - USE_DISCOVER_VISUALS = 1
// --------------------------------------------------------
TEST_CASE("[Warrior : Minion] - ULD_195 : Frightened Flunky")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::PRIEST;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Frightened Flunky"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curPlayer->choice);

    auto cards = TestUtils::GetChoiceCards(game);
    for (auto& card : cards)
    {
        CHECK_EQ(card->GetCardType(), CardType::MINION);
        CHECK_EQ(card->HasGameTag(GameTag::TAUNT), true);
    }
}

// --------------------------------------- MINION - WARRIOR
// [ULD_206] Restless Mummy - COST:4 [ATK:3/HP:2]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Rush</b> <b>Reborn</b>
// --------------------------------------------------------
// GameTag:
// - RUSH = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Warrior : Minion] - ULD_206 : Restless Mummy")
{
    // Do nothing
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_157] Questing Explorer - COST:2 [ATK:2/HP:3]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If you control a <b>Quest</b>,
//       draw a card.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// RefTag:
// - QUEST = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_157 : Questing Explorer")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Questing Explorer"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Questing Explorer"));
    const auto card3 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Untapped Potential"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 6);

    game.Process(curPlayer, PlayCardTask::Spell(card3));
    CHECK_EQ(curHand.GetCount(), 5);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curHand.GetCount(), 5);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_174] Serpent Egg - COST:2 [ATK:0/HP:3]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Deathrattle:</b> Summon a 3/4 Sea Serpent.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_174 : Serpent Egg")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Serpent Egg"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card2, card1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->card->name, "Sea Serpent");
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_177] Octosari - COST:8 [ATK:8/HP:8]
// - Race: Beast, Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Deathrattle:</b> Draw 8 cards.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_177 : Octosari")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::HUNTER;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Octosari"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curPlayer->GetHandZone()->GetCount(), 4);
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    game.Process(opPlayer, PlayCardTask::Minion(card3));
    game.Process(opPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(opPlayer->GetHandZone()->GetCount(), 6);
    CHECK_EQ(opField.GetCount(), 3);

    game.Process(opPlayer, AttackTask(card2, card1));
    game.Process(opPlayer, AttackTask(card3, card1));
    game.Process(opPlayer, AttackTask(card4, card1));
    CHECK_EQ(curField.GetCount(), 0);
    CHECK_EQ(opField.GetCount(), 0);
    CHECK_EQ(curPlayer->GetHandZone()->GetCount(), 9);
    CHECK_EQ(opPlayer->GetHandZone()->GetCount(), 6);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_178] Siamat - COST:7 [ATK:6/HP:6]
// - Race: Elemental, Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Gain 2 of <b>Rush</b>,
//       <b>Taunt</b>, <b>Divine Shield</b>, or
//       <b>Windfury</b> <i>(your choice).</i>
// --------------------------------------------------------
// Entourage: ULD_178a2, ULD_178a, ULD_178a3, ULD_178a4
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - BATTLECRY = 1
// --------------------------------------------------------
// RefTag:
// - WINDFURY = 1
// - TAUNT = 1
// - DIVINE_SHIELD = 1
// - RUSH = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_178 : Siamat")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Siamat"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Siamat"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK(curPlayer->choice);

    TestUtils::ChooseNthChoice(game, 1);
    TestUtils::ChooseNthChoice(game, 1);

    CHECK_EQ(curField[0]->HasDivineShield(), true);
    CHECK_EQ(curField[0]->HasWindfury(), true);
    CHECK_EQ(curField[0]->HasTaunt(), false);
    CHECK_EQ(curField[0]->HasRush(), false);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK(curPlayer->choice);

    TestUtils::ChooseNthChoice(game, 4);
    TestUtils::ChooseNthChoice(game, 3);

    CHECK_EQ(curField[1]->HasDivineShield(), false);
    CHECK_EQ(curField[1]->HasWindfury(), false);
    CHECK_EQ(curField[1]->HasTaunt(), true);
    CHECK_EQ(curField[1]->HasRush(), true);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_179] Phalanx Commander - COST:5 [ATK:4/HP:5]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: Your <b>Taunt</b> minions have +2 Attack.
// --------------------------------------------------------
// GameTag:
// - AURA = 1
// - TECH_LEVEL = 3
// --------------------------------------------------------
// RefTag:
// - TAUNT = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_179 : Phalanx Commander")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Phalanx Commander"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Frostwolf Grunt"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[1]->GetAttack(), 2);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[1]->GetAttack(), 4);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card4, card1));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[1]->GetAttack(), 2);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_180] Sunstruck Henchman - COST:4 [ATK:6/HP:5]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: At the start of your turn,
//       this has a 50% chance to fall asleep.
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_180 : Phalanx Commander")
{
    GameConfig config;
    config.player1Class = CardClass::SHAMAN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Sunstruck Henchman"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->CanAttack(), false);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_182] Spitting Camel - COST:2 [ATK:2/HP:4]
// - Race: Beast, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: At the end of your turn, deal 1 damage
//       to another random friendly minion.
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_182 : Spitting Camel")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Spitting Camel"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Chillwind Yeti"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Chillwind Yeti"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField[1]->GetHealth(), 5);
    CHECK_EQ(curField[2]->GetHealth(), 5);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(curField[1]->GetHealth() + curField[2]->GetHealth(), 9);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(curField[1]->GetHealth() + curField[2]->GetHealth(), 9);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(curField[1]->GetHealth() + curField[2]->GetHealth(), 8);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_183] Anubisath Warbringer - COST:9 [ATK:9/HP:6]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Deathrattle:</b> Give all minions in your hand +3/+3.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_183 : Anubisath Warbringer")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Anubisath Warbringer"));
    [[maybe_unused]] const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetAttack(), 3);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetHealth(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card3, card1));
    CHECK_EQ(curField.GetCount(), 0);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetAttack(), 6);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetHealth(), 4);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_184] Kobold Sandtrooper - COST:2 [ATK:2/HP:1]
// - Faction: Alliance, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Deathrattle:</b> Deal 3 damage to the enemy hero.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_184 : Kobold Sandtrooper")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Kobold Sandtrooper"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card2, card1));
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 27);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_185] Temple Berserker - COST:2 [ATK:1/HP:2]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Reborn</b> Has +2 Attack while damaged.
// --------------------------------------------------------
// GameTag:
// - ENRAGED = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_185 : Temple Berserker")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Temple Berserker"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, HeroPowerTask(card1));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[0]->GetHealth(), 1);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, HeroPowerTask(card1));
    CHECK_EQ(curField[0]->GetAttack(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 2);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_188] Golden Scarab - COST:3 [ATK:2/HP:2]
// - Race: Beast, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b><b>Battlecry:</b> Discover</b> a 4-Cost card.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// - DISCOVER = 1
// - USE_DISCOVER_VISUALS = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_188 : Golden Scarab")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Golden Scarab"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK(curPlayer->choice);
    CHECK_EQ(curPlayer->choice->choices.size(), 3);

    auto cards = TestUtils::GetChoiceCards(game);
    for (auto& card : cards)
    {
        CHECK_EQ(card->GetCost(), 4);
    }
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_189] Faceless Lurker - COST:5 [ATK:3/HP:3]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Taunt</b>
//       <b>Battlecry:</b> Double this minion's Health.
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_189 : Faceless Lurker")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Faceless Lurker"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Anubisath Warbringer"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(dynamic_cast<Minion*>(card1)->GetHealth(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card3, card2));
    CHECK_EQ(dynamic_cast<Minion*>(card1)->GetHealth(), 6);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetHealth(), 12);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_190] Pit Crocolisk - COST:8 [ATK:5/HP:6]
// - Race: Beast, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Deal 5 damage.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_TO_PLAY = 0
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_190 : Pit Crocolisk")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Pit Crocolisk"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Boulderfist Ogre"));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(opField[0]->GetHealth(), 7);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card1, card2));
    CHECK_EQ(opField[0]->GetHealth(), 2);
    CHECK_EQ(curPlayer->GetHandZone()->GetCount(), 5);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_191] Beaming Sidekick - COST:1 [ATK:1/HP:2]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Give a friendly minion +2 Health.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_IF_AVAILABLE = 0
// - REQ_FRIENDLY_TARGET = 0
// - REQ_MINION_TARGET = 0
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_191 : Beaming Sidekick")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Beaming Sidekick"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Beaming Sidekick"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Beaming Sidekick"));

    game.Process(curPlayer, PlayCardTask::MinionTarget(card1, nullptr));
    CHECK_EQ(curField[0]->GetHealth(), 2);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card2, card1));
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(curField[1]->GetHealth(), 2);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card3, card2));
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(curField[1]->GetHealth(), 4);
    CHECK_EQ(curField[2]->GetHealth(), 2);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_193] Living Monument - COST:10 [ATK:10/HP:10]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Taunt</b>
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_193 : Living Monument")
{
    // Do noting
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_194] Wasteland Scorpid - COST:7 [ATK:3/HP:9]
// - Race: Beast, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Poisonous</b>
// --------------------------------------------------------
// GameTag:
// - POISONOUS = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_194 : Wasteland Scorpid")
{
    // Do nothing
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_196] Neferset Ritualist - COST:2 [ATK:2/HP:3]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Restore adjacent minions
//       to full Health.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_196 : Neferset Ritualist")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Neferset Ritualist"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Injured Blademaster"));
    const auto card3 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Injured Blademaster"));

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField[0]->GetHealth(), 3);
    CHECK_EQ(curField[1]->GetHealth(), 3);

    game.Process(curPlayer, PlayCardTask(card1, nullptr, 1));
    CHECK_EQ(curField[0]->GetHealth(), 7);
    CHECK_EQ(curField[2]->GetHealth(), 7);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_197] Quicksand Elemental - COST:2 [ATK:3/HP:2]
// - Race: Elemental, Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Give all enemy minions -2 Attack
//       this turn.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_197 : Quicksand Elemental")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Infested Goblin"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Jar Dealer"));
    const auto card3 = Generic::DrawCard(
        opPlayer, Cards::FindCardByName("Quicksand Elemental"));
    const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Walking Fountain"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 2);
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField[1]->GetAttack(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField[0]->GetAttack(), 0);
    CHECK_EQ(curField[1]->GetAttack(), 0);

    game.Process(opPlayer, PlayCardTask::Minion(card4));
    game.Process(opPlayer, AttackTask(card4, curField[0]));
    CHECK_EQ(opField[1]->GetHealth(), 8);
    game.Process(opPlayer, AttackTask(card4, curField[1]));
    CHECK_EQ(opField[1]->GetHealth(), 8);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_198] Conjured Mirage - COST:4 [ATK:3/HP:10]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Taunt</b> At the start of your turn,
//       shuffle this minion into your deck.
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_198 : Conjured Mirage")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Conjured Mirage"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Walking Fountain"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[0]->GetHealth(), 10);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    game.Process(opPlayer, AttackTask(card2, curField[0]));
    CHECK_EQ(curField[0]->GetHealth(), 6);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curField.GetCount(), 0);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_205] Candletaker - COST:3 [ATK:3/HP:2]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Reborn</b>
// --------------------------------------------------------
// GameTag:
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_205 : Candletaker")
{
    // Do nothing
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_208] Khartut Defender - COST:6 [ATK:3/HP:4]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Taunt</b>, <b>Reborn</b> <b>Deathrattle:</b>
//       Restore 3 Health to your hero.
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// - DEATHRATTLE = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_208 : Khartut Defender")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);
    curPlayer->GetHero()->SetDamage(6);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Khartut Defender"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Walking Fountain"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[0]->GetHealth(), 4);
    CHECK_EQ(curField[0]->HasReborn(), true);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    game.Process(opPlayer, AttackTask(card2, curField[0]));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[0]->GetHealth(), 1);
    CHECK_EQ(curField[0]->HasReborn(), false);
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 27);

    game.Process(opPlayer, AttackTask(card2, curField[0]));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 30);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_209] Vulpera Scoundrel - COST:3 [ATK:2/HP:3]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry</b>: <b>Discover</b> a spell or
//       pick a mystery choice.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// - DISCOVER = 1
// - USE_DISCOVER_VISUALS = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_209 : Vulpera Scoundrel")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);
    curPlayer->GetHero()->SetDamage(6);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Vulpera Scoundrel"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Vulpera Scoundrel"));

    SUBCASE("A spell card - Except ULD_209t")
    {
        game.Process(curPlayer, PlayCardTask::Spell(card1));
        CHECK(curPlayer->choice);

        auto cards = TestUtils::GetChoiceCards(game);
        for (std::size_t i = 0; i < 3; ++i)
        {
            CHECK_EQ(cards[i]->IsCardClass(CardClass::DRUID), true);
            CHECK_EQ(cards[i]->GetCardType(), CardType::SPELL);
        }
        CHECK_EQ(cards[3]->id, "ULD_209t");

        TestUtils::ChooseNthChoice(game, 1);
        CHECK_EQ(curHand.GetCount(), 2);
        CHECK_EQ(curHand[1]->card->IsCardClass(CardClass::DRUID), true);
        CHECK_EQ(curHand[1]->card->GetCardType(), CardType::SPELL);
    }

    SUBCASE("Mystery Choice! - ULD_209t")
    {
        game.Process(curPlayer, PlayCardTask::Spell(card2));
        CHECK(curPlayer->choice);

        auto cards = TestUtils::GetChoiceCards(game);
        for (std::size_t i = 0; i < 3; ++i)
        {
            CHECK_EQ(cards[i]->IsCardClass(CardClass::DRUID), true);
            CHECK_EQ(cards[i]->GetCardType(), CardType::SPELL);
        }
        CHECK_EQ(cards[3]->id, "ULD_209t");

        TestUtils::ChooseNthChoice(game, 4);
        CHECK_EQ(curHand.GetCount(), 2);
        CHECK_EQ(curHand[1]->card->IsCardClass(CardClass::DRUID), true);
        CHECK_EQ(curHand[1]->card->GetCardType(), CardType::SPELL);
    }
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_214] Generous Mummy - COST:3 [ATK:5/HP:4]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Reborn</b> Your opponent's cards cost (1) less.
// --------------------------------------------------------
// GameTag:
// - AURA = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_214 : Generous Mummy")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Generous Mummy"));
    [[maybe_unused]] const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));
    [[maybe_unused]] const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));
    [[maybe_unused]] const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Frostbolt"));

    CHECK_EQ(card2->GetCost(), 3);
    CHECK_EQ(card3->GetCost(), 4);
    CHECK_EQ(card4->GetCost(), 3);
    CHECK_EQ(card5->GetCost(), 2);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(card2->GetCost(), 3);
    CHECK_EQ(card3->GetCost(), 3);
    CHECK_EQ(card4->GetCost(), 2);
    CHECK_EQ(card5->GetCost(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card3, card1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(card2->GetCost(), 3);
    CHECK_EQ(card4->GetCost(), 2);
    CHECK_EQ(card5->GetCost(), 1);

    game.Process(opPlayer, HeroPowerTask(curField[0]));
    CHECK_EQ(curField.GetCount(), 0);
    CHECK_EQ(card2->GetCost(), 3);
    CHECK_EQ(card4->GetCost(), 3);
    CHECK_EQ(card5->GetCost(), 2);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_215] Wrapped Golem - COST:7 [ATK:7/HP:5]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Reborn</b> At the end of your turn,
//       summon a 1/1 Scarab with <b>Taunt</b>.
// --------------------------------------------------------
// GameTag:
// - REBORN = 1
// --------------------------------------------------------
// RefTag:
// - TAUNT = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_215 : Wrapped Golem")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wrapped Golem"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(curField[1]->card->name, "Scarab");
    CHECK_EQ(curField[1]->GetAttack(), 1);
    CHECK_EQ(curField[1]->GetHealth(), 1);
    CHECK_EQ(curField[1]->HasTaunt(), true);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_229] Mischief Maker - COST:3 [ATK:3/HP:3]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Swap the top card of your deck
//       with your opponent's.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_229 : Mischief Maker")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::SHAMAN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    for (int i = 0; i < 30; ++i)
    {
        config.player1Deck[i] = Cards::FindCardByName("Malygos");
        config.player2Deck[i] = Cards::FindCardByName("Wolfrider");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& opHand = *(opPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mischief Maker"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 4);
    CHECK_EQ(opHand.GetCount(), 5);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(opHand.GetCount(), 6);
    CHECK_EQ(opHand[5]->card->name, "Malygos");

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curHand.GetCount(), 5);
    CHECK_EQ(curHand[4]->card->name, "Wolfrider");

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(opHand.GetCount(), 7);
    CHECK_EQ(opHand[6]->card->name, "Wolfrider");

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(curHand.GetCount(), 6);
    CHECK_EQ(curHand[5]->card->name, "Malygos");
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_250] Infested Goblin - COST:3 [ATK:2/HP:3]
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Taunt</b> <b>Deathrattle:</b> Add two 1/1 Scarabs
//       with <b>Taunt</b> to your hand.
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_250 : Infested Goblin")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::PALADIN;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Infested Goblin"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 2);
    CHECK_EQ(curField[0]->GetHealth(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    game.Process(opPlayer, AttackTask(card2, curField[0]));
    CHECK_EQ(curPlayer->GetHandZone()->GetCount(), 6);
    CHECK_EQ(curHand[4]->card->name, "Scarab");
    CHECK_EQ(curHand[5]->card->name, "Scarab");
}

// --------------------------------------- MINION - WARRIOR
// [ULD_253] Tomb Warden - COST:8 [ATK:3/HP:6]
// - Race: Mechanical, Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: <b>Taunt</b>
//       <b>Battlecry:</b> Summon a copy of this minion.
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Warrior : Minion] - ULD_253 : Tomb Warden")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;
    config.doShuffle = false;

    for (int i = 0; i < 30; ++i)
    {
        config.player1Deck[i] = Cards::FindCardByName("Tomb Warden");
    }

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(30);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Galakrond, the Unbreakable"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(5, curHand.GetCount());
    CHECK_EQ("Tomb Warden", curHand[4]->card->name);

    game.Process(curPlayer, PlayCardTask::Minion(curHand[4]));
    CHECK_EQ(2, curField.GetCount());
    CHECK_EQ("Tomb Warden", curField[0]->card->name);
    CHECK_EQ("Tomb Warden", curField[1]->card->name);
    CHECK_EQ(7, curField[0]->GetAttack());
    CHECK_EQ(7, curField[1]->GetAttack());
    CHECK_EQ(10, curField[0]->GetHealth());
    CHECK_EQ(10, curField[1]->GetHealth());

    game.Process(curPlayer, PlayCardTask::Minion(curHand[0]));
    CHECK_EQ(4, curField.GetCount());
    CHECK_EQ("Tomb Warden", curField[2]->card->name);
    CHECK_EQ("Tomb Warden", curField[3]->card->name);
    CHECK_EQ(3, curField[2]->GetAttack());
    CHECK_EQ(3, curField[3]->GetAttack());
    CHECK_EQ(6, curField[2]->GetHealth());
    CHECK_EQ(6, curField[3]->GetHealth());
}

// ---------------------------------------- SPELL - WARRIOR
// [ULD_256] Into the Fray - COST:1
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: Give all <b>Taunt</b> minions in your hand +2/+2.
// --------------------------------------------------------
// RefTag:
// - TAUNT = 1
// --------------------------------------------------------
TEST_CASE("[Warrior : Spell] - ULD_256 : Into the Fray")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Into the Fray"));
    [[maybe_unused]] const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    [[maybe_unused]] const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Frostwolf Grunt"));

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curHand.GetCount(), 2);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetAttack(), 3);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetHealth(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[1])->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[1])->GetHealth(), 4);
}

// --------------------------------------- MINION - WARRIOR
// [ULD_258] Armagedillo - COST:6 [ATK:4/HP:7]
// - Race: Beast, Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Taunt</b> At the end of your turn,
//       give all <b>Taunt</b> minions in your hand +2/+2.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - TAUNT = 1
// --------------------------------------------------------
TEST_CASE("[Warrior : Minion] - ULD_258 : Armagedillo")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& opHand = *(opPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Armagedillo"));
    [[maybe_unused]] const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    [[maybe_unused]] const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Frostwolf Grunt"));
    [[maybe_unused]] const auto card4 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Frostwolf Grunt"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetAttack(), 3);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetHealth(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[1])->GetAttack(), 2);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[1])->GetHealth(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetAttack(), 3);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[0])->GetHealth(), 1);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[1])->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[1])->GetHealth(), 4);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(dynamic_cast<Minion*>(opHand[1])->GetAttack(), 2);
    CHECK_EQ(dynamic_cast<Minion*>(opHand[1])->GetHealth(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[1])->GetAttack(), 6);
    CHECK_EQ(dynamic_cast<Minion*>(curHand[1])->GetHealth(), 6);
}

// ---------------------------------------- SPELL - WARRIOR
// [ULD_707] Plague of Wrath - COST:5
// - Set: Uldum, Rarity: Rare
// --------------------------------------------------------
// Text: Destroy all damaged minions.
// --------------------------------------------------------
// GameTag:
// - 858 = 41425
// --------------------------------------------------------
TEST_CASE("[Warrior : Spell] - ULD_707 : Plague of Wrath")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Plague of Wrath"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Whirlwind"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Bloodfen Raptor"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Bloodfen Raptor"));
    const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Bloodfen Raptor"));

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card5));
    CHECK_EQ(opField.GetCount(), 1);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Spell(card2));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(opField.GetCount(), 1);

    game.Process(curPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(opField.GetCount(), 1);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(opField.GetCount(), 0);
}

// --------------------------------------- WEAPON - WARRIOR
// [ULD_708] Livewire Lance - COST:3 [ATK:2/HP:0]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: After your Hero attacks,
//       add a <b>Lackey</b> to your hand.
// --------------------------------------------------------
// GameTag:
// - DURABILITY = 2
// - 1359 = 1
// --------------------------------------------------------
// RefTag:
// - MARK_OF_EVIL = 1
// --------------------------------------------------------
TEST_CASE("[Warrior : Weapon] - ULD_708 : Livewire Lance")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Livewire Lance"));

    game.Process(curPlayer, PlayCardTask::Weapon(card1));
    CHECK_EQ(curHand.GetCount(), 0);

    game.Process(curPlayer,
                 AttackTask(curPlayer->GetHero(), opPlayer->GetHero()));
    CHECK_EQ(curHand.GetCount(), 1);
    CHECK_EQ(curHand[0]->card->IsLackey(), true);
}

// --------------------------------------- MINION - WARRIOR
// [ULD_709] Armored Goon - COST:6 [ATK:6/HP:7]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: Whenever your hero attacks, gain 5 Armor.
// --------------------------------------------------------
TEST_CASE("[Warrior : Minion] - ULD_709 : Armored Goon")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Armored Goon"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fiery War Axe"));

    game.Process(curPlayer, PlayCardTask::Weapon(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer,
                 AttackTask(curPlayer->GetHero(), opPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->GetArmor(), 5);
}

// ---------------------------------------- SPELL - WARRIOR
// [ULD_711] Hack the System - COST:1
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Quest:</b> Attack 5 times with your hero.
//       <b>Reward:</b> Anraphet's Core.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - QUEST = 1
// - QUEST_PROGRESS_TOTAL = 5
// - 676 = 1
// - 839 = 1
// - QUEST_REWARD_DATABASE_ID = 54416
// --------------------------------------------------------
TEST_CASE("[Warrior : Spell] - ULD_711 : Hack the System")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::PRIEST;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto curHero = curPlayer->GetHero();
    auto opHero = opPlayer->GetHero();
    auto& curField = *(curPlayer->GetFieldZone());
    const auto curSecret = curPlayer->GetSecretZone();

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Hack the System"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fiery War Axe"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fiery War Axe"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fiery War Axe"));

    auto quest = dynamic_cast<Spell*>(card1);

    game.Process(curPlayer, PlayCardTask::Spell(card1));
    CHECK(curSecret->quest);
    CHECK_EQ(quest->GetQuestProgress(), 0);
    CHECK_EQ(quest->GetQuestProgressTotal(), 5);

    game.Process(curPlayer, PlayCardTask::Weapon(card2));
    game.Process(curPlayer, AttackTask(curHero, opHero));
    CHECK_EQ(opHero->GetHealth(), 27);
    CHECK_EQ(quest->GetQuestProgress(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, AttackTask(curHero, opHero));
    CHECK_EQ(opHero->GetHealth(), 24);
    CHECK_EQ(quest->GetQuestProgress(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Weapon(card3));
    game.Process(curPlayer, AttackTask(curHero, opHero));
    CHECK_EQ(opHero->GetHealth(), 21);
    CHECK_EQ(quest->GetQuestProgress(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, AttackTask(curHero, opHero));
    CHECK_EQ(opHero->GetHealth(), 18);
    CHECK_EQ(quest->GetQuestProgress(), 4);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Weapon(card4));
    game.Process(curPlayer, AttackTask(curHero, opHero));
    CHECK(!curSecret->quest);
    CHECK_EQ(opHero->GetHealth(), 15);
    CHECK_EQ(quest->GetQuestProgress(), 5);
    CHECK_EQ(curHero->heroPower->card->id, "ULD_711p3");

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->card->name, "Stone Golem");
    CHECK_EQ(curField[0]->GetAttack(), 4);
    CHECK_EQ(curField[0]->GetHealth(), 3);

    game.Process(curPlayer, AttackTask(curHero, opHero));
    CHECK_EQ(opHero->GetHealth(), 12);

    game.Process(curPlayer, HeroPowerTask());
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(curField[1]->card->name, "Stone Golem");
    CHECK_EQ(curField[1]->GetAttack(), 4);
    CHECK_EQ(curField[1]->GetHealth(), 3);
}

// --------------------------------------- MINION - WARRIOR
// [ULD_720] Bloodsworn Mercenary - COST:3 [ATK:3/HP:3]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry</b>: Choose a damaged friendly minion.
//       Summon a copy of it.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// PlayReq:
// - REQ_TARGET_IF_AVAILABLE = 0
// - REQ_FRIENDLY_TARGET = 0
// - REQ_MINION_TARGET = 0
// - REQ_DAMAGED_TARGET = 0
// --------------------------------------------------------
TEST_CASE("[Warrior : Minion] - ULD_720 : Bloodsworn Mercenary")
{
    GameConfig config;
    config.player1Class = CardClass::WARRIOR;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Bloodsworn Mercenary"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Bloodsworn Mercenary"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Rampage"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[0]->GetHealth(), 3);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card2, card1));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, HeroPowerTask(card1));
    CHECK_EQ(curField[0]->GetAttack(), 3);
    CHECK_EQ(curField[0]->GetHealth(), 2);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::SpellTarget(card3, card1));
    CHECK_EQ(curField[0]->GetAttack(), 6);
    CHECK_EQ(curField[0]->GetHealth(), 5);

    game.Process(curPlayer, PlayCardTask::MinionTarget(card2, card1));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[1]->GetAttack(), 6);
    CHECK_EQ(curField[1]->GetHealth(), 5);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_271] Injured Tol'vir - COST:2 [ATK:2/HP:6]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Taunt</b>
//       <b>Battlecry:</b> Deal 3 damage to this minion.
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_271 : Injured Tol'vir")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Injured Tol'vir"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetDamage(), 3);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_274] Wasteland Assassin - COST:5 [ATK:4/HP:2]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Stealth</b> <b>Reborn</b>
// --------------------------------------------------------
// GameTag:
// - STEALTH = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_274 : Wasteland Assassin")
{
    // Do nothing
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_275] Bone Wraith - COST:4 [ATK:2/HP:5]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Taunt</b> <b>Reborn</b>
// --------------------------------------------------------
// GameTag:
// - TAUNT = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_275 : Bone Wraith")
{
    // Do nothing
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_282] Jar Dealer - COST:1 [ATK:1/HP:1]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Deathrattle:</b> Add a random 1-Cost minion
//       to your hand.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_282 : Jar Dealer")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::PRIEST;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Jar Dealer"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Stonetusk Boar"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField[0]->GetAttack(), 1);
    CHECK_EQ(curField[0]->GetHealth(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    game.Process(opPlayer, AttackTask(card2, curField[0]));

    CHECK_EQ(curHand.GetCount(), 5);
    CHECK_EQ(curHand[4]->card->GetCost(), 1);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_289] Fishflinger - COST:2 [ATK:3/HP:2]
// - Race: Murloc, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Add a random Murloc
//       to each player's hand.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_289 : Fishflinger")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());
    auto& opHand = *(opPlayer->GetHandZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fishflinger"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curHand.GetCount(), 5);
    CHECK_EQ(opHand.GetCount(), 6);
    CHECK_EQ(curHand[4]->card->GetRace(), Race::MURLOC);
    CHECK_EQ(opHand[5]->card->GetRace(), Race::MURLOC);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_290] History Buff - COST:3 [ATK:3/HP:4]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: Whenever you play a minion,
//       give a random minion in your hand +1/+1.
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_290 : History Buff")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("History Buff"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wisp"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fireball"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetAttack(), 3);
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetHealth(), 1);

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetHealth(), 2);

    game.Process(curPlayer,
                 PlayCardTask::SpellTarget(card4, opPlayer->GetHero()));
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetAttack(), 4);
    CHECK_EQ(dynamic_cast<Minion*>(card2)->GetHealth(), 2);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_304] King Phaoris - COST:10 [ATK:5/HP:5]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Battlecry:</b> For each spell in your hand,
//       summon a random minion of the same Cost.
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_304 : King Phaoris")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("King Phaoris"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("King Phaoris"));
    const auto card3 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Sorcerer's Apprentice"));
    [[maybe_unused]] const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Fireball"));
    [[maybe_unused]] const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Flamestrike"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[1]->card->GetCost(), 4);
    CHECK_EQ(curField[2]->card->GetCost(), 7);

    curPlayer->SetUsedMana(0);

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField.GetCount(), 4);

    curPlayer->SetUsedMana(0);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 7);
    CHECK_EQ(curField[5]->card->GetCost(),
             curField[1]->card->name == "Kirin Tor Tricaster" ? 4 : 3);
    CHECK_EQ(curField[6]->card->GetCost(),
             curField[1]->card->name == "Kirin Tor Tricaster" ? 7 : 6);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_309] Dwarven Archaeologist - COST:2 [ATK:2/HP:3]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: After you <b>Discover</b> a card,
//       reduce its cost by (1).
// --------------------------------------------------------
// RefTag:
// - DISCOVER = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_309 : Dwarven Archaeologist")
{
    GameConfig config;
    config.player1Class = CardClass::DRUID;
    config.player2Class = CardClass::WARRIOR;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curHand = *(curPlayer->GetHandZone());

    const auto card1 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Dwarven Archaeologist"));
    const auto card2 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Worthy Expedition"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Spell(card2));
    CHECK(curPlayer->choice);

    TestUtils::ChooseNthChoice(game, 1);
    CHECK_EQ(curHand.GetCount(), 1);

    const int reducedCost = curHand[0]->GetCost();
    const int originalCost = curHand[0]->card->GetCost();
    const int costDifference = originalCost - reducedCost;
    CHECK_LE(costDifference, 1);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_450] Vilefiend - COST:2 [ATK:2/HP:2]
// - Race: Demon, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Lifesteal</b>
// --------------------------------------------------------
// GameTag:
// - LIFESTEAL = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_450 : Vilefiend")
{
    GameConfig config;
    config.player1Class = CardClass::PRIEST;
    config.player2Class = CardClass::PRIEST;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Vilefiend"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Lightwarden"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer,
                 PlayCardTask::SpellTarget(card3, curPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 24);
    CHECK_EQ(curField[1]->GetAttack(), 1);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, AttackTask(card1, opPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 26);
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 28);
    CHECK_EQ(curField[1]->GetAttack(), 3);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, HeroPowerTask(curPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 28);
    CHECK_EQ(curField[1]->GetAttack(), 5);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, HeroPowerTask(curPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 30);
    CHECK_EQ(curField[1]->GetAttack(), 7);

    game.Process(curPlayer, AttackTask(card1, opPlayer->GetHero()));
    CHECK_EQ(curPlayer->GetHero()->GetHealth(), 30);
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 26);
    CHECK_EQ(curField[1]->GetAttack(), 7);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_702] Mortuary Machine - COST:5 [ATK:8/HP:8]
// - Race: Mechanical, Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: After your opponent plays a minion,
//       give it <b>Reborn</b>.
// --------------------------------------------------------
// RefTag:
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_702 : Mortuary Machine")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mortuary Machine"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));
    const auto card3 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Wolfrider"));

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(opField.GetCount(), 1);
    CHECK_EQ(opField[0]->HasReborn(), false);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(opField.GetCount(), 2);
    CHECK_EQ(opField[1]->HasReborn(), true);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_703] Desert Obelisk - COST:5 [ATK:0/HP:5]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: If you control 3 of these at the end of your turn,
//       deal 5 damage to a random enemy.
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_703 : Desert Obelisk")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Desert Obelisk"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Desert Obelisk"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Desert Obelisk"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Desert Obelisk"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 30);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    game.Process(curPlayer, PlayCardTask::Minion(card4));
    CHECK_EQ(curField.GetCount(), 4);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 10);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_705] Mogu Cultist - COST:1 [ATK:1/HP:1]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry:</b> If your board is full of Mogu Cultists,
//       sacrifice them all and summon Highkeeper Ra.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_705 : Mogu Cultist")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mogu Cultist"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mogu Cultist"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mogu Cultist"));
    const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mogu Cultist"));
    const auto card5 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mogu Cultist"));
    const auto card6 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mogu Cultist"));
    const auto card7 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Mogu Cultist"));
    const auto card8 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Malygos"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    game.Process(curPlayer, PlayCardTask::Minion(card2));
    game.Process(curPlayer, PlayCardTask::Minion(card3));
    game.Process(curPlayer, PlayCardTask::Minion(card4));
    game.Process(curPlayer, PlayCardTask::Minion(card5));
    game.Process(curPlayer, PlayCardTask::Minion(card6));
    CHECK_EQ(curField.GetCount(), 6);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card8));
    CHECK_EQ(opField.GetCount(), 1);
    CHECK_EQ(opField[0]->GetHealth(), 12);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    opPlayer->GetHero()->SetDamage(0);

    game.Process(curPlayer, PlayCardTask::Minion(card7));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->card->name, "Highkeeper Ra");

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    CHECK_EQ(opField.GetCount(), 0);
    CHECK_EQ(opPlayer->GetHero()->GetHealth(), 4);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_706] Blatant Decoy - COST:6 [ATK:5/HP:5]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Deathrattle:</b> Each player summons
//       the lowest Cost minion from their hand.
// --------------------------------------------------------
// GameTag:
// - DEATHRATTLE = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_706 : Blatant Decoy")
{
    GameConfig config;
    config.player1Class = CardClass::PALADIN;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Blatant Decoy"));
    [[maybe_unused]] const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Consecration"));
    [[maybe_unused]] const auto card3 = Generic::DrawCard(
        curPlayer, Cards::FindCardByName("Ancestral Guardian"));
    [[maybe_unused]] const auto card4 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Tirion Fordring"));
    const auto card5 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Fireball"));
    [[maybe_unused]] const auto card6 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Blizzard"));
    [[maybe_unused]] const auto card7 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Windfury Harpy"));
    [[maybe_unused]] const auto card8 = Generic::DrawCard(
        opPlayer, Cards::FindCardByName("Ravenholdt Assassin"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::SpellTarget(card5, card1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curField[0]->card->name, "Ancestral Guardian");
    CHECK_EQ(opField.GetCount(), 1);
    CHECK_EQ(opField[0]->card->name, "Windfury Harpy");
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_712] Bug Collector - COST:2 [ATK:2/HP:1]
// - Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Summon a 1/1 Locust with <b>Rush</b>.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
// RefTag:
// - RUSH = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_712 : Bug Collector")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());
    auto& opField = *(opPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Bug Collector"));
    const auto card2 =
        Generic::DrawCard(opPlayer, Cards::FindCardByName("Bug Collector"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(curField[1]->card->name, "Locust");
    CHECK_EQ(curField[1]->GetAttack(), 1);
    CHECK_EQ(curField[1]->GetHealth(), 1);
    CHECK_EQ(curField[1]->GetGameTag(GameTag::RUSH), 1);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, PlayCardTask::Minion(card2));
    CHECK_EQ(opField.GetCount(), 2);
    CHECK_EQ(opField[1]->card->name, "Locust");
    CHECK_EQ(opField[1]->GetAttack(), 1);
    CHECK_EQ(opField[1]->GetHealth(), 1);
    CHECK_EQ(opField[1]->GetGameTag(GameTag::RUSH), 1);

    game.Process(opPlayer, AttackTask(opField[1], curField[1]));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(opField.GetCount(), 1);
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_719] Desert Hare - COST:3 [ATK:1/HP:1]
// - Race: Beast, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Battlecry:</b> Summon two 1/1 Desert Hares.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_719 : Desert Hare")
{
    GameConfig config;
    config.player1Class = CardClass::MAGE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = true;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Desert Hare"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 3);
    CHECK_EQ(curField[0]->card->name, "Desert Hare");
    CHECK_EQ(curField[1]->card->name, "Desert Hare");
    CHECK_EQ(curField[2]->card->name, "Desert Hare");
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_721] Colossus of the Moon - COST:10 [ATK:10/HP:10]
// - Set: Uldum, Rarity: Legendary
// --------------------------------------------------------
// Text: <b>Divine Shield</b> <b>Reborn</b>
// --------------------------------------------------------
// GameTag:
// - ELITE = 1
// - DIVINE_SHIELD = 1
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_721 : Colossus of the Moon")
{
    // Do nothing
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_723] Murmy - COST:1 [ATK:1/HP:1]
// - Race: Murloc, Set: Uldum, Rarity: Common
// --------------------------------------------------------
// Text: <b>Reborn</b>
// --------------------------------------------------------
// GameTag:
// - REBORN = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_723 : Murmy")
{
    // Do nothing
}

// --------------------------------------- MINION - NEUTRAL
// [ULD_727] Body Wrapper - COST:4 [ATK:4/HP:4]
// - Set: Uldum, Rarity: Epic
// --------------------------------------------------------
// Text: <b>Battlecry:</b> <b>Discover</b> a friendly minion
//       that died this game. Shuffle it into your deck.
// --------------------------------------------------------
// GameTag:
// - BATTLECRY = 1
// - DISCOVER = 1
// - USE_DISCOVER_VISUALS = 1
// --------------------------------------------------------
TEST_CASE("[Neutral : Minion] - ULD_727 : Body Wrapper")
{
    GameConfig config;
    config.player1Class = CardClass::ROGUE;
    config.player2Class = CardClass::MAGE;
    config.startPlayer = PlayerType::PLAYER1;
    config.doFillDecks = false;
    config.autoRun = false;

    Game game(config);
    game.Start();
    game.ProcessUntil(Step::MAIN_ACTION);

    Player* curPlayer = game.GetCurrentPlayer();
    Player* opPlayer = game.GetOpponentPlayer();
    curPlayer->SetTotalMana(10);
    curPlayer->SetUsedMana(0);
    opPlayer->SetTotalMana(10);
    opPlayer->SetUsedMana(0);

    auto& curDeck = *(curPlayer->GetDeckZone());
    auto& curField = *(curPlayer->GetFieldZone());

    const auto card1 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Body Wrapper"));
    const auto card2 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Body Wrapper"));
    const auto card3 =
        Generic::DrawCard(curPlayer, Cards::FindCardByName("Wolfrider"));

    game.Process(curPlayer, PlayCardTask::Minion(card1));
    CHECK_EQ(curField.GetCount(), 1);
    CHECK_EQ(curDeck.GetCount(), 0);

    game.Process(curPlayer, PlayCardTask::Minion(card3));
    CHECK_EQ(curField.GetCount(), 2);

    game.Process(curPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(opPlayer, HeroPowerTask(card3));
    CHECK_EQ(curField.GetCount(), 1);

    game.Process(opPlayer, EndTurnTask());
    game.ProcessUntil(Step::MAIN_ACTION);

    game.Process(curPlayer, PlayCardTask::Minion(card2));
    CHECK(curPlayer->choice);

    auto cards = TestUtils::GetChoiceCards(game);
    CHECK_EQ(cards.size(), 1);
    CHECK_EQ(cards[0]->name, "Wolfrider");

    TestUtils::ChooseNthChoice(game, 1);
    CHECK_EQ(curField.GetCount(), 2);
    CHECK_EQ(curDeck.GetCount(), 1);
    CHECK_EQ(curDeck[0]->card->name, "Wolfrider");
}