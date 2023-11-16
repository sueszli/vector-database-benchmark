//
// Created by andre on 28.06.18.
//

#include "CharacterEquipment.h"
#include <engine/World.h>
#include <logic/PlayerController.h>
#include <logic/ScriptEngine.h>

using namespace Logic;

CharacterEquipment::CharacterEquipment(World::WorldInstance& world, Handle::EntityHandle characterEntity)
    : m_CharacterEntity(characterEntity)
    , m_World(world)
{
}

bool CharacterEquipment::equipItemToSlot(ItemHandle item, Slot slot)
{
    if (!isItemEquipable(item))
        return false;

    if (!isItemTypeCorrectForSlot(item, slot))
        return false;

    LogInfo() << "Equipping item: " << getInstanceNameOfItem(item);

    switch (getKindOfItem(item))
    {
        case Kind::MELEE:
            return equipMelee(item);

        case Kind::BOW:
            return equipBow(item);

        case Kind::AMULET:
            return equipAmulet(item);

        case Kind::RING:
            return equipRing(item, slot);

        case Kind::MAGIC:
            return equipMagic(item, slot);

        case Kind::BELT:
            return equipBelt(item);

        case Kind::ARMOR:
            return equipArmor(item);

        case Kind::OTHER:
        default:
            return false;
    }

    return true;
}

bool CharacterEquipment::equipItem(ItemHandle item)
{
    if (!isItemEquipable(item))
        return false;

    tl::optional<Slot> slot;
    switch (getKindOfItem(item))
    {
        case Kind::MELEE:
        case Kind::BOW:
        case Kind::AMULET:
        case Kind::BELT:
        case Kind::ARMOR:
            slot = getCorrectSlotForItem(item);
            break;

        case Kind::RING:
            slot = findAnyFreeRingSlot();
            break;

        case Kind::MAGIC:
            slot = findAnyFreeMagicSlot();
            break;

        case Kind::OTHER:
        default:
            return false;
    }

    if (!slot)
        return false;

    return equipItemToSlot(item, *slot);
}

void CharacterEquipment::unequipItemInSlot(Slot slot)
{
    switch (slot)
    {
        case Slot::MELEE:
            removeCharacterModelAttachment(EModelNode::Longsword);
            removeCharacterModelAttachment(EModelNode::Sword);
            break;

        case Slot::BOW:
            removeCharacterModelAttachment(EModelNode::Bow);
            removeCharacterModelAttachment(EModelNode::Crossbow);
            break;

        case Slot::ARMOR:
            switchToDefaultCharacterModel();
            break;

        default:  // Everything else can be taken off without visual changes
            break;
    }
}

void CharacterEquipment::unequipItem(ItemHandle item)
{
    auto slot = findSlotItemWasEquippedTo(item);

    if (slot)
    {
        unequipItemInSlot(*slot);
    }
}

bool CharacterEquipment::equipMelee(ItemHandle item)
{
    setItemInSlot(item, Slot::MELEE);

    return true;
}

bool CharacterEquipment::equipBow(ItemHandle item)
{
    setItemInSlot(item, Slot::BOW);

    return true;
}

bool CharacterEquipment::equipAmulet(ItemHandle item)
{
    setItemInSlot(item, Slot::AMULET);
    return true;
}

bool CharacterEquipment::equipRing(ItemHandle item, Slot slot)
{
    setItemInSlot(item, slot);
    return true;
}

bool CharacterEquipment::equipMagic(ItemHandle item, Slot slot)
{
    setItemInSlot(item, slot);
    return true;
}

bool CharacterEquipment::equipBelt(ItemHandle item)
{
    setItemInSlot(item, Slot::BELT);
    return true;
}

bool CharacterEquipment::equipArmor(ItemHandle item)
{
    auto data = getDataOfItem(item);

    if (!data)
        return false;

    switchCharacterModelArmor(data->visual_change);
    setItemInSlot(item, Slot::ARMOR);
    return true;
}

void CharacterEquipment::showMeleeWeaponOnCharacter()
{
    using Daedalus::GEngineClasses::C_Item;

    auto item = getItemInSlot(Slot::MELEE);
    auto data = getDataOfItem(item);

    if (!data)
        return;

    EModelNode node;
    if (data->flags & C_Item::Flags::ITEM_2HD_AXE)
    {
        node = EModelNode::Longsword;
    }
    else if (data->flags & C_Item::Flags::ITEM_2HD_SWD)
    {
        node = EModelNode::Longsword;
    }
    else if (data->flags & C_Item::Flags::ITEM_AXE)
    {
        node = EModelNode::Sword;
    }
    else if (data->flags & C_Item::Flags::ITEM_SWD)
    {
        node = EModelNode::Sword;
    }
    else
    {
        // What is this?
        return;
    }

    setCharacterModelAttachment(getItemVisual(item), node);
}

void CharacterEquipment::showBowWeaponOnCharacter()
{
    using Daedalus::GEngineClasses::C_Item;
    auto item = getItemInSlot(Slot::BOW);
    auto data = getDataOfItem(item);

    if (!data)
        return;

    if (data->flags & C_Item::Flags::ITEM_CROSSBOW)
    {
        setCharacterModelAttachment(getItemVisual(item), EModelNode::Crossbow);
    }
    else if (data->flags & C_Item::Flags::ITEM_BOW)
    {
        setCharacterModelAttachment(getItemVisual(item), EModelNode::Bow);
    }
    else
    {
        // What is this?
    }
}

void CharacterEquipment::putMeleeWeaponInCharactersHand()
{
    auto item = getItemInSlot(Slot::MELEE);

    putItemIntoRightHand(item);
    removeCharacterModelAttachment(EModelNode::Sword);
    removeCharacterModelAttachment(EModelNode::Longsword);
}

void CharacterEquipment::putBowWeaponInCharactersHand()
{
    auto item = getItemInSlot(Slot::BOW);

    putItemIntoRightHand(item);
    removeCharacterModelAttachment(EModelNode::Bow);
    removeCharacterModelAttachment(EModelNode::Crossbow);
}

void CharacterEquipment::removeItemInCharactersHandAndShowWeaponsOnBody()
{
    removeCharacterModelAttachment(EModelNode::Lefthand);
    removeCharacterModelAttachment(EModelNode::Righthand);

    // Need to put those back onto the character if they exist
    showBowWeaponOnCharacter();
    showMeleeWeaponOnCharacter();
}

tl::optional<CharacterEquipment::Slot> CharacterEquipment::getCorrectSlotForItem(ItemHandle item) const
{
    switch (getKindOfItem(item))
    {
        case Kind::MELEE:
            return Slot::MELEE;
        case Kind::BOW:
            return Slot::BOW;
        case Kind::AMULET:
            return Slot::AMULET;
        case Kind::RING:
            return Slot::RING_LEFT;
        case Kind::MAGIC:
            return Slot::MAGIC_0;
        case Kind::ARMOR:
            return Slot::ARMOR;
        case Kind::BELT:
            return Slot::BELT;
        case Kind::OTHER:
        default:
            return tl::nullopt;
    }
}

CharacterEquipment::Kind CharacterEquipment::getKindOfItem(ItemHandle item) const
{
    using Daedalus::GEngineClasses::C_Item;
    auto itemData = getDataOfItem(item);

    if (!itemData)
        return Kind::OTHER;

    // Only equippable items get a kind
    if ((itemData->mainflag & C_Item::ITM_CAT_EQUIPABLE) == 0)
        return Kind::OTHER;

    // Magic and armor can only be identified through the mainflags
    if ((itemData->mainflag & C_Item::ITM_CAT_RUNE) != 0)
        return Kind::MAGIC;

    if ((itemData->mainflag & C_Item::ITM_CAT_MAGIC) != 0)
        return Kind::MAGIC;

    if ((itemData->mainflag & C_Item::ITM_CAT_ARMOR) != 0)
        return Kind::ARMOR;

    std::pair<C_Item::Flags, Kind> pairs[] = {
        {C_Item::Flags::ITEM_DAG, Kind::MELEE},
        {C_Item::Flags::ITEM_SWD, Kind::MELEE},
        {C_Item::Flags::ITEM_AXE, Kind::MELEE},
        {C_Item::Flags::ITEM_2HD_SWD, Kind::MELEE},
        {C_Item::Flags::ITEM_2HD_AXE, Kind::MELEE},
        {C_Item::Flags::ITEM_BOW, Kind::BOW},
        {C_Item::Flags::ITEM_CROSSBOW, Kind::BOW},
        {C_Item::Flags::ITEM_AMULET, Kind::AMULET},
        {C_Item::Flags::ITEM_RING, Kind::RING},
        {C_Item::Flags::ITEM_BELT, Kind::BELT},
        {C_Item::Flags::ITEM_MISSION, Kind::OTHER},
    };

    for (auto& p : pairs)
    {
        if (itemData->flags & p.first)
            return p.second;
    }

    return Kind::OTHER;
}

CharacterEquipment::WeaponKind CharacterEquipment::getWeaponKindOfItem(ItemHandle item) const
{
    using Daedalus::GEngineClasses::C_Item;

    auto itemData = getDataOfItem(item);

    if (!itemData)
        return WeaponKind::NONE;

    std::pair<C_Item::Flags, WeaponKind> pairs[] = {
        {C_Item::Flags::ITEM_DAG, WeaponKind::MELEE_1H},
        {C_Item::Flags::ITEM_SWD, WeaponKind::MELEE_1H},
        {C_Item::Flags::ITEM_AXE, WeaponKind::MELEE_1H},
        {C_Item::Flags::ITEM_2HD_SWD, WeaponKind::MELEE_2H},
        {C_Item::Flags::ITEM_2HD_AXE, WeaponKind::MELEE_2H},
        {C_Item::Flags::ITEM_BOW, WeaponKind::BOW},
        {C_Item::Flags::ITEM_CROSSBOW, WeaponKind::CROSSBOW},
    };

    for (auto& p : pairs)
    {
        if (itemData->flags & p.first)
            return p.second;
    }

    return WeaponKind::NONE;
}

tl::optional<CharacterEquipment::Slot> CharacterEquipment::findAnyFreeMagicSlot() const
{
    Slot possibleSlots[] = {
        Slot::MAGIC_0,
        Slot::MAGIC_1,
        Slot::MAGIC_2,
        Slot::MAGIC_3,
        Slot::MAGIC_4,
        Slot::MAGIC_5,
        Slot::MAGIC_6,
        Slot::MAGIC_7,
        Slot::MAGIC_8,
        Slot::MAGIC_9,
    };

    for (Slot s : possibleSlots)
    {
        if (!getItemInSlot(s).isValid())
            return s;
    }

    return tl::nullopt;
}

tl::optional<CharacterEquipment::Slot> CharacterEquipment::findAnyFreeRingSlot() const
{
    Slot possibleSlots[] = {
        Slot::RING_LEFT,
        Slot::RING_RIGHT,
    };

    for (Slot s : possibleSlots)
    {
        if (!getItemInSlot(s).isValid())
            return s;
    }

    return tl::nullopt;
}

tl::optional<CharacterEquipment::Slot> CharacterEquipment::findSlotItemWasEquippedTo(ItemHandle item) const
{
    for (size_t i = 0; i < (size_t)Slot::COUNT; i++)
    {
        if (m_ItemsBySlot[i] == item)
            return (Slot)i;
    }

    return tl::nullopt;
}

bool CharacterEquipment::hasMeleeWeaponEquipped() const
{
    return getItemInSlot(Slot::MELEE).isValid();
}

bool CharacterEquipment::hasBowEquipped() const
{
    return getItemInSlot(Slot::BOW).isValid();
}

bool CharacterEquipment::hasItemEquipped(ItemHandle item) const
{
    for (auto h : m_ItemsBySlot)
    {
        if (h == item)
            return true;
    }

    return false;
}

bool CharacterEquipment::isItemTypeCorrectForSlot(ItemHandle item, Slot slot) const
{
    Kind kind = getKindOfItem(item);

    switch (kind)
    {
        case Kind::MELEE:
            return slot == Slot::MELEE;

        case Kind::BOW:
            return slot == Slot::BOW;

        case Kind::AMULET:
            return slot == Slot::AMULET;

        case Kind::RING:
        {
            if (slot == Slot::RING_LEFT)
                return true;

            if (slot == Slot::RING_RIGHT)
                return true;

            return false;
        }

        case Kind::MAGIC:
        {
            Slot possibleSlots[] = {
                Slot::MAGIC_0,
                Slot::MAGIC_1,
                Slot::MAGIC_2,
                Slot::MAGIC_3,
                Slot::MAGIC_4,
                Slot::MAGIC_5,
                Slot::MAGIC_6,
                Slot::MAGIC_7,
                Slot::MAGIC_8,
                Slot::MAGIC_9};

            for (Slot s : possibleSlots)
            {
                if (slot == s)
                    return true;
            }

            return false;
        }

        case Kind::BELT:
            return slot == Slot::BELT;

        case Kind::ARMOR:
            return slot == Slot::ARMOR;

        case Kind::OTHER:
        default:
            return false;
    }
}

bool CharacterEquipment::isItemOfKind(ItemHandle item, Kind kind) const
{
    return getKindOfItem(item) == kind;
}

bool CharacterEquipment::isItemEquipable(ItemHandle item) const
{
    return !isItemOfKind(item, Kind::OTHER);  // All but OTHER can be equipped
}

bool CharacterEquipment::doAttributesAllowUse(ItemHandle item) const
{
    using Daedalus::GEngineClasses::C_Item;
    using Daedalus::GEngineClasses::C_Npc;

    C_Item& data = m_World.getScriptEngine().getGameState().getItem(item);
    C_Npc& npc = getController().getScriptInstance();
    ScriptEngine& s = m_World.getScriptEngine();

    for (size_t i = 0; i < Daedalus::GEngineClasses::C_Item::COND_ATR_MAX; i++)
    {
        // Why is 0 not allowed? That's how gothic is doing it though, as it seems...
        if (data.cond_atr[i] > 0)
        {
            assert(data.cond_atr[i] < Daedalus::GEngineClasses::C_Npc::EATR_MAX);

            // Check for enough strength, etc.
            if (npc.attribute[data.cond_atr[i]] < data.cond_value[i])
            {
                return false;
            }
        }
    }

    return true;
}

tl::optional<CharacterEquipment::ItemData&> CharacterEquipment::getItemDataInSlot(Slot slot) const
{
    return getDataOfItem(getItemInSlot(slot));
}

std::string CharacterEquipment::getItemVisual(ItemHandle item) const
{
    auto data = getDataOfItem(item);

    if (!data)
        return "";

    return data->visual;
}

CharacterEquipment::ItemHandle CharacterEquipment::getItemInSlot(Slot slot) const
{
    return m_ItemsBySlot[(size_t)slot];
}

void CharacterEquipment::setItemInSlot(ItemHandle item, Slot slot)
{
    m_ItemsBySlot[(size_t)slot] = item;
}

PlayerController& CharacterEquipment::getController() const
{
    auto logic = m_World.getEntity<Components::LogicComponent>(m_CharacterEntity).m_pLogicController;
    assert(logic != nullptr);
    return *reinterpret_cast<PlayerController*>(logic);
}

tl::optional<CharacterEquipment::ItemData&> CharacterEquipment::getDataOfItem(ItemHandle item) const
{
    if (!item.isValid())
        return tl::nullopt;

    return m_World.getScriptEngine().getGameState().getItem(item);
}

void CharacterEquipment::setCharacterModelAttachment(const std::string& visual, EModelNode node)
{
    ModelVisual* pVisual = getController().getModelVisual();

    if (!pVisual)
        return;

    pVisual->setNodeVisual(visual, node);
}

void CharacterEquipment::setCharacterModelAttachment(ItemHandle item, EModelNode node)
{
    setCharacterModelAttachment(getItemVisual(item), node);
}

void CharacterEquipment::removeCharacterModelAttachment(EModelNode node)
{
    setCharacterModelAttachment("", node);
}

void CharacterEquipment::switchToDefaultCharacterModel()
{
    ModelVisual* pVisual = getController().getModelVisual();

    if (!pVisual)
        return;

    // TODO: Implement this. Actually not easy to get the default character model as it seems
    // ModelVisual::BodyState state = pVisual->getBodyState();
    // state.bodyVisual = getController().getScriptInstance().
}

void CharacterEquipment::switchCharacterModelArmor(const std::string& visual)
{
    ModelVisual* pVisual = getController().getModelVisual();

    if (!pVisual)
        return;

    ModelVisual::BodyState state = pVisual->getBodyState();
    state.bodyVisual = visual;

    pVisual->setBodyState(state);
}

void CharacterEquipment::putItemIntoRightHand(ItemHandle item)
{
    setCharacterModelAttachment(item, EModelNode::Righthand);
}

void CharacterEquipment::putItemIntoLeftHand(ItemHandle item)
{
    setCharacterModelAttachment(item, EModelNode::Lefthand);
}

void CharacterEquipment::exportSlots(json& j) const
{
    j["MELEE"] = getInstanceNameOfItem(getItemInSlot(Slot::MELEE));
    j["BOW"] = getInstanceNameOfItem(getItemInSlot(Slot::BOW));
    j["MAGIC_0"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_0));
    j["MAGIC_1"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_1));
    j["MAGIC_2"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_2));
    j["MAGIC_3"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_3));
    j["MAGIC_4"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_4));
    j["MAGIC_5"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_5));
    j["MAGIC_6"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_6));
    j["MAGIC_7"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_7));
    j["MAGIC_8"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_8));
    j["MAGIC_9"] = getInstanceNameOfItem(getItemInSlot(Slot::MAGIC_9));
    j["RING_LEFT"] = getInstanceNameOfItem(getItemInSlot(Slot::RING_LEFT));
    j["RING_RIGHT"] = getInstanceNameOfItem(getItemInSlot(Slot::RING_RIGHT));
    j["AMULET"] = getInstanceNameOfItem(getItemInSlot(Slot::AMULET));
    j["BELT"] = getInstanceNameOfItem(getItemInSlot(Slot::BELT));
    j["ARMOR"] = getInstanceNameOfItem(getItemInSlot(Slot::ARMOR));
}

void CharacterEquipment::importSlots(const json& j)
{
    auto& inventory = getController().getInventory();

    equipItemToSlot(inventory.getItem(j["MELEE"].get<std::string>()), Slot::MELEE);
    equipItemToSlot(inventory.getItem(j["BOW"].get<std::string>()), Slot::BOW);
    equipItemToSlot(inventory.getItem(j["MAGIC_0"].get<std::string>()), Slot::MAGIC_0);
    equipItemToSlot(inventory.getItem(j["MAGIC_1"].get<std::string>()), Slot::MAGIC_1);
    equipItemToSlot(inventory.getItem(j["MAGIC_2"].get<std::string>()), Slot::MAGIC_2);
    equipItemToSlot(inventory.getItem(j["MAGIC_3"].get<std::string>()), Slot::MAGIC_3);
    equipItemToSlot(inventory.getItem(j["MAGIC_4"].get<std::string>()), Slot::MAGIC_4);
    equipItemToSlot(inventory.getItem(j["MAGIC_5"].get<std::string>()), Slot::MAGIC_5);
    equipItemToSlot(inventory.getItem(j["MAGIC_6"].get<std::string>()), Slot::MAGIC_6);
    equipItemToSlot(inventory.getItem(j["MAGIC_7"].get<std::string>()), Slot::MAGIC_7);
    equipItemToSlot(inventory.getItem(j["MAGIC_8"].get<std::string>()), Slot::MAGIC_8);
    equipItemToSlot(inventory.getItem(j["MAGIC_9"].get<std::string>()), Slot::MAGIC_9);
    equipItemToSlot(inventory.getItem(j["RING_LEFT"].get<std::string>()), Slot::RING_LEFT);
    equipItemToSlot(inventory.getItem(j["RING_RIGHT"].get<std::string>()), Slot::RING_RIGHT);
    equipItemToSlot(inventory.getItem(j["AMULET"].get<std::string>()), Slot::AMULET);
    equipItemToSlot(inventory.getItem(j["BELT"].get<std::string>()), Slot::BELT);
    equipItemToSlot(inventory.getItem(j["ARMOR"].get<std::string>()), Slot::ARMOR);
}

std::string CharacterEquipment::getInstanceNameOfItem(ItemHandle item) const
{
    auto data = getDataOfItem(item);

    if (!data)
        return "";

    ScriptEngine& s = m_World.getScriptEngine();

    return s.getSymbolNameByIndex(data->instanceSymbol);
}
