// This code is based on Sabberstone project.
// Copyright (c) 2017-2019 SabberStone Team, darkfriend77 & rnilva
// RosettaStone is hearthstone simulator using C++ with reinforcement learning.
// Copyright (c) 2019 Chris Ohk, Youngjoong Kim, SeungHyun Jeon

#ifndef ROSETTASTONE_PLAYMODE_COST_HPP
#define ROSETTASTONE_PLAYMODE_COST_HPP

#include <Rosetta/PlayMode/Enchants/Attrs/SelfContainedIntAttr.hpp>

#include <memory>

namespace RosettaStone::PlayMode
{
//!
//! \brief Cost class.
//!
//! This class is an attribute for cost and inherits from SelfContainedIntAttr
//! class. It uses CRTP(Curiously Recurring Template Pattern) technique.
//!
class Cost : public SelfContainedIntAttr<Cost, Entity>
{
 public:
    //! Generates new effect for cost attribute.
    //! \param effectOp The effect operator of the effect.
    //! \param value The value of the effect.
    //! \return The effect that is dynamically allocated.
    static std::shared_ptr<IEffect> Effect(EffectOperator effectOp, int value)
    {
        return SelfContainedIntAttr::Effect(effectOp, value);
    }

    //! Applies the effect that affects the attribute.
    //! \param entity The entity to apply the effect.
    //! \param effectOp The effect operator to change the attribute.
    //! \param value The value to change the attribute.
    void Apply(Entity* entity, EffectOperator effectOp, int value) override
    {
        SelfContainedIntAttr::Apply(entity, effectOp, value);

        const auto playable = dynamic_cast<Playable*>(entity);

        if (auto* costManager = playable->costManager; costManager)
        {
            costManager->AddCostEnchantment(effectOp, value);
        }
    }

    //! Applies the aura that affects the attribute.
    //! \param entity The entity to apply the aura.
    //! \param effectOp The effect operator to change the attribute.
    //! \param value The value to change the attribute.
    void ApplyAura(Entity* entity, EffectOperator effectOp, int value) override
    {
        const auto playable = dynamic_cast<Playable*>(entity);

        CostManager* costManager = playable->costManager;
        if (!costManager)
        {
            costManager = new CostManager();
            playable->costManager = costManager;
        }

        costManager->AddCostAura(effectOp, value);
    }

    //! Removes the aura that affects the attribute.
    //! \param entity The entity to remove the aura.
    //! \param effectOp The effect operator to change the attribute.
    //! \param value The value to change the attribute.
    void RemoveAura(Entity* entity, EffectOperator effectOp, int value) override
    {
        const auto playable = dynamic_cast<Playable*>(entity);

        if (auto* costManager = playable->costManager; costManager)
        {
            costManager->RemoveCostAura(effectOp, value);
        }
    }

 protected:
    //! Returns the value of the attribute of the entity.
    //! \param entity The entity to get the value of the attribute.
    //! \return The value of the attribute of the entity.
    int GetValue(Entity* entity) override
    {
        const auto playable = dynamic_cast<Playable*>(entity);
        return playable->GetCost();
    }

    //! Sets the value of the attribute of the entity.
    //! \param entity The entity to set the value of the attribute.
    //! \param value The value of the attribute of the entity.
    void SetValue(Entity* entity, int value) override
    {
        auto playable = dynamic_cast<Playable*>(entity);
        playable->SetCost(value);
    }

    //! Returns the value the attribute that is affected by the aura effect.
    //! \param auraEffects The aura effects that affects the attribute.
    //! \return The value the attribute that is affected by the aura effect.
    int GetAuraValue([[maybe_unused]] AuraEffects* auraEffects) override
    {
        throw std::logic_error("Cost::GetAuraValue() - Not implemented!");
    }

    //! Sets the value the attribute that is affected by the aura effect.
    //! \param auraEffects The aura effects that affects the attribute.
    //! \param value The value the attribute that is affected by the aura
    //! effect.
    void SetAuraValue([[maybe_unused]] AuraEffects* auraEffects,
                      [[maybe_unused]] int value) override
    {
        throw std::logic_error("Cost::GetAuraValue() - Not implemented!");
    }
};
}  // namespace RosettaStone::PlayMode

#endif  // ROSETTASTONE_PLAYMODE_COST_HPP
