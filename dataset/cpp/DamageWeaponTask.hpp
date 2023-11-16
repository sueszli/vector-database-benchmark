// This code is based on Sabberstone project.
// Copyright (c) 2017-2019 SabberStone Team, darkfriend77 & rnilva
// RosettaStone is hearthstone simulator using C++ with reinforcement learning.
// Copyright (c) 2019 Chris Ohk, Youngjoong Kim, SeungHyun Jeon

#ifndef ROSETTASTONE_PLAYMODE_DAMAGE_WEAPON_TASK_HPP
#define ROSETTASTONE_PLAYMODE_DAMAGE_WEAPON_TASK_HPP

#include <Rosetta/PlayMode/Tasks/ITask.hpp>

namespace RosettaStone::PlayMode::SimpleTasks
{
//!
//! \brief DamageWeaponTask class.
//!
//! This class represents the task for taking damage of durability
//! to one of the equipped weapons.
//!
class DamageWeaponTask : public ITask
{
 public:
    //! Constructs task with given \p opponent and \p amount.
    //! \param opponent The flag that indicates whether
    //! the target is opponent's equipped weapon.
    //! \param amount The amount to damage the durability.
    DamageWeaponTask(bool opponent = true, int amount = 1);

 private:
    //! Processes task logic internally and returns meta data.
    //! \param player The player to run task.
    //! \return The result of task processing.
    TaskStatus Impl(Player* player) override;

    //! Internal method of Clone().
    //! \return The cloned task.
    std::unique_ptr<ITask> CloneImpl() override;

    bool m_opponent = true;
    int m_amount = 0;
};
}  // namespace RosettaStone::PlayMode::SimpleTasks

#endif  // ROSETTASTONE_PLAYMODE_DAMAGE_WEAPON_TASK_HPP
