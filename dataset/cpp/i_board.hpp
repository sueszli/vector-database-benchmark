﻿/*
neogfx C++ App/Game Engine - Examples - Games - Chess
Copyright(C) 2020 Leigh Johnston

This program is free software: you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <neogfx/core/event.hpp>
#include <chess/primitives.hpp>
#include <chess/i_player.hpp>

namespace chess::gui
{
    class i_board
    {
    public:
        declare_event(changed)
    public:
        virtual ~i_board() = default;
    public:
        virtual void new_game(i_player_factory& aPlayerFactory, player_type aWhitePlayer, player_type aBlackPlayer) = 0;
        virtual void setup(mailbox_position const& aPosition) = 0;
        virtual bool play(move const& aMove) = 0;
        virtual void edit(move const& aMove) = 0;
        virtual bool can_undo() const = 0;
        virtual void undo() = 0;
        virtual bool can_redo() const = 0;
        virtual void redo() = 0;
        virtual bool can_play() const = 0;
        virtual void play() = 0;
        virtual bool can_stop() const = 0;
        virtual void stop() = 0;
    public:
        virtual i_player const& current_player() const = 0;
        virtual i_player& current_player() = 0;
        virtual i_player const& next_player() const = 0;
        virtual i_player& next_player() = 0;
        virtual i_player const& white_player() const = 0;
        virtual i_player& white_player() = 0;
        virtual i_player const& black_player() const = 0;
        virtual i_player& black_player() = 0;
    };
}