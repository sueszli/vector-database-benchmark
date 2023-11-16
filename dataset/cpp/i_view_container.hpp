// i_view_container.hpp
/*
neogfx C++ App/Game Engine
Copyright (c) 2015, 2020 Leigh Johnston.  All Rights Reserved.

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

#include <neogfx/neogfx.hpp>
#include <neogfx/core/event.hpp>

namespace neogfx
{
    class i_widget;
    class i_tab_container;
}

namespace neogfx::mvc
{
    class i_view;
    class i_controller;

    enum class view_container_style
    {
        SDI,
        MDI,
        Tabbed,
        Explorer
    };

    class i_view_container
    {
    public:
        declare_event(view_added, i_view&)
        declare_event(view_removed, i_view&)
    public:
        struct controller_not_found : std::logic_error { controller_not_found() : std::logic_error("neogfx::mvc::i_view_container::controller_not_found") {} };
    public:
        virtual const i_widget& as_widget() const = 0;
        virtual i_widget& as_widget() = 0;
        virtual const i_tab_container& tab_container() const = 0;
        virtual i_tab_container& tab_container() = 0;
        virtual const i_widget& view_stack() const = 0;
        virtual i_widget& view_stack() = 0;
    public:
        virtual view_container_style style() const = 0;
        virtual void change_style(view_container_style aNewStyle) = 0;
    public:
        virtual void add_controller(i_controller& aController) = 0;
        virtual void add_controller(std::shared_ptr<i_controller> aController) = 0;
        virtual void remove_controller(i_controller& aController) = 0;
    };
}