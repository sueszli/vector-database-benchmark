// menu_item.cpp
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

#include <neogfx/neogfx.hpp>
#include <neolib/core/scoped.hpp>
#include <neogfx/gui/widget/menu_item.hpp>
#include <neogfx/gui/widget/i_menu.hpp>

namespace neogfx
{
    menu_item::menu_item(i_action& aAction) : 
        iContents{ ref_ptr<i_action>{ ref_ptr<i_action>{}, &aAction } }, iOpenAnySubMenu{ true }, iWidget{ nullptr }
    {
    }

    menu_item::menu_item(i_ref_ptr<i_action> const& aAction) : 
        iContents{ aAction }, iOpenAnySubMenu{ true }, iWidget{ nullptr }
    {
    }

    menu_item::menu_item(i_menu& aSubMenu) : 
        iContents{ std::shared_ptr<i_menu>{std::shared_ptr<i_menu>{}, &aSubMenu } }, iOpenAnySubMenu{ true }, iWidget{ nullptr }
    {
    }

    menu_item::menu_item(std::shared_ptr<i_menu> aSubMenu) : 
        iContents{ aSubMenu }, iOpenAnySubMenu{ true }, iWidget{ nullptr }
    {
    }

    menu_item_type menu_item::type() const
    {
        if (std::holds_alternative<action_pointer>(iContents))
            return menu_item_type::Action;
        else
            return menu_item_type::SubMenu;
    }

    uuid const& menu_item::group() const
    {
        if (type() == menu_item_type::Action)
            return action().group();
        else
            return sub_menu().group();
    }

    const i_action& menu_item::action() const
    {
        if (type() != menu_item_type::Action)
            throw wrong_type();
        return *static_variant_cast<const action_pointer&>(iContents);
    }

    i_action& menu_item::action()
    {
        return const_cast<i_action&>(to_const(*this).action());
    }

    const i_menu& menu_item::sub_menu() const
    {
        if (type() != menu_item_type::SubMenu)
            throw wrong_type();
        return *static_variant_cast<const menu_pointer&>(iContents);
    }

    i_menu& menu_item::sub_menu()
    {
        return const_cast<i_menu&>(to_const(*this).sub_menu());
    }

    const i_menu_item_widget& menu_item::as_widget() const
    {
        if (iWidget)
            return *iWidget;
        throw no_widget();
    }

    i_menu_item_widget& menu_item::as_widget()
    {
        if (iWidget)
            return *iWidget;
        throw no_widget();
    }

    void menu_item::set_widget(i_menu_item_widget& aWidget)
    {
        iWidget = &aWidget;
    }

    void menu_item::unset_widget()
    {
        iWidget = nullptr;
    }

    bool menu_item::available() const
    {
        if (type() == menu_item_type::Action && (action().is_separator() || action().is_disabled()))
            return false;
        if (type() == menu_item_type::SubMenu && sub_menu().count() == 0)
            return false;
        return true;
    }

    void menu_item::select(bool aOpenAnySubMenu)
    {
        neolib::scoped_flag sf{ iOpenAnySubMenu, aOpenAnySubMenu };
        Selected.trigger();
    }

    bool menu_item::open_any_sub_menu() const
    {
        return iOpenAnySubMenu;
    }
}