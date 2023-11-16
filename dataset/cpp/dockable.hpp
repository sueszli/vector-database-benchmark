// dockable.hpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2020 Leigh Johnston.  All Rights Reserved.
  
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
#include <string>
#include <neolib/task/timer.hpp>
#include <neogfx/gui/widget/i_dock.hpp>
#include <neogfx/gui/widget/i_dockable.hpp>
#include <neogfx/gui/widget/framed_widget.hpp>
#include <neogfx/gui/widget/decorated.hpp>

namespace neogfx
{
    class dockable : public decorated<framed_widget<>, reference_counted<i_dockable>>
    {
        meta_object(decorated<framed_widget<>, reference_counted<i_dockable>>)
    public:
        define_declared_event(Docked, docked, i_dock&)
        define_declared_event(Undocked, undocked, i_dock&)
    public:
        typedef i_dockable abstract_type;
    public:
        dockable(std::shared_ptr<i_widget> aDockableWidget, std::string const& aTitle = "", dock_area aAcceptableDocks = dock_area::Any);
    public:
        const neolib::string& title() const override;
    public:
        bool can_dock(const i_dock& aDock) const override;
        bool is_docked() const override;
        const i_dock& which_dock() const override;
        i_dock& which_dock() override;
        void dock(i_dock& aDock) override;
        void undock() override;
    public:
        const i_widget& docked_widget() const override;
        i_widget& docked_widget() override;
    protected:
        void focus_gained(focus_reason aFocusReason) override;
    protected:
        color frame_color() const override;
    public:
        template <typename WidgetType>
        const WidgetType& docked_widget() const
        {
            return static_cast<const WidgetType&>(docked_widget());
        }
        template <typename WidgetType>
        WidgetType& docked_widget()
        {
            return static_cast<WidgetType&>(docked_widget());
        }
    private:
        neolib::string iTitle;
        dock_area iAcceptableDocks;
        std::shared_ptr<i_widget> iDockedWidget;
        i_dock* iDock;
    };

    template <typename WidgetType, typename... Args>
    inline dockable make_dockable(std::string const& aTitle = "", dock_area aAcceptableDocks = dock_area::Any, Args&&... aArgs)
    {
        return dockable{ std::make_shared<WidgetType>(std::forward<Args>(aArgs)...), aTitle, aAcceptableDocks };
    }
    template <typename WidgetType, typename... Args>
    static std::shared_ptr<i_dockable> make_shared_dockable(std::string const& aTitle = "", dock_area aAcceptableDocks = dock_area::Any, Args&&... aArgs)
    {
        return std::make_shared<dockable>(std::make_shared<WidgetType>(std::forward<Args>(aArgs)...), aTitle, aAcceptableDocks);
    }
}