// view.cpp
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
#include <neogfx/gui/mvc/view.hpp>
#include <neogfx/gui/mvc/i_controller.hpp>
#include <neogfx/gui/mvc/i_view_container.hpp>

namespace neogfx::mvc
{
    view::view(i_controller& aController, i_model& aModel) :
        framed_scrollable_widget{
            aController.container().as_widget(),
            aController.container().style() != view_container_style::MDI ? frame_style::NoFrame : frame_style::WindowFrame },
        iController(aController), iModel(aModel), iActive(false)
    {
    }
    
    const i_widget& view::as_widget() const
    {
        return *this;
    }
    
    i_widget& view::as_widget()
    {
        return *this;
    }
    
    const i_model& view::model() const
    {
        return iModel;
    }
    
    i_model& view::model()
    {
        return iModel;
    }
    
    bool view::is_active() const
    {
        return iActive;
    }
    
    void view::activate()
    {
        if (!iActive)
        {
            iActive = true;
            Activated.trigger();
        }
    }
    
    void view::deactivate()
    {
        if (iActive)
        {
            iActive = false;
            Deactivated.trigger();
        }
    }
}