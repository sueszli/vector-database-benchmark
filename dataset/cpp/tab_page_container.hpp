// tab_page_container.hpp
/*
neoGFX Resource Compiler
Copyright(C) 2019 Leigh Johnston

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
#include <neogfx/tools/nrc/ui_element.hpp>

namespace neogfx::nrc
{
    class tab_page_container : public ui_element<>
    {
    public:
        tab_page_container(const i_ui_element_parser& aParser, i_ui_element& aParent) :
            ui_element<>{ aParser, aParent, ui_element_type::TabPageContainer },
            iClosableTabs{ aParent.parser().get<bool>("closable_tabs", false) }
        {
            add_header("neogfx/gui/widget/tab_page_container.hpp");
            add_data_names({ "closable_tabs" });
        }
    public:
        void parse(const neolib::i_string& aName, const data_t& aData) override
        {
            ui_element<>::parse(aName, aData);
        }
        void parse(const neolib::i_string& aName, const array_data_t& aData) override
        {
            ui_element<>::parse(aName, aData);
        }
    protected:
        void emit() const override
        {
        }
        void emit_preamble() const override
        {
            emit("  %1%<> %2%;\n", type_name(), id());
            ui_element<>::emit_preamble();
        }
        void emit_ctor() const override
        {
            if ((parent().type() & ui_element_type::MASK_RESERVED_GENERIC) == ui_element_type::Window)
            {
                if (!iClosableTabs)
                    emit(",\n"
                        "   %1%{ %2%.client_layout() }", id(), parent().id());
                else
                    emit(",\n"
                        "   %1%{ %2%.client_layout(), true }", id(), parent().id());
            }
            else if (is_widget_or_layout(parent().type()))
            {
                if (!iClosableTabs)
                    emit(",\n"
                        "   %1%{ %2% }", id(), parent().id());
                else
                    emit(",\n"
                        "   %1%{ %2%, true }", id(), parent().id());
            }
            ui_element<>::emit_ctor();
        }
        void emit_body() const override
        {
            ui_element<>::emit_body();
        }
    protected:
        using ui_element<>::emit;
    private:
        bool iClosableTabs;
    };
}
