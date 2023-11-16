// text_field.hpp
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
    class text_field : public ui_element<>
    {
    public:
        text_field(const i_ui_element_parser& aParser, i_ui_element& aParent, ui_element_type aElementType = ui_element_type::TextField) :
            ui_element<>{ aParser, aParent, aElementType }
        {
            add_header("neogfx/gui/widget/text_field.hpp");
            add_data_names({ "label", "input_box", "hint" });
        }
    public:
        void parse(const neolib::i_string& aName, const data_t& aData) override
        {
            ui_element<>::parse(aName, aData);
            if (aName == "tab_stop_hint")
                iTabStopHint = aData.get<neolib::i_string>();
            if (aName == "hint")
                iHint = aData.get<neolib::i_string>();
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
            emit("  %1% %2%;\n", type_name(), id());
            ui_element<>::emit_preamble();
        }
        void emit_ctor() const override
        {
            ui_element<>::emit_generic_ctor();
            ui_element<>::emit_ctor();
        }
        void emit_body() const override
        {
            ui_element<>::emit_body();
            if (iTabStopHint)
                emit("   %1%.input_box().set_tab_stop_hint(\"%2%\"_s);\n", id(), *iTabStopHint);
            if (iHint)
                emit("   %1%.hint().set_text(\"%2%\"_t);\n", id(), *iHint);
        }
    protected:
        using ui_element<>::emit;
    private:
        neolib::optional<string> iTabStopHint;
        neolib::optional<string> iHint;
    };
}
