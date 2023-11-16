// i_element_library.hpp
/*
  neoGFX Design Studio
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

#include <neogfx/neogfx.hpp>
#include <neolib/core/reference_counted.hpp>
#include <neolib/core/i_vector.hpp>
#include <neolib/core/i_set.hpp>
#include <neolib/core/i_string.hpp>
#include <neolib/plugin/i_plugin.hpp>
#include <neolib/app/i_application.hpp>
#include <neogfx/gfx/i_texture.hpp>
#include <neogfx/tools/DesignStudio/i_ide.hpp>
#include <neogfx/tools/DesignStudio/i_element.hpp>

namespace neogfx::DesignStudio
{
    class i_project;

    class i_element_library : public neolib::i_reference_counted
    {
        // types
    public:
        typedef i_element_library abstract_type;
        typedef neolib::i_set<neolib::i_string> elements_t;
        typedef neolib::i_vector<neolib::i_string> elements_ordered_t;
        // exceptions
    public:
        struct unknown_element_type : std::logic_error { unknown_element_type() : std::logic_error{ "neogfx::DesignStudio::i_element_library::unknown_element_type" } {} };
        //initialisation
    public:
        virtual neolib::i_application& application() const = 0;
        virtual void ide_ready(i_ide& aIde) = 0;
        // meta
    public:
        virtual const elements_t& elements() const = 0;
        virtual const elements_ordered_t& elements_ordered() const = 0;
        // factory
    public:
        virtual void create_element(i_project& aProject, const neolib::i_string& aElementType, const neolib::i_string& aElementId, neolib::i_ref_ptr<i_element>& aResult) = 0;
        virtual void create_element(i_element& aParent, const neolib::i_string& aElementType, const neolib::i_string& aElementId, neolib::i_ref_ptr<i_element>& aResult) = 0;
        virtual DesignStudio::element_group element_group(const neolib::i_string& aElementType) const = 0;
        virtual i_texture const& element_icon(const neolib::i_string& aElementType) const = 0;
        // helpers
    public:
        neolib::ref_ptr<i_element> create_element(i_project& aProject, std::string const& aElementType, std::string const& aElementId)
        {
            neolib::ref_ptr<i_element> result;
            create_element(aProject, string{ aElementType }, string{ aElementId }, result);
            return result;
        }
        neolib::ref_ptr<i_element> create_element(i_element& aParent, std::string const& aElementType, std::string const& aElementId)
        {
            neolib::ref_ptr<i_element> result;
            create_element(aParent, string{ aElementType }, string{ aElementId }, result);
            return result;
        }
        // interface
    public:
        static const neolib::uuid& iid() { static const neolib::uuid sId = neolib::make_uuid("0FAFD88D-DE88-4473-A70F-DCBC53BA8CB9"); return sId; }
    };
}
