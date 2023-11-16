////////////////////////////////////////////////////////////////////////////
//	Module 		: property_color.hpp
//	Created 	: 10.12.2007
//  Modified 	: 10.12.2007
//	Author		: Dmitriy Iassenev
//	Description : color property implementation class
////////////////////////////////////////////////////////////////////////////

#ifndef PROPERTY_COLOR_HPP_INCLUDED
#define PROPERTY_COLOR_HPP_INCLUDED

#include "property_color_base.hpp"

public ref class property_color : public property_color_base
{
public:
    typedef XRay::Editor::property_holder_base::color_getter_type color_getter_type;
    typedef XRay::Editor::property_holder_base::color_setter_type color_setter_type;
    typedef property_color_base inherited;

public:
    property_color(
        color_getter_type const& getter, color_setter_type const& setter, array<System::Attribute ^> ^ attributes);
    virtual ~property_color();
    !property_color();
    virtual XRay::Editor::color get_value_raw() override;
    virtual void set_value_raw(XRay::Editor::color value) override;

private:
    color_getter_type* m_getter;
    color_setter_type* m_setter;
}; // ref class property_color

#endif // ifndef PROPERTY_COLOR_HPP_INCLUDED
