/*
 * Copyright (c) 2014, Daniel Krebs
 * Copyright (c) 2014, Niklas Hauser
 * Copyright (c) 2014, Sascha Schade
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_GUI_NUMBERFIELD_HPP
#define MODM_GUI_NUMBERFIELD_HPP

#include "widget.hpp"

namespace modm
{

namespace gui
{

/**
 * @ingroup modm_ui_gui
 * @author	Daniel Krebs
 */
template<typename T>
class NumberField : public Widget
{
public:
	NumberField(T default_value, Dimension d) :
		Widget(d, false),
		value(default_value)
	{
	}

	void
	render(View* view);

	void
	setValue(T value)
	{
		if(this->value == value)
			return;
		this->value = value;
		this->markDirty();
	}

	T
	getValue()
	{
		return this->value;
	}

private:
	T value;
};

/// @ingroup modm_ui_gui
typedef NumberField<int16_t> IntegerField;

/// @ingroup modm_ui_gui
class FloatField : public NumberField<float>
{
public:
	FloatField(float value, Dimension d);

	void
	render(View* view);
};

}	// namespace gui

}	// namespace modm

#include "numberfield_impl.hpp"

#endif  // MODM_GUI_NUMBERFIELD_HPP
