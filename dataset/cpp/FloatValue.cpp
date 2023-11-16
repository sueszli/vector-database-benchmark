/*
 * Copyright 2009, Ingo Weinhold, ingo_weinhold@gmx.de.
 * Distributed under the terms of the MIT License.
 */


#include "FloatValue.h"

#include <stdio.h>


FloatValue::FloatValue(const BVariant& value)
	:
	fValue(value)
{
}


FloatValue::~FloatValue()
{
}


bool
FloatValue::ToString(BString& _string) const
{
	char buffer[128];

	switch (fValue.Type()) {
		case B_FLOAT_TYPE:
		{
			snprintf(buffer, sizeof(buffer), "%f", fValue.ToFloat());
			break;
		}
		case B_DOUBLE_TYPE:
		{
			snprintf(buffer, sizeof(buffer), "%g", fValue.ToDouble());
			break;
		}
		default:
			return false;
	}

	BString string(buffer);
	if (string.Length() == 0)
		return false;

	_string = string;
	return true;
}


bool
FloatValue::ToVariant(BVariant& _value) const
{
	_value = fValue;
	return true;
}


bool
FloatValue::operator==(const Value& other) const
{
	const FloatValue* otherFloat = dynamic_cast<const FloatValue*>(&other);
	return otherFloat != NULL ? fValue == otherFloat->fValue : false;
}
