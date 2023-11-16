/*
 * Copyright (c) 2014-2015, Niklas Hauser
 * Copyright (c) 2017, Christopher Durand
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_UNITTEST_REGISTER_HPP
#define MODM_UNITTEST_REGISTER_HPP

#include <unittest/testsuite.hpp>
#include <modm/architecture/interface/register.hpp>

namespace modm
{

/// @ingroup modm_test_test_architecture
template<typename T>
struct testing
{
protected:
	enum class
	Test : uint8_t
	{
		A = Bit0,
		B = Bit1,
		C = Bit2,
		D = Bit3,
		E = Bit4,
		F = Bit5,
		G = Bit6,
		H = Bit7,
	};
	MODM_FLAGS8(Test);

	enum class
	Test2 : uint8_t
	{
		A = Bit0,
		B = Bit1,
		C = Bit2,
		D = Bit3,
		E = Bit4,
		F = Bit5,
		G = Bit6,
		H = Bit7,
	};
	MODM_FLAGS8(Test2);

	typedef FlagsGroup< Test_t, Test2_t > Common_t;


	enum class
	Test3 : uint8_t
	{
		// Field1 : 5:7
		Bit = Bit4,
		// Config2 2:3
		// Config0 0:1
	};
	// this configuration field is available once in
	// - bit position 0, and once in
	// - bit position 2.
	enum class
	Config : uint8_t
	{
		Zero = 0,
		One = Bit0,
		Two = Bit1,
		Three = Bit1 | Bit0,
	};

	typedef Flags8<Test3> Test3_t;
	MODM_INT_TYPE_FLAGS(Test3_t);

	typedef Value< Test3_t, 3, 5 > Address;
	typedef Configuration< Test3_t, Config, Bit1 | Bit0, 2 > Config2;
	typedef Configuration< Test3_t, Config, Bit1 | Bit0 > Config0;


	enum class
	Direct : uint8_t
	{
		Zero = 0,
		One = Bit2,
		Two = Bit3,
		Three = Bit3 | Bit2,
	};
	typedef Configuration< Test3_t, Direct, Bit3 | Bit2 > Direct0;

	typedef Flags8<> Test5_t;
};

/// @ingroup modm_test_test_architecture
enum class
Test4
{
	A = 0x01,
	B = 0x02,
	C = 0x04,
	D = 0x08,
	E = 0x10,
	F = 0x20,
	G = 0x40,
	H = 0x80,
	Mask = 0xff,
};
typedef Flags8<Test4> Test4_t;
// test macro outside of struct
// all enum operations must not be declared 'friend'
MODM_TYPE_FLAGS(Test4_t)

}

// @author Niklas Hauser
/// @ingroup modm_test_test_architecture
class RegisterTest : public unittest::TestSuite, public modm::testing<void>
{
public:
	void
	testAssignments();

	void
	testOperators();

	void
	testFunctions();

	void
	testCasting();

	void
	testConfigurations();

	void
	testValue();

	uint8_t
	translateCommonArgument(Common_t common);

};

#endif // MODM_UNITTEST_REGISTER_HPP
