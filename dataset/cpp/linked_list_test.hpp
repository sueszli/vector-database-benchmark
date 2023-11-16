/*
 * Copyright (c) 2009, Martin Rosekeit
 * Copyright (c) 2009-2010, Fabian Greif
 * Copyright (c) 2012, Niklas Hauser
 * Copyright (c) 2015, Kevin Läufer
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#include <unittest/testsuite.hpp>

/// @ingroup modm_test_test_container
class LinkedListTest : public unittest::TestSuite
{
public:
	void
	setUp();

	void
	testConstructor();

	void
	testAppend();

	void
	testAppendCount();

	void
	testPrepend();

	void
	testPrependCount();

	void
	testRemoveFront();

	void
	testRemoveFrontCount();

	void
	testDestructor();

	void
	testConstIterator();

	void
	testConstIteratorAccess();

	void
	testIterator();

	void
	testIteratorAccess();

	void
	testSize();

	void
	testRemove();

	void
	testInsert();
};
