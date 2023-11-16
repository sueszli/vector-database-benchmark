/*
 * Copyright 2014, Haiku, Inc.
 * Distributed under the terms of the MIT License.
 */


#include "CalendarViewTest.h"

#include <Application.h>
#include <CalendarView.h>
#include <Window.h>

#include <cppunit/TestCaller.h>
#include <cppunit/TestSuite.h>


using namespace BPrivate;


CalendarViewTest::CalendarViewTest()
{
}


CalendarViewTest::~CalendarViewTest()
{
}


void
CalendarViewTest::TestSetters()
{
	// TODO: CalendarView probably uses some other library, test that instead
	BApplication app(
		"application/x-vnd.CalendarViewTest_TestSetters.test");
	BWindow window(BRect(50,50,550,550),
		"CalendarViewTest_TestSetters", B_TITLED_WINDOW,
		B_QUIT_ON_WINDOW_CLOSE, 0);
	BCalendarView *view = new BCalendarView("test");
	window.AddChild(view);

	NextSubTest();
	view->SetDate(2004, 2, 29);
	CPPUNIT_ASSERT_EQUAL(2004, view->Year());
	CPPUNIT_ASSERT_EQUAL(2, view->Month());
	CPPUNIT_ASSERT_EQUAL(29, view->Day());

	NextSubTest();
	// Moving from leap year to leap year on 29 feb. must not change day
	view->SetYear(2008);
	CPPUNIT_ASSERT_EQUAL(2008, view->Year());
	CPPUNIT_ASSERT_EQUAL(2, view->Month());
	CPPUNIT_ASSERT_EQUAL(29, view->Day());

	NextSubTest();
	// Moving from leap year to non-leap year on 29 feb. must go back to 28
	view->SetYear(2014);
	CPPUNIT_ASSERT_EQUAL(2014, view->Year());
	CPPUNIT_ASSERT_EQUAL(2, view->Month());
	CPPUNIT_ASSERT_EQUAL(28, view->Day());

	NextSubTest();
	view->SetDate(2014, 8, 31);
	CPPUNIT_ASSERT_EQUAL(2014, view->Year());
	CPPUNIT_ASSERT_EQUAL(8, view->Month());
	CPPUNIT_ASSERT_EQUAL(31, view->Day());

	NextSubTest();
	// Moving to month with less days should adjust day
	view->SetMonth(2);
	CPPUNIT_ASSERT_EQUAL(2014, view->Year());
	CPPUNIT_ASSERT_EQUAL(2, view->Month());
	CPPUNIT_ASSERT_EQUAL(28, view->Day());
}


/*static*/ void
CalendarViewTest::AddTests(BTestSuite& parent)
{
	CppUnit::TestSuite& suite = *new CppUnit::TestSuite("CalendarViewTest");

	suite.addTest(new CppUnit::TestCaller<CalendarViewTest>(
		"CalendarViewTest::TestSetters", &CalendarViewTest::TestSetters));

	parent.addTest("CalendarViewTest", &suite);
}
