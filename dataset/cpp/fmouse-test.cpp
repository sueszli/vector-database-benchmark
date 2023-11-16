/***********************************************************************
* fmouse-test.cpp - FMouse unit tests                                  *
*                                                                      *
* This file is part of the FINAL CUT widget toolkit                    *
*                                                                      *
* Copyright 2018-2023 Markus Gans                                      *
*                                                                      *
* FINAL CUT is free software; you can redistribute it and/or modify    *
* it under the terms of the GNU Lesser General Public License as       *
* published by the Free Software Foundation; either version 3 of       *
* the License, or (at your option) any later version.                  *
*                                                                      *
* FINAL CUT is distributed in the hope that it will be useful, but     *
* WITHOUT ANY WARRANTY; without even the implied warranty of           *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
* GNU Lesser General Public License for more details.                  *
*                                                                      *
* You should have received a copy of the GNU Lesser General Public     *
* License along with this program.  If not, see                        *
* <http://www.gnu.org/licenses/>.                                      *
***********************************************************************/

#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>

#include <final/final.h>

namespace test
{

//----------------------------------------------------------------------
// class FMouse_protected
//----------------------------------------------------------------------

class FMouse_protected : public finalcut::FMouse
{
  public:
    auto hasData() noexcept -> bool override
    { return true; }

    void setRawData (finalcut::FKeyboard::keybuffer&) noexcept override
    { }

    void processEvent (const TimeValue&) override
    { }

    auto getMaxWidth() noexcept -> uInt16
    {
      return finalcut::FMouse::getMaxWidth();
    }

    auto getMaxHeight() noexcept -> uInt16
    {
      return finalcut::FMouse::getMaxHeight();
    }

    auto getNewMousePosition() const noexcept -> const finalcut::FPoint&
    {
      return finalcut::FMouse::getNewPos();
    }

    auto getDblclickInterval() noexcept -> uInt64
    {
      return finalcut::FMouse::getDblclickInterval();
    }

    void setNewMousePosition (int x, int y) noexcept
    {
      finalcut::FMouse::setNewPos(x, y);
    }

    void useNewMousePosition() noexcept
    {
      finalcut::FMouse::useNewPos();
    }

    auto isDblclickTimeout (const TimeValue& t) -> bool
    {
      return finalcut::FMouse::isDblclickTimeout(t);
    }
};

}  // namespace test


//----------------------------------------------------------------------
// class finalcut::FMouseTest
//----------------------------------------------------------------------

class FMouseTest : public CPPUNIT_NS::TestFixture
{
  public:
    FMouseTest()
    {
      // create timer instance
      finalcut::FObjectTimer timer{};
    }

  protected:
    void classNameTest();
    void noArgumentTest();
    void doubleClickTest();
    void workspaceSizeTest();
#ifdef F_HAVE_LIBGPM
    void gpmMouseTest();
#endif
    void x11MouseTest();
    void sgrMouseTest();
    void urxvtMouseTest();
    void mouseControlTest();

  private:
    auto insertData (std::initializer_list<char>) -> finalcut::FKeyboard::keybuffer;

    // Adds code needed to register the test suite
    CPPUNIT_TEST_SUITE (FMouseTest);

    // Add a methods to the test suite
    CPPUNIT_TEST (classNameTest);
    CPPUNIT_TEST (noArgumentTest);
    CPPUNIT_TEST (doubleClickTest);
    CPPUNIT_TEST (workspaceSizeTest);
#ifdef F_HAVE_LIBGPM
    CPPUNIT_TEST (gpmMouseTest);
#endif
    CPPUNIT_TEST (x11MouseTest);
    CPPUNIT_TEST (sgrMouseTest);
    CPPUNIT_TEST (urxvtMouseTest);
    CPPUNIT_TEST (mouseControlTest);

    // End of test suite definition
    CPPUNIT_TEST_SUITE_END();
};

//----------------------------------------------------------------------
void FMouseTest::classNameTest()
{
  test::FMouse_protected m;
  const finalcut::FString& classname1 = m.getClassName();
  CPPUNIT_ASSERT ( classname1 == "FMouse" );

#ifdef F_HAVE_LIBGPM
  finalcut::FMouseGPM gpm_mouse;
  const finalcut::FString& classname2 = gpm_mouse.getClassName();
  CPPUNIT_ASSERT ( classname2 == "FMouseGPM" );
#endif

  finalcut::FMouseX11 x11_mouse;
  const finalcut::FString& classname3 = x11_mouse.getClassName();
  CPPUNIT_ASSERT ( classname3 == "FMouseX11" );

  finalcut::FMouseSGR sgr_mouse;
  const finalcut::FString& classname4 = sgr_mouse.getClassName();
  CPPUNIT_ASSERT ( classname4 == "FMouseSGR" );

  finalcut::FMouseUrxvt urxvt_mouse;
  const finalcut::FString& classname5 = urxvt_mouse.getClassName();
  CPPUNIT_ASSERT ( classname5 == "FMouseUrxvt" );

  finalcut::FMouseControl mouse_control;
  const finalcut::FString& classname6 = mouse_control.getClassName();
  CPPUNIT_ASSERT ( classname6 == "FMouseControl" );
}

//----------------------------------------------------------------------
void FMouseTest::noArgumentTest()
{
  test::FMouse_protected mouse;
  CPPUNIT_ASSERT ( mouse.getPos() == finalcut::FPoint(0, 0) );
  CPPUNIT_ASSERT ( mouse.getNewMousePosition() == finalcut::FPoint(0, 0) );
  CPPUNIT_ASSERT ( ! mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! mouse.isMoved() );
  CPPUNIT_ASSERT ( ! mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( mouse.getMouseTypeID() == finalcut::FMouse::MouseType::None );

#ifdef F_HAVE_LIBGPM
  finalcut::FMouseGPM gpm_mouse;
  CPPUNIT_ASSERT ( gpm_mouse.getMouseTypeID() == finalcut::FMouse::MouseType::Gpm );
  CPPUNIT_ASSERT ( ! gpm_mouse.hasData() );
#endif

  finalcut::FMouseX11 x11_mouse;
  CPPUNIT_ASSERT ( x11_mouse.getMouseTypeID() == finalcut::FMouse::MouseType::X11 );
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );

  finalcut::FMouseSGR sgr_mouse;
  CPPUNIT_ASSERT ( sgr_mouse.getMouseTypeID() == finalcut::FMouse::MouseType::Sgr );
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );

  finalcut::FMouseUrxvt urxvt_mouse;
  CPPUNIT_ASSERT ( urxvt_mouse.getMouseTypeID() == finalcut::FMouse::MouseType::Urxvt );
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );

  finalcut::FMouseControl mouse_control;
  CPPUNIT_ASSERT ( ! mouse_control.hasData() );

  mouse.setNewMousePosition(5, 12);
  CPPUNIT_ASSERT ( mouse.getPos() == finalcut::FPoint(0, 0) );
  CPPUNIT_ASSERT ( mouse.getNewMousePosition() == finalcut::FPoint(5, 12) );

  mouse.useNewMousePosition();
  CPPUNIT_ASSERT ( mouse.getPos() == finalcut::FPoint(5, 12) );
  CPPUNIT_ASSERT ( mouse.getNewMousePosition() == finalcut::FPoint(5, 12) );
}

//----------------------------------------------------------------------
void FMouseTest::doubleClickTest()
{
  using finalcut::operator -;

  test::FMouse_protected mouse;
  CPPUNIT_ASSERT ( mouse.getDblclickInterval() == 500000 );  // 500 ms
  TimeValue tv = {};
  CPPUNIT_ASSERT ( mouse.isDblclickTimeout(tv) );

  tv = finalcut::FObjectTimer::getCurrentTime();
  CPPUNIT_ASSERT ( ! mouse.isDblclickTimeout(tv) );

  tv -= std::chrono::seconds(1);  // Minus one second
  CPPUNIT_ASSERT ( mouse.isDblclickTimeout(tv) );

  mouse.setDblclickInterval(1000000);
  tv = finalcut::FObjectTimer::getCurrentTime();
  CPPUNIT_ASSERT ( ! mouse.isDblclickTimeout(tv) );

  auto tv_delta = std::chrono::microseconds(500000);
  tv -= tv_delta;
  CPPUNIT_ASSERT ( ! mouse.isDblclickTimeout(tv) );
  tv -= tv_delta;
  CPPUNIT_ASSERT ( mouse.isDblclickTimeout(tv) );
}

//----------------------------------------------------------------------
void FMouseTest::workspaceSizeTest()
{
  test::FMouse_protected mouse;
  CPPUNIT_ASSERT ( mouse.getMaxWidth() == 80 );
  CPPUNIT_ASSERT ( mouse.getMaxHeight() == 25 );

  mouse.setMaxWidth(92);
  mouse.setMaxHeight(30);
  CPPUNIT_ASSERT ( mouse.getMaxWidth() == 92 );
  CPPUNIT_ASSERT ( mouse.getMaxHeight() == 30 );
}

#ifdef F_HAVE_LIBGPM
//----------------------------------------------------------------------
void FMouseTest::gpmMouseTest()
{
  finalcut::FMouseGPM gpm_mouse;
  gpm_mouse.setStdinNo(fileno(stdin));
  CPPUNIT_ASSERT ( gpm_mouse.getMouseTypeID() == finalcut::FMouse::MouseType::Gpm );
  CPPUNIT_ASSERT ( ! gpm_mouse.isGpmMouseEnabled() );

  if ( gpm_mouse.enableGpmMouse() )
  {
    CPPUNIT_ASSERT ( gpm_mouse.isGpmMouseEnabled() );
    CPPUNIT_ASSERT ( ! gpm_mouse.hasEvent() );
  }
  else
    CPPUNIT_ASSERT ( ! gpm_mouse.isGpmMouseEnabled() );

  gpm_mouse.disableGpmMouse();
  CPPUNIT_ASSERT ( ! gpm_mouse.isGpmMouseEnabled() );
}
#endif

//----------------------------------------------------------------------
void FMouseTest::x11MouseTest()
{
  finalcut::FMouseX11 x11_mouse;
  CPPUNIT_ASSERT ( x11_mouse.getMouseTypeID() == finalcut::FMouse::MouseType::X11 );
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );

  auto rawdata1 = insertData ({ 0x1b, '[', 'M', 0x23, 0x50, 0x32, 0x40, 0x40 });
  x11_mouse.setRawData (rawdata1);
  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( rawdata1.getSize() == 2 );
  CPPUNIT_ASSERT ( rawdata1.strncmp_front("@@", 2) );

  auto tv = finalcut::FObjectTimer::getCurrentTime();
  x11_mouse.processEvent (tv);

  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(48, 18) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  // The same input again
  auto raw = insertData ({ 0x1b, '[', 'M', 0x23, 0x50, 0x32 });
  x11_mouse.setRawData (raw);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasEvent() );

  // Left mouse button pressed
  auto rawdata2 = insertData ({ 0x1b, '[', 'M', 0x20, 0x21, 0x21 });
  x11_mouse.setRawData (rawdata2);
  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( ! x11_mouse.hasUnprocessedInput() );
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  // Left mouse button released
  auto rawdata3 = insertData ({ 0x1b, '[', 'M', 0x23, 0x21, 0x21 });
  x11_mouse.setRawData (rawdata3);

  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( ! x11_mouse.hasUnprocessedInput() );
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  // Left mouse button pressed again (double click)
  auto rawdata4 = insertData ({ 0x1b, '[', 'M', 0x20, 0x21, 0x21 });
  x11_mouse.setRawData (rawdata4);

  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( ! x11_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );


  // Middle mouse button
  auto rawdata5 = insertData ({ 0x1b, '[', 'M', 0x21, 0x21, 0x21
                              , 0x1b, '[', 'M', 0x23, 0x21, 0x21 });
  x11_mouse.setRawData (rawdata5);

  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  x11_mouse.setRawData (rawdata5);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isMiddleButtonReleased() );

  // Right mouse button
  auto rawdata6 = insertData ({ 0x1b, '[', 'M', 0x22, 0x21, 0x21
                              , 0x1b, '[', 'M', 0x23, 0x21, 0x21 });
  x11_mouse.setRawData (rawdata6);

  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  x11_mouse.setRawData (rawdata6);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isRightButtonReleased() );

  // Mouse wheel
  auto rawdata7 = insertData ({ 0x1b, '[', 'M', 0x60, 0x70, 0x39
                              , 0x1b, '[', 'M', 0x61, 0x70, 0x39
                              , 0x1b, '[', 'M', 0x62, 0x70, 0x39
                              , 0x1b, '[', 'M', 0x63, 0x70, 0x39 });
  x11_mouse.setRawData (rawdata7);

  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(80, 25) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  x11_mouse.setRawData (rawdata7);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( x11_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );

  x11_mouse.setRawData (rawdata7);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( x11_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );

  x11_mouse.setRawData (rawdata7);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( x11_mouse.isWheelRight() );

  // Mouse move
  auto rawdata8 = insertData ({ 0x1b, '[', 'M', 0x20, 0x21, 0x21
                              , 0x1b, '[', 'M', 0x40, 0x23, 0x25
                              , 0x1b, '[', 'M', 0x23, 0x23, 0x25 });
  x11_mouse.setRawData (rawdata8);

  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  x11_mouse.setRawData (rawdata8);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(3, 5) );
  CPPUNIT_ASSERT ( x11_mouse.isMoved() );

  x11_mouse.setRawData (rawdata8);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(3, 5) );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  // Mouse + keyboard modifier key
  auto rawdata9 = insertData ({ 0x1b, '[', 'M', 0x24, 0x30, 0x40
                              , 0x1b, '[', 'M', 0x28, 0x30, 0x40
                              , 0x1b, '[', 'M', 0x30, 0x30, 0x40
                              , 0x1b, '[', 'M', 0x3c, 0x30, 0x40 });
  x11_mouse.setRawData (rawdata9);

  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! x11_mouse.hasData() );
  CPPUNIT_ASSERT ( x11_mouse.getPos() == finalcut::FPoint(16, 32) );
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  CPPUNIT_ASSERT ( x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! x11_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMoved() );

  x11_mouse.setRawData (rawdata9);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isMetaKeyPressed() );

  x11_mouse.setRawData (rawdata9);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! x11_mouse.isMetaKeyPressed() );

  x11_mouse.setRawData (rawdata9);
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( x11_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( x11_mouse.isMetaKeyPressed() );

  // Clear event test
  auto rawdata10 = insertData ({ 0x1b, '[', 'M', 0x20, 0x7f, 0x3f });
  x11_mouse.setRawData (rawdata10);
  CPPUNIT_ASSERT ( x11_mouse.hasData() );
  x11_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( x11_mouse.hasEvent() );
  x11_mouse.clearEvent();
  CPPUNIT_ASSERT ( ! x11_mouse.hasEvent() );
}

//----------------------------------------------------------------------
void FMouseTest::sgrMouseTest()
{
  finalcut::FMouseSGR sgr_mouse;
  CPPUNIT_ASSERT ( sgr_mouse.getMouseTypeID() == finalcut::FMouse::MouseType::Sgr );
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );

  // Left mouse button pressed
  auto rawdata1 = insertData ({ 0x1b, '[', '<', '0', ';', '7'
                              , '3', ';', '4', 'M', '@', '@' });
  sgr_mouse.setRawData (rawdata1);
  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( rawdata1.getSize() == 2 );
  CPPUNIT_ASSERT ( rawdata1.strncmp_front("@@", 2) );

  auto tv = finalcut::FObjectTimer::getCurrentTime();
  sgr_mouse.processEvent (tv);

  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(73, 4) );
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  CPPUNIT_ASSERT ( sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  // The same input again
  auto raw = insertData ({ 0x1b, '[', '<', '0', ';', '7', '3', ';', '4', 'M' });
  sgr_mouse.setRawData (raw);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasEvent() );

  // Left mouse button released
  auto rawdata2 = insertData ({ 0x1b, '[', '<', '0', ';', '7', '3', ';', '4', 'm' });
  sgr_mouse.setRawData (rawdata2);

  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( ! sgr_mouse.hasUnprocessedInput() );
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(73, 4) );
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  // Left mouse button pressed again (double click)
  auto rawdata4 = insertData ({ 0x1b, '[', '<', '0', ';', '7', '3', ';', '4', 'M' });
  sgr_mouse.setRawData (rawdata4);

  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( ! sgr_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(73, 4) );
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( sgr_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  // Middle mouse button
  auto rawdata5 = insertData ({ 0x1b, '[', '<', '1', ';', '1', ';', '1', 'M'
                              , 0x1b, '[', '<', '1', ';', '1', ';', '1', 'm' });
  sgr_mouse.setRawData (rawdata5);

  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  sgr_mouse.setRawData (rawdata5);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isMiddleButtonReleased() );

  // Right mouse button
  auto rawdata6 = insertData ({ 0x1b, '[', '<', '2', ';', '3', ';', '3', 'M'
                              , 0x1b, '[', '<', '2', ';', '3', ';', '4', 'm' });
  sgr_mouse.setRawData (rawdata6);

  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(3, 3) );
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  sgr_mouse.setRawData (rawdata6);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(3, 4) );
  CPPUNIT_ASSERT ( ! sgr_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isRightButtonReleased() );

  // Mouse wheel
  auto rawdata7 = insertData ({ 0x1b, '[', '<', '6', '4', ';', '4', ';', '9', 'M'
                              , 0x1b, '[', '<', '6', '5', ';', '4', ';', '9', 'M'
                              , 0x1b, '[', '<', '6', '6', ';', '4', ';', '9', 'M'
                              , 0x1b, '[', '<', '6', '7', ';', '4', ';', '9', 'M' });
  sgr_mouse.setRawData (rawdata7);

  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(4, 9) );
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  sgr_mouse.setRawData (rawdata7);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );

  sgr_mouse.setRawData (rawdata7);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );

  sgr_mouse.setRawData (rawdata7);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( sgr_mouse.isWheelRight() );

  // Mouse move
  auto rawdata8 = insertData ({ 0x1b, '[', '<', '0', ';', '1', ';', '2', 'M'
                              , 0x1b, '[', '<', '3', '2', ';', '2', ';', '3', 'M'
                              , 0x1b, '[', '<', '0', ';', '3', ';', '4', 'm' });
  sgr_mouse.setRawData (rawdata8);

  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(1, 2) );
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  CPPUNIT_ASSERT ( sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  sgr_mouse.setRawData (rawdata8);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(2, 3) );
  CPPUNIT_ASSERT ( sgr_mouse.isMoved() );

  sgr_mouse.setRawData (rawdata8);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(3, 4) );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  // Mouse + keyboard modifier key
  auto rawdata9 = insertData ({ 0x1b, '[', '<', '4', ';', '5', ';', '5', 'M'
                              , 0x1b, '[', '<', '8', ';', '5', ';', '5', 'M'
                              , 0x1b, '[', '<', '1', '6', ';', '5', ';', '5', 'M'
                              , 0x1b, '[', '<', '2', '8', ';', '5', ';', '5', 'M' });
  sgr_mouse.setRawData (rawdata9);

  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasData() );
  CPPUNIT_ASSERT ( sgr_mouse.getPos() == finalcut::FPoint(5, 5) );
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  CPPUNIT_ASSERT ( sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMoved() );

  sgr_mouse.setRawData (rawdata9);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isMetaKeyPressed() );

  sgr_mouse.setRawData (rawdata9);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! sgr_mouse.isMetaKeyPressed() );

  sgr_mouse.setRawData (rawdata9);
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( sgr_mouse.isMetaKeyPressed() );

  // Clear event test
  auto rawdata10 = insertData ({ 0x1b, '[', '<', '2', ';', '1', ';', '1', 'M' });
  sgr_mouse.setRawData (rawdata10);
  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( sgr_mouse.hasEvent() );
  sgr_mouse.clearEvent();
  CPPUNIT_ASSERT ( ! sgr_mouse.hasEvent() );

  // Wrong mouse data
  auto rawdata11 = insertData ({ 0x1b, '[', '<', '2', 'O', ';', '2', ';', '2', 'M'
                               , 0x1b, '[', '<', '1', ';', 'x', ';', '3', 'M'
                               , 0x1b, '[', '<', '6', ';', '5', ';', '@', 'M', '@' });
  sgr_mouse.setRawData (rawdata11);
  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasEvent() );

  sgr_mouse.setRawData (rawdata11);
  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasEvent() );

  sgr_mouse.setRawData (rawdata11);
  CPPUNIT_ASSERT ( sgr_mouse.hasData() );
  sgr_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! sgr_mouse.hasEvent() );

  CPPUNIT_ASSERT ( sgr_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( rawdata11.getSize() == 1 );
  CPPUNIT_ASSERT ( rawdata11.strncmp_front("@", 1) );
}

//----------------------------------------------------------------------
void FMouseTest::urxvtMouseTest()
{
  finalcut::FMouseUrxvt urxvt_mouse;
  CPPUNIT_ASSERT ( urxvt_mouse.getMouseTypeID() == finalcut::FMouse::MouseType::Urxvt );
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );

  // Left mouse button pressed
  auto rawdata1 = insertData ({ 0x1b, '[', '3', '2', ';', '4'
                              , '9', ';', '6', 'M', '@', '@' });
  urxvt_mouse.setRawData (rawdata1);
  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( rawdata1.getSize() == 2 );
  CPPUNIT_ASSERT ( rawdata1.strncmp_front("@@", 2) );

  auto tv = finalcut::FObjectTimer::getCurrentTime();
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(49, 6) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  CPPUNIT_ASSERT ( urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  // The same input again
  auto raw = insertData ({ 0x1b, '[', '3', '2', ';', '4', '9', ';', '6', 'M' });
  urxvt_mouse.setRawData (raw);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasEvent() );

  // Left mouse button released
  auto rawdata2 = insertData ({ 0x1b, '[', '3', '5', ';', '4', '9', ';', '6', 'M' });
  urxvt_mouse.setRawData (rawdata2);

  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasUnprocessedInput() );
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(49, 6) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  // Left mouse button pressed again (double click)
  auto rawdata4 = insertData ({ 0x1b, '[', '3', '2', ';', '4', '9', ';', '6', 'M' });
  urxvt_mouse.setRawData (rawdata4);

  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(49, 6) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( urxvt_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  // Middle mouse button
  auto rawdata5 = insertData ({ 0x1b, '[', '3', '3', ';', '1', ';', '1', 'M'
                              , 0x1b, '[', '3', '5', ';', '1', ';', '1', 'M' });
  urxvt_mouse.setRawData (rawdata5);

  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  urxvt_mouse.setRawData (rawdata5);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isMiddleButtonReleased() );

  // Right mouse button
  auto rawdata6 = insertData ({ 0x1b, '[', '3', '4', ';', '3', ';', '3', 'M'
                              , 0x1b, '[', '3', '5', ';', '3', ';', '4', 'M' });
  urxvt_mouse.setRawData (rawdata6);

  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(3, 3) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  urxvt_mouse.setRawData (rawdata6);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(3, 4) );
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isRightButtonReleased() );

  // Mouse wheel
  auto rawdata7 = insertData ({ 0x1b, '[', '9', '6', ';', '4', ';', '9', 'M'
                              , 0x1b, '[', '9', '7', ';', '4', ';', '9', 'M'
                              , 0x1b, '[', '9', '8', ';', '4', ';', '9', 'M'
                              , 0x1b, '[', '9', '9', ';', '4', ';', '9', 'M' });
  urxvt_mouse.setRawData (rawdata7);

  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(4, 9) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  urxvt_mouse.setRawData (rawdata7);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );

  urxvt_mouse.setRawData (rawdata7);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );

  urxvt_mouse.setRawData (rawdata7);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( urxvt_mouse.isWheelRight() );

  // Mouse move
  auto rawdata8 = insertData ({ 0x1b, '[', '3', '2', ';', '1', ';', '2', 'M'
                              , 0x1b, '[', '6', '4', ';', '2', ';', '3', 'M'
                              , 0x1b, '[', '3', '5', ';', '3', ';', '4', 'M' });
  urxvt_mouse.setRawData (rawdata8);

  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(1, 2) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  CPPUNIT_ASSERT ( urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  urxvt_mouse.setRawData (rawdata8);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(2, 3) );
  CPPUNIT_ASSERT ( urxvt_mouse.isMoved() );

  urxvt_mouse.setRawData (rawdata8);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(3, 4) );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  // Mouse + keyboard modifier key
  auto rawdata9 = insertData ({ 0x1b, '[', '3', '6', ';', '5', ';', '5', 'M'
                              , 0x1b, '[', '4', '0', ';', '5', ';', '5', 'M'
                              , 0x1b, '[', '4', '8', ';', '5', ';', '5', 'M'
                              , 0x1b, '[', '6', '0', ';', '5', ';', '5', 'M' });
  urxvt_mouse.setRawData (rawdata9);

  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(5, 5) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  CPPUNIT_ASSERT ( urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelUp() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelDown() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelLeft() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isWheelRight() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMoved() );

  urxvt_mouse.setRawData (rawdata9);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isMetaKeyPressed() );

  urxvt_mouse.setRawData (rawdata9);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! urxvt_mouse.isMetaKeyPressed() );

  urxvt_mouse.setRawData (rawdata9);
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isControlKeyPressed() );
  CPPUNIT_ASSERT ( urxvt_mouse.isMetaKeyPressed() );

  // Clear event test
  auto rawdata10 = insertData ({ 0x1b, '[', '3', '2', ';', '1', ';', '1', 'M' });
  urxvt_mouse.setRawData (rawdata10);
  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );
  urxvt_mouse.clearEvent();
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasEvent() );

  // Wrong mouse data
  auto rawdata11 = insertData ({ 0x1b, '[', '3', 'O', ';', '2', ';', '2', 'M'
                               , 0x1b, '[', '3', '3', ';', 'x', ';', '3', 'M'
                               , 0x1b, '[', '3', '4', ';', '5', ';', '@', 'M', '@' });
  urxvt_mouse.setRawData (rawdata11);
  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasEvent() );

  urxvt_mouse.setRawData (rawdata11);
  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasEvent() );

  urxvt_mouse.setRawData (rawdata11);
  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasEvent() );

  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( rawdata11.getSize() == 1 );
  CPPUNIT_ASSERT ( rawdata11.strncmp_front("@", 1) );

  // Negative values
  auto rawdata12 = insertData ({ 0x1b, '[', '3', '2', ';', '-', '5', ';', '5', 'M'
                               , 0x1b, '[', '3', '2', ';', '3', ';', '-', '3', 'M' });
  urxvt_mouse.setRawData (rawdata12);
  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.hasUnprocessedInput() );
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() != finalcut::FPoint(-5, 5) );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(1, 5) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );

  urxvt_mouse.setRawData (rawdata12);
  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() != finalcut::FPoint(3, -3) );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(3, 1) );
  CPPUNIT_ASSERT ( urxvt_mouse.hasEvent() );

  // Oversize values
  urxvt_mouse.setMaxWidth(40);
  urxvt_mouse.setMaxHeight(20);
  auto rawdata13 = insertData ({ 0x1b, '[', '3', '2', ';', '7', '0', ';', '2', '5', 'M' });
  urxvt_mouse.setRawData (rawdata13);
  CPPUNIT_ASSERT ( urxvt_mouse.hasData() );
  urxvt_mouse.processEvent (tv);
  CPPUNIT_ASSERT ( ! urxvt_mouse.hasData() );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() != finalcut::FPoint(70, 25) );
  CPPUNIT_ASSERT ( urxvt_mouse.getPos() == finalcut::FPoint(40, 20) );
}

//----------------------------------------------------------------------
void FMouseTest::mouseControlTest()
{
  char* pram_0 = finalcut::C_STR("./a.out");
  char** parms = &pram_0;
  finalcut::FApplication app(1, parms);
  CPPUNIT_ASSERT ( ! finalcut::FApplication::isQuit() );  // Need in processQueuedInput()

  finalcut::FMouseControl mouse_control;
  bool left_pressed{false};
  bool middle_pressed{false};
  bool right_pressed{false};
  bool has_current_mouse_event{false};
  auto cmd = [ &left_pressed
             , &middle_pressed
             , &right_pressed
             , &has_current_mouse_event
             , &mouse_control ] (const finalcut::FMouseData& md)
             {
               left_pressed = md.isLeftButtonPressed();
               middle_pressed = md.isMiddleButtonPressed();
               right_pressed = md.isRightButtonPressed();
               has_current_mouse_event = mouse_control.getCurrentMouseEvent() != nullptr;
             };
  finalcut::FMouseCommand mouse_cmd (cmd);
  mouse_control.setEventCommand (mouse_cmd);
  mouse_control.setStdinNo(fileno(stdin));
  mouse_control.setMaxWidth(100);
  mouse_control.setMaxHeight(40);
  mouse_control.clearEvent();
  mouse_control.setDblclickInterval(750000);
  mouse_control.useGpmMouse(true);
  mouse_control.useXtermMouse(true);
  mouse_control.enable();

  CPPUNIT_ASSERT ( ! mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(0, 0) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.hasEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelRight() );
  CPPUNIT_ASSERT ( ! mouse_control.isMoved() );
  CPPUNIT_ASSERT ( ! mouse_control.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! mouse_control.hasDataInQueue() );

  if ( mouse_control.isGpmMouseEnabled() )
  {
    CPPUNIT_ASSERT ( ! mouse_control.getGpmKeyPressed(false) );
    mouse_control.drawPointer();
  }

  // Left mouse button pressed on an X11 mouse
  auto rawdata1 = insertData ({ 0x1b, '[', 'M', 0x20, 0x25, 0x28
                              , 0x1b, '[', 'M', 0x23, 0x25, 0x28 });
  mouse_control.setRawData (finalcut::FMouse::MouseType::X11, rawdata1);

  CPPUNIT_ASSERT ( mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.hasUnprocessedInput() );
  auto tv = finalcut::FObjectTimer::getCurrentTime();
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( ! mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(5, 8) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( mouse_control.hasEvent() );
  CPPUNIT_ASSERT ( mouse_control.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelRight() );
  CPPUNIT_ASSERT ( ! mouse_control.isMoved() );
  CPPUNIT_ASSERT ( mouse_control.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( mouse_control.hasDataInQueue() );

  mouse_control.processQueuedInput();
  CPPUNIT_ASSERT ( left_pressed );
  CPPUNIT_ASSERT ( ! middle_pressed );
  CPPUNIT_ASSERT ( ! right_pressed );
  CPPUNIT_ASSERT ( has_current_mouse_event );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.hasDataInQueue() );

  mouse_control.setRawData (finalcut::FMouse::MouseType::X11, rawdata1);
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( ! mouse_control.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( mouse_control.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonDoubleClick() );

  // Middle mouse button on an SGR mouse
  auto rawdata2 = insertData ({ 0x1b, '[', '<', '1', ';', '1', ';', '1', 'M'
                              , 0x1b, '[', '<', '1', ';', '1', ';', '1', 'm' });
  mouse_control.setRawData (finalcut::FMouse::MouseType::Sgr, rawdata2);
  CPPUNIT_ASSERT ( mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( ! mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(1, 1) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( mouse_control.hasEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonReleased() );
  CPPUNIT_ASSERT ( mouse_control.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelRight() );
  CPPUNIT_ASSERT ( ! mouse_control.isMoved() );
  CPPUNIT_ASSERT ( mouse_control.hasDataInQueue() );

  mouse_control.processQueuedInput();
  CPPUNIT_ASSERT ( ! left_pressed );
  CPPUNIT_ASSERT ( middle_pressed );
  CPPUNIT_ASSERT ( ! right_pressed );
  CPPUNIT_ASSERT ( has_current_mouse_event );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.hasDataInQueue() );

  mouse_control.setRawData (finalcut::FMouse::MouseType::Sgr, rawdata2);
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( ! mouse_control.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( mouse_control.isMiddleButtonReleased() );

  // Right mouse button on a urxvt mouse
  auto rawdata3 = insertData ({ 0x1b, '[', '3', '4', ';', '3', ';', '3', 'M'
                              , 0x1b, '[', '3', '5', ';', '3', ';', '4', 'M' });
  mouse_control.setRawData (finalcut::FMouse::MouseType::Urxvt, rawdata3);
  CPPUNIT_ASSERT ( mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( ! mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(3, 3) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( mouse_control.hasEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( mouse_control.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelRight() );
  CPPUNIT_ASSERT ( ! mouse_control.isMoved() );
  CPPUNIT_ASSERT ( mouse_control.hasDataInQueue() );

  mouse_control.processQueuedInput();
  CPPUNIT_ASSERT ( ! left_pressed );
  CPPUNIT_ASSERT ( ! middle_pressed );
  CPPUNIT_ASSERT ( right_pressed );
  CPPUNIT_ASSERT ( has_current_mouse_event );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.hasDataInQueue() );

  mouse_control.setRawData (finalcut::FMouse::MouseType::Urxvt, rawdata3);
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(3, 4) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonPressed() );
  CPPUNIT_ASSERT ( mouse_control.isRightButtonReleased() );

  // Mouse wheel on an X11 mouse
  auto rawdata4 = insertData ({ 0x1b, '[', 'M', 0x60, 0x70, 0x39
                              , 0x1b, '[', 'M', 0x61, 0x70, 0x39
                              , 0x1b, '[', 'M', 0x62, 0x70, 0x39
                              , 0x1b, '[', 'M', 0x63, 0x70, 0x39 });
  mouse_control.setRawData (finalcut::FMouse::MouseType::X11, rawdata4);
  CPPUNIT_ASSERT ( mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( ! mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(80, 25) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( mouse_control.hasEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelRight() );
  CPPUNIT_ASSERT ( ! mouse_control.isMoved() );
  CPPUNIT_ASSERT ( mouse_control.hasDataInQueue() );

  mouse_control.processQueuedInput();
  CPPUNIT_ASSERT ( ! left_pressed );
  CPPUNIT_ASSERT ( ! middle_pressed );
  CPPUNIT_ASSERT ( ! right_pressed );
  CPPUNIT_ASSERT ( has_current_mouse_event );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.hasDataInQueue() );

  mouse_control.setRawData (finalcut::FMouse::MouseType::X11, rawdata4);
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( mouse_control.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelRight() );

  mouse_control.setRawData (finalcut::FMouse::MouseType::X11, rawdata4);
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( mouse_control.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelRight() );

  mouse_control.setRawData (finalcut::FMouse::MouseType::X11, rawdata4);
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( ! mouse_control.hasUnprocessedInput() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( mouse_control.isWheelRight() );

  // Mouse move on an SGR mouse
  auto rawdata5 = insertData ({ 0x1b, '[', '<', '0', ';', '1', ';', '2', 'M'
                              , 0x1b, '[', '<', '3', '2', ';', '2', ';', '3', 'M'
                              , 0x1b, '[', '<', '0', ';', '3', ';', '4', 'm' });
  mouse_control.setRawData (finalcut::FMouse::MouseType::Sgr, rawdata5);
  CPPUNIT_ASSERT ( mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.hasUnprocessedInput() );
  tv = finalcut::FObjectTimer::getCurrentTime();
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( ! mouse_control.hasData() );
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(1, 2) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( mouse_control.hasEvent() );
  CPPUNIT_ASSERT ( mouse_control.isLeftButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isLeftButtonDoubleClick() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isRightButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMiddleButtonReleased() );
  CPPUNIT_ASSERT ( ! mouse_control.isShiftKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isControlKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isMetaKeyPressed() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelUp() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelDown() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelLeft() );
  CPPUNIT_ASSERT ( ! mouse_control.isWheelRight() );
  CPPUNIT_ASSERT ( ! mouse_control.isMoved() );
  CPPUNIT_ASSERT ( mouse_control.hasDataInQueue() );

  mouse_control.processQueuedInput();
  CPPUNIT_ASSERT ( left_pressed );
  CPPUNIT_ASSERT ( ! middle_pressed );
  CPPUNIT_ASSERT ( ! right_pressed );
  CPPUNIT_ASSERT ( has_current_mouse_event );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.hasDataInQueue() );

  mouse_control.setRawData (finalcut::FMouse::MouseType::Sgr, rawdata5);
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(2, 3) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( mouse_control.isMoved() );

  mouse_control.setRawData (finalcut::FMouse::MouseType::Sgr, rawdata5);
  mouse_control.processEvent (tv);
  CPPUNIT_ASSERT ( mouse_control.getPos() == finalcut::FPoint(3, 4) );
  CPPUNIT_ASSERT ( ! mouse_control.getCurrentMouseEvent() );
  CPPUNIT_ASSERT ( ! mouse_control.isMoved() );

  mouse_control.disable();
}

//----------------------------------------------------------------------
auto FMouseTest::insertData (std::initializer_list<char> list) -> finalcut::FKeyboard::keybuffer
{
  finalcut::FKeyboard::keybuffer buffer;

  for (const char& ch : list)
    buffer.push(ch);

  return buffer;
}

// Put the test suite in the registry
CPPUNIT_TEST_SUITE_REGISTRATION (FMouseTest);

// The general unit test main part
#include <main-test.inc>
