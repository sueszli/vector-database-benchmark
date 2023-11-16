/***********************************************************************
* fcheckbox.cpp - Widget FCheckBox                                     *
*                                                                      *
* This file is part of the FINAL CUT widget toolkit                    *
*                                                                      *
* Copyright 2014-2023 Markus Gans                                      *
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

#include "final/fc.h"
#include "final/widget/fcheckbox.h"

namespace finalcut
{

//----------------------------------------------------------------------
// class FCheckBox
//----------------------------------------------------------------------

// constructor and destructor
//----------------------------------------------------------------------
FCheckBox::FCheckBox(FWidget* parent)
  : FToggleButton{parent}
{
  init();
}

//----------------------------------------------------------------------
FCheckBox::FCheckBox (const FString& txt, FWidget* parent)
  : FToggleButton{txt, parent}
{
  init();
}

//----------------------------------------------------------------------
FCheckBox::~FCheckBox() noexcept = default;  // destructor


// private methods of FCheckBox
//----------------------------------------------------------------------
void FCheckBox::init()
{
  setLabelOffsetPos(4);
  setButtonWidth(4);
  FWidget::setVisibleCursor();
}

//----------------------------------------------------------------------
void FCheckBox::draw()
{
  if ( ! isVisible() )
    return;

  drawCheckButton();
  drawLabel();
  FToggleButton::draw();
}

//----------------------------------------------------------------------
void FCheckBox::drawCheckButton()
{
  print() << FPoint{1, 1};
  setColor();

  if ( FVTerm::getFOutput()->isMonochron() )
  {
    if ( hasFocus() )
      setReverse(false);
    else
      setReverse(true);
  }

  if ( isChecked() )
    drawChecked();
  else
    drawUnchecked();

  if ( FVTerm::getFOutput()->isMonochron() )
    setReverse(false);
}

//----------------------------------------------------------------------
inline void FCheckBox::drawChecked()
{
  if ( FVTerm::getFOutput()->isNewFont() )
    print (CHECKBOX_ON);
  else
  {
    print ('[');
    print (UniChar::Times);  // Times ×
    print (']');
  }
}

//----------------------------------------------------------------------
inline void FCheckBox::drawUnchecked()
{
  if ( FVTerm::getFOutput()->isNewFont() )
    print (CHECKBOX);
  else
  {
    print ('[');
    print (' ');
    print (']');
  }
}

}  // namespace finalcut
