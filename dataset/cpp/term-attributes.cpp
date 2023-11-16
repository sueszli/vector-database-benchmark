/***********************************************************************
* term-attributes.cpp - Test the video attributes of the terminal      *
*                                                                      *
* This file is part of the FINAL CUT widget toolkit                    *
*                                                                      *
* Copyright 2015-2023 Markus Gans                                      *
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

#include <functional>
#include <vector>
#include <final/final.h>

using finalcut::FPoint;
using finalcut::FSize;
using finalcut::FColorPair;
using finalcut::FColor;


//----------------------------------------------------------------------
// class AttribDlg
//----------------------------------------------------------------------

class AttribDlg final : public finalcut::FDialog
{
  public:
    // Constructor
    explicit AttribDlg (finalcut::FWidget* = nullptr);

    // Methods
    auto getBGColor() const -> FColor;

    // Event handlers
    void onKeyPress (finalcut::FKeyEvent*) override;
    void onWheel (finalcut::FWheelEvent*) override;
    void onClose (finalcut::FCloseEvent*) override;

    // Callback methods
    void cb_next();
    void cb_back();

  private:
    // Method
    void initLayout() override;
    void adjustSize() override;
    void draw() override;

    // Data members
    FColor bgcolor{FColor::Undefined};
    finalcut::FButton next_button{"&Next >", this};
    finalcut::FButton back_button{"< &Back", this};
};

//----------------------------------------------------------------------
AttribDlg::AttribDlg (finalcut::FWidget* parent)
  : finalcut::FDialog{parent}
{
  next_button.addAccelerator (finalcut::FKey::Right);
  back_button.addAccelerator (finalcut::FKey::Left);

  // Add function callbacks
  next_button.addCallback
  (
    "clicked",
    this, &AttribDlg::cb_next
  );

  back_button.addCallback
  (
    "clicked",
    this, &AttribDlg::cb_back
  );
}

//----------------------------------------------------------------------
auto AttribDlg::getBGColor() const -> FColor
{
  return bgcolor;
}

//----------------------------------------------------------------------
void AttribDlg::onKeyPress (finalcut::FKeyEvent* ev)
{
  if ( ! ev )
    return;

  if ( ev->key() == finalcut::FKey('q') )
  {
    close();
    ev->accept();
  }
  else
    finalcut::FDialog::onKeyPress(ev);
}

//----------------------------------------------------------------------
void AttribDlg::onWheel (finalcut::FWheelEvent* ev)
{
  const finalcut::MouseWheel wheel = ev->getWheel();

  if ( wheel == finalcut::MouseWheel::Up )
    cb_next();
  else if ( wheel == finalcut::MouseWheel::Down )
    cb_back();
}

//----------------------------------------------------------------------
void AttribDlg::onClose (finalcut::FCloseEvent* ev)
{
  finalcut::FApplication::closeConfirmationDialog (this, ev);
}

//----------------------------------------------------------------------
void AttribDlg::cb_next()
{
  if ( finalcut::FVTerm::getFOutput()->isMonochron() )
    return;

  if ( bgcolor == FColor(finalcut::FVTerm::getFOutput()->getMaxColor() - 1) )
    bgcolor = FColor::Default;
  else if ( bgcolor == FColor::Default )
    bgcolor = FColor::Black;
  else
    ++bgcolor;

  redraw();
}

//----------------------------------------------------------------------
void AttribDlg::cb_back()
{
  if ( finalcut::FVTerm::getFOutput()->isMonochron() )
    return;

  if ( bgcolor == 0 )
    bgcolor = FColor::Default;
  else if ( bgcolor == FColor::Default )
    bgcolor = FColor(finalcut::FVTerm::getFOutput()->getMaxColor() - 1);
  else
    --bgcolor;

  redraw();
}

//----------------------------------------------------------------------
void AttribDlg::initLayout()
{
  next_button.setGeometry ( FPoint{int(getWidth()) - 13, int(getHeight()) - 4}
                          , FSize{10, 1} );
  back_button.setGeometry ( FPoint{int(getWidth()) - 25, int(getHeight()) - 4}
                          , FSize{10, 1} );
  FDialog::initLayout();
}

//----------------------------------------------------------------------
void AttribDlg::adjustSize()
{
  auto x = int((getDesktopWidth() - getWidth()) / 2);
  auto y = int((getDesktopHeight() - getHeight()) / 2) + 1;

  if ( x < 1 )
    x = 1;

  if ( y < 1 )
    y = 1;

  setGeometry(FPoint{x, y}, FSize{69, 21}, false);
  next_button.setGeometry ( FPoint{int(getWidth()) - 13, int(getHeight()) - 4}
                          , FSize{10, 1}, false );
  back_button.setGeometry ( FPoint{int(getWidth()) - 25, int(getHeight()) - 4}
                          , FSize{10, 1}, false );
  finalcut::FDialog::adjustSize();
}

//----------------------------------------------------------------------
void AttribDlg::draw()
{
  if ( bgcolor == FColor::Undefined )
  {
    // Get the color after initializing the color theme in show()
    if ( finalcut::FVTerm::getFOutput()->isMonochron() )
      bgcolor = FColor::Default;
    else
      bgcolor = getColorTheme()->label.bg;

    // Get the terminal type after the terminal detection in show()
    FDialog::setText ( "A terminal attributes test ("
                     + finalcut::FString{finalcut::FTerm::getTermType()}
                     + ")");
  }

  FDialog::draw();
}


//----------------------------------------------------------------------
// class AttribDemo
//----------------------------------------------------------------------

class AttribDemo final : public finalcut::FWidget
{
  public:
    // Constructor
    explicit AttribDemo (FWidget* = nullptr);

    // Event handler
    void onWheel (finalcut::FWheelEvent* ev) override
    {
      auto p = static_cast<AttribDlg*>(getParentWidget());

      if ( p )
        p->onWheel(ev);
    }

  private:
    // Methods
    void printColorLine();
    void printAltCharset();
    void printDim();
    void printNormal();
    void printBold();
    void printBoldDim();
    void printItalic();
    void printUnderline();
    void printDblUnderline();
    void printCrossesOut();
    void printBlink();
    void printReverse();
    void printStandout();
    void printInvisible();
    void printProtected();
    void draw() override;

    // Data member
    FColor last_color{FColor::Blue};
};

//----------------------------------------------------------------------
AttribDemo::AttribDemo (finalcut::FWidget* parent)
  : finalcut::FWidget{parent}
{
  unsetFocusable();
}

//----------------------------------------------------------------------
void AttribDemo::printColorLine()
{
  const auto& parent = static_cast<AttribDlg*>(getParent());

  for (FColor color{FColor::Black}; color < last_color; ++color)
  {
    print() << FColorPair{color, parent->getBGColor()} << " # ";
  }
}

//----------------------------------------------------------------------
void AttribDemo::printAltCharset()
{
  const auto& wc = getColorTheme();
  const auto& parent = static_cast<AttribDlg*>(getParent());

  if ( ! finalcut::FVTerm::getFOutput()->isMonochron() )
    setColor (wc->label.fg, wc->label.bg);

  print() << FPoint{1, 1} << "Alternate charset: ";

  if ( parent->getBGColor() == FColor::Default )
  {
    setColor (FColor::Default, FColor::Default);
  }
  else
  {
    if ( (parent->getBGColor() <= 8)
      || (parent->getBGColor() >= 16 && parent->getBGColor() <= 231
        && (parent->getBGColor() - 16) % 36 <= 17)
      || (parent->getBGColor() >= 232 && parent->getBGColor() <= 243) )
      setColor (FColor::White, parent->getBGColor());
    else
      setColor (FColor::Black, parent->getBGColor());
  }

  setAltCharset();
  print("`abcdefghijklmnopqrstuvwxyz{|}~");
  unsetAltCharset();
  print("                 ");
}

//----------------------------------------------------------------------
void AttribDemo::printDim()
{
  print("              Dim: ");
  setDim();
  printColorLine();
  unsetDim();
}

//----------------------------------------------------------------------
void AttribDemo::printNormal()
{
  print("           Normal: ");
  setNormal();
  printColorLine();
}

//----------------------------------------------------------------------
void AttribDemo::printBold()
{
  print("             Bold: ");
  setBold();
  printColorLine();
  unsetBold();
}

//----------------------------------------------------------------------
void AttribDemo::printBoldDim()
{
  print("         Bold+Dim: ");
  setBold();
  setDim();
  printColorLine();
  unsetDim();
  unsetBold();
}

//----------------------------------------------------------------------
void AttribDemo::printItalic()
{
  print("           Italic: ");
  setItalic();
  printColorLine();
  unsetItalic();
}

//----------------------------------------------------------------------
void AttribDemo::printUnderline()
{
  print("        Underline: ");
  setUnderline();
  printColorLine();
  unsetUnderline();
}

//----------------------------------------------------------------------
void AttribDemo::printDblUnderline()
{
  print(" Double underline: ");
  setDoubleUnderline();
  printColorLine();
  unsetDoubleUnderline();
}

//----------------------------------------------------------------------
void AttribDemo::printCrossesOut()
{
  print("      Crossed-out: ");
  setCrossedOut();
  printColorLine();
  unsetCrossedOut();
}

//----------------------------------------------------------------------
void AttribDemo::printBlink()
{
  print("            Blink: ");
  setBlink();
  printColorLine();
  unsetBlink();
}

//----------------------------------------------------------------------
void AttribDemo::printReverse()
{
  print("          Reverse: ");
  setReverse();
  printColorLine();
  unsetReverse();
}

//----------------------------------------------------------------------
void AttribDemo::printStandout()
{
  print("         Standout: ");
  setStandout();
  printColorLine();
  unsetStandout();
}

//----------------------------------------------------------------------
void AttribDemo::printInvisible()
{
  print("        Invisible: ");
  setInvisible();
  printColorLine();
  unsetInvisible();
}

//----------------------------------------------------------------------
void AttribDemo::printProtected()
{
  print("        Protected: ");
  setProtected();
  printColorLine();
  unsetProtected();
}

//----------------------------------------------------------------------
void AttribDemo::draw()
{
  const auto& wc = getColorTheme();
  last_color = FColor(finalcut::FVTerm::getFOutput()->getMaxColor());

  if ( finalcut::FVTerm::getFOutput()->isMonochron() )
    last_color = FColor(1);
  else if ( last_color > 16 )
    last_color = FColor(16);

  // test alternate character set
  printAltCharset();

  const std::vector<std::function<void()> > effect
  {
    [this] { printNormal(); },
    [this] { printDim(); },
    [this] { printBold(); },
    [this] { printBoldDim(); },
    [this] { printItalic(); },
    [this] { printUnderline(); },
    [this] { printDblUnderline(); },
    [this] { printCrossesOut(); },
    [this] { printBlink(); },
    [this] { printReverse(); },
    [this] { printStandout(); },
    [this] { printInvisible(); },
    [this] { printProtected(); },
  };

  for (std::size_t y{0}; y < getParentWidget()->getHeight() - 7; y++)
  {
    print() << FPoint{1, 2 + int(y)};

    if ( ! finalcut::FVTerm::getFOutput()->isMonochron() )
      setColor (wc->label.fg, wc->label.bg);

    if ( y < effect.size() )
      effect[y]();
  }

  if ( ! finalcut::FVTerm::getFOutput()->isMonochron() )
    setColor(wc->label.fg, wc->label.bg);

  print() << FPoint{1, 15};
  const FColor bg = static_cast<AttribDlg*>(getParent())->getBGColor();
  print (" Background color:");

  if ( bg == FColor::Default )
    print (" default");
  else
    printf ( " %u", bg);

  print() << FPoint{16, 17} << "Change background color ->";
}


//----------------------------------------------------------------------
//                               main part
//----------------------------------------------------------------------
auto main (int argc, char* argv[]) -> int
{
  // Create the application object
  finalcut::FApplication app {argc, argv};

  // Create a dialog box object.
  // This object will be automatically deleted by
  // the parent object "app" (FObject destructor).
  AttribDlg dialog{&app};
  dialog.setSize (FSize{69, 21});
  dialog.setShadow();  // Instead of the transparent window shadow

  // Create the attribute demo widget as a child object from the dialog
  AttribDemo demo(&dialog);
  demo.setGeometry (FPoint{1, 1}, FSize{67, 19});

  // Set the dialog object as main widget
  finalcut::FWidget::setMainWidget(&dialog);

  // Show and start the application
  dialog.show();
  return app.exec();
}
