/***********************************************************************
* listview.cpp - Example for using a multi-column FListView widget     *
*                                                                      *
* This file is part of the FINAL CUT widget toolkit                    *
*                                                                      *
* Copyright 2017-2022 Markus Gans                                      *
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

#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <final/final.h>

using finalcut::FPoint;
using finalcut::FSize;


//----------------------------------------------------------------------
// class Listview
//----------------------------------------------------------------------

class Listview final : public finalcut::FDialog
{
  public:
    // Constructor
    explicit Listview (finalcut::FWidget* = nullptr);

  private:
    // Method
    void populate();
    void initLayout() override;

    // Event handlers
    void onClose (finalcut::FCloseEvent*) override;

    // Callback method
    void cb_showInMessagebox();
    void cb_showHideColumns();

    // Data members
    finalcut::FListView listview{this};
    finalcut::FButton   columns{this};
    finalcut::FButton   quit{this};
};

//----------------------------------------------------------------------
Listview::Listview (finalcut::FWidget* parent)
  : finalcut::FDialog{parent}
{
  // Add columns to the view
  listview.addColumn ("City");
  listview.addColumn ("Condition");
  listview.addColumn ("Temp.");
  listview.addColumn ("Humidity");
  listview.addColumn ("Pressure", 10);

  // Set right alignment for the third, fourth, and fifth column
  listview.setColumnAlignment (3, finalcut::Align::Right);
  listview.setColumnAlignment (4, finalcut::Align::Right);
  listview.setColumnAlignment (5, finalcut::Align::Right);

  // Set the type of sorting
  listview.setColumnSortType (1, finalcut::SortType::Name);
  listview.setColumnSortType (2, finalcut::SortType::Name);
  listview.setColumnSortType (3, finalcut::SortType::Number);
  listview.setColumnSortType (4, finalcut::SortType::Number);
  listview.setColumnSortType (5, finalcut::SortType::Number);

  // Sort in ascending order by the 1st column
  listview.setColumnSort (1, finalcut::SortOrder::Ascending);
  // Sorting follows later automatically on insert().
  // Otherwise you could start the sorting directly with sort()

  // Allways show the sort indicator (▼/▲)
  listview.hideSortIndicator(false);

  // Populate FListView with a list of items
  populate();

  // Set push button text
  columns.setText (L"&Columns");
  quit.setText (L"&Quit");

  // Add some function callbacks
  quit.addCallback
  (
    "clicked",
    finalcut::getFApplication(),
    &finalcut::FApplication::cb_exitApp,
    this
  );

  columns.addCallback
  (
    "clicked",
    this, &Listview::cb_showHideColumns
  );

  listview.addCallback
  (
    "clicked",
    this, &Listview::cb_showInMessagebox
  );
}

//----------------------------------------------------------------------
void Listview::populate()
{
  constexpr std::array<std::array<const char*, 5>, 41> weather =
  {{
    {{ "Alexandria", "Sunny", "31°C", "61%", "1006.4 mb" }},
    {{ "Amsterdam", "Cloudy", "21°C", "82%", "1021.3 mb" }},
    {{ "Baghdad", "Fair", "47°C", "9%", "1001.0 mb" }},
    {{ "Bangkok", "Partly Cloudy", "30°C", "69%", "1002.0 mb" }},
    {{ "Beijing", "Fair", "31°C", "68%", "1007.1 mb" }},
    {{ "Berlin", "Cloudy", "22°C", "53%", "1022.0 mb" }},
    {{ "Bogotá", "Fair", "9°C", "95%", "1028.5 mb" }},
    {{ "Budapest", "Partly Cloudy", "23°C", "37%", "1020.7 mb" }},
    {{ "Buenos Aires", "Cloudy", "7°C", "73%", "1019.0 mb" }},
    {{ "Cairo", "Fair", "39°C", "22%", "1006.1 mb" }},
    {{ "Cape Town", "Partly Cloudy", "12°C", "45%", "1030.1 mb" }},
    {{ "Chicago", "Mostly Cloudy", "21°C", "81%", "1014.9 mb" }},
    {{ "Delhi", "Haze", "33°C", "68%", "998.0 mb" }},
    {{ "Dhaka", "Haze", "32°C", "64%", "996.3 mb" }},
    {{ "Houston", "Cloudy", "23°C", "100%", "1014.2 mb" }},
    {{ "Istanbul", "Mostly Cloudy", "27°C", "61%", "1011.2 mb" }},
    {{ "Jakarta", "Fair", "28°C", "71%", "1009.1 mb" }},
    {{ "Jerusalem", "Sunny", "35°C", "17%", "1005.8 mb" }},
    {{ "Johannesburg", "Fair", "18°C", "16%", "1020.0 mb" }},
    {{ "Karachi", "Mostly Cloudy", "29°C", "76%", "998.0 mb" }},
    {{ "Lagos", "Mostly Cloudy", "27°C", "86%", "1014.6 mb" }},
    {{ "Lima", "Cloudy", "17°C", "83%", "1017.3 mb" }},
    {{ "London", "Cloudy", "23°C", "71%", "1023.0 mb" }},
    {{ "Los Angeles", "Fair", "21°C", "78%", "1011.9 mb" }},
    {{ "Madrid", "Fair", "32°C", "35%", "1020.0 mb" }},
    {{ "Mexico City", "Partly Cloudy", "14°C", "79%", "1028.5 mb" }},
    {{ "Moscow", "Partly Cloudy", "24°C", "54%", "1014.2 mb" }},
    {{ "Mumbai", "Haze", "28°C", "77%", "1003.0 mb" }},
    {{ "New York City", "Sunny", "21°C", "80%", "1014.2 mb" }},
    {{ "Paris", "Partly Cloudy", "27°C", "57%", "1024.4 mb" }},
    {{ "Reykjavík", "Mostly Cloudy", "11°C", "76%", "998.6 mb" }},
    {{ "Rio de Janeiro", "Fair", "24°C", "64%", "1022.0 mb" }},
    {{ "Rome", "Fair", "32°C", "18%", "1014.2 mb" }},
    {{ "Saint Petersburg", "Mostly Cloudy", "18°C", "55%", "1014.6 mb" }},
    {{ "São Paulo", "Fair", "19°C", "53%", "1024.0 mb" }},
    {{ "Seoul", "Cloudy", "26°C", "87%", "1012.2 mb" }},
    {{ "Shanghai", "Fair", "32°C", "69%", "1009.1 mb" }},
    {{ "Singapore", "Mostly Cloudy", "29°C", "73%", "1009.1 mb" }},
    {{ "Tehran", "Fair", "36°C", "14%", "1013.2 mb" }},
    {{ "Tokyo", "Mostly Cloudy", "28°C", "67%", "1009.1 mb" }},
    {{ "Zurich", "Mostly Cloudy", "23°C", "44%", "1023.7 mb" }}
  }};

  for (const auto& place : weather)
  {
    const finalcut::FStringList line (place.cbegin(), place.cend());
    listview.insert (line);
  }
}

//----------------------------------------------------------------------
void Listview::initLayout()
{
  // Set FListView geometry
  listview.setGeometry(FPoint{2, 1}, FSize{33, 14});
  // Set columns button geometry
  columns.setGeometry(FPoint{2, 16}, FSize{11, 1});
  // Set quit button geometry
  quit.setGeometry(FPoint{24, 16}, FSize{10, 1});
  FDialog::initLayout();
}

//----------------------------------------------------------------------
void Listview::onClose (finalcut::FCloseEvent* ev)
{
  finalcut::FApplication::closeConfirmationDialog (this, ev);
}

//----------------------------------------------------------------------
void Listview::cb_showInMessagebox()
{
  const auto& item = listview.getCurrentItem();
  finalcut::FMessageBox info ( "Weather in " + item->getText(1)
                             , "  Condition: " + item->getText(2) + "\n"
                               "Temperature: " + item->getText(3) + "\n"
                               "   Humidity: " + item->getText(4) + "\n"
                               "   Pressure: " + item->getText(5)
                             , finalcut::FMessageBox::ButtonType::Ok
                             , finalcut::FMessageBox::ButtonType::Reject
                             , finalcut::FMessageBox::ButtonType::Reject
                             , this );
  info.show();
}

//----------------------------------------------------------------------
void Listview::cb_showHideColumns()
{
  finalcut::FMessageBox column_header_dlg \
  (
    "Show colums"
    , "\n\n\n\n"
    , finalcut::FMessageBox::ButtonType::Ok
    , finalcut::FMessageBox::ButtonType::Cancel
    , finalcut::FMessageBox::ButtonType::Reject
    , this
  );

  auto number_of_columns = listview.getColumnCount();
  std::vector<std::shared_ptr<finalcut::FCheckBox>> checkboxes{};

  for (std::size_t column{0}; column < number_of_columns; column++)
  {
    auto col_name = listview.getColumnText(int(column) + 1);
    checkboxes.emplace_back(std::make_shared<finalcut::FCheckBox>(col_name, &column_header_dlg));
    checkboxes[column]->setGeometry (FPoint{6, 4 + int(column)}, FSize{20, 1});

    if ( ! listview.isColumnHidden(int(column) + 1) )
      checkboxes[column]->setChecked();
  }

  column_header_dlg.setHeadline("Select columns to view");
  const auto& ret = column_header_dlg.exec();

  if ( ret != finalcut::FMessageBox::ButtonType::Ok )
    return;

  for (std::size_t column{0}; column < number_of_columns; column++)
  {
    if ( listview.isColumnHidden(int(column) + 1) && checkboxes[column]->isChecked() )
      listview.showColumn(int(column) + 1);
    else if ( ! listview.isColumnHidden(int(column) + 1) && ! checkboxes[column]->isChecked() )
      listview.hideColumn(int(column) + 1);
  }
}

//----------------------------------------------------------------------
//                               main part
//----------------------------------------------------------------------

auto main (int argc, char* argv[]) -> int
{
  // Create the application object
  finalcut::FApplication app(argc, argv);

  // Create main dialog object
  Listview d(&app);
  d.setText (L"Weather data");
  d.setGeometry ( FPoint{int(1 + (app.getWidth() - 37) / 2), 3}
                , FSize{37, 20} );
  d.setShadow();

  // Set dialog d as main widget
  finalcut::FWidget::setMainWidget(&d);

  // Show and start the application
  d.show();
  return app.exec();
}
