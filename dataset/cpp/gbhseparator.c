
/*  Gtk+ User Interface Builder
 *  Copyright (C) 1998  Damon Chaplin
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#include <gtk/gtkhseparator.h>
#include "../gb.h"

/* Include the 21x21 icon pixmap for this widget, to be used in the palette */
#include "../graphics/hseparator.xpm"

/*
 * This is the GbWidget struct for this widget (see ../gbwidget.h).
 * It is initialized in the init() function at the end of this file
 */
static GbWidget gbwidget;



/******
 * NOTE: To use these functions you need to uncomment them AND add a pointer
 * to the function in the GbWidget struct at the end of this file.
 ******/

/*
 * Creates a new GtkWidget of class GtkHSeparator, performing any specialized
 * initialization needed for the widget to work correctly in this environment.
 * If a dialog box is used to initialize the widget, return NULL from this
 * function, and call data->callback with your new widget when it is done.
 * If the widget needs a special destroy handler, add a signal here.
 */
GtkWidget*
gb_hseparator_new(GbWidgetNewData *data)
{
  return gtk_hseparator_new ();
}



/*
 * Creates the components needed to edit the extra properties of this widget.
 */
/*
   static void
   gb_hseparator_create_properties(GtkWidget *widget, GbWidgetCreateArgData *data)
   {

   }
 */



/*
 * Gets the properties of the widget. This is used for both displaying the
 * properties in the property editor, and also for saving the properties.
 */
/*
   static void
   gb_hseparator_get_properties(GtkWidget *widget, GbWidgetGetArgData *data)
   {

   }
 */



/*
 * Sets the properties of the widget. This is used for both applying the
 * properties changed in the property editor, and also for loading.
 */
/*
   static void
   gb_hseparator_set_properties(GtkWidget *widget, GbWidgetSetArgData *data)
   {

   }
 */



/*
 * Adds menu items to a context menu which is just about to appear!
 * Add commands to aid in editing a GtkHSeparator, with signals pointing to
 * other functions in this file.
 */
/*
   static void
   gb_hseparator_create_popup_menu(GtkWidget *widget, GbWidgetCreateMenuData *data)
   {

   }
 */



/*
 * Writes the source code needed to create this widget.
 * You have to output everything necessary to create the widget here, though
 * there are some convenience functions to help.
 */
static void
gb_hseparator_write_source (GtkWidget * widget, GbWidgetWriteSourceData * data)
{
  if (data->create_widget)
    {
      source_add (data, "  %s = gtk_hseparator_new ();\n", data->wname);
    }
  gb_widget_write_standard_source (widget, data);
}



/*
 * Initializes the GbWidget structure.
 * I've placed this at the end of the file so we don't have to include
 * declarations of all the functions.
 */
GbWidget *
gb_hseparator_init ()
{
  /* Initialise the GTK type */
  volatile GtkType type;
  type = gtk_hseparator_get_type ();

  /* Initialize the GbWidget structure */
  gb_widget_init_struct (&gbwidget);

  /* Fill in the pixmap struct & tooltip */
  gbwidget.pixmap_struct = hseparator_xpm;
  gbwidget.tooltip = _("Horizonal Separator");

  /* Fill in any functions that this GbWidget has */
   gbwidget.gb_widget_new               = gb_hseparator_new;
/*
   gbwidget.gb_widget_create_properties = gb_hseparator_create_properties;
   gbwidget.gb_widget_get_properties    = gb_hseparator_get_properties;
   gbwidget.gb_widget_set_properties    = gb_hseparator_set_properties;
   gbwidget.gb_widget_create_popup_menu = gb_hseparator_create_popup_menu;
 */
  gbwidget.gb_widget_write_source = gb_hseparator_write_source;

  return &gbwidget;
}
