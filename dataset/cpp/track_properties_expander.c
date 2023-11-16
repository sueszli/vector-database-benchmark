// SPDX-FileCopyrightText: © 2019-2023 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense

#include "dsp/engine.h"
#include "dsp/track.h"
#include "gui/widgets/channel_slot.h"
#include "gui/widgets/editable_label.h"
#include "gui/widgets/route_target_selector.h"
#include "gui/widgets/track_properties_expander.h"
#include "plugins/plugin_gtk.h"
#include "project.h"
#include "utils/gtk.h"
#include "utils/string.h"

#include <glib/gi18n.h>

G_DEFINE_TYPE (
  TrackPropertiesExpanderWidget,
  track_properties_expander_widget,
  TWO_COL_EXPANDER_BOX_WIDGET_TYPE)

/**
 * Refreshes each field.
 */
void
track_properties_expander_widget_refresh (
  TrackPropertiesExpanderWidget * self,
  Track *                         track)
{
  g_return_if_fail (self);
  self->track = track;

  if (track)
    {
      g_return_if_fail (self->direct_out);
      route_target_selector_widget_refresh (
        self->direct_out, track_type_has_channel (track->type) ? track : NULL);

      editable_label_widget_setup (
        self->name, track, (GenericStringGetter) track_get_name,
        (GenericStringSetter) track_set_name_with_action);

      bool is_instrument = track->type == TRACK_TYPE_INSTRUMENT;
      gtk_widget_set_visible (GTK_WIDGET (self->instrument_slot), is_instrument);
      gtk_widget_set_visible (
        GTK_WIDGET (self->instrument_label), is_instrument);
      if (is_instrument)
        {
          channel_slot_widget_set_instrument (self->instrument_slot, track);
        }
    }
}

/**
 * Sets up the TrackPropertiesExpanderWidget.
 */
void
track_properties_expander_widget_setup (
  TrackPropertiesExpanderWidget * self,
  Track *                         track)
{
  g_warn_if_fail (track);
  self->track = track;

  GtkWidget * lbl;

#define CREATE_LABEL(x) \
  lbl = plugin_gtk_new_label (x, true, false, 0.f, 0.5f); \
  gtk_widget_add_css_class (lbl, "inspector_label"); \
  gtk_widget_set_margin_start (lbl, 2); \
  gtk_widget_set_visible (lbl, 1)

  /* add track name */
  self->name = editable_label_widget_new (NULL, NULL, NULL, 11);
  gtk_label_set_xalign (self->name->label, 0);
  gtk_widget_set_margin_start (GTK_WIDGET (self->name->label), 4);
  CREATE_LABEL (_ ("Track Name"));
  two_col_expander_box_widget_add_single (
    Z_TWO_COL_EXPANDER_BOX_WIDGET (self), lbl);
  two_col_expander_box_widget_add_single (
    Z_TWO_COL_EXPANDER_BOX_WIDGET (self), GTK_WIDGET (self->name));
  gtk_widget_set_hexpand (GTK_WIDGET (self->name->label), true);

  /* add direct out */
  self->direct_out = g_object_new (ROUTE_TARGET_SELECTOR_WIDGET_TYPE, NULL);
  CREATE_LABEL (_ ("Direct Out"));
  two_col_expander_box_widget_add_single (
    Z_TWO_COL_EXPANDER_BOX_WIDGET (self), lbl);
  two_col_expander_box_widget_add_single (
    Z_TWO_COL_EXPANDER_BOX_WIDGET (self), GTK_WIDGET (self->direct_out));

  /* add instrument slot */
  self->instrument_slot = channel_slot_widget_new_instrument ();
  CREATE_LABEL (_ ("Instrument"));
  two_col_expander_box_widget_add_single (
    Z_TWO_COL_EXPANDER_BOX_WIDGET (self), lbl);
  self->instrument_label = GTK_LABEL (lbl);
  two_col_expander_box_widget_add_single (
    Z_TWO_COL_EXPANDER_BOX_WIDGET (self), GTK_WIDGET (self->instrument_slot));

#undef CREATE_LABEL

  /* set name and icon */
  expander_box_widget_set_label (
    Z_EXPANDER_BOX_WIDGET (self), _ ("Track Properties"));
  expander_box_widget_set_icon_name (Z_EXPANDER_BOX_WIDGET (self), "info");
  expander_box_widget_set_orientation (
    Z_EXPANDER_BOX_WIDGET (self), GTK_ORIENTATION_VERTICAL);

  track_properties_expander_widget_refresh (self, track);
}

static void
track_properties_expander_widget_class_init (
  TrackPropertiesExpanderWidgetClass * klass)
{
}

static void
track_properties_expander_widget_init (TrackPropertiesExpanderWidget * self)
{
}
