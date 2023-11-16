// SPDX-FileCopyrightText: © 2022-2023 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense

#include "gui/widgets/zoom_buttons.h"
#include "utils/gtk.h"
#include "utils/resources.h"

#include <glib/gi18n.h>

G_DEFINE_TYPE (ZoomButtonsWidget, zoom_buttons_widget, GTK_TYPE_BOX)

/**
 * @param orientation Orientation the zoom buttons will zoom
 *   in.
 */
void
zoom_buttons_widget_setup (
  ZoomButtonsWidget * self,
  bool                timeline,
  GtkOrientation      orientation)
{
  const char * type = timeline ? "timeline" : "editor";

  char detailed_action[700];

#define SET_ACTION(name, widget) \
  sprintf ( \
    detailed_action, "app." name "::%s%s", type, \
    orientation == GTK_ORIENTATION_HORIZONTAL ? "" : "v"); \
  gtk_actionable_set_detailed_action_name ( \
    GTK_ACTIONABLE (self->widget), detailed_action)

  SET_ACTION ("zoom-in", zoom_in);
  SET_ACTION ("zoom-out", zoom_out);
  SET_ACTION ("original-size", original_size);
  SET_ACTION ("best-fit", best_fit);

#undef SET_ACTION

  gtk_orientable_set_orientation (GTK_ORIENTABLE (self), orientation);
}

static void
zoom_buttons_widget_init (ZoomButtonsWidget * self)
{
  gtk_widget_init_template (GTK_WIDGET (self));

#define SET_TOOLTIP(x, action, tooltip) \
  z_gtk_widget_set_tooltip_for_action ( \
    GTK_WIDGET (self->x), "app." action "::global", tooltip)

  SET_TOOLTIP (zoom_in, "zoom-in", _ ("Zoom In"));
  SET_TOOLTIP (zoom_out, "zoom-out", _ ("Zoom Out"));
  SET_TOOLTIP (original_size, "original-size", _ ("Original Size"));
  SET_TOOLTIP (best_fit, "best-fit", _ ("Best Fit"));

#undef SET_TOOLTIP
}

static void
zoom_buttons_widget_class_init (ZoomButtonsWidgetClass * _klass)
{
  GtkWidgetClass * klass = GTK_WIDGET_CLASS (_klass);
  resources_set_class_template (klass, "zoom_buttons.ui");

  gtk_widget_class_set_css_name (klass, "zoom-buttons");

#define BIND_CHILD(x) \
  gtk_widget_class_bind_template_child (klass, ZoomButtonsWidget, x)

  BIND_CHILD (zoom_in);
  BIND_CHILD (zoom_out);
  BIND_CHILD (original_size);
  BIND_CHILD (best_fit);

#undef BIND_CHILD
}
