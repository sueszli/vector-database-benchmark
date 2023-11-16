// SPDX-FileCopyrightText: © 2019-2020 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense

#include "dsp/control_port.h"
#include "dsp/track.h"
#include "gui/widgets/knob.h"
#include "gui/widgets/knob_with_name.h"
#include "gui/widgets/live_waveform.h"
#include "gui/widgets/modulator.h"
#include "gui/widgets/modulator_inner.h"
#include "gui/widgets/popovers/port_connections_popover.h"
#include "plugins/plugin.h"
#include "utils/arrays.h"
#include "utils/flags.h"
#include "utils/gtk.h"
#include "utils/string.h"

#include <glib/gi18n.h>
#include <gtk/gtk.h>

G_DEFINE_TYPE (ModulatorWidget, modulator_widget, TWO_COL_EXPANDER_BOX_WIDGET_TYPE)

void
modulator_widget_refresh (ModulatorWidget * self)
{
  modulator_inner_widget_refresh (self->inner);
}

ModulatorWidget *
modulator_widget_new (Plugin * modulator)
{
  g_return_val_if_fail (IS_PLUGIN (modulator), NULL);

  ModulatorWidget * self = g_object_new (MODULATOR_WIDGET_TYPE, NULL);

  self->modulator = modulator;

  self->inner = modulator_inner_widget_new (self);

  expander_box_widget_set_label (
    Z_EXPANDER_BOX_WIDGET (self), modulator->setting->descr->name);
  expander_box_widget_set_icon_name (Z_EXPANDER_BOX_WIDGET (self), "modulator");

  two_col_expander_box_widget_add_single (
    Z_TWO_COL_EXPANDER_BOX_WIDGET (self), GTK_WIDGET (self->inner));
  two_col_expander_box_widget_set_scroll_policy (
    Z_TWO_COL_EXPANDER_BOX_WIDGET (self), GTK_POLICY_NEVER, GTK_POLICY_NEVER);

  /* TODO */
#if 0
  GtkBox * box =
    two_col_expander_box_widget_get_content_box (
      Z_TWO_COL_EXPANDER_BOX_WIDGET (self));
  gtk_box_set_child_packing (
    box, GTK_WIDGET (self->inner),
    F_NO_EXPAND, F_FILL, 0, GTK_PACK_START);
#endif

  g_object_ref (self);

  return self;
}

static void
finalize (ModulatorWidget * self)
{
  G_OBJECT_CLASS (modulator_widget_parent_class)->finalize (G_OBJECT (self));
}

static void
modulator_widget_class_init (ModulatorWidgetClass * _klass)
{
  GObjectClass * klass = G_OBJECT_CLASS (_klass);

  klass->finalize = (GObjectFinalizeFunc) finalize;
}

static void
modulator_widget_init (ModulatorWidget * self)
{
  expander_box_widget_set_orientation (
    Z_EXPANDER_BOX_WIDGET (self), GTK_ORIENTATION_HORIZONTAL);
}
