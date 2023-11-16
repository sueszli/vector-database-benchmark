// SPDX-FileCopyrightText: © 2019-2023 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense

#include <stdlib.h>

#include "dsp/automation_track.h"
#include "dsp/modulator_macro_processor.h"
#include "dsp/modulator_track.h"
#include "dsp/port.h"
#include "dsp/router.h"
#include "dsp/track.h"
#include "gui/backend/event.h"
#include "gui/backend/event_manager.h"
#include "gui/widgets/main_window.h"
#include "project.h"
#include "utils/arrays.h"
#include "utils/dialogs.h"
#include "utils/flags.h"
#include "utils/gtk.h"
#include "utils/mem.h"
#include "utils/objects.h"
#include "zrythm_app.h"

#include <glib/gi18n.h>
#include <gtk/gtk.h>

/**
 * Inits the modulator track.
 */
void
modulator_track_init (Track * self)
{
  self->type = TRACK_TYPE_MODULATOR;
  self->main_height = TRACK_DEF_HEIGHT / 2;

  gdk_rgba_parse (&self->color, "#222222");
  self->icon_name = g_strdup ("modulator");

  const int max_macros = 8;
  self->num_visible_modulator_macros = max_macros;
  for (int i = 0; i < max_macros; i++)
    {
      self->modulator_macros[i] = modulator_macro_processor_new (self, i);
    }
  self->num_modulator_macros = max_macros;

  /* set invisible */
  self->visible = false;
}

/**
 * Creates the default modulator track.
 */
Track *
modulator_track_default (int track_pos)
{
  Track * self = track_new (
    TRACK_TYPE_MODULATOR, track_pos, _ ("Modulators"), F_WITHOUT_LANE);

  return self;
}

typedef struct ModulatorImportData
{
  Track *  track;
  int      slot;
  Plugin * modulator;
  bool     replace_mode;
  bool     confirm;
  bool     gen_automatables;
  bool     recalc_graph;
  bool     pub_events;
} ModulatorImportData;

static void
modulator_import_data_free (void * _data)
{
  ModulatorImportData * self = (ModulatorImportData *) _data;
  object_zero_and_free (self);
}

static void
do_insert (ModulatorImportData * data)
{
  Track * self = data->track;
  if (data->replace_mode)
    {
      Plugin * existing_pl =
        data->slot < self->num_modulators ? self->modulators[data->slot] : NULL;
      if (existing_pl)
        {
          /* free current plugin */
          if (existing_pl)
            {
              modulator_track_remove_modulator (
                self, data->slot, F_REPLACING, F_DELETING_PLUGIN,
                F_NOT_DELETING_TRACK, F_NO_RECALC_GRAPH);
            }
        }

      g_message (
        "Inserting modulator %s at %s:%d",
        data->modulator->setting->descr->name, self->name, data->slot);
      if (data->slot == self->num_modulators)
        {
          array_double_size_if_full (
            self->modulators, self->num_modulators, self->modulators_size,
            Modulator *);
          self->num_modulators++;
        }
    }
  else
    {
      array_double_size_if_full (
        self->modulators, self->num_modulators, self->modulators_size,
        Modulator *);

      /* push other modulators forward (make
       * space for new modulator) */
      self->num_modulators++;
      for (int i = self->num_modulators - 1; i > data->slot; i--)
        {
          self->modulators[i] = self->modulators[i - 1];
          g_message (
            "setting modulator %s from slot %d "
            "to slot %d",
            self->modulators[i]->setting->descr->name, i - 1, i);
          plugin_set_track_and_slot (
            self->modulators[i], track_get_name_hash (self),
            PLUGIN_SLOT_MODULATOR, i);
        }
    }

  /* add the modulator */
  self->modulators[data->slot] = data->modulator;
  g_message (
    "setting modulator %s to slot %d", data->modulator->setting->descr->name,
    data->slot);

  plugin_set_track_and_slot (
    data->modulator, track_get_name_hash (self), PLUGIN_SLOT_MODULATOR,
    data->slot);

  if (data->gen_automatables)
    {
      plugin_generate_automation_tracks (data->modulator, self);
    }

  if (data->pub_events)
    {
      EVENTS_PUSH (ET_MODULATOR_ADDED, data->modulator);
    }

  if (data->recalc_graph)
    {
      router_recalc_graph (ROUTER, F_NOT_SOFT);
    }
}

static void
overwrite_plugin_response_cb (
  AdwMessageDialog * dialog,
  char *             response,
  gpointer           user_data)
{
  ModulatorImportData * data = (ModulatorImportData *) user_data;
  if (!string_is_equal (response, "overwrite"))
    {
      return;
    }

  do_insert (data);
}

/**
 * Inserts and connects a Modulator to the Track.
 *
 * @param replace_mode Whether to perform the
 *   operation in replace mode (replace current
 *   modulator if true, not touching other
 *   modulators, or push other modulators forward
 *   if false).
 */
void
modulator_track_insert_modulator (
  Track *  self,
  int      slot,
  Plugin * modulator,
  bool     replace_mode,
  bool     confirm,
  bool     gen_automatables,
  bool     recalc_graph,
  bool     pub_events)
{
  g_return_if_fail (
    IS_TRACK (self) && IS_PLUGIN (modulator) && slot <= self->num_modulators);

  ModulatorImportData * data = object_new (ModulatorImportData);
  data->track = self;
  data->slot = slot;
  data->modulator = modulator;
  data->replace_mode = replace_mode;
  data->confirm = confirm;
  data->gen_automatables = gen_automatables;
  data->recalc_graph = recalc_graph;
  data->pub_events = pub_events;

  if (replace_mode)
    {
      Plugin * existing_pl =
        slot < self->num_modulators ? self->modulators[slot] : NULL;
      if (existing_pl && confirm)
        {
          AdwMessageDialog * dialog =
            dialogs_get_overwrite_plugin_dialog (GTK_WINDOW (MAIN_WINDOW));
          gtk_window_present (GTK_WINDOW (dialog));
          g_signal_connect_data (
            dialog, "response", G_CALLBACK (overwrite_plugin_response_cb), data,
            (GClosureNotify) modulator_import_data_free, G_CONNECT_DEFAULT);
          return;
        }
    }

  do_insert (data);
  modulator_import_data_free (data);
}

/**
 * Removes a plugin at pos from the track.
 *
 * @param replacing Whether replacing the modulator.
 *   If this is false, modulators after this slot
 *   will be pushed back.
 * @param deleting_modulator
 * @param deleting_track If true, the automation
 *   tracks associated with the plugin are not
 *   deleted at this time.
 * @param recalc_graph Recalculate mixer graph.
 */
void
modulator_track_remove_modulator (
  Track * self,
  int     slot,
  bool    replacing,
  bool    deleting_modulator,
  bool    deleting_track,
  bool    recalc_graph)
{
  Plugin * plugin = self->modulators[slot];
  g_return_if_fail (IS_PLUGIN (plugin));
  g_return_if_fail (plugin->id.track_name_hash == track_get_name_hash (self));

  plugin_remove_ats_from_automation_tracklist (
    plugin, deleting_modulator, !deleting_track && !deleting_modulator);

  g_message (
    "Removing %s from %s:%d", plugin->setting->descr->name, self->name, slot);

  /* unexpose all JACK ports */
  plugin_expose_ports (plugin, F_NOT_EXPOSE, true, true);

  /* if deleting plugin disconnect the plugin
   * entirely */
  if (deleting_modulator)
    {
      if (plugin_is_selected (plugin))
        {
          mixer_selections_remove_slot (
            MIXER_SELECTIONS, plugin->id.slot, PLUGIN_SLOT_MODULATOR,
            F_PUBLISH_EVENTS);
        }

      plugin_disconnect (plugin);
      object_free_w_func_and_null (plugin_free, plugin);
    }

  if (!replacing)
    {
      for (int i = slot; i < self->num_modulators - 1; i++)
        {
          self->modulators[i] = self->modulators[i + 1];
          plugin_set_track_and_slot (
            self->modulators[i], track_get_name_hash (self),
            PLUGIN_SLOT_MODULATOR, i);
        }
      self->num_modulators--;
    }

  if (recalc_graph)
    {
      router_recalc_graph (ROUTER, F_NOT_SOFT);
    }
}
