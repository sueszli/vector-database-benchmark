// SPDX-FileCopyrightText: © 2022-2023 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense

#include "gui/backend/event.h"
#include "gui/backend/event_manager.h"
#include "project.h"
#include "settings/chord_preset_pack_manager.h"
#include "utils/error.h"
#include "utils/io.h"
#include "utils/objects.h"
#include "zrythm_app.h"

#include <glib/gi18n.h>

#define USER_PACKS_DIR_NAME "chord-preset-packs"
#define USER_PACK_FILENAME "chord-presets.yaml"

static char *
get_user_packs_path (void)
{
  char * zrythm_dir = zrythm_get_dir (ZRYTHM_DIR_USER_TOP);
  g_return_val_if_fail (zrythm_dir, NULL);

  return g_build_filename (zrythm_dir, USER_PACKS_DIR_NAME, NULL);
}

static bool
is_yaml_our_version (const char * yaml)
{
  bool same_version = false;
  char version_str[120];
  sprintf (
    version_str, "schema_version: %d\n", CHORD_PRESET_PACK_SCHEMA_VERSION);
  same_version = g_str_has_prefix (yaml, version_str);
  if (!same_version)
    {
      sprintf (
        version_str, "---\nschema_version: %d\n",
        CHORD_PRESET_PACK_SCHEMA_VERSION);
      same_version = g_str_has_prefix (yaml, version_str);
    }

  return same_version;
}

static void
add_standard_packs (ChordPresetPackManager * self)
{
#define ADD_SIMPLE_CHORD(i, root, chord_type) \
  pset->descr[i] = \
    chord_descriptor_new (root, false, root, chord_type, CHORD_ACC_NONE, 0);

#define ADD_SIMPLE_CHORDS( \
  n0, t0, n1, t1, n2, t2, n3, t3, n4, t4, n5, t5, n6, t6, n7, t7, n8, t8, n9, \
  t9, n10, t10, n11, t11) \
  ADD_SIMPLE_CHORD (0, n0, t0); \
  ADD_SIMPLE_CHORD (1, n1, t1); \
  ADD_SIMPLE_CHORD (2, n2, t2); \
  ADD_SIMPLE_CHORD (3, n3, t3); \
  ADD_SIMPLE_CHORD (4, n4, t4); \
  ADD_SIMPLE_CHORD (5, n5, t5); \
  ADD_SIMPLE_CHORD (6, n6, t6); \
  ADD_SIMPLE_CHORD (7, n7, t7); \
  ADD_SIMPLE_CHORD (8, n8, t8); \
  ADD_SIMPLE_CHORD (9, n9, t9); \
  ADD_SIMPLE_CHORD (10, n10, t10); \
  ADD_SIMPLE_CHORD (11, n11, t11)

#define ADD_SIMPLE_4CHORDS(n0, t0, n1, t1, n2, t2, n3, t3) \
  ADD_SIMPLE_CHORDS ( \
    n0, t0, n1, t1, n2, t2, n3, t3, NOTE_C, CHORD_TYPE_NONE, NOTE_C, \
    CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, \
    CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, \
    CHORD_TYPE_NONE)

  ChordPresetPack * pack;
  ChordPreset *     pset;

  /* --- euro pop pack --- */
  pack = chord_preset_pack_new (_ ("Euro Pop"), true);

  pset = chord_preset_new (_ ("4 Chord Song"));
  ADD_SIMPLE_CHORDS (
    NOTE_A, CHORD_TYPE_MIN, NOTE_C, CHORD_TYPE_MAJ, NOTE_F, CHORD_TYPE_MAJ,
    NOTE_G, CHORD_TYPE_MAJ, NOTE_G, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  pset->descr[4]->accent = CHORD_ACC_7;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* Johann Pachelbel, My Chemical Romance */
  pset = chord_preset_new (_ ("Canon in D"));
  ADD_SIMPLE_CHORDS (
    NOTE_D, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MAJ, NOTE_B, CHORD_TYPE_MIN,
    NOTE_FS, CHORD_TYPE_MIN, NOTE_G, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MAJ,
    NOTE_G, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  pset = chord_preset_new (_ ("Love Progression"));
  ADD_SIMPLE_4CHORDS (
    NOTE_C, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MIN, NOTE_F, CHORD_TYPE_MAJ,
    NOTE_G, CHORD_TYPE_MAJ);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  pset = chord_preset_new (_ ("Pop Chords 1"));
  ADD_SIMPLE_CHORDS (
    NOTE_C, CHORD_TYPE_MAJ, NOTE_G, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MIN,
    NOTE_F, CHORD_TYPE_MAJ, NOTE_E, CHORD_TYPE_MAJ, NOTE_B, CHORD_TYPE_MAJ,
    NOTE_CS, CHORD_TYPE_MIN, NOTE_A, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  pset = chord_preset_new (_ ("Most Often Used Chords"));
  ADD_SIMPLE_CHORDS (
    NOTE_G, CHORD_TYPE_MAJ, NOTE_F, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_MAJ,
    NOTE_A, CHORD_TYPE_MIN, NOTE_D, CHORD_TYPE_MIN, NOTE_E, CHORD_TYPE_MIN,
    NOTE_E, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MAJ, NOTE_AS, CHORD_TYPE_MAJ,
    NOTE_A, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  g_ptr_array_add (self->pset_packs, pack);

  /* --- j/k pop --- */

  pack = chord_preset_pack_new (_ ("Eastern Pop"), true);

  /* fight together */
  pset = chord_preset_new ("Fight Together");
  ADD_SIMPLE_CHORDS (
    NOTE_G, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MAJ,
    NOTE_G, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MAJ, NOTE_B, CHORD_TYPE_MIN,
    NOTE_D, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* gee */
  pset = chord_preset_new ("Gee");
  ADD_SIMPLE_CHORDS (
    NOTE_A, CHORD_TYPE_MAJ, NOTE_FS, CHORD_TYPE_MIN, NOTE_GS, CHORD_TYPE_MIN,
    NOTE_GS, CHORD_TYPE_MIN, NOTE_CS, CHORD_TYPE_MIN, NOTE_CS, CHORD_TYPE_MIN,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  pset->descr[0]->accent = CHORD_ACC_7;
  pset->descr[3]->accent = CHORD_ACC_7;
  pset->descr[5]->accent = CHORD_ACC_7;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* yuriyurarararayuruyuri */
  pset = chord_preset_new ("Daijiken");
  ADD_SIMPLE_CHORDS (
    NOTE_DS, CHORD_TYPE_MAJ, NOTE_AS, CHORD_TYPE_MIN, NOTE_F, CHORD_TYPE_MAJ,
    NOTE_AS, CHORD_TYPE_MIN, NOTE_GS, CHORD_TYPE_MAJ, NOTE_DS, CHORD_TYPE_MAJ,
    NOTE_G, CHORD_TYPE_MIN, NOTE_F, CHORD_TYPE_MIN, NOTE_AS, CHORD_TYPE_MAJ,
    NOTE_GS, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_MIN, NOTE_C, CHORD_TYPE_MAJ);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  g_ptr_array_add (self->pset_packs, pack);

  /* --- dance --- */

  pack = chord_preset_pack_new (_ ("Dance"), true);

  /* the idolm@ster 2 */
  pset = chord_preset_new ("Idol 2");
  ADD_SIMPLE_4CHORDS (
    NOTE_C, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MAJ, NOTE_B, CHORD_TYPE_MIN,
    NOTE_E, CHORD_TYPE_MIN);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  g_ptr_array_add (self->pset_packs, pack);

  /* --- ballad --- */

  pack = chord_preset_pack_new (_ ("Ballad"), true);

  /* snow halation */
  pset = chord_preset_new ("Snow Halation");
  ADD_SIMPLE_CHORDS (
    NOTE_D, CHORD_TYPE_MAJ, NOTE_E, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MIN,
    NOTE_FS, CHORD_TYPE_MIN, NOTE_B, CHORD_TYPE_MIN, NOTE_CS, CHORD_TYPE_MAJ,
    NOTE_E, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MIN, NOTE_CS, CHORD_TYPE_MAJ,
    NOTE_B, CHORD_TYPE_MAJ, NOTE_E, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MAJ);
  pset->descr[4]->accent = CHORD_ACC_7;
  pset->descr[5]->accent = CHORD_ACC_7;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* connect */
  pset = chord_preset_new ("Connect");
  ADD_SIMPLE_CHORDS (
    NOTE_B, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MAJ, NOTE_AS, CHORD_TYPE_MIN,
    NOTE_DS, CHORD_TYPE_MIN, NOTE_GS, CHORD_TYPE_MIN, NOTE_B, CHORD_TYPE_MAJ,
    NOTE_CS, CHORD_TYPE_MAJ, NOTE_DS, CHORD_TYPE_MAJ, NOTE_GS, CHORD_TYPE_MAJ,
    NOTE_AS, CHORD_TYPE_MAJ, NOTE_G, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_MIN);
  pset->descr[8]->accent = CHORD_ACC_7;
  pset->descr[10]->accent = CHORD_ACC_7;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* secret base */
  pset = chord_preset_new ("Secret Base");
  ADD_SIMPLE_CHORDS (
    NOTE_B, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MAJ, NOTE_DS, CHORD_TYPE_MIN,
    NOTE_CS, CHORD_TYPE_MAJ, NOTE_B, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MAJ,
    NOTE_FS, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  pset->descr[2]->accent = CHORD_ACC_7;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  g_ptr_array_add (self->pset_packs, pack);

  /* --- eurodance --- */

  pack = chord_preset_pack_new (_ ("Eurodance"), true);

  /* what is love */
  pset = chord_preset_new ("What is Love");
  ADD_SIMPLE_4CHORDS (
    NOTE_G, CHORD_TYPE_MIN, NOTE_AS, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MIN,
    NOTE_F, CHORD_TYPE_MAJ);
  pset->descr[2]->accent = CHORD_ACC_7;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* blue */
  pset = chord_preset_new ("Blue");
  ADD_SIMPLE_CHORDS (
    NOTE_G, CHORD_TYPE_MIN, NOTE_F, CHORD_TYPE_MAJ, NOTE_DS, CHORD_TYPE_MAJ,
    NOTE_C, CHORD_TYPE_MIN, NOTE_C, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MIN,
    NOTE_GS, CHORD_TYPE_MAJ, NOTE_AS, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  g_ptr_array_add (self->pset_packs, pack);

  /* --- eurobeat --- */

  pack = chord_preset_pack_new (_ ("Eurobeat"), true);

  pset = chord_preset_new ("Burning Night");
  ADD_SIMPLE_CHORDS (
    NOTE_CS, CHORD_TYPE_MAJ, NOTE_DS, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_MIN,
    NOTE_F, CHORD_TYPE_MIN, NOTE_AS, CHORD_TYPE_MIN, NOTE_B, CHORD_TYPE_DIM,
    NOTE_C, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* believe / dreamin' of you */
  pset = chord_preset_new ("Dreamin' Of You");
  ADD_SIMPLE_CHORDS (
    NOTE_F, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MIN,
    NOTE_AS, CHORD_TYPE_MAJ, NOTE_G, CHORD_TYPE_MIN, NOTE_A, CHORD_TYPE_MAJ,
    NOTE_G, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MAJ, NOTE_FS, CHORD_TYPE_MIN,
    NOTE_B, CHORD_TYPE_MIN, NOTE_E, CHORD_TYPE_MIN, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* get me power */
  pset = chord_preset_new ("Get Me Power");
  ADD_SIMPLE_CHORDS (
    NOTE_B, CHORD_TYPE_MIN, NOTE_E, CHORD_TYPE_MIN, NOTE_D, CHORD_TYPE_MAJ,
    NOTE_A, CHORD_TYPE_MAJ, NOTE_G, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_MAJ,
    NOTE_FS, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MIN, NOTE_D, CHORD_TYPE_MIN,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* night of fire */
  pset = chord_preset_new ("Night of Fire");
  ADD_SIMPLE_CHORDS (
    NOTE_DS, CHORD_TYPE_MIN, NOTE_B, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MAJ,
    NOTE_FS, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MAJ, NOTE_DS, CHORD_TYPE_MIN,
    NOTE_FS, CHORD_TYPE_MAJ, NOTE_GS, CHORD_TYPE_MAJ, NOTE_B, CHORD_TYPE_MAJ,
    NOTE_GS, CHORD_TYPE_MIN, NOTE_AS, CHORD_TYPE_MIN, NOTE_CS, CHORD_TYPE_MAJ);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* super fever night */
  pset = chord_preset_new ("Super Fever Night");
  ADD_SIMPLE_CHORDS (
    NOTE_B, CHORD_TYPE_MIN, NOTE_G, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MAJ,
    NOTE_B, CHORD_TYPE_MIN, NOTE_G, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MAJ,
    NOTE_D, CHORD_TYPE_MAJ, NOTE_E, CHORD_TYPE_MAJ, NOTE_FS, CHORD_TYPE_MIN,
    NOTE_FS, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  pset->descr[4]->accent = CHORD_ACC_7;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* break in2 the nite */
  pset = chord_preset_new ("Step in2 the Nite");
  ADD_SIMPLE_CHORDS (
    NOTE_D, CHORD_TYPE_MIN, NOTE_F, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_MAJ,
    NOTE_D, CHORD_TYPE_MIN, NOTE_AS, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_MAJ,
    NOTE_D, CHORD_TYPE_MIN, NOTE_G, CHORD_TYPE_MIN, NOTE_F, CHORD_TYPE_MAJ,
    NOTE_C, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MIN, NOTE_C, CHORD_TYPE_NONE);
  pset->descr[4]->inversion = -2;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  g_ptr_array_add (self->pset_packs, pack);

  /* --- progressive trance --- */

  pack = chord_preset_pack_new (_ ("Progressive Trance"), true);

  pset = chord_preset_new ("Sajek Valley");
  ADD_SIMPLE_CHORDS (
    NOTE_A, CHORD_TYPE_MIN, NOTE_D, CHORD_TYPE_MIN, NOTE_F, CHORD_TYPE_MAJ,
    NOTE_C, CHORD_TYPE_MAJ, NOTE_G, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MIN,
    NOTE_E, CHORD_TYPE_MIN, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  pset->descr[1]->accent = CHORD_ACC_7;
  pset->descr[3]->inversion = 1;
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  g_ptr_array_add (self->pset_packs, pack);

  /* --- rock --- */

  pack = chord_preset_pack_new (_ ("Rock"), true);

  pset = chord_preset_new ("Overdrive");
  ADD_SIMPLE_CHORDS (
    NOTE_FS, CHORD_TYPE_MAJ, NOTE_GS, CHORD_TYPE_MAJ, NOTE_AS, CHORD_TYPE_MIN,
    NOTE_AS, CHORD_TYPE_MAJ, NOTE_FS, CHORD_TYPE_MAJ, NOTE_GS, CHORD_TYPE_MAJ,
    NOTE_AS, CHORD_TYPE_MIN, NOTE_CS, CHORD_TYPE_MAJ, NOTE_GS, CHORD_TYPE_MAJ,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* kokoro */
  pset = chord_preset_new ("Kokoro");
  ADD_SIMPLE_CHORDS (
    NOTE_FS, CHORD_TYPE_MAJ, NOTE_F, CHORD_TYPE_MIN, NOTE_AS, CHORD_TYPE_MIN,
    NOTE_DS, CHORD_TYPE_MIN, NOTE_GS, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MAJ,
    NOTE_FS, CHORD_TYPE_MAJ, NOTE_FS, CHORD_TYPE_MAJ, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  pset = chord_preset_new ("Pray");
  ADD_SIMPLE_CHORDS (
    NOTE_B, CHORD_TYPE_MIN, NOTE_G, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MAJ,
    NOTE_A, CHORD_TYPE_MAJ, NOTE_E, CHORD_TYPE_MAJ, NOTE_B, CHORD_TYPE_MAJ,
    NOTE_CS, CHORD_TYPE_MIN, NOTE_CS, CHORD_TYPE_MAJ, NOTE_DS, CHORD_TYPE_MAJ,
    NOTE_GS, CHORD_TYPE_MIN, NOTE_FS, CHORD_TYPE_MAJ, NOTE_E, CHORD_TYPE_MAJ);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* no thank you */
  pset = chord_preset_new ("No Thank You");
  ADD_SIMPLE_CHORDS (
    NOTE_E, CHORD_TYPE_MIN, NOTE_D, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MAJ,
    NOTE_C, CHORD_TYPE_MAJ, NOTE_G, CHORD_TYPE_MAJ, NOTE_A, CHORD_TYPE_MIN,
    NOTE_B, CHORD_TYPE_MIN, NOTE_C, CHORD_TYPE_MAJ, NOTE_D, CHORD_TYPE_MAJ,
    NOTE_G, CHORD_TYPE_MAJ, NOTE_B, CHORD_TYPE_MIN, NOTE_E, CHORD_TYPE_MIN);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  /* boulevard of broken dreams */
  pset = chord_preset_new ("Broken Dreams");
  ADD_SIMPLE_CHORDS (
    NOTE_F, CHORD_TYPE_MIN, NOTE_GS, CHORD_TYPE_MAJ, NOTE_DS, CHORD_TYPE_MAJ,
    NOTE_AS, CHORD_TYPE_MAJ, NOTE_CS, CHORD_TYPE_MAJ, NOTE_GS, CHORD_TYPE_MAJ,
    NOTE_DS, CHORD_TYPE_MAJ, NOTE_F, CHORD_TYPE_MIN, NOTE_C, CHORD_TYPE_NONE,
    NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE, NOTE_C, CHORD_TYPE_NONE);
  chord_preset_pack_add_preset (pack, pset);
  chord_preset_free (pset);

  g_ptr_array_add (self->pset_packs, pack);

#undef ADD_SIMPLE_CHORD
}

/**
 * Creates a new chord preset pack manager.
 *
 * @param scan_for_packs Whether to scan for preset
 *   packs.
 */
ChordPresetPackManager *
chord_preset_pack_manager_new (bool scan_for_packs)
{
  ChordPresetPackManager * self = object_new (ChordPresetPackManager);

  self->pset_packs =
    g_ptr_array_new_with_free_func (chord_preset_pack_destroy_cb);

  /* add standard preset packs */
  add_standard_packs (self);

  if (!ZRYTHM_TESTING)
    {
      /* add user preset packs */
      char * main_path = get_user_packs_path ();
      g_debug ("Reading user chord packs from %s...", main_path);

      char ** pack_paths =
        io_get_files_in_dir_ending_in (main_path, true, ".yaml", false);
      if (pack_paths)
        {
          char * pack_path = NULL;
          int    i = 0;
          while ((pack_path = pack_paths[i++]))
            {
              if (
                !g_file_test (pack_path, G_FILE_TEST_EXISTS)
                || g_file_test (pack_path, G_FILE_TEST_IS_DIR))
                {
                  continue;
                }

              g_debug ("checking file %s", pack_path);

              char *   yaml = NULL;
              GError * err = NULL;
              g_file_get_contents (pack_path, &yaml, NULL, &err);
              if (err != NULL)
                {
                  g_warning (
                    "Failed to read yaml "
                    "from %s",
                    pack_path);
                  continue;
                }

              /* if not same version ignore */
              if (!is_yaml_our_version (yaml))
                {
                  g_warning (
                    "old chord preset version for %s, "
                    "skipping",
                    pack_path);
                  continue;
                }

              ChordPresetPack * pack = (ChordPresetPack *) yaml_deserialize (
                yaml, &chord_preset_pack_schema, &err);
              if (!pack)
                {
                  g_critical (
                    "failed to load chord preset pack: %s", err->message);
                  return NULL;
                }
              if (!pack->presets)
                {
                  pack->presets_size = 12;
                  pack->presets =
                    object_new_n (pack->presets_size, ChordPreset *);
                }
              else
                {
                  pack->presets_size = (size_t) pack->num_presets;
                }
              g_ptr_array_add (self->pset_packs, pack);

              g_free (yaml);
            }

          g_strfreev (pack_paths);
        }
      else
        {
          g_message ("no user chord presets found");
        }
    }

  return self;
}

int
chord_preset_pack_manager_get_num_packs (const ChordPresetPackManager * self)
{
  return (int) self->pset_packs->len;
}

ChordPresetPack *
chord_preset_pack_manager_get_pack_at (
  const ChordPresetPackManager * self,
  int                            idx)
{
  return (ChordPresetPack *) g_ptr_array_index (self->pset_packs, idx);
}

/**
 * Add a copy of the given pack.
 */
void
chord_preset_pack_manager_add_pack (
  ChordPresetPackManager * self,
  const ChordPresetPack *  pack,
  bool                     serialize)
{
  ChordPresetPack * new_pack = chord_preset_pack_clone (pack);
  g_ptr_array_add (self->pset_packs, new_pack);

  if (serialize)
    {
      GError * err = NULL;
      bool     success = chord_preset_pack_manager_serialize (self, &err);
      if (!success)
        {
          HANDLE_ERROR_LITERAL (err, "Failed to serialize chord preset packs");
        }
    }

  EVENTS_PUSH (ET_CHORD_PRESET_PACK_ADDED, NULL);
}

void
chord_preset_pack_manager_delete_pack (
  ChordPresetPackManager * self,
  ChordPresetPack *        pack,
  bool                     serialize)
{
  g_ptr_array_remove (self->pset_packs, pack);

  if (serialize)
    {
      GError * err = NULL;
      bool     success = chord_preset_pack_manager_serialize (self, &err);
      if (!success)
        {
          HANDLE_ERROR_LITERAL (err, "Failed to serialize chord preset packs");
        }
    }

  EVENTS_PUSH (ET_CHORD_PRESET_PACK_REMOVED, NULL);
}

ChordPresetPack *
chord_preset_pack_manager_get_pack_for_preset (
  ChordPresetPackManager * self,
  const ChordPreset *      pset)
{
  for (size_t i = 0; i < self->pset_packs->len; i++)
    {
      ChordPresetPack * pack =
        (ChordPresetPack *) g_ptr_array_index (self->pset_packs, i);

      if (chord_preset_pack_contains_preset (pack, pset))
        {
          return pack;
        }
    }

  g_return_val_if_reached (NULL);
}

int
chord_preset_pack_manager_get_pack_index (
  ChordPresetPackManager * self,
  ChordPresetPack *        pack)
{
  for (size_t i = 0; i < self->pset_packs->len; i++)
    {
      ChordPresetPack * cur_pack =
        (ChordPresetPack *) g_ptr_array_index (self->pset_packs, i);
      if (cur_pack == pack)
        return i;
    }
  g_return_val_if_reached (-1);
}

/**
 * Returns the preset index in its pack.
 */
int
chord_preset_pack_manager_get_pset_index (
  ChordPresetPackManager * self,
  ChordPreset *            pset)
{
  ChordPresetPack * pack =
    chord_preset_pack_manager_get_pack_for_preset (self, pset);
  g_return_val_if_fail (pack, -1);

  for (int i = 0; i < pack->num_presets; i++)
    {
      ChordPreset * cur_pset = pack->presets[i];
      if (cur_pset == pset)
        return i;
    }
  g_return_val_if_reached (-1);
}

/**
 * Add a copy of the given preset.
 */
void
chord_preset_pack_manager_add_preset (
  ChordPresetPackManager * self,
  ChordPresetPack *        pack,
  const ChordPreset *      pset,
  bool                     serialize)
{
  chord_preset_pack_add_preset (pack, pset);

  if (serialize)
    {
      GError * err = NULL;
      bool     success = chord_preset_pack_manager_serialize (self, &err);
      if (!success)
        {
          HANDLE_ERROR_LITERAL (err, "Failed to serialize chord preset packs");
        }
    }
}

void
chord_preset_pack_manager_delete_preset (
  ChordPresetPackManager * self,
  ChordPreset *            pset,
  bool                     serialize)
{
  ChordPresetPack * pack =
    chord_preset_pack_manager_get_pack_for_preset (self, pset);
  if (!pack)
    return;

  chord_preset_pack_delete_preset (pack, pset);

  if (serialize)
    {
      GError * err = NULL;
      bool     success = chord_preset_pack_manager_serialize (self, &err);
      if (!success)
        {
          HANDLE_ERROR_LITERAL (err, "Failed to serialize chord preset packs");
        }
    }
}

/**
 * Serializes the chord presets.
 *
 * @return Whether successful.
 */
bool
chord_preset_pack_manager_serialize (
  ChordPresetPackManager * self,
  GError **                error)
{
  /* TODO backup existing packs first */

  g_message ("Serializing user preset packs...");
  char * main_path = get_user_packs_path ();
  g_return_val_if_fail (main_path && strlen (main_path) > 2, false);
  g_message ("Writing user chord packs to %s...", main_path);

  for (size_t i = 0; i < self->pset_packs->len; i++)
    {
      ChordPresetPack * pack =
        (ChordPresetPack *) g_ptr_array_index (self->pset_packs, i);
      if (pack->is_standard)
        continue;

      g_return_val_if_fail (pack->name && strlen (pack->name) > 0, false);

      char *   pack_dir = g_build_filename (main_path, pack->name, NULL);
      GError * err = NULL;
      bool     success = io_mkdir (pack_dir, &err);
      if (!success)
        {
          PROPAGATE_PREFIXED_ERROR (
            error, err, "Failed to create directory %s", pack_dir);
          return false;
        }
      char * pack_yaml = yaml_serialize (pack, &chord_preset_pack_schema, &err);
      if (!pack_yaml)
        {
          PROPAGATE_PREFIXED_ERROR_LITERAL (
            error, err, _ ("Failed to serialize chord preset packs"));
          return false;
        }
      char * pack_path = g_build_filename (pack_dir, USER_PACK_FILENAME, NULL);
      success = g_file_set_contents (pack_path, pack_yaml, -1, &err);
      if (!success)
        {
          PROPAGATE_PREFIXED_ERROR (
            error, err, "Unable to write chord preset pack %s", pack_path);
          return false;
        }
      g_free (pack_path);
      g_free (pack_yaml);
      g_free (pack_dir);
    }

  g_free (main_path);

  return true;
}

void
chord_preset_pack_manager_free (const ChordPresetPackManager * self)
{
  g_ptr_array_unref (self->pset_packs);

  object_zero_and_free (self);
}
