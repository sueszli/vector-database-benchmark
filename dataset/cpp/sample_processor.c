// SPDX-FileCopyrightText: © 2021-2022 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense

#include "zrythm-test-config.h"

#include "dsp/track.h"
#include "project.h"
#include "utils/flags.h"
#include "zrythm.h"

#include <glib.h>

#include "tests/helpers/plugin_manager.h"
#include "tests/helpers/project.h"

#include <locale.h>

static void
test_queue_file (void)
{
  test_helper_zrythm_init ();

  char *          filepath = g_build_filename (TESTS_SRCDIR, "test.wav", NULL);
  SupportedFile * file = supported_file_new_from_path (filepath);
  g_free (filepath);
  for (int i = 0; i < 5; i++)
    {
      sample_processor_queue_file (SAMPLE_PROCESSOR, file);
    }

  supported_file_free (file);

  test_helper_zrythm_cleanup ();
}

static void
test_queue_midi_and_roll_transport (void)
{
#ifdef HAVE_HELM
  test_helper_zrythm_init ();

  /* stop dummy audio engine processing so we can
   * process manually */
  test_project_stop_dummy_engine ();

  SAMPLE_PROCESSOR->instrument_setting =
    test_plugin_manager_get_plugin_setting (HELM_BUNDLE, HELM_URI, false);

  char * filepath =
    g_build_filename (TESTS_SRCDIR, "1_track_with_data.mid", NULL);
  SupportedFile * file = supported_file_new_from_path (filepath);
  g_free (filepath);

  transport_request_roll (TRANSPORT, true);

  g_message ("=============== queueing file =============");

  sample_processor_queue_file (SAMPLE_PROCESSOR, file);
  g_assert_cmpint (SAMPLE_PROCESSOR->tracklist->num_tracks, ==, 3);

  g_message ("============= starting process ===========");

  /* process a few times */
  engine_process (AUDIO_ENGINE, AUDIO_ENGINE->block_length);
  engine_process (AUDIO_ENGINE, AUDIO_ENGINE->block_length);
  engine_process (AUDIO_ENGINE, AUDIO_ENGINE->block_length);

  g_message ("============= done process ===========");

  supported_file_free (file);

  test_helper_zrythm_cleanup ();
#endif
}

int
main (int argc, char * argv[])
{
  g_test_init (&argc, &argv, NULL);

#define TEST_PREFIX "/audio/sample_processor/"

  g_test_add_func (
    TEST_PREFIX "test queue midi and roll transport",
    (GTestFunc) test_queue_midi_and_roll_transport);
  g_test_add_func (TEST_PREFIX "test queue file", (GTestFunc) test_queue_file);

  return g_test_run ();
}
