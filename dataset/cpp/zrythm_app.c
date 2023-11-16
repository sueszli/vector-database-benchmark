// SPDX-FileCopyrightText: © 2021 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense

#include "zrythm-test-config.h"

#include "dsp/fader.h"
#include "dsp/midi_event.h"
#include "dsp/router.h"
#include "utils/math.h"

#include <glib.h>

#include "tests/helpers/plugin_manager.h"
#include "tests/helpers/zrythm.h"

#include "ext/whereami/whereami.h"
#include <lilv/lilv.h>

static void
on_finished_conversion_from_zpj_to_yaml (void)
{
  char * exe_path = NULL;
  int    dirname_length, length;
  length = wai_getExecutablePath (NULL, 0, &dirname_length);
  if (length > 0)
    {
      exe_path = (char *) malloc ((size_t) length + 1);
      wai_getExecutablePath (exe_path, length, &dirname_length);
      exe_path[length] = '\0';
    }
  g_assert_nonnull (exe_path);

  char project_zpj[600];
  sprintf (project_zpj, "%s/project.zpj", PROJECT->dir);
  char arg1[900];
  sprintf (arg1, "--zpj-to-yaml=%s", project_zpj);

  char project_yaml[600];
  sprintf (project_yaml, "%s/project.yaml", PROJECT->dir);
  char arg2[900];
  sprintf (arg2, "--output=%s", project_yaml);

  /* --- convert back and check --- */

  sprintf (arg1, "--yaml-to-zpj=%s", project_yaml);
  sprintf (arg2, "--output=%s", project_zpj);

  int    argc = 3;
  char * argv_after[] = { exe_path, arg1, arg2 };

  ZrythmApp * app = zrythm_app_new (argc, (const char **) argv_after);
  int         ret = g_application_run (G_APPLICATION (app), argc, argv_after);
  g_assert_cmpint (ret, ==, 0);
  g_object_unref (app);

  test_helper_zrythm_cleanup ();
}

static void
test_project_conversion (void)
{
  test_helper_zrythm_init ();

  atexit (on_finished_conversion_from_zpj_to_yaml);

  char * exe_path = NULL;
  int    dirname_length, length;
  length = wai_getExecutablePath (NULL, 0, &dirname_length);
  if (length > 0)
    {
      exe_path = (char *) malloc ((size_t) length + 1);
      wai_getExecutablePath (exe_path, length, &dirname_length);
      exe_path[length] = '\0';
    }
  g_assert_nonnull (exe_path);

  char project_zpj[600];
  sprintf (project_zpj, "%s/project.zpj", PROJECT->dir);
  char arg1[900];
  sprintf (arg1, "--zpj-to-yaml=%s", project_zpj);

  char project_yaml[600];
  sprintf (project_yaml, "%s/project.yaml", PROJECT->dir);
  char arg2[900];
  sprintf (arg2, "--output=%s", project_yaml);

  int    argc = 3;
  char * argv[] = { exe_path, arg1, arg2 };

  ZrythmApp * app = zrythm_app_new (argc, (const char **) argv);
  int         ret = g_application_run (G_APPLICATION (app), argc, argv);
  g_assert_cmpint (ret, ==, 0);
  g_object_unref (app);

  /* --- convert back and check --- */

  sprintf (arg1, "--yaml-to-zpj=%s", project_yaml);
  sprintf (arg2, "--output=%s", project_zpj);

  argc = 3;
  char * argv_after[] = { exe_path, arg1, arg2 };

  app = zrythm_app_new (argc, (const char **) argv_after);
  ret = g_application_run (G_APPLICATION (app), argc, argv_after);
  g_assert_cmpint (ret, ==, 0);
  g_object_unref (app);

  test_helper_zrythm_cleanup ();
}

int
main (int argc, char * argv[])
{
  g_test_init (&argc, &argv, NULL);

#define TEST_PREFIX "/zrythm_app/"

  g_test_add_func (
    TEST_PREFIX "test project conversion", (GTestFunc) test_project_conversion);

  return g_test_run ();
}
