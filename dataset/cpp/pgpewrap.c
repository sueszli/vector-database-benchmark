/**
 * @file
 * Standalone tool to manipulate a program's command line
 *
 * C version by Wessel Dankers <wsl@fruit.eu.org>
 *
 * This code is in the public domain.
 */

#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void print_usage(const char *progname)
{
  fprintf(stderr, "Command line usage: %s [flags] -- prefix [recipients]\n", progname);
  exit(1);
}

int main(int argc, char **argv)
{
  char **opts = NULL, **opt = NULL;
  char *pfx = NULL;

  if (argc < 2)
  {
    print_usage((argc != 0) ? argv[0] : "pgpewrap");
  }

  opts = malloc((2 * argc + 1) * sizeof(*opts));
  if (!opts)
  {
    perror(argv[0]);
    exit(2);
  }

  opt = opts;
  *opt++ = argv[1];
  pfx = NULL;

  for (int i = 2; i < argc;)
  {
    if (strcmp(argv[i], "--") == 0)
    {
      i += 2;
      if (i > argc)
      {
        free(opts);
        print_usage(argv[0]);
      }
      pfx = argv[i - 1];
    }
    if (pfx)
      *opt++ = pfx;
    *opt++ = argv[i++];
  }
  *opt = NULL;

  execvp(opts[0], opts);
  perror(argv[0]);
  free(opts);
  return 2;
}
