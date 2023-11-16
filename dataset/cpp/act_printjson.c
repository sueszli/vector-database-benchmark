/* Print comprehensive information to stdout in JSON format
 * This file is part of jdupes; see jdupes.c for license information */

#ifndef NO_JSON

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <libjodycode.h>
#include "likely_unlikely.h"
#include "jdupes.h"
#include "version.h"
#include "act_printjson.h"

#define IS_CONT(a)  ((a & 0xc0) == 0x80)
#define GET_CONT(a) (a & 0x3f)
#define TO_HEX(a) (char)(((a) & 0x0f) <= 0x09 ? ((a) & 0x0f) + 0x30 : ((a) & 0x0f) + 0x57)

#if defined(__GNU__) && !defined(PATH_MAX)
#define PATH_MAX 1024
#endif

/** Decodes a single UTF-8 codepoint, consuming bytes. */
static inline uint32_t decode_utf8(const char * restrict * const string) {
  uint32_t ret = 0;
  /** Eat problems up silently. */
  assert(!IS_CONT(**string));
  while (unlikely(IS_CONT(**string)))
    (*string)++;

  /** ASCII. */
  if (likely(!(**string & 0x80)))
    return (uint32_t)*(*string)++;

  /** Multibyte 2, 3, 4. */
  if ((**string & 0xe0) == 0xc0) {
    ret = *(*string)++ & 0x1f;
    ret = (ret << 6) | GET_CONT(*(*string)++);
    return ret;
  }

  if ((**string & 0xf0) == 0xe0) {
    ret = *(*string)++ & 0x0f;
    ret = (ret << 6) | GET_CONT(*(*string)++);
    ret = (ret << 6) | GET_CONT(*(*string)++);
    return ret;
  }

  if ((**string & 0xf8) == 0xf0) {
    ret = *(*string)++ & 0x07;
    ret = (ret << 6) | GET_CONT(*(*string)++);
    ret = (ret << 6) | GET_CONT(*(*string)++);
    ret = (ret << 6) | GET_CONT(*(*string)++);
    return ret;
  }

  /** We shouldn't be here... Because 5 and 6 bytes are impossible... */
  assert(0);
  return 0xffffffff;
}

/** Escapes a single UTF-16 code unit for JSON. */
static inline void escape_uni16(uint16_t u16, char ** const json) {
  *(*json)++ = '\\';
  *(*json)++ = 'u';
  *(*json)++ = TO_HEX(u16 >> 12);
  *(*json)++ = TO_HEX(u16 >> 8);
  *(*json)++ = TO_HEX(u16 >> 4);
  *(*json)++ = TO_HEX(u16);
}

/** Escapes a UTF-8 string to ASCII JSON format. */
static void json_escape(const char * restrict string, char * restrict const target)
{
  uint32_t curr = 0;
  char *escaped = target;
  while (*string != '\0' && (escaped - target) < (PATH_MAX * 2 - 1)) {
    switch (*string) {
      case '\"':
      case '\\':
        *escaped++ = '\\';
        *escaped++ = *string++;
        break;
      default:
	curr = decode_utf8(&string);
	if (curr == 0xffffffff) break;
	if (likely(curr < 0xffff)) {
	  if (likely(curr < 0x20 || curr > 0x7f))
	    escape_uni16((uint16_t)curr, &escaped);
	  else
	    *escaped++ = (char)curr;
	} else {
	  curr -= 0x10000;
	  escape_uni16((uint16_t)(0xD800 + ((curr >> 10) & 0x03ff)), &escaped);
	  escape_uni16((uint16_t)(0xDC00 + (curr & 0x03ff)), &escaped);
	}
        break;
    }
  }
  *escaped = '\0';
  return;
}

void printjson(file_t * restrict files, const int argc, char **argv)
{
  file_t * restrict tmpfile;
  int arg = 0, comma = 0, len = 0;
  char *temp = malloc(PATH_MAX * 2);
  char *temp2 = malloc(PATH_MAX * 2);
  char *temp_insert = temp;

  LOUD(fprintf(stderr, "printjson: %p\n", files));

  /* Output information about the jdupes command environment */
  printf("{\n  \"jdupesVersion\": \"%s\",\n  \"jdupesVersionDate\": \"%s\",\n", VER, VERDATE);

  printf("  \"commandLine\": \"");
  while (arg < argc) {
    len = sprintf(temp_insert, " %s", argv[arg]);
    assert(len >= 0);
    temp_insert += len;
    arg++;
  }
  json_escape(temp + 1, temp2); /* Skip the starting space */
  printf("%s\",\n", temp2);
  printf("  \"extensionFlags\": \"");
#ifndef NO_HELPTEXT
  if (feature_flags[0] == NULL) printf("none\",\n");
  else for (int c = 0; feature_flags[c] != NULL; c++)
    printf("%s%s", feature_flags[c], feature_flags[c+1] == NULL ? "\",\n" : " ");
#else
  printf("unavailable\",\n");
#endif

  printf("  \"matchSets\": [\n");
  while (files != NULL) {
    if (ISFLAG(files->flags, FF_HAS_DUPES)) {
      if (comma) printf(",\n");
      printf("    {\n      \"fileSize\": %" PRIdMAX ",\n      \"fileList\": [\n        { \"filePath\": \"", (intmax_t)files->size);
      sprintf(temp, "%s", files->d_name);
      json_escape(temp, temp2);
      jc_fwprint(stdout, temp2, 0);
      printf("\"");
      tmpfile = files->duplicates;
      while (tmpfile != NULL) {
        printf(" },\n        { \"filePath\": \"");
        sprintf(temp, "%s", tmpfile->d_name);
        json_escape(temp, temp2);
        jc_fwprint(stdout, temp2, 0);
        printf("\"");
        tmpfile = tmpfile->duplicates;
      }
      printf(" }\n      ]\n    }");
      comma = 1;
    }
    files = files->next;
  }

  printf("\n  ]\n}\n");

  free(temp); free(temp2);
  return;
}

#endif /* NO_JSON */
