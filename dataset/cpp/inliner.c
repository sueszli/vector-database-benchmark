/* inliner - inline libc source into C file.
 *
 * The inliner is used at build-time, and developers can use the
 * "inline C" feature to implement target-specific parts such as
 * C runtime and essential libraries.
 */

#include <stdio.h>
#include <stdlib.h>

#define MAX_LINE_LEN 200
#define MAX_SIZE 65536

char *SOURCE;
int source_idx;

void write_char(char c)
{
    SOURCE[source_idx++] = c;
}

void write_str(char *str)
{
    int i = 0;
    while (str[i])
        write_char(str[i++]);
}

void write_line(char *src)
{
    int i;

    write_str("  __c(\"");
    for (i = 0; src[i]; i++) {
        if (src[i] == '\"') {
            write_char('\\');
            write_char('\"');
        } else if (src[i] != '\n') {
            write_char(src[i]);
        }
    }

    write_char('\\');
    write_char('n');
    write_str("\");\n");
}

void load_from(char *file)
{
    char buffer[MAX_LINE_LEN];
    FILE *f = fopen(file, "rb");
    for (;;) {
        if (!fgets(buffer, MAX_LINE_LEN, f)) {
            fclose(f);
            return;
        }
        write_line(buffer);
    }
    fclose(f);
}

void save_to(char *file)
{
    int i;
    FILE *f = fopen(file, "wb");
    for (i = 0; i < source_idx; i++)
        fputc(SOURCE[i], f);
    fclose(f);
}

int main(int argc, char *argv[])
{
    if (argc <= 2) {
        printf("Usage: inliner <input.c> <output.inc>\n");
        return -1;
    }

    source_idx = 0;
    SOURCE = malloc(MAX_SIZE);

    write_str("/* Created by tools/inliner - DO NOT EDIT. */\n");

    /* __c is inspired by __asm keyword, which invokes the inline assembler.
     * Here, it is meant to "inline C code." Example:
     *   __c("int strlen(char *str) {\n");
     *   __c("    int i = 0;\n");
     *   __c("    while (str[i])\n");
     *   __c("        i++;\n");
     *   __c("    return i;\n");
     *   __c("}\n");
     */
    write_str("void __c(char *src) {\n");
    write_str("    int i;\n");
    write_str("    for (i = 0; src[i]; i++)\n");
    write_str("        SOURCE[source_idx++] = src[i];\n");
    write_str("}\n");

    write_str("void libc_generate() {\n");
    load_from(argv[1]);
    write_str("}\n");
    save_to(argv[2]);

    return 0;
}
