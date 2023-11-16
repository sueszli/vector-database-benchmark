/**
 * @file
 * @brief Run specified shell
 *
 * @date 28.04.2014
 * @author Alexander Kalmuk
 */

#include <cmd/shell.h>
#include <stdio.h>

static void print_help(void) {
	printf("Usage: shell_run <shell>\n");
}

int main(int argc, char **argv) {
	const struct shell *shell;

	if (argc != 2) {
		print_help();
		return -1;
	}

	shell = shell_lookup(argv[1]);

	if (!shell) {
		printf("shell_run: shell does not exists - %s\n", argv[1]);
		return -1;
	}

	return shell_run(shell);
}
