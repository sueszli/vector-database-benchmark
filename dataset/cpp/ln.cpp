/*
	This file is part of duckOS.

	duckOS is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	duckOS is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with duckOS.  If not, see <https://www.gnu.org/licenses/>.

	Copyright (c) Byteduck 2016-2021. All rights reserved.
*/

//A program that makes a hard or symbolic link to a file.

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

int main(int argc, char** argv) {
	bool symbolic = false;
	if(argc >= 2) symbolic = strcmp(argv[1], "-s") == 0;
	if(argc < (symbolic ? 4 : 3)) {
		printf("Missing operands\nUsage: ln [-s] FILE LINK_NAME\n");
		return 1;
	}

	int res;
	if(symbolic)
		res = symlink(argv[2], argv[3]);
	else
		res = link(argv[1], argv[2]);
	if(res == 0) return 0;
	perror("ln");
	return errno;
}
