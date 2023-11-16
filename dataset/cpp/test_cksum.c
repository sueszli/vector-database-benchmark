#include <stdio.h>

int main(int argc, char **argv) {
	FILE *fp = fopen(argv[1],"rb");
	unsigned char checksum = 0;
	while (!feof(fp) && !ferror(fp)) {
	   checksum ^= fgetc(fp);
	}

	fclose(fp);
	printf("%d\n", checksum);
	return 0;
}
