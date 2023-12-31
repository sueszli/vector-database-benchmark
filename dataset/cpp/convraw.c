/*
 * Copyright (c) 1999, 2000 Greg Haerr <greg@censoft.com>
 *
 * INT10.ORG Raw font file converter
 * Note: ascent field of produced C file must be hand-editted
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include "../../include/device.h"

/* font constants*/
#define FONT_FILE	"EDDA9.F14"
#define	ROM_CHAR_WIDTH	8	/* number of pixels for character width */
#define MAX_ROM_HEIGHT	14	/* numer of scan lines in each font character*/
int	ROM_CHAR_HEIGHT = MAX_ROM_HEIGHT;
#define	FONT_CHARS	256	/* number of characters in font tables */

/* local data*/
char rom_char_addr[MAX_ROM_HEIGHT * FONT_CHARS];

void print_rom_table(void);
void print_char(int ch,MWIMAGEBITS *b, int w, int h);
void print_bits(MWIMAGEBITS *bits, int width, int height);

int main(int ac, char **av)
{
	int fd;

	fd = open(FONT_FILE, O_RDONLY);
	if (fd < 0) {
		printf("Can't open %s\n", FONT_FILE);
		exit(1);
	}
	read(fd, rom_char_addr, sizeof(rom_char_addr));

	printf("/* Generated by convraw*/\n");
	printf("//#include \"device.h\"\n\n");
	printf("/* ROM %dx%d*/\n\n", ROM_CHAR_WIDTH, ROM_CHAR_HEIGHT);
	//printf("static MWIMAGEBITS rom%dx%d_bits[] = {\n\n", ROM_CHAR_WIDTH, ROM_CHAR_HEIGHT);
	printf("unsigned short rom%dx%d_bits[] = {\n\n", ROM_CHAR_WIDTH, ROM_CHAR_HEIGHT);

	print_rom_table();

	printf("};\n\n");
	printf("#if 0\n");
	printf("/* Exported structure definition. */\n"
		"MWCFONT font_rom%dx%d = {\n",
		ROM_CHAR_WIDTH, ROM_CHAR_HEIGHT);
	printf("\t\"rom%dx%d\",\n", ROM_CHAR_WIDTH, ROM_CHAR_HEIGHT);
	printf("\t%d,\n", ROM_CHAR_WIDTH);
	printf("\t%d,\n", ROM_CHAR_HEIGHT);
	printf("\t%d,\n", ROM_CHAR_HEIGHT);	/* ascent*/
	printf("\t0,\n\t256,\n");
	printf("\trom%dx%d_bits,\n", ROM_CHAR_WIDTH, ROM_CHAR_HEIGHT);
	printf("\t0,\n\t0\n");
	printf("};\n");
	printf("#endif\n");

	return 0;
}

void
print_rom_table(void)
{
	char *		bits;
	int		n;
	int		ch;
	MWIMAGEBITS *	p;
	MWIMAGEBITS	image[MAX_ROM_HEIGHT];

	for(ch=0; ch < 256; ++ch) {
		bits = rom_char_addr + ch * ROM_CHAR_HEIGHT;
		p = image;
		for(n=0; n<ROM_CHAR_HEIGHT; ++n)
		    *p++ = *bits++ << 8;
		print_char(ch, image, ROM_CHAR_WIDTH, ROM_CHAR_HEIGHT);
		print_bits(image, ROM_CHAR_WIDTH, ROM_CHAR_HEIGHT);
		printf("\n");
	}
}

/* Character ! (0x21):
   ht=16, width=8
   +----------------+
   |                |
   |                |
   | *              |
   | *              |
   | *              |
   | *              |
   | *              |
   | *              |
   |                |
   | *              |
   |                |
   |                |
   +----------------+ */

void
print_char(int ch,MWIMAGEBITS *bits, int width, int height)
{
	MWCOORD 		x;
	int 		bitcount;	/* number of bits left in bitmap word */
	MWIMAGEBITS	bitvalue;	/* bitmap word value */

	printf("/* Character %c (0x%02x):\n", ch? ch: ' ', ch);
	printf("   ht=%d, width=%d\n", height, width);
	printf("   +");
	for(x=0; x<width; ++x)
		printf("-");
	printf("+\n");
	x = 0;
	bitcount = 0;
	while (height > 0) {
	    if (bitcount <= 0) {
		    printf("   |");
		    bitcount = MWIMAGE_BITSPERIMAGE;
		    bitvalue = *bits++;
	    }
		if (MWIMAGE_TESTBIT(bitvalue))
			    printf("*");
		else printf(" ");
	    bitvalue = MWIMAGE_SHIFTBIT(bitvalue);
	    --bitcount;
	    if (x++ == width-1) {
		    x = 0;
		    --height;
		    bitcount = 0;
		    printf("|\n");
	    }
	}
	printf("   +");
	for(x=0; x<width; ++x)
		printf("-");
	printf("+ */\n");
}

#define	MWIMAGE_GETBIT4(m)	(((m) & 0xf000) >> 12)
#define	MWIMAGE_SHIFTBIT4(m)	((MWIMAGEBITS) ((m) << 4))

void
print_bits(MWIMAGEBITS *bits, int width, int height)
{
	MWCOORD 		x;
	int 		bitcount;	/* number of bits left in bitmap word */
	MWIMAGEBITS	bitvalue;	/* bitmap word value */

	x = 0;
	bitcount = 0;
	while (height > 0) {
	    if (bitcount <= 0) {
		    printf("0x");
		    bitcount = MWIMAGE_BITSPERIMAGE;
		    bitvalue = *bits++;
	    }
		printf("%x", MWIMAGE_GETBIT4(bitvalue));
	    bitvalue = MWIMAGE_SHIFTBIT4(bitvalue);
	    bitcount -= 4;
		x += 4;
	    if (x >= width-1) {
			if(MWIMAGE_BITSPERIMAGE > width)
				for(x=MWIMAGE_BITSPERIMAGE-width; x>0; ) {
					printf("0");
					x -= 4;
				}
		    x = 0;
		    --height;
		    bitcount = 0;
		    printf(",\n");
	    }
	}
}
