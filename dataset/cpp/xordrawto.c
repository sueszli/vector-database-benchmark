/*
 *	Enterprise 64/128 graphics libraries
 *
 *	xordrawto(x,y)
 *
 *	Stefano Bodrato - March 2011
 *
 *	$Id: xordrawto.c $
 */

#include <enterprise.h>
//#include <graphics.h>
extern void __LIB__ xordrawto(int x2, int y2) __smallc;


void xordrawto(int x,int y)
{
	esccmd_cmd='I';	// INK colour
	esccmd_x=1;
	exos_write_block(DEFAULT_VIDEO, 3, esccmd);

	esccmd_cmd='M'; // set beam style
	esccmd_x=3;	// XOR
	exos_write_block(DEFAULT_VIDEO, 3, esccmd);

	esccmd_cmd='S'; // set beam on
	exos_write_block(DEFAULT_VIDEO, 2, esccmd);

	esccmd_cmd='A'; // set beam position
	esccmd_x=x*2;
	esccmd_y=971-y*4;
	exos_write_block(DEFAULT_VIDEO, 6, esccmd);

	esccmd_cmd='s'; // set beam off
	exos_write_block(DEFAULT_VIDEO, 2, esccmd);

	esccmd_cmd='M'; // set beam style
	esccmd_x=0;	// PLOT
	exos_write_block(DEFAULT_VIDEO, 3, esccmd);
}
