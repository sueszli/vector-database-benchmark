#include <stdio.h>
#include <graphics.h>
#include <X11/Xz88dk.h>


#pragma redirect fputc_cons=xputc

/*
   A proportional font is simply a sprite set.
   You can import a whole font with the sprite editor provided in {z88dk}/support/sprites
   Use 'N' to import from PrintMaster or NewsMaster, 'O' for GEOS, 'L' for BDF and other formats.
   Move on the last sprite to save the whole set, then press F5.
   
   Then, paste the generated set over the current font[] declaration and rename the first sprite into "font".
   Change _yh_proportional for an appropriate line spacing.
   
   (Use F1 for a summary of the functions)
   
*/

// Generated by Daniel McKinnon's z88dk Sprite Editor
char font[] = { 6, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 
, 0x00,
 3, 12, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0x00 , 0xC0 
, 0xC0,
 7, 4, 0xCC , 0xCC , 0xCC , 0xCC,
  10, 12, 0x1B , 0x00 , 0x1B , 0x00 , 0x1B , 0x00 , 0x7F , 0x80 , 0x36 , 0x00 , 0x36 
, 0x00 , 0x36 , 0x00 , 0x36 , 0x00 , 0xFF , 0x00 , 0x6C , 0x00 , 0x6C , 0x00 
, 0x6C , 0x00,
 8, 13, 0x10 , 0x7C , 0xD6 , 0xD6 , 0xD0 , 0xD0 , 0x7C , 0x16 , 0x16 , 0xD6 , 0xD6 
, 0x7C , 0x10,
 14, 12, 0x38 , 0x18 , 0x6C , 0x30 , 0x6C , 0x60 , 0x6C , 0xC0 , 0x39 , 0x80 , 0x03 
, 0x00 , 0x06 , 0x00 , 0x0C , 0xE0 , 0x19 , 0xB0 , 0x31 , 0xB0 , 0x61 , 0xB0 
, 0xC0 , 0xE0,
 10, 12, 0x38 , 0x00 , 0x6C , 0x00 , 0x6C , 0x00 , 0x6C , 0x00 , 0x38 , 0x00 , 0x38 
, 0x00 , 0x78 , 0x00 , 0x6D , 0x80 , 0xCF , 0x00 , 0xC6 , 0x00 , 0xCF , 0x00 
, 0x7B , 0x00,
 3, 4, 0xC0 , 0xC0 , 0xC0 , 0xC0,
  5, 16, 0x30 , 0x60 , 0x60 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 
, 0xC0 , 0xC0 , 0x60 , 0x60 , 0x30,
 5, 16, 0xC0 , 0x60 , 0x60 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 
, 0x30 , 0x30 , 0x60 , 0x60 , 0xC0,
 9, 10, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x66 , 0x00 , 0x66 , 0x00 , 0x3C 
, 0x00 , 0xFF , 0x00 , 0x3C , 0x00 , 0x66 , 0x00 , 0x66 , 0x00,
 9, 11, 0x00 , 0x00 , 0x00 , 0x00 , 0x18 , 0x00 , 0x18 , 0x00 , 0x18 , 0x00 , 0x18 
, 0x00 , 0xFF , 0x00 , 0x18 , 0x00 , 0x18 , 0x00 , 0x18 , 0x00 , 0x18 , 0x00,
 4, 13, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x60 
, 0x60 , 0xC0,
 6, 8, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xF8,
 3, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xC0 
, 0xC0,
 5, 12, 0x30 , 0x30 , 0x30 , 0x30 , 0x60 , 0x60 , 0x60 , 0x60 , 0xC0 , 0xC0 , 0xC0 
, 0xC0,
 8, 12, 0x38 , 0x6C , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0x6C 
, 0x38,
 5, 12, 0x30 , 0xF0 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 
, 0x30,
 8, 12, 0x7C , 0xC6 , 0x86 , 0x06 , 0x06 , 0x0C , 0x18 , 0x30 , 0x60 , 0xC0 , 0xC0 
, 0xFE,
 8, 12, 0x7C , 0xC6 , 0x86 , 0x06 , 0x06 , 0x3C , 0x06 , 0x06 , 0x06 , 0x86 , 0xC6 
, 0x7C,
 8, 12, 0x0C , 0x0C , 0x1C , 0x1C , 0x3C , 0x6C , 0x6C , 0xCC , 0xFE , 0x0C , 0x0C 
, 0x0C,
 8, 12, 0xFE , 0xC0 , 0xC0 , 0xC0 , 0xFC , 0xC6 , 0x06 , 0x06 , 0x06 , 0x86 , 0xC6 
, 0x7C,
 8, 12, 0x3C , 0x60 , 0xC0 , 0xC0 , 0xC0 , 0xFC , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0xC6 
, 0x7C,
 8, 12, 0xFE , 0x06 , 0x06 , 0x0C , 0x0C , 0x18 , 0x18 , 0x18 , 0x30 , 0x30 , 0x30 
, 0x30,
 8, 12, 0x7C , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0x7C , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0xC6 
, 0x7C,
 8, 12, 0x7C , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0xC6 , 0x7E , 0x06 , 0x06 , 0x06 , 0x0C 
, 0x78,
 3, 12, 0x00 , 0x00 , 0x00 , 0xC0 , 0xC0 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xC0 
, 0xC0,
 4, 13, 0x00 , 0x00 , 0x00 , 0x60 , 0x60 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x60 
, 0x60 , 0xC0,
 8, 12, 0x00 , 0x06 , 0x0C , 0x18 , 0x30 , 0x60 , 0xC0 , 0x60 , 0x30 , 0x18 , 0x0C 
, 0x06,
 8, 8, 0x00 , 0x00 , 0x00 , 0x00 , 0xFE , 0x00 , 0x00 , 0xFE,
 8, 12, 0x00 , 0xC0 , 0x60 , 0x30 , 0x18 , 0x0C , 0x06 , 0x0C , 0x18 , 0x30 , 0x60 
, 0xC0,
 9, 12, 0x3C , 0x00 , 0x66 , 0x00 , 0xC3 , 0x00 , 0x03 , 0x00 , 0x03 , 0x00 , 0x06 
, 0x00 , 0x0C , 0x00 , 0x18 , 0x00 , 0x18 , 0x00 , 0x00 , 0x00 , 0x18 , 0x00 
, 0x18 , 0x00,
 15, 13, 0x0F , 0xC0 , 0x38 , 0x70 , 0x60 , 0x18 , 0x63 , 0x98 , 0xC6 , 0xCC , 0xCC 
, 0xCC , 0xCC , 0xCC , 0xCC , 0xCC , 0xCC , 0xD8 , 0x67 , 0x70 , 0x60 , 0x00 
, 0x38 , 0x00 , 0x0F , 0xC0,
 12, 12, 0x04 , 0x00 , 0x0E , 0x00 , 0x0E , 0x00 , 0x1B , 0x00 , 0x1B , 0x00 , 0x31 
, 0x80 , 0x31 , 0x80 , 0x60 , 0xC0 , 0x7F , 0xC0 , 0xC0 , 0x60 , 0xC0 , 0x60 
, 0xC0 , 0x60,
 10, 12, 0xFE , 0x00 , 0xC3 , 0x00 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC3 , 0x00 , 0xFE 
, 0x00 , 0xC3 , 0x00 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC3 , 0x00 
, 0xFE , 0x00,
 11, 12, 0x1F , 0x00 , 0x31 , 0x80 , 0x60 , 0xC0 , 0x60 , 0xC0 , 0xC0 , 0x00 , 0xC0 
, 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0x60 , 0xC0 , 0x60 , 0xC0 , 0x31 , 0x80 
, 0x1F , 0x00,
 10, 12, 0xFC , 0x00 , 0xC6 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC1 , 0x80 , 0xC1 
, 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC6 , 0x00 
, 0xFC , 0x00,
 9, 12, 0xFF , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xFE 
, 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 
, 0xFF , 0x00,
 9, 12, 0xFF , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xFE 
, 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 
, 0xC0 , 0x00,
 11, 12, 0x1F , 0x00 , 0x31 , 0x80 , 0x60 , 0xC0 , 0x60 , 0xC0 , 0xC0 , 0x00 , 0xC0 
, 0x00 , 0xC7 , 0xC0 , 0xC0 , 0xC0 , 0x60 , 0xC0 , 0x61 , 0xC0 , 0x31 , 0xC0 
, 0x1F , 0x40,
 10, 12, 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xFF 
, 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 
, 0xC1 , 0x80,
 3, 12, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 
, 0xC0,
 8, 12, 0x06 , 0x06 , 0x06 , 0x06 , 0x06 , 0x06 , 0x06 , 0x06 , 0xC6 , 0xC6 , 0x6C 
, 0x38,
 10, 12, 0xC3 , 0x00 , 0xC6 , 0x00 , 0xCC , 0x00 , 0xD8 , 0x00 , 0xF0 , 0x00 , 0xE0 
, 0x00 , 0xF0 , 0x00 , 0xD8 , 0x00 , 0xCC , 0x00 , 0xC6 , 0x00 , 0xC3 , 0x00 
, 0xC1 , 0x80,
 9, 12, 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 
, 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 
, 0xFF , 0x00,
 12, 12, 0xC0 , 0x60 , 0xC0 , 0x60 , 0xE0 , 0xE0 , 0xE0 , 0xE0 , 0xF1 , 0xE0 , 0xF1 
, 0xE0 , 0xDB , 0x60 , 0xDB , 0x60 , 0xCE , 0x60 , 0xCE , 0x60 , 0xC4 , 0x60 
, 0xC4 , 0x60,
 10, 12, 0xC1 , 0x80 , 0xC1 , 0x80 , 0xE1 , 0x80 , 0xF1 , 0x80 , 0xF9 , 0x80 , 0xD9 
, 0x80 , 0xCD , 0x80 , 0xCF , 0x80 , 0xC7 , 0x80 , 0xC3 , 0x80 , 0xC1 , 0x80 
, 0xC1 , 0x80,
 11, 12, 0x1E , 0x00 , 0x33 , 0x00 , 0x61 , 0x80 , 0x61 , 0x80 , 0xC0 , 0xC0 , 0xC0 
, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0x61 , 0x80 , 0x61 , 0x80 , 0x33 , 0x00 
, 0x1E , 0x00,
 10, 12, 0xFE , 0x00 , 0xC3 , 0x00 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC3 
, 0x00 , 0xFE , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 
, 0xC0 , 0x00,
 11, 12, 0x1E , 0x00 , 0x33 , 0x00 , 0x61 , 0x80 , 0x61 , 0x80 , 0xC0 , 0xC0 , 0xC0 
, 0xC0 , 0xC0 , 0xC0 , 0xCC , 0xC0 , 0x67 , 0x80 , 0x63 , 0x80 , 0x33 , 0x80 
, 0x1E , 0xC0,
 11, 12, 0xFE , 0x00 , 0xC3 , 0x00 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC3 
, 0x00 , 0xFE , 0x00 , 0xC3 , 0x00 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 
, 0xC0 , 0xC0,
 9, 12, 0x3C , 0x00 , 0x66 , 0x00 , 0xC3 , 0x00 , 0xC0 , 0x00 , 0x60 , 0x00 , 0x3C 
, 0x00 , 0x06 , 0x00 , 0x03 , 0x00 , 0x03 , 0x00 , 0xC3 , 0x00 , 0x66 , 0x00 
, 0x3C , 0x00,
 11, 12, 0xFF , 0xC0 , 0x0C , 0x00 , 0x0C , 0x00 , 0x0C , 0x00 , 0x0C , 0x00 , 0x0C 
, 0x00 , 0x0C , 0x00 , 0x0C , 0x00 , 0x0C , 0x00 , 0x0C , 0x00 , 0x0C , 0x00 
, 0x0C , 0x00,
 10, 12, 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 
, 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0x63 , 0x00 
, 0x3E , 0x00,
 11, 12, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0x61 , 0x80 , 0x61 , 0x80 , 0x61 , 0x80 , 0x33 
, 0x00 , 0x33 , 0x00 , 0x12 , 0x00 , 0x1E , 0x00 , 0x1E , 0x00 , 0x0C , 0x00 
, 0x0C , 0x00,
 15, 12, 0xC0 , 0x0C , 0xC3 , 0x0C , 0xC3 , 0x0C , 0x63 , 0x18 , 0x67 , 0x98 , 0x67 
, 0x98 , 0x34 , 0xB0 , 0x3C , 0xF0 , 0x3C , 0xF0 , 0x18 , 0x60 , 0x18 , 0x60 
, 0x18 , 0x60,
 11, 12, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0x61 , 0x80 , 0x33 , 0x00 , 0x1E , 0x00 , 0x0C 
, 0x00 , 0x0C , 0x00 , 0x1E , 0x00 , 0x33 , 0x00 , 0x61 , 0x80 , 0xC0 , 0xC0 
, 0xC0 , 0xC0,
 11, 12, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0x61 , 0x80 , 0x33 , 0x00 , 0x33 , 0x00 , 0x1E 
, 0x00 , 0x1E , 0x00 , 0x0C , 0x00 , 0x0C , 0x00 , 0x0C , 0x00 , 0x0C , 0x00 
, 0x0C , 0x00,
 10, 12, 0xFF , 0x80 , 0x01 , 0x80 , 0x03 , 0x00 , 0x06 , 0x00 , 0x06 , 0x00 , 0x0C 
, 0x00 , 0x18 , 0x00 , 0x30 , 0x00 , 0x30 , 0x00 , 0x60 , 0x00 , 0xC0 , 0x00 
, 0xFF , 0x80,
 5, 16, 0xF0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 
, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xF0,
 5, 12, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0x60 , 0x60 , 0x60 , 0x60 , 0x30 , 0x30 , 0x30 
, 0x30,
 5, 16, 0xF0 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 
, 0x30 , 0x30 , 0x30 , 0x30 , 0xF0,
 6, 3, 0x20 , 0x70 , 0xD8,
 10, 14, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 
, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 
, 0x00 , 0x00 , 0x00 , 0x00 , 0xFF , 0x80,
 5, 3, 0xE0 , 0x60 , 0x30,
 10, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x7E , 0x00 , 0xC3 , 0x00 , 0x03 
, 0x00 , 0x1F , 0x00 , 0x73 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 
, 0x7D , 0x80,
 9, 12, 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xDC , 0x00 , 0xE6 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xE6 , 0x00 
, 0xDC , 0x00,
 8, 12, 0x00 , 0x00 , 0x00 , 0x3C , 0x66 , 0xC2 , 0xC0 , 0xC0 , 0xC0 , 0xC2 , 0x66 
, 0x3C,
 9, 12, 0x03 , 0x00 , 0x03 , 0x00 , 0x03 , 0x00 , 0x3B , 0x00 , 0x67 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0x67 , 0x00 
, 0x3B , 0x00,
 9, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x3C , 0x00 , 0x66 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xFF , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0x63 , 0x00 
, 0x3E , 0x00,
 7, 12, 0x1C , 0x30 , 0x30 , 0xFC , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 
, 0x30,
 9, 16, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x3B , 0x00 , 0x67 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0x67 , 0x00 
, 0x3B , 0x00 , 0x03 , 0x00 , 0x03 , 0x00 , 0xC6 , 0x00 , 0x7C , 0x00,
 9, 12, 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xDC , 0x00 , 0xE6 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 
, 0xC3 , 0x00,
 3, 12, 0xC0 , 0xC0 , 0x00 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 
, 0xC0,
 4, 16, 0x60 , 0x60 , 0x00 , 0x60 , 0x60 , 0x60 , 0x60 , 0x60 , 0x60 , 0x60 , 0x60 
, 0x60 , 0x60 , 0x60 , 0x60 , 0xC0,
 9, 12, 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC6 , 0x00 , 0xCC , 0x00 , 0xD8 
, 0x00 , 0xF0 , 0x00 , 0xF0 , 0x00 , 0xD8 , 0x00 , 0xCC , 0x00 , 0xC6 , 0x00 
, 0xC3 , 0x00,
 3, 12, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 
, 0xC0,
 13, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xDC , 0xE0 , 0xE7 , 0x30 , 0xC6 
, 0x30 , 0xC6 , 0x30 , 0xC6 , 0x30 , 0xC6 , 0x30 , 0xC6 , 0x30 , 0xC6 , 0x30 
, 0xC6 , 0x30,
 9, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xDC , 0x00 , 0xE6 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 
, 0xC3 , 0x00,
 9, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x3C , 0x00 , 0x66 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0x66 , 0x00 
, 0x3C , 0x00,
 9, 16, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xDC , 0x00 , 0xE6 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xE6 , 0x00 
, 0xDC , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00 , 0xC0 , 0x00,
 9, 16, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x3B , 0x00 , 0x67 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0x67 , 0x00 
, 0x3B , 0x00 , 0x03 , 0x00 , 0x03 , 0x00 , 0x03 , 0x00 , 0x03 , 0x00,
 6, 12, 0x00 , 0x00 , 0x00 , 0xD8 , 0xF0 , 0xE0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 
, 0xC0,
 8, 12, 0x00 , 0x00 , 0x00 , 0x7C , 0xC6 , 0xC0 , 0xE0 , 0x7C , 0x0E , 0x06 , 0xC6 
, 0x7C,
 7, 12, 0x30 , 0x30 , 0x30 , 0xFC , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 
, 0x1C,
 9, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 
, 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0x67 , 0x00 
, 0x3B , 0x00,
 10, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0x63 
, 0x00 , 0x63 , 0x00 , 0x63 , 0x00 , 0x36 , 0x00 , 0x36 , 0x00 , 0x1C , 0x00 
, 0x1C , 0x00,
 15, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xC0 , 0x0C , 0xC3 , 0x0C , 0x63 
, 0x18 , 0x67 , 0x98 , 0x67 , 0x98 , 0x34 , 0xB0 , 0x3C , 0xF0 , 0x18 , 0x60 
, 0x18 , 0x60,
 9, 12, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xC3 , 0x00 , 0xC3 , 0x00 , 0x66 
, 0x00 , 0x3C , 0x00 , 0x18 , 0x00 , 0x3C , 0x00 , 0x66 , 0x00 , 0xC3 , 0x00 
, 0xC3 , 0x00,
 10, 16, 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0x00 , 0xC1 , 0x80 , 0xC1 , 0x80 , 0x63 
, 0x00 , 0x63 , 0x00 , 0x63 , 0x00 , 0x36 , 0x00 , 0x36 , 0x00 , 0x1C , 0x00 
, 0x1C , 0x00 , 0x18 , 0x00 , 0x18 , 0x00 , 0x30 , 0x00 , 0x60 , 0x00,
 7, 12, 0x00 , 0x00 , 0x00 , 0xFC , 0x0C , 0x18 , 0x18 , 0x30 , 0x60 , 0x60 , 0xC0 
, 0xFC,
 6, 15, 0x18 , 0x30 , 0x60 , 0x60 , 0x60 , 0x60 , 0x60 , 0xC0 , 0x60 , 0x60 , 0x60 
, 0x60 , 0x60 , 0x30 , 0x18,
 3, 16, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0 
, 0xC0 , 0xC0 , 0xC0 , 0xC0 , 0xC0,
 6, 15, 0xC0 , 0x60 , 0x30 , 0x30 , 0x30 , 0x30 , 0x30 , 0x18 , 0x30 , 0x30 , 0x30 
, 0x30 , 0x30 , 0x60 , 0xC0,
 6, 2, 0xE8 , 0xB8  };


int bold_flg=0;

extern char __LIB__ xputc (char c);

extern char xputc (char c) {
	_yh_proportional=17;
	_xfputc (c, font, bold_flg);  // char, font, bold flag
	return (c);
}



main()
{
	clg();
	printf("Importing proportional fonts !\n\n");
	printf("abcdefghijklmnopqrstuvwxyz\n");
	bold_flg=1;
	printf("abcdefghijklmnopqrstuvwxyz\n");
	while (getk()!=0){};
	while (getk()==0){};
}

