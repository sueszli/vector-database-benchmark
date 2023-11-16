/*
 *   CP/M disc generator
 *
 *   $Id: cpm2.c $
 */

#include "appmake.h"



static char             *c_binary_name      = NULL;
static char             *c_crt_filename      = NULL;
static char             *c_disc_format       = NULL;
static char             *c_output_file      = NULL;
static char             *c_boot_filename     = NULL;
static char             *c_disc_container    = "dsk";
static char             *c_extension         = NULL;
static char            **c_additional_files  = NULL;
static int               c_additional_files_num = 0;

static char              c_force_com_extension   = 0;
static char              c_disable_com_file_creation = 0;
static char              help         = 0;


static void c_add_file(char *option);

/* Options that are available for this module */
option_t cpm2_options[] = {
    { 'h', "help",     "Display this help",          OPT_BOOL,  &help},
    { 'f', "format",   "Disk format",                OPT_STR,   &c_disc_format},
    { 'b', "binfile",  "Linked binary file",         OPT_STR|OPT_INPUT,   &c_binary_name },
    { 'c', "crt0file", "crt0 file used in linking",  OPT_STR,   &c_crt_filename },
    { 'o', "output",   "Name of output file",        OPT_STR|OPT_OUTPUT,   &c_output_file },
    { 's', "bootfile", "Name of the boot file",      OPT_STR,   &c_boot_filename },
    {  0,  "container", "Type of container (raw,dsk)", OPT_STR, &c_disc_container },
    {  0,  "extension", "Extension for the output file", OPT_STR, &c_extension},
    {  0,  "force-com-ext", "Always force COM extension", OPT_BOOL, &c_force_com_extension},
    {  0,  "no-com-file", "Don't create a separate .com file", OPT_BOOL, &c_disable_com_file_creation },
                              /* ISO C does not require that a void pointer can be cast to a function pointer
                                 (and vice versa), but conforming compilers are required to warn you about it.
                                 If this is your case, you can probably ignore the warning. */
    { 'a', "add-file", "Add additional files [hostfile:cpmfile] or [hostfile]", OPT_FUNCTION, (void *)c_add_file },
    {  0 ,  NULL,       NULL,                        OPT_NONE,  NULL }
};

static void              dump_formats(void);
static void              bic_write_system_file(disc_handle *h);



static disc_spec einstein_spec = {
    .name = "Einstein",
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 64,
    .extent_size = 2048
};


static disc_spec actrix_spec = {
    .name = "Actrix",
    .disk_mode = MFM250,
    .sectors_per_track = 9,
    .tracks = 40,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0x55,
    .boottracks = 2,
    .directory_entries = 32,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 3, 6, 1, 4, 7, 2, 5, 8 }
};


static disc_spec ampro_spec = {
    .name = "Ampro",
    .disk_mode = MFM300,	
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 17,
    .alternate_sides = 1,
};


// Apple II CP/M on Softcard
static disc_spec apple2_spec = {
    .name = "A2Softcard",
    .sectors_per_track = 16,
    .tracks = 35,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0,6,12,3,9,15,14,5,11,2,8,7,13,4,10,1 }
};


static disc_spec attache_spec = {
    .name = "Attache",
    .disk_mode = MFM300,	
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


// "Aussie Byte" - SMF Knight 2000
static disc_spec aussie_spec = {
    .name = "Knight_2000",
    .disk_mode = MFM250,
    .sectors_per_track = 5,
    .tracks = 80,
    .sides = 2,
    .sector_size = 1024,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .directory_entries = 256,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
    .alternate_sides = 1,
    .has_skew = 1,
    .skew_tab = { 0, 2, 4, 1, 3 }
};


// SSSD osborne 1 disks, see section 7.7 of the Osborne 1 technical manual
// http://dunfield.classiccmp.org/osborne/o1techm.pdf

static disc_spec osborne_spec = {
    .name = "Osborne_DD",
    .disk_mode = MFM300,
    .sectors_per_track = 5,
    .tracks = 40,
    .sides = 1,
    .sector_size = 1024,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};

static disc_spec osborne_sd_spec = {
    .name = "Osborne_SD",
    .disk_mode = FM250,
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 }
};


static disc_spec dmv_spec = {
    .name = "NCR DMV",
    .disk_mode = MFM300,
    .sectors_per_track = 8,
    .tracks = 40,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x50,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 256,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


// Eagle IIE-2 SSDD 96tpi
static disc_spec eagle2_spec = {
    .name = "Eagle II",
    .disk_mode = MFM250,
    .sectors_per_track = 5,
    .tracks = 80,
    .sides = 1,
    .sector_size = 1024,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 192,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 2, 4, 1, 3 }
};


static disc_spec cpcsystem_spec = {
    .name = "CPCSystem",
    .sectors_per_track = 9,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 0x41,
};


static disc_spec pcw40_spec = {
    .name = "PCW40",
    .sectors_per_track = 9,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 1,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


static disc_spec pcw80_spec = {
    .name = "PCW80",
    .sectors_per_track = 9,
    .tracks = 80,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 1,
    .directory_entries = 256,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
    .alternate_sides = 1,
};


static disc_spec microbee_spec = {
    .name = "Microbee",
    .sectors_per_track = 10,
    .tracks = 80,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .directory_entries = 128,
    .extent_size = 4096,
    .byte_size_extents = 1,
    .first_sector_offset = 0x15,
    .boot_tracks_sector_offset = 1,
    .alternate_sides = 1,
    .has_skew = 1,
    .skew_track_start = 5,
    .skew_tab = { 1, 4, 7, 0, 3, 6, 9, 2, 5, 8 }
};


// PMC-101 MicroMate (Type "A")
static disc_spec pmc101a_spec = {
    .name = "PMC-101_A",
    .disk_mode = MFM250,
    .sectors_per_track = 5,
    .tracks = 40,
    .sides = 2,
    .sector_size = 1024,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
};


static disc_spec md2_spec = {
    .name = "Morrow_MD2",
    .disk_mode = MFM250,
    .sectors_per_track = 5,
    .tracks = 40,
    .sides = 1,
    .sector_size = 1024,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 3, 1, 4, 2 }
};

// Tested on ampro emulator, works also in MFM300 mode
static disc_spec md3_spec = {
    .name = "Morrow_MD3",
    .disk_mode = MFM250,
    .sectors_per_track = 5,
    .tracks = 40,
    .sides = 2,
    .sector_size = 1024,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 192,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
    .has_skew = 1,
    .skew_tab = { 0, 3, 1, 4, 2 }
};


static disc_spec mbc1000_spec = {
    .name = "MBC-1000",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 40,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
    .has_skew = 1,
    .skew_tab = { 0,3,6,9,12,15,2,5,8,11,14,1,4,7,10,13 }
};


static disc_spec altos5_spec = {
    .name = "Altos5",
    .disk_mode = MFM250,
    .sectors_per_track = 9,
    .tracks = 80,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 177,
    .extent_size = 4096,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1
};


static disc_spec altos580_spec = {
    .name = "Altos 580",
    .disk_mode = MFM250,
    .sectors_per_track = 9,
    .tracks = 80,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 256,
    .extent_size = 4096,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
    .skew_tab = { 0, 2, 4, 6, 8, 1, 3, 5, 7 }
};


static disc_spec mbc1200_spec = {
    .name = "MBC-1200",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 80,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .directory_entries = 128,
    .extent_size = 4096,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
    .has_skew = 1,
    .skew_tab = { 0,3,6,9,12,15,2,5,8,11,14,1,4,7,10,13 }
};

  
static disc_spec mbc2000_spec = {
    .name = "MBC-2000",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 40,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
    .has_skew = 1,
    .skew_tab = { 0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11 }
};

// Unverified gap size
static disc_spec bondwell12_spec = {
    .name = "Bondwell12",
    .sectors_per_track = 18,
    .tracks = 40,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 12,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
};


static disc_spec bondwell2_spec = {
    .name = "Bondwell2",
    .sectors_per_track = 18,
    .tracks = 80,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
};


static disc_spec kayproii_spec = {
    .name = "KayproII",
    .disk_mode = MFM300,
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 1,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
};


static disc_spec kaypro4_spec = {
    .name = "Kaypro4/10",
    .disk_mode = MFM250,
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 2,
    .alternate_sides = 1,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 1,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
	.side2_sector_numbering = 1
};


static disc_spec mz2500cpm_spec = {
    .name = "MZ2500CPM",
    .sectors_per_track = 16,
    .tracks = 80,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
    .alternate_sides = 1
};


static disc_spec televideo_spec = {
    .name = "Televideo",
    .disk_mode = MFM250,
    .sectors_per_track = 18,
    .tracks = 40,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .directory_entries = 128,
    .alternate_sides = 1,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


static disc_spec nascom_spec = {
    .name = "Nascom",
    .disk_mode = MFM300,
    .sectors_per_track = 10,
    .tracks = 77,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
};


// Toshiba T100/PASOPIA
static disc_spec pasopia_spec = {
    .name = "PASOPIA",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 35,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x23,
    .filler_byte = 0xE5,
    .boottracks = 6,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
    .has_skew = 1,
    .skew_tab = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 }
};


// NEC PC-6001/6601
static disc_spec pc6001_spec = {
    .name = "NEC PC6001",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 40,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x23,
    .filler_byte = 0xFF,
    .boottracks = 2,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


// NEC PC-8001, SSDD
static disc_spec pc8001_spec = {
    .name = "NEC PC8001",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 40,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x23,
    .filler_byte = 0xE5,
    .boottracks = 2,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1
};


// NEC PC-8801, DSDD
static disc_spec pc88_spec = {
    .name = "NEC PC8801",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 40,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x4E,
    .filler_byte = 0xFF,
    .boottracks = 4,
    .alternate_sides = 1,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


static disc_spec qc10_spec = {
    .name = "QC10",
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x3e,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
};


static disc_spec tiki100_ss_spec = {
    .name = "Tiki100",
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x3e,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


static disc_spec tiki100_ds_spec = {
    .name = "Tiki100",
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x3e,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
};


// TRS80 Model I Omikron Mapper CP/M
static disc_spec omikron_spec = {
    .name = "Omikron",
    .disk_mode = FM250,
    .sectors_per_track = 18,
    .tracks = 35,
    .sides = 1,
    .sector_size = 128,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 4, 8, 12, 16, 2, 6, 10, 14, 1, 5, 9, 13, 17, 3, 7, 11, 15 }
};

// TRS80 Model II Lifeboat CP/M
static disc_spec lifeboat_spec = {
    .name = "Lifeboat",
    .disk_mode = MFM500,
    .sectors_per_track = 8,
    .tracks = 77,
    .sides = 1,
    .sector_size = 1024,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_track_start = 0,
    .skew_tab = { 0, 3, 6, 1, 4, 7, 2, 5 }
};

// TRS80 Model II FMG CP/M
static disc_spec fmgcpm_spec = {
    .name = "TRS80IIFMG",
    .disk_mode = MFM500,
    .sectors_per_track = 26,
    .tracks = 77,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};

// TRS80 Model II Pickles & Trout. CP/M
static disc_spec ptcpm_spec = {
    .name = "Pickles&Trout",
    .disk_mode = MFM500,
    .sectors_per_track = 16,
    .tracks = 77,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
};

// TRS80 Holmes VID-80 DDSS
static disc_spec holmes_spec = {
    .name = "TRS80Holmes",
    .disk_mode = MFM250,
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 1,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
};

// TRS80 Memory Merchant Shuffleboard DDSS
static disc_spec merchant_spec = {
    .name = "MemMerchant",
    .disk_mode = MFM300,
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x3e,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};

// TRS80 Model III Hurricane Compactor I&II (DSDD)
static disc_spec hurricane_spec = {
    .name = "Hurricane",
    .disk_mode = MFM300,
    .sectors_per_track = 5,
    .tracks = 40,
    .sides = 1,
    .sector_size = 1024,
    .gap3_length = 0x3e,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 3, 1, 4, 2 }
};

// TRS80 Model 4 Montezuma CP/M
static disc_spec montezuma_spec = {
    .name = "Montezuma",
    .disk_mode = MFM250,
    .sectors_per_track = 18,
    .tracks = 40,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11, 13, 15, 17 }
};

// TRS80 Model 4 CP/M Plus
static disc_spec trs80_cpm3_spec = {
    .name = "TRS80M4CPM3",
    .disk_mode = MFM250,
    .sectors_per_track = 8,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 1,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};

// LNW-80 (TRS80 clone)
static disc_spec lnw80_spec = {
    .name = "LNW 80",
    .disk_mode = MFM250,
    .sectors_per_track = 18,
    .tracks = 40,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0,5,10,15,2,7,12,17,4,9,14,1,6,11,16,3,8,13 }
};

static disc_spec svi40ss_spec = {
    .name = "SVI40SS",
    .sectors_per_track = 17,
    .tracks = 40,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x52,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


static disc_spec col1_spec = {
    .name = "ColAdam",
    .disk_mode = MFM300,
    .sectors_per_track = 8,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x52,
    .filler_byte = 0xe5,
    .boottracks = 0,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .offset = 13312,
    .has_skew = 1,
    .skew_track_start = 0,
    .skew_tab = { 0, 5, 2, 7, 4, 1, 6, 3 }
};


static disc_spec smc777_spec = {
    .name = "SMC-777",
    .sectors_per_track = 16,
    .tracks = 70,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_track_start = 3,
    .skew_tab = { 0,3,6,9,0xc,0xf,2,5,8,0xb,0xE,1,4,7,0xa,0xd }
};


static disc_spec plus3_spec = {
    .name = "ZX+3",
    .sectors_per_track = 9,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 1,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


// BBC Micro, Acorn Z80 2nd processor
static disc_spec bbc_spec = {
    .name = "BBC Micro",
    .sectors_per_track = 5,
    .tracks = 80,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0,2,4,1,3 }
};


static disc_spec bic_spec = {
    .name = "A1505/BIC",
    .sectors_per_track = 5,
    .tracks = 80,
    .sides = 2,
    .sector_size = 1024,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .directory_entries = 128,
    .alternate_sides = 1,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
};


// 8" floppy disk on Xerox 820 or Ferguson BigBoard
static disc_spec bigboard_spec = {
    .name = "BigBoard",
    .disk_mode = FM500,
    .sectors_per_track = 26,
    .tracks = 77,
    .sides = 1,
    .sector_size = 128,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 64,
    .alternate_sides = 0,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0x00, 0x06, 0x0C, 0x12, 0x18, 0x04, 0x0A, 0x10, 0x16, 0x02, 0x08, 0x0E, 0x14, 0x01, 0x07, 0x0d, 0x13, 0x19, 0x05, 0x0b, 0x11, 0x17, 0x03, 0x09, 0x0f, 0x15 }
};


static disc_spec excali_spec = {
    .name = "Excalibur64",
    .disk_mode = MFM300,
    .sectors_per_track = 5,
    .tracks = 80,
    .sides = 2,
    .sector_size = 1024,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .alternate_sides = 1,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 3, 1, 4, 2 }
};


static disc_spec gemini_spec = {
    .name = "Gemini",
    .sectors_per_track = 10,
    .tracks = 35,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
    .alternate_sides = 1,
};


// HP 120/125, DSDD
static disc_spec hp125_spec = {
    .name = "HP125",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 35,
    .sides = 2,
    .alternate_sides = 1,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 128,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
};


static disc_spec lynx_spec = {
    .name = "CampLynx",
    .disk_mode = FM500,           // possibly wrong information gathered online, IMD format is UNTESTED
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


// DEC Rainbow 100
// Some of the disk images are marked as MFM300
static disc_spec rainbow_spec = {
    .name = "Rainbow100",
    .disk_mode = MFM250,	
    .sectors_per_track = 10,
    .tracks = 80,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 }
};


static disc_spec rc700_spec = {
    .name = "RC-700",
    .sectors_per_track = 9,
    .tracks = 35,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .alternate_sides = 1,
    .directory_entries = 112,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
    .has_skew = 1,
    .skew_tab = { 0, 2, 4, 6, 8, 1, 3, 5, 7 }
};


static disc_spec alphatro_spec = {
    .name = "Alphatronic PC",
    .disk_mode = MFM300,           // 300 kbps MFM, visible only when using the IMD format
    .sectors_per_track = 16,
    .tracks = 40,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
};


static disc_spec sharpx1_spec = {
    .name = "Sharp-X1",
    .disk_mode = MFM250,
    .sectors_per_track = 16,
    .tracks = 40,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .alternate_sides = 1,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


static disc_spec fp1100_spec = {
    .name = "Casio-FP1100",
    .sectors_per_track = 16,
    .tracks = 40,
    .sides = 2,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .alternate_sides = 1,
    .directory_entries = 64,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


static disc_spec vector06c_spec = {
    .name = "Vector06c",
    .sectors_per_track = 10,
    .tracks = 80,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 8,
    .directory_entries = 128,
    .alternate_sides = 1,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
};


static disc_spec v1050_spec = {
    .name = "Visual1050",
    .disk_mode = MFM250,
    .sectors_per_track = 10,
    .tracks = 80,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};


// Heath H89/Zenith Z89 SSSD (H17 disk unit)
// H77/Z-77 or H87/Z-87 drives with soft-sectored controller
static disc_spec hz17_spec = {
    .name = "HZenith17",
    .disk_mode = FM250,
    .sectors_per_track = 10,
    .tracks = 40,
    .sides = 1,
    .sector_size = 256,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};

// Heath H89/Zenith Z89 SSDD (Magnolia disk unit)
static disc_spec magnolia_spec = {
    .name = "Magnolia",
    .disk_mode = MFM250,
    .sectors_per_track = 9,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 3,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
};

// Heath H100/Zenith Z100 5" DSDD
static disc_spec z100_spec = {
    .name = "Z-100",
    .disk_mode = MFM250,
    .sectors_per_track = 8,
    .tracks = 40,
    .sides = 2,
    .sector_size = 512,
    .gap3_length = 0x17,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 256,
    .extent_size = 2048,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .alternate_sides = 1,
};

static disc_spec z80pack_spec = {
    .name = "Z80pack",
    .sectors_per_track = 26,
    .tracks = 77,
    .sides = 1,
    .sector_size = 128,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 64,
    .alternate_sides = 0,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 0,
    .has_skew = 1,
    .skew_tab = { 0x00, 0x06, 0x0C, 0x12, 0x18, 0x04, 0x0A, 0x10, 0x16, 0x02, 0x08, 0x0E, 0x14, 0x01, 0x07, 0x0d, 0x13, 0x19, 0x05, 0x0b, 0x11, 0x17, 0x03, 0x09, 0x0f, 0x15 }
};

// CAOS, NANOS, Z1013 CP/M
static disc_spec caos_spec = {
    .name = "CAOS",
    .sectors_per_track = 5,
    .tracks = 80,
    .sides = 2,
    .sector_size = 1024,
    .gap3_length = 0x52,
    .filler_byte = 0xe5,
    .boottracks = 4,
    .alternate_sides = 1,
    .directory_entries = 128,
    .extent_size = 2048,
    .byte_size_extents = 0,
    .first_sector_offset = 1,
};

static disc_spec corvette_spec = {
     .name = "Corvette",
     .sectors_per_track = 5,
     .tracks = 80,
     .sides = 2,
     .sector_size = 1024,
     .gap3_length = 0x2a,   //?
     .filler_byte = 0xe5,
     .boottracks = 1,
     .directory_entries = 128,
     .alternate_sides = 1,
     .extent_size = 2048,
     .byte_size_extents = 0,  //?
     .first_sector_offset = 1,
};

static disc_spec tim011_spec = {
     .name = "Tim011",
     .sectors_per_track = 5,
     .tracks = 80,
     .sides = 2,
     .sector_size = 1024,
     .gap3_length = 0x2a,   
     .filler_byte = 0xe5,
     .boottracks = 2,
     .directory_entries = 256,
     .alternate_sides = 1,
     .extent_size = 2048,
     .byte_size_extents = 0,  
     .first_sector_offset = 1,
};

static disc_spec corvetteBOOT_spec = {
     .name = "Corvette-boot",
     .sectors_per_track = 5,
     .tracks = 80,
     .sides = 2,
     .sector_size = 1024,
     .gap3_length = 0x2a,   //?
     .filler_byte = 0xe5,
     .boottracks = 2,
     .directory_entries = 128,
     .alternate_sides = 1,
     .extent_size = 2048,
     .byte_size_extents = 0,  //?
     .first_sector_offset = 1,
};


static disc_spec nabupc_spec = {
     .name = "Nabu PC",
     .sectors_per_track = 5,
     .tracks = 40,
     .sides = 1,
     .sector_size = 1024,
     .gap3_length = 0x2a,   //?
     .filler_byte = 0xe5,
     .boottracks = 1,
     .directory_entries = 96,
     .alternate_sides = 0,
     .extent_size = 1024,
     .byte_size_extents = 1, 
     .first_sector_offset = 0,
     .has_skew = 1,
     .skew_track_start = 2,
     .skew_tab = { 0, 2, 4, 1, 3 }
};

static disc_spec naburn_spec = {
     .name = "Nabu PC",
     .sectors_per_track = 4,
     .tracks = 16384,
     .sides = 1,
     .sector_size = 128,
     .gap3_length = 0x2a,   //?
     .filler_byte = 0xe5,
     .boottracks = 0,
     .directory_entries = 512,
     .alternate_sides = 0,
     .extent_size = 4096,
     .byte_size_extents = 0, 
     .first_sector_offset = 0,
     .has_skew = 0,
};

static disc_spec nshd8_spec = {
     .name = "Northstar Virtual Disk 8",
     .sectors_per_track = 16,
     .tracks = 1024,
     .sides = 1,
     .sector_size = 512,
     .gap3_length = 0x2a,   //?
     .filler_byte = 0xe5,
     .boottracks = 0,
     .directory_entries = 256,
     .alternate_sides = 0,
     .extent_size = 8192,
     .byte_size_extents = 0,
     .first_sector_offset = 0,
     .has_skew = 0,
};


static disc_spec idpfdd_spec = {
      .name = "Iskra Delta Partner FDD",
      .sectors_per_track = 18,
      .tracks = 73, // 146
      .sides = 2,
      .sector_size = 256,
      .gap3_length = 0x2a, //?
      .filler_byte = 0xe5,
      .boottracks = 2,
      .directory_entries = 128,
      .alternate_sides = 2,
      .extent_size = 2048,
      .byte_size_extents = 0, 
      .first_sector_offset = 0,
      .has_skew = 0
 };

static disc_spec vt180_spec = {
    .name = "VT-180",
    .disk_mode = MFM250,
    .sectors_per_track = 9,
    .tracks = 40,
    .sides = 1,
    .sector_size = 512,
    .gap3_length = 0x2a,
    .filler_byte = 0xe5,
    .boottracks = 2,
    .directory_entries = 64,
    .extent_size = 1024,
    .byte_size_extents = 1,
    .first_sector_offset = 1,
    .has_skew = 1,
    .skew_tab = { 0, 2, 4, 6, 8, 1, 3, 5, 7 }
};

static disc_spec x820_spec = {
     .name = "Xerox820",
     .disk_mode = FM250,
     .sectors_per_track = 18,
     .tracks = 40,
     .sides = 1,
     .sector_size = 128,
     .gap3_length = 0x2a,
     .filler_byte = 0xe5,
     .boottracks = 3,
     .directory_entries = 32,
     .extent_size = 1024,
     .byte_size_extents = 1,
     .first_sector_offset = 1,
     .has_skew = 1,
     .skew_tab = { 0,5,10,15,2,7,12,17,4,9,14,1,6,11,16,3,8,13 }
};


static struct formats {
     const char    *name;
     const char    *description;
     disc_spec  *spec;
     size_t         bootlen; 
     void          *bootsector;
     char           force_com_extension;
     void         (*extra_hook)(disc_handle *handle);
} formats[] = {
    { "actrix",    "Actrix Access",         &actrix_spec, 0, NULL, 1 },
    { "alphatro",  "Alphatronic PC",        &alphatro_spec, 0, NULL, 1 },
    { "altos5",    "Altos 5",               &altos5_spec, 0, NULL, 1 },
    { "altos580",  "Altos 580",             &altos580_spec, 0, NULL, 1 },
    { "ampro",     "Ampro 48tpi",           &ampro_spec, 0, NULL, 1 },
    { "apple2",    "Apple II Softcard",     &apple2_spec, 0, NULL, 1 },
    { "attache",   "Otrona Attache'",       &attache_spec, 0, NULL, 1 },
    { "aussie",    "AussieByte Knight2000", &aussie_spec, 0, NULL, 1 },
    { "bbc",       "BBC Micro Z80CPU SSSD", &bbc_spec, 0, NULL, 1 },
    { "bic",       "BIC / A5105",           &bic_spec, 0, NULL, 1, bic_write_system_file },
    { "bigboard",  "X820/Bigboard, 8in",    &bigboard_spec, 0, NULL, 1 },
    { "bw12",      "Bondwell 12/14",        &bondwell12_spec, 0, NULL, 1 },
    { "bw2",       "Bondwell Model 2",      &bondwell2_spec, 0, NULL, 1 },
    { "caos",      "CAOS/NANOS/z1013 CP/M", &caos_spec, 0, NULL, 1 },
    { "cpcsystem", "CPC System Disc",       &cpcsystem_spec, 0, NULL, 0 },
    { "col1",      "Coleco ADAM 40T SSDD",  &col1_spec, 0, NULL, 1 },
    { "corvette",  "Corvette", &corvette_spec, 32,         "\x80\xc3\x00\xda\x0a\x00\x00\x01\x01\x01\x03\x01\x05\x00\x50\x00\x28\x00\x04\x0f\x00\x8c\x01\x7f\x00\xc0\x00\x20\x00\x01\x00\x11", 1 },
    { "corvboot",  "Corvette Boot", &corvetteBOOT_spec, 32,"\x80\xc3\x00\xda\x0a\x00\x00\x01\x01\x01\x03\x01\x05\x00\x50\x00\x28\x00\x04\x0f\x00\x8a\x01\x7f\x00\xc0\x00\x20\x00\x02\x00\x10", 1 }, // Needs a CP/M bootstrap file specified to auto-boot
    { "dmv",       "NCR Decision Mate",     &dmv_spec, 16, "\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5NCR F3", 1 },
    { "eagle2",    "Eagle II",              &eagle2_spec, 0, NULL, 1 },
    { "einstein",  "Tatung Einstein",       &einstein_spec, 0, NULL, 1 },
    { "excali64",  "Excalibur 64",          &excali_spec, 0, NULL, 1 },
    { "fp1100",    "Casio FP1100",          &fp1100_spec, 0, NULL, 1 },
    { "gemini",    "GeminiGalaxy",          &gemini_spec, 0, NULL, 1 },
    { "hp125",     "HP 125/120",            &hp125_spec, 0, NULL, 1 },
    { "idpfdd",    "Iskra Delta Partner",   &idpfdd_spec, 0, NULL, 1 },
    { "kayproii",  "Kaypro ii",             &kayproii_spec, 0, NULL, 1 },
    { "kaypro4",   "Kaypro 4/10",           &kaypro4_spec,  0, NULL, 1 },
    { "lnw80",     "LNW80 TRS80 Clone",     &lnw80_spec, 0, NULL, 1 },
    { "lynx",      "Camputers Lynx",        &lynx_spec, 0, NULL, 1 },
    { "microbee-ds80",  "Microbee DS80",    &microbee_spec, 0, NULL, 1 },
    { "morrow2",   "Morrow MD 2 (SS)",      &md2_spec, 0, NULL, 1 },
    { "morrow3",   "Morrow MD 3 (DS)",      &md3_spec, 0, NULL, 1 },
    { "mbc1000",   "Sanyo MBC-1000/1150",   &mbc1000_spec, 0, NULL, 1 },
    { "mbc1200",   "Sanyo MBC-200/1250",    &mbc1200_spec, 0, NULL, 1 },
    { "mbc2000",   "Sanyo MBC-2000",        &mbc2000_spec, 0, NULL, 1 },
    { "naburn",    "Nabu PC (8mb)",         &naburn_spec, 0, NULL, 1 },
    { "nabupc",    "Nabu PC",               &nabupc_spec, 0, NULL, 1 },
    { "nascomcpm", "Nascom CPM",            &nascom_spec, 0, NULL, 1 },
    { "nshd8",     "Northstar Virtual 8",   &nshd8_spec, 0, NULL, 1 },
    { "mz2500cpm", "Sharp MZ2500 - CPM",    &mz2500cpm_spec, 0, NULL, 1 },
    { "osborne1",  "Osborne 1 DD",          &osborne_spec, 0, NULL, 1 },
    { "osborne1sd", "Osborne 1 SD",         &osborne_sd_spec, 0, NULL, 1 },
    { "pasopia",   "Toshiba Pasopia/T100",  &pasopia_spec, 0, NULL, 1 },
    { "pc6001",    "NEC PC6001/6601",       &pc6001_spec, 0, NULL, 1 },
    { "pc8001",    "NEC PC8001",            &pc8001_spec, 0, NULL, 1 },
    { "pc88",      "NEC PC8001/8801,FM7/8", &pc88_spec, 0, NULL, 1 },
    { "pcw80",     "Amstrad PCW, 80T",      &pcw80_spec, 16, "\x03\x81\x50\x09\x02\x01\x04\x04\x2A\x52\x00\x00\x00\x00\x00\x00", 1 },
    { "pcw40",     "Amstrad PCW, 40T",      &pcw40_spec, 16, "\x00\x00\x28\x09\x02\x01\x03\x02\x2A\x52\x00\x00\x00\x00\x00\x00", 1 },
    { "plus3",     "Spectrum +3 173k",      &plus3_spec, 0, NULL, 1 },
    { "qc10",      "Epson QC-10, QX-10",    &qc10_spec, 0, NULL, 1 },
    { "rainbow",   "DEC Rainbow 100",       &rainbow_spec, 0, NULL, 1 },
    { "rc700",     "Regnecentralen RC-700", &rc700_spec, 0, NULL, 1 },
    { "sharpx1",   "Sharp X1",              &sharpx1_spec, 0, NULL, 1 },
    { "smc777",    "Sony SMC-70/SMC-777",   &smc777_spec, 0, NULL, 1 },
    { "svi-40ss",  "SVI 40ss (174k)",       &svi40ss_spec, 0, NULL, 1 },
    { "televideo", "Televideo TS80x/TPC1",  &televideo_spec, 0, NULL, 1 },
    { "tiki100ss", "Tiki 100 (200k)",       &tiki100_ss_spec, 0, NULL, 1 },
    { "tiki100ds", "Tiki 100 (400k)",       &tiki100_ds_spec, 0, NULL, 1 },
    { "tim011",    "TIM011 (DSDD 3.5\")",   &tim011_spec, 0, NULL, 1 },
    { "omikron",   "TRS80 I Omikron",       &omikron_spec, 0, NULL, 1 },
    { "lifeboat",  "TRS80 II Lifeboat",     &lifeboat_spec, 0, NULL, 1 },
    { "fmgcpm",    "TRS80 II FMG CP/M",     &fmgcpm_spec, 0, NULL, 1 },
    { "ptcpm",     "TRS80 II PickelsTrout", &ptcpm_spec, 0, NULL, 1 },
    { "holmes",    "TRS80 Holmes VID-80",   &holmes_spec, 0, NULL, 1 },
    { "merchant",  "TRS80 III MemMerchant", &merchant_spec, 0, NULL, 1 },
    { "compactor", "TRS80 III Hurricane C", &hurricane_spec, 0, NULL, 1 },
    { "montezuma", "TRS80 4 Montezuma",     &montezuma_spec, 0, NULL, 1 },
    { "m4cpm3",    "TRS80 4 CP/M Plus",     &trs80_cpm3_spec, 0, NULL, 1 },
    { "vector06c", "Vector 06c",            &vector06c_spec, 0, NULL, 1 },
    { "v1050",     "Visual 1050",           &v1050_spec, 0, NULL, 1 },
    { "vt180",     "DEC VT-180",            &vt180_spec, 0, NULL, 1 },
    { "x820",      "Xerox 820",             &x820_spec, 0, NULL, 1 },
    { "hz89",      "Zenith Z89, Z17-SSSD",  &hz17_spec, 0, NULL, 1 },
    { "hz100",     "Zenith Z100, DSDD",     &z100_spec, 0, NULL, 1 },
    { "magnolia",  "Zenith Z89, magnolia",  &magnolia_spec, 0, NULL, 1 },
    { "z80pack",   "z80pack 8\" format",    &z80pack_spec, 0, NULL, 1 },
    { NULL, NULL }
};

static void dump_formats(void)
{
    struct formats* f = &formats[0];

    printf("Supported CP/M formats:\n\n");

    while (f->name) {
        printf("%-20s%s\n", f->name, f->description);
        printf("%d tracks, %d sectors/track, %d bytes/sector, %d entries, %d bytes/extent\n\n", f->spec->tracks, f->spec->sectors_per_track, f->spec->sector_size, f->spec->directory_entries, f->spec->extent_size);
        f++;
    }

    printf("\nSupported containers:\n\n");
    disc_print_writers(stdout);
    exit(1);
}

// Called for each additional file
static void c_add_file(char *param)
{
    char *colon = strchr(param, ':');
    char  cpm_filename[20];
    char  filename[FILENAME_MAX+1];
    int   i;

    if ( colon == NULL ) {
        // We need to create a CP/M filename from the argument given
        char *basename;

        basename = zbasename(param);
        cpm_create_filename(basename, cpm_filename, 0, 0);
        strcpy(filename, param);
    } else {
        snprintf(filename, sizeof(filename),"%.*s", (int)(colon - param), param);
        snprintf(cpm_filename, sizeof(cpm_filename),"%s",colon+1);
    }

    i = c_additional_files_num;
    c_additional_files_num += 2;
    c_additional_files = realloc(c_additional_files, sizeof(c_additional_files[0]) * c_additional_files_num);
    c_additional_files[i] = strdup(filename);
    c_additional_files[i+1] = strdup(cpm_filename);
}

int cpm2_exec(char* target)
{

    if (help) {
        dump_formats();
        return -1;
    }
    if (c_binary_name == NULL) {
        return -1;
    }
    if (c_disc_format == NULL || c_disc_container == NULL ) {
        dump_formats();
        return -1;
    }

    return cpm_write_file_to_image(c_disc_format, c_disc_container, c_output_file, c_binary_name, c_crt_filename, c_boot_filename);
}


// TODO: Needs bootsector handling
disc_handle *cpm_create_with_format(const char *disc_format) 
{
    disc_spec* spec = NULL;
    struct formats* f = &formats[0];

    while (f->name != NULL) {
        if (strcasecmp(disc_format, f->name) == 0) {
            spec = f->spec;
            break;
        }
        f++;
    }
    if (spec == NULL) {
        return NULL;
    }
    return cpm_create(spec);
}

static void write_extra_files(disc_handle *h)
{
    int     i;
    void   *filebuf;
    FILE   *binary_fp;
    size_t  binlen;

    for ( i = 0; i < c_additional_files_num; i+= 2) {
        // Open the binary file
        if ((binary_fp = fopen(c_additional_files[i], "rb")) == NULL) {
            exit_log(1, "Can't open input file %s\n", c_additional_files[i]);
        }
        if (fseek(binary_fp, 0, SEEK_END)) {
            fclose(binary_fp);
            exit_log(1, "Couldn't determine size of file: %s\n",c_additional_files[i]);
        }
        binlen = ftell(binary_fp);
        fseek(binary_fp, 0L, SEEK_SET);
        filebuf = malloc(binlen);
        if (1 != fread(filebuf, binlen, 1, binary_fp))  { fclose(binary_fp); exit_log(1, "Could not read required data from <%s>\n",c_additional_files[i]); }
        fclose(binary_fp);

        disc_write_file(h, c_additional_files[i+1], filebuf, binlen);
        free(filebuf);
    }

}


int cpm_write_file_to_image(const char *disc_format, const char *container, const char* output_file, const char* binary_name, const char* crt_filename, const char* boot_filename)
{
    disc_spec* spec = NULL;
    struct formats* f = &formats[0];
    const char      *extension;
    disc_writer_func writer = disc_get_writer(container, &extension);
    char disc_name[FILENAME_MAX + 1];
    char cpm_filename[20] = "APP     COM";
    void* filebuf;
    FILE* binary_fp;
    disc_handle* h;
    size_t binlen;

    while (f->name != NULL) {
        if (strcasecmp(disc_format, f->name) == 0) {
            spec = f->spec;
            break;
        }
        f++;
    }
    if (spec == NULL) {
        return -1;
    }

    if (writer == NULL) {
        return -1;
    }


    if (output_file == NULL) {
        strcpy(disc_name, binary_name);
        suffix_change(disc_name, c_extension ? c_extension : extension);
    } else {
        strcpy(disc_name, output_file);
    }
    cpm_create_filename(binary_name, cpm_filename, (f->force_com_extension || c_force_com_extension), 0);

    // Open the binary file
    if ((binary_fp = fopen_bin(binary_name, crt_filename)) == NULL) {
        exit_log(1, "Can't open input file %s\n", binary_name);
    }
    if (fseek(binary_fp, 0, SEEK_END)) {
        fclose(binary_fp);
        exit_log(1, "Couldn't determine size of file\n");
    }
    binlen = ftell(binary_fp);
    fseek(binary_fp, 0L, SEEK_SET);
    filebuf = malloc(binlen);
    if (1 != fread(filebuf, binlen, 1, binary_fp))  { fclose(binary_fp); exit_log(1, "Could not read required data from <%s>\n",binary_name); }
    fclose(binary_fp);

    h = cpm_create(spec);
    if (boot_filename != NULL) {
        size_t bootlen;
        size_t max_bootsize = spec->boottracks * spec->sectors_per_track * spec->sector_size * (spec->alternate_sides + 1);
        if ((binary_fp = fopen(boot_filename, "rb")) != NULL) {
            void* bootbuf;
            if (fseek(binary_fp, 0, SEEK_END)) {
                fclose(binary_fp);
                exit_log(1, "Couldn't determine size of file\n");
            }
            bootlen = ftell(binary_fp);
            fseek(binary_fp, 0L, SEEK_SET);
            if (bootlen > max_bootsize) {
                exit_log(1, "Boot file is too large\n");
            }
            bootbuf = malloc(max_bootsize);
            if (1 != fread(bootbuf, bootlen, 1, binary_fp)) { fclose(binary_fp); exit_log(1, "Could not read required data from <%s>\n",binary_name); }
            fclose(binary_fp);
            disc_write_boot_track(h, bootbuf, bootlen);
            free(bootbuf);
        }
    } else if (f->bootsector) {
        disc_write_boot_track(h, f->bootsector, f->bootlen);
    }

    disc_write_file(h, cpm_filename, filebuf, binlen);

  

    if ( f->extra_hook ) {
        f->extra_hook(h);
    }

    write_extra_files(h);
    
    if (writer(h, disc_name) < 0) {
        exit_log(1, "Can't write disc image\n");
    }
    disc_free(h);

    if ( c_disable_com_file_creation == 0 ) {
        FILE *fpout;
        int   i;

        // Create a .com file alongside the binary so that we have a complete file for copying in other ways
        any_suffix_change(disc_name, ".com", '.');

        if ((fpout = fopen(disc_name, "wb")) == NULL) {
            exit_log(1,"Can't open output file: %s\n",disc_name);
        }

        for (i = 0; i < binlen; i++) {
            writebyte(((unsigned char *)filebuf)[i], fpout);
        }
        fclose(fpout);
    }


    return 0;
}

static void bic_write_system_file(disc_handle *h)
{
    char buf[128] = {0};

    buf[0] = 26; // Soft-EOF

    disc_write_file(h, "SCPX5105SYS", buf, 1);
}

void cpm_create_filename(const char* binary, char* cpm_filename, char force_com_extension, char include_dot)
{
    int count = 0;
    int dest = 0;
    char *ptr;

    ptr = zbasename((char *)binary);

    while (count < 8 && count < strlen(ptr) && ptr[count] != '.') {
        if (ptr[count] > 127) {
            cpm_filename[count] = '_';
        } else {
            cpm_filename[count] = toupper(ptr[count]);
        }
        count++;
    }
    dest = count;

    if ( include_dot ) {
        cpm_filename[dest++] = '.';
    } else {
       while (dest < 8) {
           cpm_filename[dest++] = ' ';
       }
    }
    if (force_com_extension) {
        cpm_filename[dest++] = 'C';
        cpm_filename[dest++] = 'O';
        cpm_filename[dest++] = 'M';
    } else {
        while (count < strlen(ptr) && ptr[count] != '.') {
            count++;
        }
        if (count < strlen(ptr)) {
            while (dest < (12 + include_dot) && count < strlen(ptr)) {
                if (ptr[count] == '.') {
                    count++;
                    continue;
                }
                if (ptr[count] > 127) {
                    cpm_filename[dest] = '_';
                } else {
                    cpm_filename[dest] = toupper(ptr[count]);
                }
                dest++;
                count++;
            }
        }
        if ( !include_dot ) {
            while (dest < 12) {
                cpm_filename[dest++] = ' ';
            }
        }
    }
    cpm_filename[dest++] = 0;
}
