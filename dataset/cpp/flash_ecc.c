/************************************************************************************//**
* \file         Source/HCS12/flash_ecc.c
* \brief        Bootloader flash driver source file for HCS12 derivatives with error
*               correction code in flash memory, such as the HCS12Pxx. This flash memory
*               uses a different addressing scheme than other HCS12 derivatives.
* \ingroup      Target_HCS12
* \internal
*----------------------------------------------------------------------------------------
*                          C O P Y R I G H T
*----------------------------------------------------------------------------------------
*   Copyright (c) 2013  by Feaser    http://www.feaser.com    All rights reserved
*
*----------------------------------------------------------------------------------------
*                            L I C E N S E
*----------------------------------------------------------------------------------------
* This file is part of OpenBLT. OpenBLT is free software: you can redistribute it and/or
* modify it under the terms of the GNU General Public License as published by the Free
* Software Foundation, either version 3 of the License, or (at your option) any later
* version.
*
* OpenBLT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE. See the GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License along with OpenBLT. It
* should be located in ".\Doc\license.html". If not, contact Feaser to obtain a copy.
*
* \endinternal
****************************************************************************************/

/****************************************************************************************
* Include files
****************************************************************************************/
#include "boot.h"                                /* bootloader generic header          */


/****************************************************************************************
* Macro definitions
****************************************************************************************/
/** \brief Value for an invalid flash sector. */
#define FLASH_INVALID_SECTOR_IDX        (0xff)
/** \brief Value for an invalid flash address. */
#define FLASH_INVALID_ADDRESS           (0xffffffff)
/** \brief Standard size of a flash block for writing. */
#define FLASH_WRITE_BLOCK_SIZE          (512)
/** \brief Total numbers of sectors in array flashLayout[]. */
#define FLASH_TOTAL_SECTORS             (sizeof(flashLayout)/sizeof(flashLayout[0]))
#define FLASH_LAST_SECTOR_IDX           (FLASH_TOTAL_SECTORS-1)
#define FLASH_ERASE_BLOCK_SIZE          (256)
/** \brief Offset into the user program's vector table where the checksum is located.
 *         Note that the value can be overriden in blt_conf.h, because the size of the 
 *         vector table could vary. When changing this value, don't forget to update the
 *         location of the checksum in the user program accordingly. Otherwise the 
 *         checksum verification will always fail.
 */
#ifndef BOOT_FLASH_VECTOR_TABLE_CS_OFFSET
#define BOOT_FLASH_VECTOR_TABLE_CS_OFFSET (0x82)
#endif
/** \brief Total size of the vector table, excluding the bootloader specific checksum. */
#define FLASH_VECTOR_TABLE_SIZE         (0x80)
/** \brief Start address of the bootloader programmable flash. */
#define FLASH_START_ADDRESS             (flashLayout[0].sector_start)
/** \brief End address of the bootloader programmable flash. */
#define FLASH_END_ADDRESS               (flashLayout[FLASH_LAST_SECTOR_IDX].sector_start + \
                                         flashLayout[FLASH_LAST_SECTOR_IDX].sector_size - 1)
/** \brief Size of a flash page on the HCS12. */
#define FLASH_PAGE_SIZE                 (0x4000)      /* flash page size in bytes      */
/** \brief Physical start address of the HCS12 page window. */
#define FLASH_PAGE_OFFSET               (0x8000)      /* physical start addr. of pages */
/** \brief PPAGE register to select a specific flash page. */
#define FLASH_PPAGE_REG                 (*(volatile blt_int8u *)(0x0015))
/** \brief Base address of the flash related control registers. */
#define FLASH_REGS_BASE_ADDRESS         (0x0100)
/** \brief Macro for accessing the flash related control registers. */
#define FLASH                           ((volatile tFlashRegs *)FLASH_REGS_BASE_ADDRESS)
/** \brief Bitmask for flash clock divider bits. */
#define FLASH_FDIV_MASK                 (0x3f)
/** \brief Invalid value for the flash clock divider bits. */
#define FLASH_FDIV_INVALID              (0xff)
/** \brief Maximum number of flash command parameters. */
#define FLASH_CMD_MAX_PARAMS            (4)
/** \brief A phrase is an aligned group of 4 16-bit words, so 8 bytes. */
#define FLASH_PHRASE_SIZE               (8)
/** \brief Erase sector flash command. */
#define FLASH_ERASE_SECTOR_CMD          (0x0A)
/** \brief Program phrase flash command. */
#define FLASH_PROGRAM_PHRASE_CMD        (0x06)


/****************************************************************************************
* Register definitions
****************************************************************************************/
/** \brief FSTAT - command complete irg flag bit. */
#define CCIF_BIT       (0x80)
/** \brief FCLKDIV - clock divider loaded bit. */
#define FDIVLD_BIT     (0x80)
/** \brief FSTAT - flash access error flag bit. */
#define ACCERR_BIT     (0x20)
/** \brief FSTAT - flash protection violation flag bit. */
#define FPVIOL_BIT     (0x10)


/****************************************************************************************
* Plausibility checks
****************************************************************************************/
#if (BOOT_FLASH_VECTOR_TABLE_CS_OFFSET >= FLASH_WRITE_BLOCK_SIZE)
#error "BOOT_FLASH_VECTOR_TABLE_CS_OFFSET is set too high. It must be located in the first writable block."
#endif

#ifndef BOOT_FLASH_CUSTOM_LAYOUT_ENABLE
#define BOOT_FLASH_CUSTOM_LAYOUT_ENABLE (0u)
#endif


/****************************************************************************************
* Type definitions
****************************************************************************************/
/** \brief Structure type for the flash sectors in the flash layout table. */
typedef struct
{
  blt_addr   sector_start;                       /**< sector start address             */
  blt_int32u sector_size;                        /**< sector size in bytes             */
} tFlashSector;

/** \brief    Structure type for grouping flash block information.
 *  \details  Programming is done per block of max FLASH_WRITE_BLOCK_SIZE. for this a
 *            flash block manager is implemented in this driver. this flash block manager
 *            depends on this flash block info structure. It holds the base address of
 *            the flash block and the data that should be programmed into the flash
 *            block. The .base_addr must be a multiple of FLASH_WRITE_BLOCK_SIZE.
 */
typedef struct
{
  blt_addr  base_addr;
  blt_int8u data[FLASH_WRITE_BLOCK_SIZE];
} tFlashBlockInfo;

/** \brief Structure type for the flash control registers. */
typedef volatile struct
{
  volatile blt_int8u  fclkdiv;                   /**< flash clock devider register     */
  volatile blt_int8u  fsec;                      /**< flash security register          */
  volatile blt_int8u  fccobix;                   /**< flash CCOB index register        */
  volatile blt_int8u  frsv0;                     /**< flash reserver register          */
  volatile blt_int8u  fcnfg;                     /**< flash configuration register     */
  volatile blt_int8u  fercnfg;                   /**< flash error configuration reg.   */
  volatile blt_int8u  fstat;                     /**< flash status register            */
  volatile blt_int8u  ferstat;                   /**< flash error status register      */
  volatile blt_int8u  fprot;                     /**< program flash protection register*/
  volatile blt_int8u  dfprot;                    /**< data flash protection register   */
  volatile blt_int16u fccob;                     /**< flash command common object reg. */
  volatile blt_int8u  frsv1;                     /**< flash reserver register          */
  volatile blt_int8u  frsv2;                     /**< flash reserver register          */
  volatile blt_int8u  frsv3;                     /**< flash reserver register          */
  volatile blt_int8u  frsv4;                     /**< flash reserver register          */
  volatile blt_int8u  fopt;                      /**< flash option register            */
  volatile blt_int8u  frsv5;                     /**< flash reserver register          */
  volatile blt_int8u  frsv6;                     /**< flash reserver register          */
  volatile blt_int8u  frsv7;                     /**< flash reserver register          */
} tFlashRegs;

/** \brief Pointer type to flash command execution function. */
typedef void (*pFlashExeCmdFct)(void);

/** \brief Mapping table for finding the corect flash clock divider prescaler. */
typedef struct
{
  blt_int16u sysclock_min;                       /**< min busclock for this prescaler  */
  blt_int16u sysclock_max;                       /**< max busclock for this prescaler  */
  blt_int8u  prescaler;                          /**< prescaler for this busclock range*/
} tFlashPrescalerSysclockMapping;


/****************************************************************************************
* Function prototypes
****************************************************************************************/
static blt_bool   FlashInitBlock(tFlashBlockInfo *block, blt_addr address);
static tFlashBlockInfo *FlashSwitchBlock(tFlashBlockInfo *block, blt_addr base_addr);
static blt_bool   FlashAddToBlock(tFlashBlockInfo *block, blt_addr address,
                                  blt_int8u *data, blt_int32u len);
static blt_bool   FlashWriteBlock(tFlashBlockInfo *block);
static blt_int8u  FlashGetGlobalAddrByte(blt_addr addr);
static blt_int8u  FlashGetPhysPage(blt_addr addr);
static blt_int16u FlashGetPhysAddr(blt_addr addr);
static void       FlashExecuteCommand(void);
static blt_bool   FlashOperate(blt_int8u cmd, blt_addr addr, blt_int16u params[],
                               blt_int8u param_count);


/****************************************************************************************
* Local constant declarations
****************************************************************************************/
/** \brief   If desired, it is possible to set BOOT_FLASH_CUSTOM_LAYOUT_ENABLE to > 0
 *           in blt_conf.h and then implement your own version of the flashLayout[] table
 *           in a source-file with the name flash_layout.c. This way you customize the
 *           flash memory size reserved for the bootloader, without having to modify
 *           the flashLayout[] table in this file directly. This file will then include
 *           flash_layout.c so there is no need to compile it additionally with your
 *           project.
 */
#if (BOOT_FLASH_CUSTOM_LAYOUT_ENABLE == 0)
/** \brief   Array wit the layout of the flash memory.
 *  \details Also controls what part of the flash memory is reserved for the bootloader.
 *           If the bootloader size changes, the reserved sectors for the bootloader
 *           might need adjustment to make sure the bootloader doesn't get overwritten.
 *           This layout uses global addresses only. Note that the last part is where the
 *           bootloader also resides and it has been entered as 8 chunks of 2kb.
 *           This allows flexibility for reserving more/less space for the bootloader in
 *           case its size changes in the future.
 */
static const tFlashSector flashLayout[] =
{
#if (BOOT_NVM_SIZE_KB > 128)
#error "BOOT_NVM_SIZE_KB > 128 is currently not supported."
#endif
#if (BOOT_NVM_SIZE_KB >= 128)
  { 0x20000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x21000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x22000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x23000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x24000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x25000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x26000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x27000, 0x1000 },                  /* flash sector - 4kb                          */
#endif
#if (BOOT_NVM_SIZE_KB >= 96)
  { 0x28000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x29000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x2A000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x2B000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x2C000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x2D000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x2E000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x2F000, 0x1000 },                  /* flash sector - 4kb                          */
#endif
#if (BOOT_NVM_SIZE_KB >= 64)
  { 0x30000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x31000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x32000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x33000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x34000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x35000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x36000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x37000, 0x1000 },                  /* flash sector - 4kb                          */
#endif
  { 0x38000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x39000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x3A000, 0x1000 },                  /* flash sector - 4kb                          */
  { 0x3B000, 0x1000 },                  /* flash sector - 4kb                          */

  { 0x3C000, 0x0800 },                  /* flash sector - 2kb                          */
  { 0x3C800, 0x0800 },                  /* flash sector - 2kb                          */
  { 0x3D000, 0x0800 },                  /* flash sector - 2kb                          */
  { 0x3D800, 0x0800 },                  /* flash sector - 2kb                          */
  { 0x3E000, 0x0800 },                  /* flash sector - 2kb                          */
  /* { 0x3E800, 0x0800 },                  flash sector - 2kb (reserved for bootloader)*/
  /* { 0x3F000, 0x0800 },                  flash sector - 2kb (reserved for bootloader)*/
  /* { 0x3F800, 0x0800 },                  flash sector - 2kb (reserved for bootloader)*/
};


/** \brief   Lookup table for determining the flash clock divider setting based on the
 *           system clock speed. The flash clock must be around 1MHz and is scaled down
 *           from the system clock using a prescaler value. Note that clock speeds in
 *           the table are in kHz.
 */
static const tFlashPrescalerSysclockMapping flashFDIVlookup[] =
{
  {  1000,  1600, 0x00 },                 /* FDIV[5:0] = prescaler = 0x00                */
  {  1600,  2600, 0x01 },                 /* FDIV[5:0] = prescaler = 0x01                */
  {  2600,  3600, 0x02 },                 /* FDIV[5:0] = prescaler = 0x02                */
  {  3600,  4600, 0x03 },                 /* FDIV[5:0] = prescaler = 0x03                */
  {  4600,  5600, 0x04 },                 /* FDIV[5:0] = prescaler = 0x04                */
  {  5600,  6600, 0x05 },                 /* FDIV[5:0] = prescaler = 0x05                */
  {  6600,  7600, 0x06 },                 /* FDIV[5:0] = prescaler = 0x06                */
  {  7600,  8600, 0x07 },                 /* FDIV[5:0] = prescaler = 0x07                */
  {  8600,  9600, 0x08 },                 /* FDIV[5:0] = prescaler = 0x08                */
  {  9600, 10600, 0x09 },                 /* FDIV[5:0] = prescaler = 0x09                */
  { 10600, 11600, 0x0A },                 /* FDIV[5:0] = prescaler = 0x0A                */
  { 11600, 12600, 0x0B },                 /* FDIV[5:0] = prescaler = 0x0B                */
  { 12600, 13600, 0x0C },                 /* FDIV[5:0] = prescaler = 0x0C                */
  { 13600, 14600, 0x0D },                 /* FDIV[5:0] = prescaler = 0x0D                */
  { 14600, 15600, 0x0E },                 /* FDIV[5:0] = prescaler = 0x0E                */
  { 15600, 16600, 0x0F },                 /* FDIV[5:0] = prescaler = 0x0F                */
  { 16600, 17600, 0x10 },                 /* FDIV[5:0] = prescaler = 0x10                */
  { 17600, 18600, 0x11 },                 /* FDIV[5:0] = prescaler = 0x11                */
  { 18600, 19600, 0x12 },                 /* FDIV[5:0] = prescaler = 0x12                */
  { 19600, 20600, 0x13 },                 /* FDIV[5:0] = prescaler = 0x13                */
  { 20600, 21600, 0x14 },                 /* FDIV[5:0] = prescaler = 0x14                */
  { 21600, 22600, 0x15 },                 /* FDIV[5:0] = prescaler = 0x15                */
  { 22600, 23600, 0x16 },                 /* FDIV[5:0] = prescaler = 0x16                */
  { 23600, 24600, 0x17 },                 /* FDIV[5:0] = prescaler = 0x17                */
  { 24600, 25600, 0x18 },                 /* FDIV[5:0] = prescaler = 0x18                */
  { 25600, 26600, 0x19 },                 /* FDIV[5:0] = prescaler = 0x19                */
  { 26600, 27600, 0x1A },                 /* FDIV[5:0] = prescaler = 0x1A                */
  { 27600, 28600, 0x1B },                 /* FDIV[5:0] = prescaler = 0x1B                */
  { 28600, 29600, 0x1C },                 /* FDIV[5:0] = prescaler = 0x1C                */
  { 29600, 30600, 0x1D },                 /* FDIV[5:0] = prescaler = 0x1D                */
  { 30600, 31600, 0x1E },                 /* FDIV[5:0] = prescaler = 0x1E                */
  { 31600, 32600, 0x1F }                  /* FDIV[5:0] = prescaler = 0x1F                */
};
#else
#include "flash_layout.c"
#endif /* BOOT_FLASH_CUSTOM_LAYOUT_ENABLE == 0 */


/** \brief  Array with executable code for performing flash operations.
 *  \details This array contains the machine code to perform the actual command on the
 *           flash device, such as program or erase. the code is compiler and location
 *           independent. This allows us to copy it to a ram buffer and execute the code
 *           from ram. This way the flash driver can be located in flash memory without
 *           running into problems when erasing/programming the same flash block that
 *           contains the flash driver. the source code for the machine code is as
 *           follows:
 *             // launch the command
 *             FLASH->fstat = CCIF_BIT;
 *             // wait at least 4 cycles (per AN2720)
 *             asm("nop");
 *             asm("nop");
 *             asm("nop");
 *             asm("nop");
 *             // wait for command to complete
 *             while ((FLASH->fstat & CCIF_BIT) != CCIF_BIT);
 */
static const blt_int8u flashExecCmd[] =
{
  /* asm("psha");              backup A  */
  0x36,
  /* asm("pshx");              backup X */
  0x34,
  /* asm("ldx #0x100");        load flash register base in X */
  0xce, 0x01, 0x00,
  /* asm("leax 6,x");          point X to FSTAT register */
  0x1a, 0x06,
  /* asm("ldaa #0x80");        load CCIF mask in A */
  0x86, 0x80,
  /* asm("staa 0,x");          set CCIF bit in FSTAT to launch the command */
  0x6a, 0x00,
  /* asm("nop"); [4 times]     wait at least 4 cycles */
  0xa7,0xa7, 0xa7, 0xa7,
  /* asm("brclr 0,x,#0x80,*"); wait for command completion: CCIF in FSTAT equals 1 */
  0x0f, 0x00, 0x80, 0xfc,
  /* asm("pulx");              restore X */
  0x30,
  /* asm("pula");              restore A */
  0x32,
  /* asm("rts");               return */
  0x3d
};


/****************************************************************************************
* Local data declarations
****************************************************************************************/
/** \brief   Local variable with information about the flash block that is currently
 *           being operated on.
 *  \details The smallest amount of flash that can be programmed is
 *           FLASH_WRITE_BLOCK_SIZE. A flash block manager is implemented in this driver
 *           and stores info in this variable. Whenever new data should be flashed, it
 *           is first added to a RAM buffer, which is part of this variable. Whenever
 *           the RAM buffer, which has the size of a flash block, is full or  data needs
 *           to be written to a different block, the contents of the RAM buffer are
 *           programmed to flash. The flash block manager requires some software
 *           overhead, yet results is faster flash programming because data is first
 *           harvested, ideally until there is enough to program an entire flash block,
 *           before the flash device is actually operated on.
 */
static tFlashBlockInfo blockInfo;

/** \brief   Local variable with information about the flash boot block.
 *  \details The first block of the user program holds the vector table, which on the
 *           STM32 is also the where the checksum is written to. Is it likely that
 *           the vector table is first flashed and then, at the end of the programming
 *           sequence, the checksum. This means that this flash block need to be written
 *           to twice. Normally this is not a problem with flash memory, as long as you
 *           write the same values to those bytes that are not supposed to be changed
 *           and the locations where you do write to are still in the erased 0xFF state.
 *           Unfortunately, writing twice to flash this way, does not work reliably on
 *           all micros. This is why we need to have an extra block, the bootblock,
 *           placed under the management of the block manager. This way is it possible
 *           to implement functionality so that the bootblock is only written to once
 *           at the end of the programming sequence.
 */
static tFlashBlockInfo bootBlockInfo;

/** \brief RAM buffer where the executable flash operation code is copied to. */
static blt_int8u flashExecCmdRam[(sizeof(flashExecCmd)/sizeof(flashExecCmd[0]))];


/************************************************************************************//**
** \brief     Initializes the flash driver.
** \return    none.
**
****************************************************************************************/
void FlashInit(void)
{
  blt_int8u fdiv_bits = FLASH_FDIV_INVALID;
  blt_int8u cnt;

  /* init the flash block info structs by setting the address to an invalid address */
  blockInfo.base_addr = FLASH_INVALID_ADDRESS;
  bootBlockInfo.base_addr = FLASH_INVALID_ADDRESS;

  /* try to find correct flash clock divider setting using the lookup table */
  for (cnt=0; cnt<(sizeof(flashFDIVlookup)/sizeof(flashFDIVlookup[0])); cnt++)
  {
    if ((BOOT_CPU_SYSTEM_SPEED_KHZ > flashFDIVlookup[cnt].sysclock_min) &&
        (BOOT_CPU_SYSTEM_SPEED_KHZ <= flashFDIVlookup[cnt].sysclock_max))
    {
      /* matching configuration found in the lookup table so store it */
      fdiv_bits = flashFDIVlookup[cnt].prescaler;
      /* lookup completed so no need to continue searching */
      break;
    }
  }

  /* make sure that a valid configuration was found */
  ASSERT_RT(fdiv_bits != FLASH_FDIV_INVALID);

  /* wait until all flash commands are finished */
  while ((FLASH->fstat & CCIF_BIT) == 0)
  {
    ;
  }

  /* reset the clock divider bits and then write the new configuration */
  FLASH->fclkdiv &= ~FLASH_FDIV_MASK;
  FLASH->fclkdiv |= fdiv_bits;

  /* double check that the configuration was correctly stored because access to these
   * clock divider bits can be locked.
   */
  ASSERT_RT((FLASH->fclkdiv & FLASH_FDIV_MASK) == fdiv_bits);

  /* double check that the flash module registered the clock configuration otherwise
   * subsequent flash commands will fail.
   */
  ASSERT_RT((FLASH->fclkdiv & FDIVLD_BIT) != 0);
} /*** end of FlashInit ***/


/************************************************************************************//**
** \brief     Reinitializes the flash driver.
** \return    none.
**
****************************************************************************************/
void FlashReinit(void)
{
  /* init the flash block info structs by setting the address to an invalid address */
  blockInfo.base_addr = FLASH_INVALID_ADDRESS;
  bootBlockInfo.base_addr = FLASH_INVALID_ADDRESS;
} /*** end of FlashReinit ***/


/************************************************************************************//**
** \brief     Writes the data to flash through a flash block manager. Note that this
**            function also checks that no data is programmed outside the flash
**            memory region, so the bootloader can never be overwritten.
** \param     addr Start address.
** \param     len  Length in bytes.
** \param     data Pointer to the data buffer.
** \return    BLT_TRUE if successful, BLT_FALSE otherwise.
**
****************************************************************************************/
blt_bool FlashWrite(blt_addr addr, blt_int32u len, blt_int8u *data)
{
  blt_addr base_addr;
  blt_addr last_block_base_addr;

  /* make sure the addresses are within the flash device */
  if ((addr < FLASH_START_ADDRESS) || ((addr+len-1) > FLASH_END_ADDRESS))
  {
    return BLT_FALSE;
  }

  /* determine the start address of the last block in flash */
  last_block_base_addr = flashLayout[FLASH_LAST_SECTOR_IDX].sector_start + \
                         flashLayout[FLASH_LAST_SECTOR_IDX].sector_size -  \
                         FLASH_WRITE_BLOCK_SIZE;

  /* if this is the bootblock, then let the boot block manager handle it */
  base_addr = (addr/FLASH_WRITE_BLOCK_SIZE)*FLASH_WRITE_BLOCK_SIZE;
  if (base_addr == last_block_base_addr)
  {
    /* let the boot block manager handle it */
    return FlashAddToBlock(&bootBlockInfo, addr, data, len);
  }
  /* let the block manager handle it */
  return FlashAddToBlock(&blockInfo, addr, data, len);
} /*** end of FlashWrite ***/


/************************************************************************************//**
** \brief     Erases the flash memory. Note that this function also checks that no
**            data is erased outside the flash memory region, so the bootloader can
**            never be erased.
** \param     addr Start address.
** \param     len  Length in bytes.
** \return    BLT_TRUE if successful, BLT_FALSE otherwise.
**
****************************************************************************************/
blt_bool FlashErase(blt_addr addr, blt_int32u len)
{
  blt_addr erase_base_addr;
  blt_int16u nr_of_erase_blocks;
  blt_int32u total_erase_len;
  blt_int16u block_cnt;

  /* determine the base address for the erase operation, by aligning to
   * FLASH_ERASE_BLOCK_SIZE.
   */
  erase_base_addr = (addr/FLASH_ERASE_BLOCK_SIZE)*FLASH_ERASE_BLOCK_SIZE;

  /* make sure the addresses are within the flash device */
  if ((erase_base_addr < FLASH_START_ADDRESS) || ((addr+len-1) > FLASH_END_ADDRESS))
  {
    return BLT_FALSE;
  }

  /* determine number of bytes to erase from base address */
  total_erase_len = len + (addr - erase_base_addr);

  /* determine the number of blocks to erase */
  nr_of_erase_blocks = (blt_int16u)(total_erase_len / FLASH_ERASE_BLOCK_SIZE);
  if ((total_erase_len % FLASH_ERASE_BLOCK_SIZE) > 0)
  {
    nr_of_erase_blocks++;
  }

  /* erase all blocks one by one */
  for (block_cnt=0; block_cnt<nr_of_erase_blocks; block_cnt++)
  {
    /* keep the watchdog happy */
    CopService();

    /* erase the block */
    if (FlashOperate(FLASH_ERASE_SECTOR_CMD, erase_base_addr, BLT_NULL, 0) == BLT_FALSE)
    {
      /* error occurred */
      return BLT_FALSE;
    }

    /* point to the next block's base address */
    erase_base_addr += FLASH_ERASE_BLOCK_SIZE;
  }

  /* erase successful */
  return BLT_TRUE;
} /*** end of FlashErase ***/


/************************************************************************************//**
** \brief     Writes a checksum of the user program to non-volatile memory. This is
**            performed once the entire user program has been programmed. Through
**            the checksum, the bootloader can check if the programming session
**            was completed, which indicates that a valid user programming is
**            present and can be started.
** \return    BLT_TRUE if successful, BLT_FALSE otherwise.
**
****************************************************************************************/
blt_bool FlashWriteChecksum(void)
{
  blt_int16u signature_checksum = 0;
  blt_int8u  byte_counter;
  blt_int16u vectab_offset;
  blt_addr   checksum_address;

  /* for the HCS12 target we defined the checksum as the 16-bit Two's complement value
   * of the sum of all the 64 interrupt vector addresses, so basically a checksum over
   * the contents of the entire user program interrupt vector table.
   *
   * the bootloader writes this 16-bit checksum value right after the vector table
   * of the user program. note that this means one extra dummy entry must be added
   * at the end of the user program's vector table to reserve storage space for the
   * checksum.
   */

  /* first check that the bootblock contains valid data. if not, this means the
   * bootblock is not part of the reprogramming this time and therefore no
   * new checksum needs to be written
   */
  if (bootBlockInfo.base_addr == FLASH_INVALID_ADDRESS)
  {
    return BLT_TRUE;
  }

  /* the bootblock contains the data for the last sector in flashLayout. the
   * user program vector table and the checkum will be located at the end
   * of this block. first determine the offset in the bootblock data to
   * reach the start of the vector table.
   */
  vectab_offset = FLASH_WRITE_BLOCK_SIZE - FLASH_VECTOR_TABLE_SIZE;

  /* compute the checksum. note that the user program's vectors are not yet written
   * to flash but are present in the bootblock data structure at this point.
   */
  for (byte_counter=0; byte_counter<FLASH_VECTOR_TABLE_SIZE; byte_counter++)
  {

    signature_checksum += bootBlockInfo.data[vectab_offset + byte_counter];
  }
  signature_checksum  = ~signature_checksum; /* one's complement */
  signature_checksum += 1; /* two's complement */

  /* write the checksum */
  checksum_address = flashLayout[FLASH_LAST_SECTOR_IDX].sector_start + \
                     flashLayout[FLASH_LAST_SECTOR_IDX].sector_size -  \
                     BOOT_FLASH_VECTOR_TABLE_CS_OFFSET;
  return FlashWrite(checksum_address, sizeof(signature_checksum),
                    (blt_int8u *)&signature_checksum);
} /*** end of FlashWriteChecksum ***/


/************************************************************************************//**
** \brief     Verifies the checksum, which indicates that a valid user program is
**            present and can be started.
** \return    BLT_TRUE if successful, BLT_FALSE otherwise.
**
****************************************************************************************/
blt_bool FlashVerifyChecksum(void)
{
  blt_int16u signature_checksum = 0;
  blt_int8u  byte_counter;
  blt_addr   checksum_addr_glob;
  blt_addr   vector_table_addr_glob;

  /* get global address of the checksum */
  checksum_addr_glob = (flashLayout[FLASH_LAST_SECTOR_IDX].sector_start + \
                        flashLayout[FLASH_LAST_SECTOR_IDX].sector_size -  \
                        BOOT_FLASH_VECTOR_TABLE_CS_OFFSET);
  /* get global address of the vector table start */
  vector_table_addr_glob = (flashLayout[FLASH_LAST_SECTOR_IDX].sector_start + \
                            flashLayout[FLASH_LAST_SECTOR_IDX].sector_size -  \
                            FLASH_VECTOR_TABLE_SIZE);
  /* compute the checksum based on how it was written by FlashWriteChecksum() */
  for (byte_counter=0; byte_counter<FLASH_VECTOR_TABLE_SIZE; byte_counter++)
  {
    signature_checksum += FlashGetGlobalAddrByte(vector_table_addr_glob + byte_counter);
  }
  /* add the 16-bit checksum value */
  signature_checksum += (((blt_int16u)FlashGetGlobalAddrByte(checksum_addr_glob) << 8) +
                         FlashGetGlobalAddrByte(checksum_addr_glob + 1));
  /* sum should add up to an unsigned 16-bit value of 0 */
  if (signature_checksum == 0)
  {
    /* checksum okay */
    return BLT_TRUE;
  }
  /* checksum incorrect */
  return BLT_FALSE;
} /*** end of FlashVerifyChecksum ***/


/************************************************************************************//**
** \brief     Finalizes the flash driver operations. There could still be data in
**            the currently active block that needs to be flashed.
** \return    BLT_TRUE if successful, BLT_FALSE otherwise.
**
****************************************************************************************/
blt_bool FlashDone(void)
{
  blt_int8u cnt;

  /* check if there is still data waiting to be programmed in the boot block */
  if (bootBlockInfo.base_addr != FLASH_INVALID_ADDRESS)
  {
    if (FlashWriteBlock(&bootBlockInfo) == BLT_FALSE)
    {
      return BLT_FALSE;
    }
  }

  /* check if there is still data waiting to be programmed */
  if (blockInfo.base_addr != FLASH_INVALID_ADDRESS)
  {
    if (FlashWriteBlock(&blockInfo) == BLT_FALSE)
    {
      return BLT_FALSE;
    }
  }

  /* flash operations complete, so clear the RAM buffer with operation execution code */
  for (cnt=0; cnt<(sizeof(flashExecCmd)/sizeof(flashExecCmd[0])); cnt++)
  {
    flashExecCmdRam[cnt] = 0;
  }

  /* still here so all is okay */
  return BLT_TRUE;
} /*** end of FlashDone ***/


/************************************************************************************//**
** \brief     Obtains the base address of the flash memory available to the user program.
**            This is basically the last address in the flashLayout table converted to
**            the physical address on the last page, because this is where the address 
**            will be in.
** \return    Base address.
**
****************************************************************************************/
blt_addr FlashGetUserProgBaseAddress(void)
{
  blt_addr end_address_global;
  blt_addr end_address_physical_page_window;
  blt_addr end_address_physical_page_last;

  end_address_global = FLASH_END_ADDRESS + 1;
  end_address_physical_page_window = FlashGetPhysAddr(end_address_global);
  end_address_physical_page_last = end_address_physical_page_window + FLASH_PAGE_SIZE;
  return end_address_physical_page_last;
} /*** end of FlashGetUserProgBaseAddress ***/


/************************************************************************************//**
** \brief     Copies data currently in flash to the block->data and sets the
**            base address.
** \param     block   Pointer to flash block info structure to operate on.
** \param     address Base address of the block data.
** \return    BLT_TRUE if successful, BLT_FALSE otherwise.
**
****************************************************************************************/
static blt_bool FlashInitBlock(tFlashBlockInfo *block, blt_addr address)
{
  blt_int8u oldPage;

  /* check address alignment */
  if ((address % FLASH_WRITE_BLOCK_SIZE) != 0)
  {
    return BLT_FALSE;
  }
  /* make sure that we are initializing a new block and not the same one */
  if (block->base_addr == address)
  {
    /* block already initialized, so nothing to do */
    return BLT_TRUE;
  }
  /* set the base address */
  block->base_addr = address;

  /* backup originally selected page */
  oldPage = FLASH_PPAGE_REG;
  /* select correct page */
  FLASH_PPAGE_REG = FlashGetPhysPage(address);
  /* copy the current data from flash */
  CpuMemCopy((blt_addr)block->data, (blt_addr)FlashGetPhysAddr(address), FLASH_WRITE_BLOCK_SIZE);
  /* restore originally selected page */
  FLASH_PPAGE_REG = oldPage;

  return BLT_TRUE;
} /*** end of FlashInitBlock ***/


/************************************************************************************//**
** \brief     Switches blocks by programming the current one and initializing the
**            next.
** \param     block   Pointer to flash block info structure to operate on.
** \param     base_addr Base address of the next block.
** \return    The pointer of the block info struct that is no being used, or a NULL
**            pointer in case of error.
**
****************************************************************************************/
static tFlashBlockInfo *FlashSwitchBlock(tFlashBlockInfo *block, blt_addr base_addr)
{
  blt_addr last_block_base_addr;

  /* get base address of the last write block. this is the base address of the boot
   * block on this target.
   */
  last_block_base_addr = flashLayout[FLASH_LAST_SECTOR_IDX].sector_start + \
                         flashLayout[FLASH_LAST_SECTOR_IDX].sector_size -  \
                         FLASH_WRITE_BLOCK_SIZE;

  /* check if a switch needs to be made away from the boot block. in this case the boot
   * block shouldn't be written yet, because this is done at the end of the programming
   * session by FlashDone(), this is right after the checksum was written.
   */
  if (block == &bootBlockInfo)
  {
    /* switch from the boot block to the generic block info structure */
    block = &blockInfo;
  }
  /* check if a switch back into the bootblock is needed. in this case the generic block
   * doesn't need to be written here yet.
   */
  else if (base_addr == last_block_base_addr)
  {
    /* switch from the generic block to the boot block info structure */
    block = &bootBlockInfo;
  }
  else
  {
    /* need to switch to a new block, so program the current one and init the next */
    if (FlashWriteBlock(block) == BLT_FALSE)
    {
      return BLT_NULL;
    }
  }

  /* initialize tne new block when necessary */
  if (FlashInitBlock(block, base_addr) == BLT_FALSE)
  {
    return BLT_NULL;
  }

  /* still here to all is okay  */
  return block;
} /*** end of FlashSwitchBlock ***/


/************************************************************************************//**
** \brief     Programming is done per block. This function adds data to the block
**            that is currently collecting data to be written to flash. If the
**            address is outside of the current block, the current block is written
**            to flash an a new block is initialized.
** \param     block   Pointer to flash block info structure to operate on.
** \param     address Flash destination address.
** \param     data    Pointer to the byte array with data.
** \param     len     Number of bytes to add to the block.
** \return    BLT_TRUE if successful, BLT_FALSE otherwise.
**
****************************************************************************************/
static blt_bool FlashAddToBlock(tFlashBlockInfo *block, blt_addr address,
                                blt_int8u *data, blt_int32u len)
{
  blt_addr   current_base_addr;
  blt_int8u  *dst;
  blt_int8u  *src;

  /* determine the current base address */
  current_base_addr = (address/FLASH_WRITE_BLOCK_SIZE)*FLASH_WRITE_BLOCK_SIZE;

  /* make sure the blockInfo is not uninitialized */
  if (block->base_addr == FLASH_INVALID_ADDRESS)
  {
    /* initialize the blockInfo struct for the current block */
    if (FlashInitBlock(block, current_base_addr) == BLT_FALSE)
    {
      return BLT_FALSE;
    }
  }

  /* check if the new data fits in the current block */
  if (block->base_addr != current_base_addr)
  {
    /* need to switch to a new block, so program the current one and init the next */
    block = FlashSwitchBlock(block, current_base_addr);
    if (block == BLT_NULL)
    {
      return BLT_FALSE;
    }
  }

  /* add the data to the current block, but check for block overflow */
  dst = &(block->data[address - block->base_addr]);
  src = data;
  do
  {
    /* keep the watchdog happy */
    CopService();
    /* buffer overflow? */
    if ((blt_addr)(dst-&(block->data[0])) >= FLASH_WRITE_BLOCK_SIZE)
    {
      /* need to switch to a new block, so program the current one and init the next */
      block = FlashSwitchBlock(block, current_base_addr+FLASH_WRITE_BLOCK_SIZE);
      if (block == BLT_NULL)
      {
        return BLT_FALSE;
      }
      /* reset destination pointer */
      dst = &(block->data[0]);
    }
    /* write the data to the buffer */
    *dst = *src;
    /* update pointers */
    dst++;
    src++;
    /* decrement byte counter */
    len--;
  }
  while (len > 0);
  /* still here so all is good */
  return BLT_TRUE;
} /*** end of FlashAddToBlock ***/


/************************************************************************************//**
** \brief     Programs FLASH_WRITE_BLOCK_SIZE bytes to flash from the block->data
**            array.
** \param     block   Pointer to flash block info structure to operate on.
** \return    BLT_TRUE if successful, BLT_FALSE otherwise.
**
****************************************************************************************/
static blt_bool FlashWriteBlock(tFlashBlockInfo *block)
{
  blt_bool   result = BLT_TRUE;
  blt_bool   cmd_result;
  blt_int16u phrase_cnt;
  blt_addr   phrase_addr;
  blt_int8u  phrase_data[FLASH_PHRASE_SIZE];
  blt_int8u  byte_cnt;

  /* make sure the blockInfo is not uninitialized */
  if (block->base_addr == FLASH_INVALID_ADDRESS)
  {
    return BLT_FALSE;
  }

  /* program all phrases in the block one by one */
  for (phrase_cnt=0; phrase_cnt<(FLASH_WRITE_BLOCK_SIZE/FLASH_PHRASE_SIZE); phrase_cnt++)
  {
    /* determine the starting address of the phrase */
    phrase_addr = block->base_addr + (phrase_cnt * FLASH_PHRASE_SIZE);
    /* copy the phrase data to a buffer */
    for (byte_cnt=0; byte_cnt<FLASH_PHRASE_SIZE; byte_cnt++)
    {
      phrase_data[byte_cnt] = block->data[(phrase_cnt * FLASH_PHRASE_SIZE) + byte_cnt];
    }

    /* keep the watchdog happy */
    CopService();

    /* program the phrase to flash */
    cmd_result = FlashOperate(FLASH_PROGRAM_PHRASE_CMD, phrase_addr,
                              (unsigned short *)phrase_data, 4);
    if (cmd_result == BLT_FALSE)
    {
      /* error occurred */
      result = BLT_FALSE;
      break;
    }

    /* verify that the written data is actually there */
    for (byte_cnt=0; byte_cnt<FLASH_PHRASE_SIZE; byte_cnt++)
    {
      if (FlashGetGlobalAddrByte(phrase_addr+byte_cnt) != phrase_data[byte_cnt])
      {
        /* write verification occurred */
        result = BLT_FALSE;
        break;
      }
    }
  }

  /* still here so all is okay */
  return result;
} /*** end of FlashWriteBlock ***/


/************************************************************************************//**
** \brief     Reads the byte value from the linear address.
** \param     addr Linear address.
** \return    The byte value located at the linear address.
**
****************************************************************************************/
static blt_int8u FlashGetGlobalAddrByte(blt_addr addr)
{
  blt_int8u oldPage;
  blt_int8u result;

  /* backup originally selected page */
  oldPage = FLASH_PPAGE_REG;

  /* select correct page */
  FLASH_PPAGE_REG = FlashGetPhysPage(addr);

  /* read the byte value from the page address */
  result = *((blt_int8u *)FlashGetPhysAddr(addr));

  /* restore originally selected page */
  FLASH_PPAGE_REG = oldPage;

  /* return the read byte value */
  return result;
} /*** end of FlashGetGlobalAddrByte ***/


/************************************************************************************//**
** \brief     Extracts the physical flash page number from a linear address.
** \param     addr Linear address.
** \return    The page number.
**
****************************************************************************************/
static blt_int8u FlashGetPhysPage(blt_addr addr)
{
  return (blt_int8u)(addr / FLASH_PAGE_SIZE);
} /*** end of FlashGetPhysPage ***/


/************************************************************************************//**
** \brief     Extracts the physical address on the flash page number from a
**            linear address.
** \param     addr Linear address.
** \return    The physical address.
**
****************************************************************************************/
static blt_int16u FlashGetPhysAddr(blt_addr addr)
{
  return (blt_int16u)(((blt_int16u)addr % FLASH_PAGE_SIZE) + FLASH_PAGE_OFFSET);
} /*** end of FlashGetPhysAddr ***/


/************************************************************************************//**
** \brief     Executes the command. The actual code for the command execution is
**            stored as location independant machine code in array flashExecCmd[].
**            The contents of this array are temporarily copied to RAM. This way the
**            function can be executed from RAM avoiding problem when try to perform
**            a flash operation on the same flash block that this driver is located on.
** \return    none.
**
****************************************************************************************/
static void FlashExecuteCommand(void)
{
  /* pointer to command execution function */
  pFlashExeCmdFct pExecCommandFct;
  blt_int8u cnt;

  /* copy code for command execution to ram buffer */
  for (cnt=0; cnt<(sizeof(flashExecCmd)/sizeof(flashExecCmd[0])); cnt++)
  {
    flashExecCmdRam[cnt] = flashExecCmd[cnt];
  }

  /* init the function pointer */
  pExecCommandFct = (pFlashExeCmdFct)((void *)flashExecCmdRam);
  /* call the command execution function */
  pExecCommandFct();
} /*** end of FlashExecuteCommand ***/


/************************************************************************************//**
** \brief     Prepares the flash command and executes it.
** \param     cmd  Command to be launched.
** \param     addr Global address to operate on (18-bit).
** \param     params Array with additional command parameters.
** \param     param_count Number of parameters in the array.
** \return    BLT_TRUE if operation was successful, otherwise BLT_FALSE.
**
****************************************************************************************/
static blt_bool FlashOperate(blt_int8u cmd, blt_addr addr, blt_int16u params[], blt_int8u param_count)
{
  blt_bool  result;
  blt_int8u idx;

  /* set default result to error */
  result = BLT_FALSE;

  /* make sure the number of flash parameters is within bounds */
  ASSERT_RT(param_count <= FLASH_CMD_MAX_PARAMS);

  /* it is not expected that flash commands are being executed now, just make sure this
   * is really the case.
   */
  if ((FLASH->fstat & CCIF_BIT) == 0)
  {
    return result;
  }

  /* clear possibly pending error flags from the previous command */
  FLASH->fstat = (FPVIOL_BIT | ACCERR_BIT);

  /* write the command and the address to operate on */
  FLASH->fccobix = 0;
  FLASH->fccob = (cmd << 8) | (((blt_int8u)(addr >> 16)) & 0x03);
  FLASH->fccobix = 1;
  FLASH->fccob = (blt_int16u)addr;

  /* write command paramaters one by one */
  for (idx=0; idx<param_count; idx++)
  {
    /* select the correct command parameter index */
    FLASH->fccobix = idx + 2;
    /* write the command parameter for this index */
    FLASH->fccob = params[idx];
  }

  /* execute the command */
  FlashExecuteCommand();

  /* check the results */
  if ((FLASH->fstat & (ACCERR_BIT | FPVIOL_BIT)) == 0)
  {
    /* operation was successful */
    result = BLT_TRUE;
  }

  return result;
} /*** end of FlashOperate ***/


/*********************************** end of flash.c ************************************/
