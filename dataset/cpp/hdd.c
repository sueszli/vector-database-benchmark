/*
 * File: hdd.c 
 * Low level routines to read and write a sector into/from the disk.
 * Two functions are used in this document.
 * >> my_read_hd_sector() is called by read_lba().
 * >> my_write_hd_sector() is called by write_lba().
 * read_lba() and write_lba are called by fs.c.
 * History:
 *    2013 - Created by Fred Nora.
 */


/*
 hd info:
 =======
 PIIX3 ATA: 
 LUN#0: disk, PCHS=963/4/17, total number of sectors 65536. (Oracle Virtualbox)
 estatistica de hoje:
 (apenas leitura, usando PIO mode)
00:01:59.902737 /Devices/IDE0/ATA0/Unit0/AtapiDMA            0 times
00:01:59.902742 /Devices/IDE0/ATA0/Unit0/AtapiPIO            0 times
00:01:59.902747 /Devices/IDE0/ATA0/Unit0/DMA                 0 times
00:01:59.902753 /Devices/IDE0/ATA0/Unit0/PIO              1699 times  <<
00:01:59.902760 /Devices/IDE0/ATA0/Unit0/ReadBytes      869376 bytes  <<
00:01:59.902766 /Devices/IDE0/ATA0/Unit0/WrittenBytes        0 bytes
 */ 


#include <bootloader.h>


/*
 * Externs
 */
 
extern void os_read_sector();
extern void os_write_sector();
extern void reset_ide0();

// Usadas por read e write.
extern unsigned long hd_buffer;
extern unsigned long hd_lba;

// Internal
int hddStatus=0;
int hddError=0;
//...

// ------------------------------------------------------------

// #bugbug
// inline is not good

static void hdd_ata_pio_read ( 
    int p, 
    void *buffer, 
    int bytes )
{
    asm volatile (\
        "cld;\
         rep; insw":: "D" (buffer),\
         "d" ( ide_ports[p].base_port + 0 ),\
          "c" (bytes/2));
}


// #bugbug
// inline is not good

void 
hdd_ata_pio_write ( 
    int p, 
    void *buffer, 
    int bytes )
{
    asm volatile (\
                "cld;\
                rep; outsw"::"S"(buffer),\
                "d"( ide_ports[p].base_port + 0 ),\
                "c"(bytes/2));
}

uint8_t hdd_ata_status_read(int p)
{
// #bugbug: 
//rever o offset

    //return inb(ata[p].cmd_block_base_addr + ATA_REG_STATUS);

    return (uint8_t) in8( (int) ide_ports[p].base_port + 7 );
}

int hdd_ata_wait_not_busy(int p)
{
    while ( hdd_ata_status_read(p) & ATA_SR_BSY )
        if ( hdd_ata_status_read(p) & ATA_SR_ERR )
            return 1;

    return 0;
}

void hdd_ata_cmd_write ( int port, int cmd_val )
{

// no_busy 
    hdd_ata_wait_not_busy(port);

    //outb(ata.cmd_block_base_address + ATA_REG_CMD,cmd_val);

    out8 ( (int) ide_ports[port].base_port + 7 , (int) cmd_val );
    ata_wait (400);  
}

int hdd_ata_wait_no_drq (int p)
{
    while ( hdd_ata_status_read(p) &ATA_SR_DRQ)
        if (hdd_ata_status_read(p) &ATA_SR_ERR)
            return 1;

    return 0;
}

/*
 * pio_rw_sector
 * IN:
 *     buffer - Buffer address
 *     lba    - LBA number 
 *     rw     - Flag read or write.
 *   //inline unsigned char inportb (int port)
 *   //out8 ( int port, int data )
 *   (IDE PIO)
 */
int 
pio_rw_sector ( 
    unsigned long buffer, 
    unsigned long lba, 
    int rw, 
    int port,
    int slave )
{
    unsigned long tmplba = (unsigned long) lba;

// #bugbug
// s� funcionaram as portas 0 e 2.
// para primary e secondary.

    if ( port < 0 || port >= 4 )
    {
        // #todo: Message
        return -1;
    }

// Selecionar se � master ou slave.
//  outb (0x1F6, slavebit<<4)
// 0 - 3    In CHS addressing, bits 0 to 3 of the head. 
//          In LBA addressing, bits 24 to 27 of the block number
// 4  DRV  Selects the drive number.
// 5  1	Always set.
// 6  LBA Uses CHS addressing if clear or LBA addressing if set.
// 7  1 Always set.
// 0x01F6; 
// Port to send drive and bit 24 - 27 of LBA

    tmplba = (unsigned long) (tmplba >> 24);

// no bit 4.
// 0 = master | 1 = slave

// master. bit 4 = 0
    if (slave == 0){
        tmplba = (unsigned long)(tmplba | 0x000000E0);    //1110 0000b;
    }

// slave. bit 4 = 1
    if (slave == 1){
        tmplba = (unsigned long)(tmplba | 0x000000F0);    //1111 0000b;
    }

// In 32bit machine
// int and long has the same size.

    out8( 
        (int) ( ide_ports[port].base_port + 6 ), 
        (int) tmplba );

// #test
    //out8( 
        // (int) ide_ports[port].base_port + 6, 
        // (int) 0xE0 | (master << 4) | ((tmplba >> 24) & 0x0F));
 
// 0x01F2
// Port to send number of sectors.

    out8( 
        (int) ( ide_ports[port].base_port + 2 ), 
        (int) 1 );

// 0x1F3  
// Port to send bit 0 - 7 of LBA.

    tmplba = lba;
    tmplba = tmplba & 0x000000FF;
    out8( (int) ide_ports[port].base_port + 3 , (int) tmplba );

// 0x1F4
// Port to send bit 8 - 15 of LBA.

    tmplba = lba;
    tmplba = tmplba >> 8;
    tmplba = tmplba & 0x000000FF;
    out8( (int) ide_ports[port].base_port + 4 , (int) tmplba );

// 0x1F5:
// Port to send bit 16 - 23 of LBA

    tmplba = lba;
    tmplba = tmplba >> 16;
    tmplba = tmplba & 0x000000FF;
    out8( (int) ide_ports[port].base_port + 5 , (int) tmplba );

// 0x1F7:
// Command port
// rw

    rw = rw & 0x000000FF;
    out8( (int) ide_ports[port].base_port + 7 , (int) rw );

// PIO or DMA ??
// If the command is going to use DMA, set the Features Register to 1, otherwise 0 for PIO.
    // outb (0x1F1, isDMA)

// timeout sim, n�o podemos esperar para sempre.
// #todo
// Colocar essas declara��es no in�cio da fun��o.

    unsigned char c=0;
    unsigned long timeout = (4444*512);

again:

    c = (unsigned char) in8( (int) ide_ports[port].base_port + 7);

// Select a bit.
    c = (c & 8);

    if (c == 0)
    {
        timeout--;
        if (timeout == 0)
        {
            printf("pio_rw_sector: [FAIL] rw sector timeout\n");
            return -3;
        }

        // #bugbug: 
        // Isso pode enrroscar aqui.

        goto again;
    }

//
// read or write.
//

    switch (rw){

        // read
        case 0x20:
            hdd_ata_pio_read ( 
                (int)    port, 
                (void *) buffer, 
                (int)    512 );
            return 0;
            break;

        // write
        case 0x30:
 
            hdd_ata_pio_write ( 
                (int)    port, 
                (void *) buffer, 
                (int)    512 );

            //Flush Cache
            hdd_ata_cmd_write( (int) port, (int) ATA_CMD_FLUSH_CACHE );
            hdd_ata_wait_not_busy(port);
            if ( hdd_ata_wait_no_drq(port) != 0)
            {
                // #todo: Message.
                return -1;
            }
            return 0;
            break;

        // fail
        default:
            printf ("pio_rw_sector: fail *hang");
            die();
            break;
    };

    return 0;
}

/*
 * my_read_hd_sector:
 * eax - buffer
 * ebx - lba
 * ecx - null
 * edx - null
 */
void 
my_read_hd_sector ( 
    unsigned long ax, 
    unsigned long bx, 
    unsigned long cx, 
    unsigned long dx )
{

//====================== WARNING ==============================
// #IMPORTANTE:
// #todo
// So falta conseguirmos as variaveis que indicam o canal e 
// se eh master ou slave.

// IN:
// (buffer, lba, rw flag, port number, master )

    pio_rw_sector ( 
        (unsigned long) ax, 
        (unsigned long) bx, 
        (int) 0x20, 
        (int) g_current_ide_channel,     // 0
        (int) g_current_ide_device );    // 1

/*
//antigo.

    // Passando os argumentos.	
	hd_buffer = (unsigned long) ax;    //arg1 = buffer. 
	hd_lba = (unsigned long) bx;       //arg2 = lba.

	// Read sector. (ASM)
	os_read_sector();

	//#todo: deletar esse return.
	//testar sem ele antes.
*/

}

/*
 * my_write_hd_sector:
 * eax - buffer
 * ebx - lba
 * ecx - null
 * edx - null
 */

void 
my_write_hd_sector ( 
    unsigned long ax, 
    unsigned long bx, 
    unsigned long cx, 
    unsigned long dx )
{

// =========================== WARNING ==============================
// #IMPORTANTE:
// #todo
// So falta conseguirmos as variaveis que indicam o canal e 
// se eh master ou slave.

// #bugbug:
// a rotina de salvar um arquivo invocada pelo shell 
// apresentou problemas. Estamos testando ...

// read test (buffer, lba, rw flag, port number )
    // pio_rw_sector ( (unsigned long) ax, (unsigned long) bx, (int) 0x30, (int) 0 );

    pio_rw_sector ( 
        (unsigned long) ax, 
        (unsigned long) bx, 
        (int) 0x30, 
        (int) g_current_ide_channel,   //0 
        (int) g_current_ide_device );  //1

/*
// Antigo.
// Passando os argumentos.
	hd_buffer = (unsigned long) ax;    //arg1 = buffer. 
	hd_lba = (unsigned long) bx;       //arg2 = lba.

	// Write sector. (ASM)
    // entry/x86/head/hwlib.inc

	os_write_sector(); 
*/

}

/*
 * read_lba: 
 * Read a LBA from the disk.
 */
void read_lba( unsigned long address, unsigned long lba )
{
// Called by fsLoadFile in fs.c
    // if ( address == 0 ){}
    my_read_hd_sector ( address, lba, 0, 0 );
}

/*
 * write_lba: 
 * Write a LBA into the disk.
 */
void write_lba ( unsigned long address, unsigned long lba )
{
    // if ( address == 0 ){}
    my_write_hd_sector ( address, lba, 0, 0 );
}


/*
 * init_hdd:
 *     Inicializa o driver de hd.
 */
// Called by OS_Loader_Main in main.c.
int init_hdd(void)
{

// #todo: 
// We need to do something here. haha

// See: ide.c
    diskATADialog( 1, FORCEPIO, FORCEPIO );
    g_driver_hdd_initialized = (int) TRUE;

    return 0;
}

//
// End
//

