
// disk_r.c

#include <kernel.h>

/*
 * __load_sequential_sectors:
 *    Carrega uma qunatidade de setores, sequencialmente.
 */
// Podemos usar isso para carregarmos o diretório raiz
// a fat, metafiles ou banco de dados.
// #bugbug
// Aqui estamos falando de uma sequência de setores.
// Isso serve para carregar o diretório raiz em fat16.
// Mas nao server para carregar subdiretorios.
// #todo
// Create __read_sequential_sectors and 
// __write_sequential_sectors
// global worker.
// #todo
// Change the return value:
// Use TRUE or FALSE.

int
__load_sequential_sectors ( 
    unsigned long address, 
    unsigned long lba, 
    unsigned long sectors )
{
    unsigned long i=0;
    unsigned long b=0;
    unsigned long next_lba=0;

    // debug_print ("__load_sequential_sectors:\n");

    for ( i=0; i < sectors; i++ )
    {
        next_lba = (unsigned long) (lba + i);
        ataReadSector ( address + b, next_lba, 0, 0 );
        b = (b +512);
    };

    return 0;
}

/*
 * fatClustToSect:
 *     Calcula qual é o setor inicial de um dado cluster.
 *     Informações para o calculo: 
 *     ~ Número do cluster.
 *     ~ Número de setores por cluster.
 *     ~ Lba inicial da area de dados.
 */
 
unsigned long 
fatClustToSect ( 
    unsigned short cluster, 
    unsigned long spc, 
    unsigned long first_data_sector )
{
    unsigned long C = (unsigned long) cluster;
    C -= 2;
// #todo: 
// Check limits.
    return (unsigned long) (C * spc) + first_data_sector;
}

/*
 * read_lba:
 *     Carrega um um setor na memória, dado o LBA.
 *     Obs: 
 *     Talvez essa rotina tenha que ter algum retorno no caso de falhas. 
 */
// #bugbug
// Essa rotina e' independente do sistema de arquivos.
// Change name to dest_buffer
// #bugbug
// Disk info?
// Qual é o disco?
// Qual é a porta IDE?
// ...
// #todo
// use 'buffer_va'.

void 
read_lba ( 
    unsigned long address, 
    unsigned long lba )
{
// #todo
// Fazer algum filtro de argumentos?

    // if ( address == 0 ){}

// See: volume.h

    switch (g_currentvolume_fatbits){

    case 32:
        debug_print ("read_lba: [FAIL] FAT32 not supported\n");
        return;
        break;

    // atahdd.c
    case 16:
        //#todo: return value.
        //#todo: IN: buffer,lba,?,?,
        ataReadSector ( address, lba, 0, 0 );
        return;
        break;

    // Nothing.
    case 12:
        debug_print ("read_lba: [FAIL] FAT12 not supported\n");
        return;
        break;

    default:
        debug_print ("read_lba: [FAIL] g_currentvolume_fatbits not supported\n");
        break;
    };
}

/*
 * fatLoadCluster:
 *     Carrega um cluster.
 *     Argumentos:
 *         setor   ~ Primeiro setor do cluster.
 *         address ~ Endereço do primeiro setor do cluster.
 *         spc     ~ Número de setores por cluster.
 * Começa do primeiro setor do cluster.
 */
 
void 
fatLoadCluster ( 
    unsigned long sector, 
    unsigned long address, 
    unsigned long spc )
{
    unsigned long i=0;
    unsigned long SectorSize = 512;  //#todo: via argument.
    for ( i=0; i<spc; i++ )
    {
        read_lba( address, sector + i );
        address = (address + SectorSize); 
    };
}

// Load metafile
void 
fs_load_metafile (
    unsigned long buffer, 
    unsigned long first_lba, 
    unsigned long size )
{
    unsigned long SizeInSectors = (size & 0xFFFFFFFF);

    debug_print ("fs_load_metafile:\n");

    if (buffer == 0){
        debug_print ("fs_load_metafile: [ERROR] buffer\n");
        return;
    }
    if (size == 0){
        debug_print ("fs_load_metafile: [ERROR] size\n");
        return;
    }

    __load_sequential_sectors ( 
        buffer, 
        first_lba, 
        SizeInSectors );
}

void fs_load_mbr(unsigned long mbr_address)
{
    unsigned long MBR_Address=0;
    unsigned long MBR_Lba=0;
    size_t        MBR_Size=0;  // In sectors.

    MBR_Address = mbr_address;
    MBR_Lba     = MBR_LBA;  // LBA = 0.
    MBR_Size    = 1;        // Number of sectors = 1.

    debug_print ("fs_load_rootdir:\n");

    __load_sequential_sectors ( 
        MBR_Address, 
        MBR_Lba, 
        MBR_Size );
}

/*
 * fs_load_fat:
 *    Carrega a fat na memória.
 *    Sistema de arquivos fat16.
 *    ? qual disco ?
 *    ? qual volume ? 
 *    #obs: Essa rotina poderia carregar a fat do volume atual do 
 * disco atual. É só checar na estrutura.
 * current disk, current volume, fat status.
 * + Se o status da fat para o vulume atual indicar que 
 * ela já está carregada, então não precisamos carregar novamente.
 */
// #todo
// Precisamos de uma estrutura com as informações sobre
// a FAT atual.

void 
fs_load_fat( 
    unsigned long fat_address, 
    unsigned long fat_lba, 
    size_t fat_size )
{

// #todo
// We need a structure to track the state
// of the table in the memory.
// maybe 'FATState'

    unsigned long __fatAddress=0;
    unsigned long __fatLBA=0;
    size_t        __fatSizeInSectors=0;

    __fatAddress       = fat_address;
    __fatLBA           = fat_lba;
    __fatSizeInSectors = (fat_size & 0xFFFF);   //fat size in sectors. 246?

    debug_print ("fs_load_fat:\n");

//
// Check cache state.
//
    
    // Se ja está na memória, então não precisamos carregar novamente.
    if (g_fat_cache_loaded == FAT_CACHE_LOADED){
         debug_print("fs_load_fat: FAT cache already loaded!\n");
         return;
    }

    //__load_sequential_sectors ( 
    //    VOLUME1_FAT_ADDRESS, 
    //    VOLUME1_FAT_LBA, 
    //    128 );

    __load_sequential_sectors ( 
        __fatAddress, 
        __fatLBA, 
        __fatSizeInSectors );

// Changing the status
    g_fat_cache_loaded = FAT_CACHE_LOADED;
}

/*
 * fs_load_rootdir:
 */
// Carrega o diretório raiz.
// address, lba, number of sectors.
// #todo
// Precisamos de uma estrutura com as informações sobre
// o diretório raiz atual.
// O ponteiro para essa estrutura sera salvo na estrutura de processo
// juntamente com o ponteiro da estrutura de cwd.

void 
fs_load_rootdir(
    unsigned long root_address, 
    unsigned long root_lba, 
    size_t root_size )  // in sectors.
{

// #todo
// We need a structure to track the state
// of the directory in the memory.
// maybe 'RootdirState'

    unsigned long RootAddress=0;
    unsigned long RootLBA=0;
    size_t        RootSize=0;

    RootAddress = root_address;
    RootLBA     = root_lba;
    RootSize    = root_size;    // number of sectors.

    debug_print ("fs_load_rootdir:\n");

    __load_sequential_sectors ( 
        RootAddress, 
        RootLBA, 
        RootSize );
}

//
// End
//

