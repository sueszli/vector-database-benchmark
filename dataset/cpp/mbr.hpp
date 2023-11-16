// 
// 

#pragma once
#ifndef FS_MBR_HPP
#define FS_MBR_HPP

#include <string>
#include <cstdint>

namespace fs {

  struct MBR {
    static constexpr int PARTITIONS {4};
  
    struct partition {
      uint8_t  flags;
      uint8_t  CHS_BEG[3];
      uint8_t  type;
      uint8_t  CHS_END[3];
      uint32_t lba_begin;
      uint32_t sectors;
      
      
      uint32_t get_LBA() const noexcept {
        return lba_begin;
      }
      uint32_t get_sectors() const noexcept {
        return sectors;
      }
      
    } __attribute__((packed));
  
    /** Legacy BIOS Parameter Block */
    struct BPB {
      uint16_t bytes_per_sector;
      uint8_t  sectors_per_cluster;
      uint16_t reserved_sectors;
      uint8_t  fa_tables;
      uint16_t root_entries;
      uint16_t small_sectors;
      uint8_t  media_type;       // 0xF8 == hard drive
      uint16_t sectors_per_fat;
      uint16_t sectors_per_track;
      uint16_t num_heads;
      uint32_t hidden_sectors;
      uint32_t large_sectors;    // Used if small_sectors == 0
      uint8_t  disk_number;      // Starts at 0x80
      uint8_t  current_head;
      uint8_t  signature;        // Must be 0x28 or 0x29
      uint32_t serial_number;    // Unique ID created by mkfs
      char     volume_label[11]; // Deprecated
      char     system_id[8];     // FAT12 or FAT16
    } __attribute__((packed));
  
    struct mbr {
      uint8_t   jump[3];
      char      oem_name[8];
      uint8_t   boot[435];       // Boot-code
      partition part[PARTITIONS];
      uint16_t  magic;           // 0xAA55

      inline BPB* bpb() noexcept
      { return reinterpret_cast<BPB*>(boot); }
    } __attribute__((packed));

    static std::string id_to_name(uint8_t);
  }; //< struct MBR

} //< namespace fs

#endif //< FS_MBR_HPP
