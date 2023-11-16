/**
 * @file mpq/mpq_writer.hpp
 *
 * Interface of functions for creating and editing MPQ files.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "mpq/mpq_common.hpp"
#include "utils/logged_fstream.hpp"

namespace devilution {
class MpqWriter {
public:
	explicit MpqWriter(const char *path);
	explicit MpqWriter(const std::string &path)
	    : MpqWriter(path.c_str())
	{
	}
	MpqWriter(MpqWriter &&other) = default;
	MpqWriter &operator=(MpqWriter &&other) = default;
	~MpqWriter();

	bool HasFile(std::string_view name) const;

	void RemoveHashEntry(std::string_view filename);
	void RemoveHashEntries(bool (*fnGetName)(uint8_t, char *));
	bool WriteFile(std::string_view filename, const std::byte *data, size_t size);
	void RenameFile(std::string_view name, std::string_view newName);

private:
	bool IsValidMpqHeader(MpqFileHeader *hdr) const;
	uint32_t GetHashIndex(MpqFileHash fileHash) const;
	uint32_t FetchHandle(std::string_view filename) const;

	bool ReadMPQHeader(MpqFileHeader *hdr);
	MpqBlockEntry *AddFile(std::string_view filename, MpqBlockEntry *block, uint32_t blockIndex);
	bool WriteFileContents(const std::byte *fileData, size_t fileSize, MpqBlockEntry *block);

	// Returns an unused entry in the block entry table.
	MpqBlockEntry *NewBlock(uint32_t *blockIndex = nullptr);

	// Marks space at `blockOffset` of size `blockSize` as free (unused) space.
	void AllocBlock(uint32_t blockOffset, uint32_t blockSize);

	// Returns the file offset that is followed by empty space of at least the given size.
	uint32_t FindFreeBlock(uint32_t size);

	bool WriteHeaderAndTables();
	bool WriteHeader();
	bool WriteBlockTable();
	bool WriteHashTable();
	void InitDefaultMpqHeader(MpqFileHeader *hdr);

	LoggedFStream stream_;
	std::string name_;
	std::uintmax_t size_ {};
	std::unique_ptr<MpqHashEntry[]> hashTable_;
	std::unique_ptr<MpqBlockEntry[]> blockTable_;

// Amiga cannot Seekp beyond EOF.
// See https://github.com/bebbo/libnix/issues/30
#ifndef __AMIGA__
#define CAN_SEEKP_BEYOND_EOF
#endif

#ifndef CAN_SEEKP_BEYOND_EOF
	long streamBegin_;
#endif
};

} // namespace devilution
