// HARFANG(R) Copyright (C) 2022 NWNC. Released under GPL/LGPL/Commercial Licence, see licence.txt for details.

#include "foundation/file_rw_interface.h"
#include "foundation/file.h"

namespace hg {

const Reader g_file_reader = {
	[](Handle hnd, void *data, size_t size) ->size_t { return Read(reinterpret_cast<File &>(hnd), data, size); },
	[](Handle hnd) -> size_t { return GetSize(reinterpret_cast<File &>(hnd)); },
	[](Handle hnd, ptrdiff_t offset, SeekMode mode) -> bool { return Seek(reinterpret_cast<File &>(hnd), offset, mode); },
	[](Handle hnd) -> size_t { return Tell(reinterpret_cast<File &>(hnd)); },
	[](Handle hnd) -> bool { return reinterpret_cast<File &>(hnd).ref != invalid_gen_ref; },
	[](Handle hnd) -> bool { return IsEOF(reinterpret_cast<File &>(hnd)); },
};

const Writer g_file_writer = {
	[](Handle hnd, const void *data, size_t size) -> size_t { return Write(reinterpret_cast<File &>(hnd), data, size); },
	[](Handle hnd, ptrdiff_t offset, SeekMode mode) -> bool { return Seek(reinterpret_cast<File &>(hnd), offset, mode); },
	[](Handle hnd) -> size_t { return Tell(reinterpret_cast<File &>(hnd)); },
	[](Handle hnd) -> bool { return reinterpret_cast<File &>(hnd).ref != invalid_gen_ref; },
};

const ReadProvider g_file_read_provider = {
	[](const char *path, bool silent) -> Handle {
		Handle hnd;
		reinterpret_cast<File &>(hnd) = Open(path, silent);
		return hnd;
	},
	[](Handle hnd) { Close(reinterpret_cast<File &>(hnd)); },
	[](const char *path) -> bool { return IsFile(path); },
};

const WriteProvider g_file_write_provider = {
	[](const char *path) -> Handle {
		Handle hnd;
		reinterpret_cast<File &>(hnd) = OpenWrite(path);
		return hnd;
	},
	[](Handle hnd) { Close(reinterpret_cast<File &>(hnd)); },
};

} // namespace hg
