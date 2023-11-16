#include "mpq/mpq_common.hpp"

#include <string_view>

#include <libmpq/mpq.h>

namespace devilution {

#if !defined(UNPACKED_MPQS) || !defined(UNPACKED_SAVES)
MpqFileHash CalculateMpqFileHash(std::string_view filename)
{
	MpqFileHash fileHash;
	libmpq__file_hash_s(filename.data(), filename.size(), &fileHash[0], &fileHash[1], &fileHash[2]);
	return fileHash;
}
#endif

} // namespace devilution
