#include <eepp/core/memorymanager.hpp>
#include <eepp/core/string.hpp>
#include <eepp/system/filesystem.hpp>
#include <eepp/system/iostreamfile.hpp>

namespace EE { namespace System {

IOStreamFile* IOStreamFile::New( const std::string& path, const std::string& modes ) {
	return eeNew( IOStreamFile, ( path, modes ) );
}

IOStreamFile::IOStreamFile( const std::string& path, const std::string& modes ) :
	mFS( NULL ), mSize( 0 ) {
	mFS = FileSystem::fopenUtf8( path, modes );
}

IOStreamFile::~IOStreamFile() {
	close();
}

ios_size IOStreamFile::read( char* data, ios_size size ) {
	if ( isOpen() ) {
		return std::fread( data, 1, static_cast<std::size_t>( size ), mFS );
	}

	return 0;
}

ios_size IOStreamFile::write( const char* data, ios_size size ) {
	if ( isOpen() ) {
		std::fwrite( data, 1, size, mFS );
	}

	return size;
}

ios_size IOStreamFile::seek( ios_size position ) {
	if ( isOpen() ) {
		std::fseek( mFS, position, SEEK_SET );
	}

	return position;
}

ios_size IOStreamFile::tell() {
	if ( mFS ) {
		ios_size Pos = std::ftell( mFS );
		return Pos;
	}

	return -1;
}

ios_size IOStreamFile::getSize() {
	if ( isOpen() ) {
		if ( 0 == mSize && mFS ) {
			Int64 position = tell();

			std::fseek( mFS, 0, SEEK_END );

			mSize = tell();

			seek( position );
		}

		return mSize;
	}

	return 0;
}

bool IOStreamFile::isOpen() {
	return NULL != mFS;
}

void IOStreamFile::flush() {
	if ( mFS )
		std::fflush( mFS );
}

void IOStreamFile::close() {
	if ( isOpen() ) {
		std::fclose( mFS );

		mFS = NULL;
	}
}

}} // namespace EE::System
