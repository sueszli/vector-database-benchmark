#ifndef EE_AUDIO_SOUNDFILEREADER_HPP
#define EE_AUDIO_SOUNDFILEREADER_HPP

#include <eepp/config.hpp>
#include <string>

namespace EE { namespace System {
class IOStream;
}} // namespace EE::System

using namespace EE::System;

namespace EE { namespace Audio {

/// \brief Abstract base class for sound file decoding
class EE_API SoundFileReader {
  public:
	/// \brief Structure holding the audio properties of a sound file
	struct Info {
		Uint64 sampleCount;		   ///< Total number of samples in the file
		unsigned int channelCount; ///< Number of channels of the sound
		unsigned int sampleRate;   ///< Samples rate of the sound, in samples per second
	};

	virtual ~SoundFileReader() {}

	////////////////////////////////////////////////////////////
	/// \brief Open a sound file for reading
	///
	/// The provided stream reference is valid as long as the
	/// SoundFileReader is alive, so it is safe to use/store it
	/// during the whole lifetime of the reader.
	///
	/// \param stream Source stream to read from
	/// \param info   Structure to fill with the properties of the loaded sound
	///
	/// \return True if the file was successfully opened
	///
	////////////////////////////////////////////////////////////
	virtual bool open( IOStream& stream, Info& info ) = 0;

	////////////////////////////////////////////////////////////
	/// \brief Change the current read position to the given sample offset
	///
	/// The sample offset takes the channels into account.
	/// If you have a time offset instead, you can easily find
	/// the corresponding sample offset with the following formula:
	/// `timeInSeconds * sampleRate * channelCount`
	/// If the given offset exceeds to total number of samples,
	/// this function must jump to the end of the file.
	///
	/// \param sampleOffset Index of the sample to jump to, relative to the beginning
	///
	////////////////////////////////////////////////////////////
	virtual void seek( Uint64 sampleOffset ) = 0;

	////////////////////////////////////////////////////////////
	/// \brief Read audio samples from the open file
	///
	/// \param samples  Pointer to the sample array to fill
	/// \param maxCount Maximum number of samples to read
	///
	/// \return Number of samples actually read (may be less than \a maxCount)
	///
	////////////////////////////////////////////////////////////
	virtual Uint64 read( Int16* samples, Uint64 maxCount ) = 0;
};

}} // namespace EE::Audio

#endif

////////////////////////////////////////////////////////////
/// @class EE::Audio::SoundFileReader
///
/// This class allows users to read audio file formats not natively
/// supported by EEPP, and thus extend the set of supported readable
/// audio formats.
///
/// A valid sound file reader must override the open, seek and write functions,
/// as well as providing a static check function; the latter is used by
/// EEPP to find a suitable writer for a given input file.
///
/// To register a new reader, use the SoundFileFactory::registerReader
/// template function.
///
/// Usage example:
/// \code
/// class MySoundFileReader : public SoundFileReader
/// {
/// public:
///
///	 static bool check(IOStream& stream)
///	 {
///		 // typically, read the first few header bytes and check fields that identify the format
///		 // return true if the reader can handle the format
///	 }
///
///	 virtual bool open(IOStream& stream, Info& info)
///	 {
///		 // read the sound file header and fill the sound attributes
///		 // (channel count, sample count and sample rate)
///		 // return true on success
///	 }
///
///	 virtual void seek(Uint64 sampleOffset)
///	 {
///		 // advance to the sampleOffset-th sample from the beginning of the sound
///	 }
///
///	 virtual Uint64 read(Int16* samples, Uint64 maxCount)
///	 {
///		 // read up to 'maxCount' samples into the 'samples' array,
///		 // convert them (for example from normalized float) if they are not stored
///		 // as 16-bits signed integers in the file
///		 // return the actual number of samples read
///	 }
/// };
///
/// SoundFileFactory::registerReader<MySoundFileReader>();
/// \endcode
///
/// \see InputSoundFile, SoundFileFactory, SoundFileWriter
///
////////////////////////////////////////////////////////////
