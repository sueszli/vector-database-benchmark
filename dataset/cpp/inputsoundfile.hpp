#ifndef EE_AUDIO_INPUTSOUNDFILE_HPP
#define EE_AUDIO_INPUTSOUNDFILE_HPP

#include <algorithm>
#include <eepp/config.hpp>
#include <eepp/core/noncopyable.hpp>
#include <eepp/system/time.hpp>
#include <string>

namespace EE { namespace System {
class IOStream;
}} // namespace EE::System

using namespace EE::System;

namespace EE { namespace Audio {

class SoundFileReader;

/// \brief Provide read access to sound files
class EE_API InputSoundFile : NonCopyable {
  public:
	InputSoundFile();

	~InputSoundFile();

	////////////////////////////////////////////////////////////
	/// \brief Open a sound file from the disk for reading
	///
	/// The supported audio formats are: WAV (PCM only), OGG/Vorbis, FLAC.
	/// The supported sample sizes for FLAC and WAV are 8, 16, 24 and 32 bit.
	///
	/// \param filename Path of the sound file to load
	///
	/// \return True if the file was successfully opened
	///
	////////////////////////////////////////////////////////////
	bool openFromFile( const std::string& filename );

	////////////////////////////////////////////////////////////
	/// \brief Open a sound file in memory for reading
	///
	/// The supported audio formats are: WAV (PCM only), OGG/Vorbis, FLAC.
	/// The supported sample sizes for FLAC and WAV are 8, 16, 24 and 32 bit.
	///
	/// \param data		Pointer to the file data in memory
	/// \param sizeInBytes Size of the data to load, in bytes
	///
	/// \return True if the file was successfully opened
	///
	////////////////////////////////////////////////////////////
	bool openFromMemory( const void* data, std::size_t sizeInBytes );

	////////////////////////////////////////////////////////////
	/// \brief Open a sound file from a custom stream for reading
	///
	/// The supported audio formats are: WAV (PCM only), OGG/Vorbis, FLAC.
	/// The supported sample sizes for FLAC and WAV are 8, 16, 24 and 32 bit.
	///
	/// \param stream Source stream to read from
	///
	/// \return True if the file was successfully opened
	///
	////////////////////////////////////////////////////////////
	bool openFromStream( IOStream& stream );

	////////////////////////////////////////////////////////////
	/// \brief Get the total number of audio samples in the file
	///
	/// \return Number of samples
	///
	////////////////////////////////////////////////////////////
	Uint64 getSampleCount() const;

	////////////////////////////////////////////////////////////
	/// \brief Get the number of channels used by the sound
	///
	/// \return Number of channels (1 = mono, 2 = stereo)
	///
	////////////////////////////////////////////////////////////
	unsigned int getChannelCount() const;

	////////////////////////////////////////////////////////////
	/// \brief Get the sample rate of the sound
	///
	/// \return Sample rate, in samples per second
	///
	////////////////////////////////////////////////////////////
	unsigned int getSampleRate() const;

	////////////////////////////////////////////////////////////
	/// \brief Get the total duration of the sound file
	///
	/// This function is provided for convenience, the duration is
	/// deduced from the other sound file attributes.
	///
	/// \return Duration of the sound file
	///
	////////////////////////////////////////////////////////////
	Time getDuration() const;

	////////////////////////////////////////////////////////////
	/// \brief Get the read offset of the file in time
	///
	/// \return Time position
	///
	////////////////////////////////////////////////////////////
	Time getTimeOffset() const;

	////////////////////////////////////////////////////////////
	/// \brief Get the read offset of the file in samples
	///
	/// \return Sample position
	///
	////////////////////////////////////////////////////////////
	Uint64 getSampleOffset() const;

	////////////////////////////////////////////////////////////
	/// \brief Change the current read position to the given sample offset
	///
	/// This function takes a sample offset to provide maximum
	/// precision. If you need to jump to a given time, use the
	/// other overload.
	///
	/// The sample offset takes the channels into account.
	/// If you have a time offset instead, you can easily find
	/// the corresponding sample offset with the following formula:
	/// `timeInSeconds * sampleRate * channelCount`
	/// If the given offset exceeds to total number of samples,
	/// this function jumps to the end of the sound file.
	///
	/// \param sampleOffset Index of the sample to jump to, relative to the beginning
	///
	////////////////////////////////////////////////////////////
	void seek( Uint64 sampleOffset );

	////////////////////////////////////////////////////////////
	/// \brief Change the current read position to the given time offset
	///
	/// Using a time offset is handy but imprecise. If you need an accurate
	/// result, consider using the overload which takes a sample offset.
	///
	/// If the given time exceeds to total duration, this function jumps
	/// to the end of the sound file.
	///
	/// \param timeOffset Time to jump to, relative to the beginning
	///
	////////////////////////////////////////////////////////////
	void seek( Time timeOffset );

	////////////////////////////////////////////////////////////
	/// \brief Read audio samples from the open file
	///
	/// \param samples  Pointer to the sample array to fill
	/// \param maxCount Maximum number of samples to read
	///
	/// \return Number of samples actually read (may be less than \a maxCount)
	///
	////////////////////////////////////////////////////////////
	Uint64 read( Int16* samples, Uint64 maxCount );

  private:
	////////////////////////////////////////////////////////////
	/// \brief Close the current file
	///
	////////////////////////////////////////////////////////////
	void close();

	////////////////////////////////////////////////////////////
	// Member data
	////////////////////////////////////////////////////////////
	SoundFileReader* mReader;	///< Reader that handles I/O on the file's format
	IOStream* mStream;			///< Input stream used to access the file's data
	bool mStreamOwned;			///< Is the stream internal or external?
	Uint64 mSampleOffset;		///< Sample Read Position
	Uint64 mSampleCount;		///< Total number of samples in the file
	unsigned int mChannelCount; ///< Number of channels of the sound
	unsigned int mSampleRate;	///< Number of samples per second
};

}} // namespace EE::Audio

#endif

////////////////////////////////////////////////////////////
/// @class EE::Audio::InputSoundFile
///
/// This class decodes audio samples from a sound file. It is
/// used internally by higher-level classes such as SoundBuffer
/// and Music, but can also be useful if you want to process
/// or analyze audio files without playing them, or if you want to
/// implement your own version of Music with more specific
/// features.
///
/// Usage example:
/// \code
/// // Open a sound file
/// InputSoundFile file;
/// if (!file.openFromFile("music.ogg"))
///	 /* error */;
///
/// // Print the sound attributes
/// std::cout << "duration: " << file.getDuration().asSeconds() << std::endl;
/// std::cout << "channels: " << file.getChannelCount() << std::endl;
/// std::cout << "sample rate: " << file.getSampleRate() << std::endl;
/// std::cout << "sample count: " << file.getSampleCount() << std::endl;
///
/// // Read and process batches of samples until the end of file is reached
/// Int16 samples[1024];
/// Uint64 count;
/// do
/// {
///	 count = file.read(samples, 1024);
///
///	 // process, analyze, play, convert, or whatever
///	 // you want to do with the samples...
/// }
/// while (count > 0);
/// \endcode
///
/// \see SoundFileReader, OutputSoundFile
///
////////////////////////////////////////////////////////////
