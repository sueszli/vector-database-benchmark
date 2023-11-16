/*
 * Copyright (C) 2001 Carlos Hasan
 * Copyright (C) 2001 François Revol
 * Copyright (C) 2001 Axel Dörfler
 * Copyright (C) 2004 Marcus Overhagen
 * Copyright (C) 2009 Stephan Amßus <superstippi@gmx.de>
 * Copyright (C) 2014 Colin Günther <coling@gmx.de>
 * Copyright (C) 2015 Adrien Destugues <pulkomandy@pulkomandy.tk>
 *
 * All rights reserved. Distributed under the terms of the MIT License.
 */

//! libavcodec based decoder for Haiku


#include "AVCodecDecoder.h"

#include <new>

#include <assert.h>
#include <string.h>

#include <Bitmap.h>
#include <Debug.h>
#include <String.h>

#include "Utilities.h"


#undef TRACE
//#define TRACE_AV_CODEC
#ifdef TRACE_AV_CODEC
#	define TRACE(x...)	printf(x)
#	define TRACE_AUDIO(x...)	printf(x)
#	define TRACE_VIDEO(x...)	printf(x)
#else
#	define TRACE(x...)
#	define TRACE_AUDIO(x...)
#	define TRACE_VIDEO(x...)
#endif

//#define LOG_STREAM_TO_FILE
#ifdef LOG_STREAM_TO_FILE
#	include <File.h>
	static BFile sAudioStreamLogFile(
		"/boot/home/Desktop/AVCodecDebugAudioStream.raw",
		B_CREATE_FILE | B_ERASE_FILE | B_WRITE_ONLY);
	static BFile sVideoStreamLogFile(
		"/boot/home/Desktop/AVCodecDebugVideoStream.raw",
		B_CREATE_FILE | B_ERASE_FILE | B_WRITE_ONLY);
	static int sDumpedPackets = 0;
#endif

typedef AVCodecID CodecID;

struct wave_format_ex {
	uint16 format_tag;
	uint16 channels;
	uint32 frames_per_sec;
	uint32 avg_bytes_per_sec;
	uint16 block_align;
	uint16 bits_per_sample;
	uint16 extra_size;
	// extra_data[extra_size]
} _PACKED;

struct avformat_codec_context {
	int sample_rate;
	int channels;
};


// profiling related globals
#define DO_PROFILING 0
#if DO_PROFILING
static bigtime_t decodingTime = 0;
static bigtime_t conversionTime = 0;
static long profileCounter = 0;
#endif


AVCodecDecoder::AVCodecDecoder()
	:
	fHeader(),
	fInputFormat(),
	fFrame(0),
	fIsAudio(false),
	fCodec(NULL),
	fCodecContext(avcodec_alloc_context3(NULL)),
	fResampleContext(NULL),
	fDecodedData(NULL),
	fDecodedDataSizeInBytes(0),
	fPostProcessedDecodedPicture(av_frame_alloc()),
	fRawDecodedPicture(av_frame_alloc()),
	fRawDecodedAudio(av_frame_alloc()),

	fCodecInitDone(false),

#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
	fSwsContext(NULL),
#else
	fFormatConversionFunc(NULL),
#endif

	fExtraData(NULL),
	fExtraDataSize(0),
	fBlockAlign(0),

	fOutputColorSpace(B_NO_COLOR_SPACE),
	fOutputFrameCount(0),
	fOutputFrameRate(1.0),
	fOutputFrameSize(0),
	fInputFrameSize(0),

	fChunkBuffer(NULL),
	fChunkBufferSize(0),
	fAudioDecodeError(false),

	fDecodedDataBuffer(av_frame_alloc()),
	fDecodedDataBufferOffset(0),
	fDecodedDataBufferSize(0),
	fBufferSinkContext(NULL),
	fBufferSourceContext(NULL),
	fFilterGraph(NULL),
	fFilterFrame(NULL)
{
	TRACE("AVCodecDecoder::AVCodecDecoder()\n");

	system_info info;
	get_system_info(&info);

	fCodecContext->err_recognition = AV_EF_CAREFUL;
	fCodecContext->error_concealment = 3;
	fCodecContext->thread_count = info.cpu_count;
}


AVCodecDecoder::~AVCodecDecoder()
{
	TRACE("[%c] AVCodecDecoder::~AVCodecDecoder()\n", fIsAudio?('a'):('v'));

#if DO_PROFILING
	if (profileCounter > 0) {
		printf("[%c] profile: d1 = %lld, d2 = %lld (%lld)\n",
			fIsAudio?('a'):('v'), decodingTime / profileCounter,
			conversionTime / profileCounter, fFrame);
	}
#endif

	if (fCodecInitDone)
		avcodec_close(fCodecContext);

	swr_free(&fResampleContext);
	free(fChunkBuffer);
	free(fDecodedData);

	av_frame_free(&fPostProcessedDecodedPicture);
	av_frame_free(&fRawDecodedPicture);
	av_free(fRawDecodedAudio->opaque);
	av_frame_free(&fRawDecodedAudio);
	avcodec_free_context(&fCodecContext);
	av_frame_free(&fDecodedDataBuffer);

	av_frame_free(&fFilterFrame);
	avfilter_graph_free(&fFilterGraph);

#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
	if (fSwsContext != NULL)
		sws_freeContext(fSwsContext);
#endif

	delete[] fExtraData;
}


void
AVCodecDecoder::GetCodecInfo(media_codec_info* mci)
{
	snprintf(mci->short_name, 32, "%s", fCodec->name);
	snprintf(mci->pretty_name, 96, "%s", fCodec->long_name);
	mci->id = 0;
	mci->sub_id = fCodec->id;
}


status_t
AVCodecDecoder::Setup(media_format* ioEncodedFormat, const void* infoBuffer,
	size_t infoSize)
{
	if (ioEncodedFormat->type != B_MEDIA_ENCODED_AUDIO
		&& ioEncodedFormat->type != B_MEDIA_ENCODED_VIDEO)
		return B_ERROR;

	fIsAudio = (ioEncodedFormat->type == B_MEDIA_ENCODED_AUDIO);
	TRACE("[%c] AVCodecDecoder::Setup()\n", fIsAudio?('a'):('v'));

#ifdef TRACE_AV_CODEC
	char buffer[1024];
	string_for_format(*ioEncodedFormat, buffer, sizeof(buffer));
	TRACE("[%c]   input_format = %s\n", fIsAudio?('a'):('v'), buffer);
	TRACE("[%c]   infoSize = %ld\n", fIsAudio?('a'):('v'), infoSize);
	TRACE("[%c]   user_data_type = %08lx\n", fIsAudio?('a'):('v'),
		ioEncodedFormat->user_data_type);
	TRACE("[%c]   meta_data_size = %ld\n", fIsAudio?('a'):('v'),
		ioEncodedFormat->MetaDataSize());
#endif

	media_format_description description;
	if (BMediaFormats().GetCodeFor(*ioEncodedFormat,
			B_MISC_FORMAT_FAMILY, &description) == B_OK) {
		if (description.u.misc.file_format != 'ffmp')
			return B_NOT_SUPPORTED;
		fCodec = avcodec_find_decoder(static_cast<CodecID>(
			description.u.misc.codec));
		if (fCodec == NULL) {
			TRACE("  unable to find the correct FFmpeg "
				"decoder (id = %lu)\n", description.u.misc.codec);
			return B_ERROR;
		}
		TRACE("  found decoder %s\n", fCodec->name);

		const void* extraData = infoBuffer;
		fExtraDataSize = infoSize;
		if (description.family == B_WAV_FORMAT_FAMILY
				&& infoSize >= sizeof(wave_format_ex)) {
			TRACE("  trying to use wave_format_ex\n");
			// Special case extra data in B_WAV_FORMAT_FAMILY
			const wave_format_ex* waveFormatData
				= (const wave_format_ex*)infoBuffer;

			size_t waveFormatSize = infoSize;
			if (waveFormatData != NULL && waveFormatSize > 0) {
				fBlockAlign = waveFormatData->block_align;
				TRACE("  found block align: %d\n", fBlockAlign);
				fExtraDataSize = waveFormatData->extra_size;
				// skip the wave_format_ex from the extra data.
				extraData = waveFormatData + 1;
			}
		} else {
			if (fIsAudio) {
				fBlockAlign
					= ioEncodedFormat->u.encoded_audio.output.buffer_size;
				TRACE("  using buffer_size as block align: %d\n",
					fBlockAlign);
			}
		}
		if (extraData != NULL && fExtraDataSize > 0) {
			TRACE("AVCodecDecoder: extra data size %ld\n", infoSize);
			delete[] fExtraData;
			fExtraData = new(std::nothrow) char[fExtraDataSize];
			if (fExtraData != NULL)
				memcpy(fExtraData, infoBuffer, fExtraDataSize);
			else
				fExtraDataSize = 0;
		}

		fInputFormat = *ioEncodedFormat;
		return B_OK;
	} else {
		TRACE("AVCodecDecoder: BMediaFormats().GetCodeFor() failed.\n");
	}

	printf("AVCodecDecoder::Setup failed!\n");
	return B_ERROR;
}


status_t
AVCodecDecoder::SeekedTo(int64 frame, bigtime_t time)
{
	status_t ret = B_OK;
	// Reset the FFmpeg codec to flush buffers, so we keep the sync
	if (fCodecInitDone) {
		avcodec_flush_buffers(fCodecContext);
		_ResetTempPacket();
	}

	// Flush internal buffers as well.
	free(fChunkBuffer);
	fChunkBuffer = NULL;
	fChunkBufferSize = 0;
	fDecodedDataBufferOffset = 0;
	fDecodedDataBufferSize = 0;
	fDecodedDataSizeInBytes = 0;

	fFrame = frame;

	return ret;
}


status_t
AVCodecDecoder::NegotiateOutputFormat(media_format* inOutFormat)
{
	TRACE("AVCodecDecoder::NegotiateOutputFormat() [%c] \n",
		fIsAudio?('a'):('v'));

#ifdef TRACE_AV_CODEC
	char buffer[1024];
	string_for_format(*inOutFormat, buffer, sizeof(buffer));
	TRACE("  [%c]  requested format = %s\n", fIsAudio?('a'):('v'), buffer);
#endif

	if (fIsAudio)
		return _NegotiateAudioOutputFormat(inOutFormat);
	else
		return _NegotiateVideoOutputFormat(inOutFormat);
}


status_t
AVCodecDecoder::Decode(void* outBuffer, int64* outFrameCount,
	media_header* mediaHeader, media_decode_info* info)
{
	if (!fCodecInitDone)
		return B_NO_INIT;

	status_t ret;
	if (fIsAudio)
		ret = _DecodeAudio(outBuffer, outFrameCount, mediaHeader, info);
	else
		ret = _DecodeVideo(outBuffer, outFrameCount, mediaHeader, info);

	return ret;
}


// #pragma mark -


void
AVCodecDecoder::_ResetTempPacket()
{
	av_init_packet(&fTempPacket);
	fTempPacket.size = 0;
	fTempPacket.data = NULL;
}


status_t
AVCodecDecoder::_NegotiateAudioOutputFormat(media_format* inOutFormat)
{
	TRACE("AVCodecDecoder::_NegotiateAudioOutputFormat()\n");

	_ApplyEssentialAudioContainerPropertiesToContext();
		// This makes audio formats play that encode the audio properties in
		// the audio container (e.g. WMA) and not in the audio frames
		// themself (e.g. MP3).
		// Note: Doing this step unconditionally is OK, because the first call
		// to _DecodeNextAudioFrameChunk() will update the essential audio
		// format properties accordingly regardless of the settings here.

	// close any previous instance
	if (fCodecInitDone) {
		fCodecInitDone = false;
		avcodec_close(fCodecContext);
	}

	if (avcodec_open2(fCodecContext, fCodec, NULL) >= 0)
		fCodecInitDone = true;
	else {
		TRACE("avcodec_open() failed to init codec!\n");
		return B_ERROR;
	}

	free(fChunkBuffer);
	fChunkBuffer = NULL;
	fChunkBufferSize = 0;
	fAudioDecodeError = false;
	fDecodedDataBufferOffset = 0;
	fDecodedDataBufferSize = 0;

	_ResetTempPacket();

	status_t statusOfDecodingFirstFrameChunk = _DecodeNextAudioFrameChunk();
	if (statusOfDecodingFirstFrameChunk != B_OK) {
		TRACE("[a] decoding first audio frame chunk failed\n");
		return B_ERROR;
	}

	media_multi_audio_format outputAudioFormat;
	outputAudioFormat = media_raw_audio_format::wildcard;
	outputAudioFormat.byte_order = B_MEDIA_HOST_ENDIAN;
	outputAudioFormat.frame_rate = fCodecContext->sample_rate;
	outputAudioFormat.channel_count = fCodecContext->channels;
	ConvertAVSampleFormatToRawAudioFormat(fCodecContext->sample_fmt,
		outputAudioFormat.format);
	// Check that format is not still a wild card!
	if (outputAudioFormat.format == 0) {
		TRACE("  format still a wild-card, assuming B_AUDIO_SHORT.\n");
		outputAudioFormat.format = media_raw_audio_format::B_AUDIO_SHORT;
	}
	outputAudioFormat.buffer_size = inOutFormat->u.raw_audio.buffer_size;
	// Check that buffer_size has a sane value
	size_t sampleSize = outputAudioFormat.format
		& media_raw_audio_format::B_AUDIO_SIZE_MASK;
	if (outputAudioFormat.buffer_size == 0) {
		outputAudioFormat.buffer_size = 512 * sampleSize
			* outputAudioFormat.channel_count;
	}

	inOutFormat->type = B_MEDIA_RAW_AUDIO;
	inOutFormat->u.raw_audio = outputAudioFormat;
	inOutFormat->require_flags = 0;
	inOutFormat->deny_flags = B_MEDIA_MAUI_UNDEFINED_FLAGS;

	// Initialize variables needed to manage decoding as much audio frames as
	// needed to fill the buffer_size.
	fOutputFrameSize = sampleSize * outputAudioFormat.channel_count;
	fOutputFrameCount = outputAudioFormat.buffer_size / fOutputFrameSize;
	fOutputFrameRate = outputAudioFormat.frame_rate;
	if (av_sample_fmt_is_planar(fCodecContext->sample_fmt))
		fInputFrameSize = sampleSize;
	else
		fInputFrameSize = fOutputFrameSize;

	fRawDecodedAudio->opaque
		= av_realloc(fRawDecodedAudio->opaque, sizeof(avformat_codec_context));
	if (fRawDecodedAudio->opaque == NULL)
		return B_NO_MEMORY;

	if (av_sample_fmt_is_planar(fCodecContext->sample_fmt)) {
		fResampleContext = swr_alloc_set_opts(NULL,
			fCodecContext->channel_layout,
			fCodecContext->request_sample_fmt,
			fCodecContext->sample_rate,
			fCodecContext->channel_layout,
			fCodecContext->sample_fmt,
			fCodecContext->sample_rate,
			0, NULL);
		swr_init(fResampleContext);
	}

	TRACE("  bit_rate = %d, sample_rate = %d, channels = %d, "
		"output frame size: %d, count: %ld, rate: %.2f\n",
		fCodecContext->bit_rate, fCodecContext->sample_rate, fCodecContext->channels,
		fOutputFrameSize, fOutputFrameCount, fOutputFrameRate);

	return B_OK;
}


status_t
AVCodecDecoder::_NegotiateVideoOutputFormat(media_format* inOutFormat)
{
	TRACE("AVCodecDecoder::_NegotiateVideoOutputFormat()\n");

	TRACE("  requested video format 0x%x\n",
		inOutFormat->u.raw_video.display.format);

	_ApplyEssentialVideoContainerPropertiesToContext();
		// This makes video formats play that encode the video properties in
		// the video container (e.g. WMV) and not in the video frames
		// themself (e.g. MPEG2).
		// Note: Doing this step unconditionally is OK, because the first call
		// to _DecodeNextVideoFrame() will update the essential video format
		// properties accordingly regardless of the settings here.

	bool codecCanHandleIncompleteFrames
		= (fCodec->capabilities & AV_CODEC_CAP_TRUNCATED) != 0;
	if (codecCanHandleIncompleteFrames) {
		// Expect and handle video frames to be splitted across consecutive
		// data chunks.
		fCodecContext->flags |= AV_CODEC_FLAG_TRUNCATED;
	}

	// close any previous instance
	if (fCodecInitDone) {
		fCodecInitDone = false;
		avcodec_close(fCodecContext);
	}

	if (avcodec_open2(fCodecContext, fCodec, NULL) >= 0)
		fCodecInitDone = true;
	else {
		TRACE("avcodec_open() failed to init codec!\n");
		return B_ERROR;
	}

#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
	fOutputColorSpace = B_RGB32;
#else
	// Make MediaPlayer happy (if not in rgb32 screen depth and no overlay,
	// it will only ask for YCbCr, which DrawBitmap doesn't handle, so the
	// default colordepth is RGB32).
	if (inOutFormat->u.raw_video.display.format == B_YCbCr422)
		fOutputColorSpace = B_YCbCr422;
	else
		fOutputColorSpace = B_RGB32;
#endif

#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
	if (fSwsContext != NULL)
		sws_freeContext(fSwsContext);
	fSwsContext = NULL;
#else
	fFormatConversionFunc = 0;
#endif

	free(fChunkBuffer);
	fChunkBuffer = NULL;
	fChunkBufferSize = 0;

	_ResetTempPacket();

	status_t statusOfDecodingFirstFrame = _DecodeNextVideoFrame();
	if (statusOfDecodingFirstFrame != B_OK) {
		TRACE("[v] decoding first video frame failed\n");
		return B_ERROR;
	}

	// Note: fSwsContext / fFormatConversionFunc should have been initialized
	// by first call to _DecodeNextVideoFrame() above.
#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
	if (fSwsContext == NULL) {
		TRACE("No SWS Scale context or decoder has not set the pixel format "
			"yet!\n");
	}
#else
	if (fFormatConversionFunc == NULL) {
		TRACE("no pixel format conversion function found or decoder has "
			"not set the pixel format yet!\n");
	}
#endif

	inOutFormat->type = B_MEDIA_RAW_VIDEO;
	inOutFormat->require_flags = 0;
	inOutFormat->deny_flags = B_MEDIA_MAUI_UNDEFINED_FLAGS;
	inOutFormat->u.raw_video = fInputFormat.u.encoded_video.output;
	inOutFormat->u.raw_video.interlace = 1;
		// Progressive (non-interlaced) video frames are delivered
	inOutFormat->u.raw_video.first_active
		= fHeader.u.raw_video.first_active_line;
	inOutFormat->u.raw_video.last_active = fHeader.u.raw_video.line_count;
	inOutFormat->u.raw_video.pixel_width_aspect
		= fHeader.u.raw_video.pixel_width_aspect;
	inOutFormat->u.raw_video.pixel_height_aspect
		= fHeader.u.raw_video.pixel_height_aspect;
#if 0
	// This was added by Colin Günther in order to handle streams with a
	// variable frame rate. fOutputFrameRate is computed from the stream
	// time_base, but it actually assumes a timebase equal to the FPS. As far
	// as I can see, a stream with a variable frame rate would have a higher
	// resolution time_base and increment the pts (presentation time) of each
	// frame by a value bigger than one.
	//
	// Fixed rate stream:
	// time_base = 1/50s, frame PTS = 1, 2, 3... (for 50Hz)
	//
	// Variable rate stream:
	// time_base = 1/300s, frame PTS = 6, 12, 18, ... (for 50Hz)
	// time_base = 1/300s, frame PTS = 5, 10, 15, ... (for 60Hz)
	//
	// The fOutputFrameRate currently does not take this into account and
	// ignores the PTS. This results in playing the above sample at 300Hz
	// instead of 50 or 60.
	//
	// However, comparing the PTS for two consecutive implies we have already
	// decoded 2 frames, which may not be the case when this method is first
	// called.
	inOutFormat->u.raw_video.field_rate = fOutputFrameRate;
		// Was calculated by first call to _DecodeNextVideoFrame()
#endif
	inOutFormat->u.raw_video.display.format = fOutputColorSpace;
	inOutFormat->u.raw_video.display.line_width
		= fHeader.u.raw_video.display_line_width;
	inOutFormat->u.raw_video.display.line_count
		= fHeader.u.raw_video.display_line_count;
	inOutFormat->u.raw_video.display.bytes_per_row
		= fHeader.u.raw_video.bytes_per_row;

#ifdef TRACE_AV_CODEC
	char buffer[1024];
	string_for_format(*inOutFormat, buffer, sizeof(buffer));
	TRACE("[v]  outFormat = %s\n", buffer);
	TRACE("  returned  video format 0x%x\n",
		inOutFormat->u.raw_video.display.format);
#endif

	return B_OK;
}


/*! \brief Fills the outBuffer with one or more already decoded audio frames.

	Besides the main duty described above, this method also fills out the other
	output parameters as documented below.

	\param outBuffer Pointer to the output buffer to copy the decoded audio
		frames to.
	\param outFrameCount Pointer to the output variable to assign the number of
		copied audio frames (usually several audio frames at once).
	\param mediaHeader Pointer to the output media header that contains the
		properties of the decoded audio frame being the first in the outBuffer.
	\param info Specifies additional decoding parameters. (Note: unused).

	\returns B_OK Decoding audio frames succeeded.
	\returns B_LAST_BUFFER_ERROR There are no more audio frames available.
	\returns Other error codes
*/
status_t
AVCodecDecoder::_DecodeAudio(void* outBuffer, int64* outFrameCount,
	media_header* mediaHeader, media_decode_info* info)
{
	TRACE_AUDIO("AVCodecDecoder::_DecodeAudio(audio start_time %.6fs)\n",
		mediaHeader->start_time / 1000000.0);

	status_t audioDecodingStatus
		= fDecodedDataSizeInBytes > 0 ? B_OK : _DecodeNextAudioFrame();

	if (audioDecodingStatus != B_OK)
		return audioDecodingStatus;

	*outFrameCount = fDecodedDataSizeInBytes / fOutputFrameSize;
	*mediaHeader = fHeader;
	memcpy(outBuffer, fDecodedData, fDecodedDataSizeInBytes);

	fDecodedDataSizeInBytes = 0;

	return B_OK;
}


/*! \brief Fills the outBuffer with an already decoded video frame.

	Besides the main duty described above, this method also fills out the other
	output parameters as documented below.

	\param outBuffer Pointer to the output buffer to copy the decoded video
		frame to.
	\param outFrameCount Pointer to the output variable to assign the number of
		copied video frames (usually one video frame).
	\param mediaHeader Pointer to the output media header that contains the
		decoded video frame properties.
	\param info Specifies additional decoding parameters. (Note: unused).

	\returns B_OK Decoding a video frame succeeded.
	\returns B_LAST_BUFFER_ERROR There are no more video frames available.
	\returns Other error codes
*/
status_t
AVCodecDecoder::_DecodeVideo(void* outBuffer, int64* outFrameCount,
	media_header* mediaHeader, media_decode_info* info)
{
	status_t videoDecodingStatus
		= fDecodedDataSizeInBytes > 0 ? B_OK : _DecodeNextVideoFrame();

	if (videoDecodingStatus != B_OK)
		return videoDecodingStatus;

	*outFrameCount = 1;
	*mediaHeader = fHeader;
	memcpy(outBuffer, fDecodedData, mediaHeader->size_used);

	fDecodedDataSizeInBytes = 0;

	return B_OK;
}


/*!	\brief Decodes next audio frame.

	We decode at least one audio frame into fDecodedData. To achieve this goal,
    we might need to request several chunks of encoded data resulting in a
    variable execution time of this function.

    The length of the decoded audio frame(s) is stored in
    fDecodedDataSizeInBytes. If this variable is greater than zero you can
    assert that all audio frames in fDecodedData are valid.

	It is assumed that the number of expected audio frames is stored in
	fOutputFrameCount. So _DecodeNextAudioFrame() must be called only after
	fOutputFrameCount has been set.

	Note: fOutputFrameCount contains the maximum number of frames a caller
	of BMediaDecoder::Decode() expects to receive. There is a direct
	relationship between fOutputFrameCount and the buffer size a caller of
	BMediaDecoder::Decode() will provide so we make sure to respect this limit
	for fDecodedDataSizeInBytes.

	On return with status code B_OK the following conditions hold true:
		1. fDecodedData contains as much audio frames as the caller of
		   BMediaDecoder::Decode() expects.
		2. fDecodedData contains lesser audio frames as the caller of
		   BMediaDecoder::Decode() expects only when one of the following
		   conditions hold true:
		       i  No more audio frames left. Consecutive calls to
		          _DecodeNextAudioFrame() will then result in the return of
		          status code B_LAST_BUFFER_ERROR.
		       ii TODO: A change in the size of the audio frames.
		3. fHeader is populated with the audio frame properties of the first
		   audio frame in fDecodedData. Especially the start_time field of
		   fHeader relates to that first audio frame. Start times of
		   consecutive audio frames in fDecodedData have to be calculated
		   manually (using the frame rate and the frame duration) if the
		   caller needs them.

	TODO: Handle change of channel_count. Such a change results in a change of
	the audio frame size and thus has different buffer requirements.
	The most sane approach for implementing this is to return the audio frames
	that were still decoded with the previous channel_count and inform the
	client of BMediaDecoder::Decode() about the change so that it can adapt to
	it. Furthermore we need to adapt our fDecodedData to the new buffer size
	requirements accordingly.

	\returns B_OK when we successfully decoded enough audio frames
	\returns B_LAST_BUFFER_ERROR when there are no more audio frames available.
	\returns Other Errors
*/
status_t
AVCodecDecoder::_DecodeNextAudioFrame()
{
	assert(fTempPacket.size >= 0);
	assert(fDecodedDataSizeInBytes == 0);
		// _DecodeNextAudioFrame needs to be called on empty fDecodedData only!
		// If this assert holds wrong we have a bug somewhere.

	status_t resetStatus = _ResetRawDecodedAudio();
	if (resetStatus != B_OK)
		return resetStatus;

	while (fRawDecodedAudio->nb_samples < fOutputFrameCount) {
		_CheckAndFixConditionsThatHintAtBrokenAudioCodeBelow();

		bool decodedDataBufferHasData = fDecodedDataBufferSize > 0;
		if (decodedDataBufferHasData) {
			_MoveAudioFramesToRawDecodedAudioAndUpdateStartTimes();
			continue;
		}

		status_t decodeAudioChunkStatus = _DecodeNextAudioFrameChunk();
		if (decodeAudioChunkStatus == B_LAST_BUFFER_ERROR
				&& fRawDecodedAudio->nb_samples > 0)
			break;
		if (decodeAudioChunkStatus != B_OK)
			return decodeAudioChunkStatus;
	}

	fFrame += fRawDecodedAudio->nb_samples;
	fDecodedDataSizeInBytes = fRawDecodedAudio->linesize[0];

	_UpdateMediaHeaderForAudioFrame();

#ifdef DEBUG
	dump_ffframe_audio(fRawDecodedAudio, "ffaudi");
#endif

	TRACE_AUDIO("  frame count: %ld current: %lld\n",
		fRawDecodedAudio->nb_samples, fFrame);

	return B_OK;
}


/*!	\brief Applies all essential audio input properties to fCodecContext that were
		passed to AVCodecDecoder when Setup() was called.

	Note: This function must be called before the AVCodec is opened via
	avcodec_open2(). Otherwise the behaviour of FFMPEG's audio decoding
	function avcodec_receive_frame() is undefined.

	Essential properties applied from fInputFormat.u.encoded_audio:
		- bit_rate copied to fCodecContext->bit_rate
		- frame_size copied to fCodecContext->frame_size
		- output.format converted to fCodecContext->sample_fmt
		- output.frame_rate copied to fCodecContext->sample_rate
		- output.channel_count copied to fCodecContext->channels

	Other essential properties being applied:
		- fBlockAlign to fCodecContext->block_align
		- fExtraData to fCodecContext->extradata
		- fExtraDataSize to fCodecContext->extradata_size

	TODO: Either the following documentation section should be removed or this
	TODO when it is clear whether fInputFormat.MetaData() and
	fInputFormat.MetaDataSize() have to be applied to fCodecContext. See the related
	TODO in the method implementation.
	Only applied when fInputFormat.MetaDataSize() is greater than zero:
		- fInputFormat.MetaData() to fCodecContext->extradata
		- fInputFormat.MetaDataSize() to fCodecContext->extradata_size
*/
void
AVCodecDecoder::_ApplyEssentialAudioContainerPropertiesToContext()
{
	media_encoded_audio_format containerProperties
		= fInputFormat.u.encoded_audio;

	fCodecContext->bit_rate
		= static_cast<int>(containerProperties.bit_rate);
	fCodecContext->frame_size
		= static_cast<int>(containerProperties.frame_size);
	ConvertRawAudioFormatToAVSampleFormat(
		containerProperties.output.format, fCodecContext->sample_fmt);
	ConvertRawAudioFormatToAVSampleFormat(
		containerProperties.output.format, fCodecContext->request_sample_fmt);
	fCodecContext->sample_rate
		= static_cast<int>(containerProperties.output.frame_rate);
	fCodecContext->channels
		= static_cast<int>(containerProperties.output.channel_count);
	// Check that channel count is not still a wild card!
	if (fCodecContext->channels == 0) {
		TRACE("  channel_count still a wild-card, assuming stereo.\n");
		fCodecContext->channels = 2;
	}

	fCodecContext->block_align = fBlockAlign;
	fCodecContext->extradata = reinterpret_cast<uint8_t*>(fExtraData);
	fCodecContext->extradata_size = fExtraDataSize;

	// TODO: This probably needs to go away, there is some misconception
	// about extra data / info buffer and meta data. See
	// Reader::GetStreamInfo(). The AVFormatReader puts extradata and
	// extradata_size into media_format::MetaData(), but used to ignore
	// the infoBuffer passed to GetStreamInfo(). I think this may be why
	// the code below was added.
	if (fInputFormat.MetaDataSize() > 0) {
		fCodecContext->extradata = static_cast<uint8_t*>(
			const_cast<void*>(fInputFormat.MetaData()));
		fCodecContext->extradata_size = fInputFormat.MetaDataSize();
	}

	TRACE("  bit_rate %d, sample_rate %d, channels %d, block_align %d, "
		"extradata_size %d\n",
		fCodecContext->bit_rate,
		fCodecContext->sample_rate,
		fCodecContext->channels,
		fCodecContext->block_align,
		fCodecContext->extradata_size);
}


/*!	\brief Resets important fields in fRawDecodedVideo to their default values.

	Note: Also initializes fDecodedData if not done already.

	\returns B_OK Resetting successfully completed.
	\returns B_NO_MEMORY No memory left for correct operation.
*/
status_t
AVCodecDecoder::_ResetRawDecodedAudio()
{
	if (fDecodedData == NULL) {
		size_t maximumSizeOfDecodedData = fOutputFrameCount * fOutputFrameSize;
		fDecodedData
			= static_cast<uint8_t*>(malloc(maximumSizeOfDecodedData));
	}
	if (fDecodedData == NULL)
		return B_NO_MEMORY;

	fRawDecodedAudio->data[0] = fDecodedData;
	fRawDecodedAudio->linesize[0] = 0;
	fRawDecodedAudio->format = AV_SAMPLE_FMT_NONE;
	fRawDecodedAudio->pkt_dts = AV_NOPTS_VALUE;
	fRawDecodedAudio->nb_samples = 0;
	memset(fRawDecodedAudio->opaque, 0, sizeof(avformat_codec_context));

	return B_OK;
}


/*!	\brief Checks fDecodedDataBufferSize and fTempPacket for invalid values,
		reports them and assigns valid values.

	Note: This method is intended to be called before any code is executed that
	deals with moving, loading or decoding any audio frames.
*/
void
AVCodecDecoder::_CheckAndFixConditionsThatHintAtBrokenAudioCodeBelow()
{
	if (fDecodedDataBufferSize < 0) {
		fprintf(stderr, "Decoding read past the end of the decoded data "
			"buffer! %" B_PRId32 "\n", fDecodedDataBufferSize);
		fDecodedDataBufferSize = 0;
	}
	if (fTempPacket.size < 0) {
		fprintf(stderr, "Decoding read past the end of the temp packet! %d\n",
			fTempPacket.size);
		fTempPacket.size = 0;
	}
}


/*!	\brief Moves audio frames from fDecodedDataBuffer to fRawDecodedAudio (and
		thus to fDecodedData) and updates the start times of fRawDecodedAudio,
		fDecodedDataBuffer and fTempPacket accordingly.

	When moving audio frames to fRawDecodedAudio this method also makes sure
	that the following important fields of fRawDecodedAudio are populated and
	updated with correct values:
		- fRawDecodedAudio->data[0]: Points to first free byte of fDecodedData
		- fRawDecodedAudio->linesize[0]: Total size of frames in fDecodedData
		- fRawDecodedAudio->format: Format of first audio frame
		- fRawDecodedAudio->pkt_dts: Start time of first audio frame
		- fRawDecodedAudio->nb_samples: Number of audio frames
		- fRawDecodedAudio->opaque: Contains the following fields for the first
		  audio frame:
		      - channels: Channel count of first audio frame
		      - sample_rate: Frame rate of first audio frame

	This function assumes to be called only when the following assumptions
	hold true:
		1. There are decoded audio frames available in fDecodedDataBuffer
		   meaning that fDecodedDataBufferSize is greater than zero.
		2. There is space left in fRawDecodedAudio to move some audio frames
		   in. This means that fRawDecodedAudio has lesser audio frames than
		   the maximum allowed (specified by fOutputFrameCount).
		3. The audio frame rate is known so that we can calculate the time
		   range (covered by the moved audio frames) to update the start times
		   accordingly.
		4. The field fRawDecodedAudio->opaque points to a memory block
		   representing a structure of type avformat_codec_context.

	After this function returns the caller can safely make the following
	assumptions:
		1. The number of decoded audio frames in fDecodedDataBuffer is
		   decreased though it may still be greater then zero.
		2. The number of frames in fRawDecodedAudio has increased and all
		   important fields are updated (see listing above).
		3. Start times of fDecodedDataBuffer and fTempPacket were increased
		   with the time range covered by the moved audio frames.

	Note: This function raises an exception (by calling the debugger), when
	fDecodedDataBufferSize is not a multiple of fOutputFrameSize.
*/
void
AVCodecDecoder::_MoveAudioFramesToRawDecodedAudioAndUpdateStartTimes()
{
	assert(fDecodedDataBufferSize > 0);
	assert(fRawDecodedAudio->nb_samples < fOutputFrameCount);
	assert(fOutputFrameRate > 0);

	int32 outFrames = fOutputFrameCount - fRawDecodedAudio->nb_samples;
	int32 inFrames = fDecodedDataBufferSize;

	int32 frames = min_c(outFrames, inFrames);
	if (frames == 0)
		debugger("fDecodedDataBufferSize not multiple of frame size!");

	// Some decoders do not support format conversion on themselves, or use
	// "planar" audio (each channel separated instead of interleaved samples).
	// In that case, we use swresample to convert the data
	if (av_sample_fmt_is_planar(fCodecContext->sample_fmt)) {
#if 0
		const uint8_t* ptr[8];
		for (int i = 0; i < 8; i++) {
			if (fDecodedDataBuffer->data[i] == NULL)
				ptr[i] = NULL;
			else
				ptr[i] = fDecodedDataBuffer->data[i] + fDecodedDataBufferOffset;
		}

		// When there are more input frames than space in the output buffer,
		// we could feed everything to swr and it would buffer the extra data.
		// However, there is no easy way to flush that data without feeding more
		// input, and it makes our timestamp computations fail.
		// So, we feed only as much frames as we can get out, and handle the
		// buffering ourselves.
		// TODO Ideally, we should try to size our output buffer so that it can
		// always hold all the output (swr provides helper functions for this)
		inFrames = frames;
		frames = swr_convert(fResampleContext, fRawDecodedAudio->data,
			outFrames, ptr, inFrames);

		if (frames < 0)
			debugger("resampling failed");
#else
		// interleave planar audio with same format
		uintptr_t out = (uintptr_t)fRawDecodedAudio->data[0];
		int32 offset = fDecodedDataBufferOffset;
		for (int i = 0; i < frames; i++) {
			for (int j = 0; j < fCodecContext->channels; j++) {
				memcpy((void*)out, fDecodedDataBuffer->data[j]
					+ offset, fInputFrameSize);
				out += fInputFrameSize;
			}
			offset += fInputFrameSize;
		}
		outFrames = frames;
		inFrames = frames;
#endif
	} else {
		memcpy(fRawDecodedAudio->data[0], fDecodedDataBuffer->data[0]
				+ fDecodedDataBufferOffset, frames * fOutputFrameSize);
		outFrames = frames;
		inFrames = frames;
	}

	size_t remainingSize = inFrames * fInputFrameSize;
	size_t decodedSize = outFrames * fOutputFrameSize;
	fDecodedDataBufferSize -= inFrames;

	bool firstAudioFramesCopiedToRawDecodedAudio
		= fRawDecodedAudio->data[0] != fDecodedData;
	if (!firstAudioFramesCopiedToRawDecodedAudio) {
		fRawDecodedAudio->format = fDecodedDataBuffer->format;
		fRawDecodedAudio->pkt_dts = fDecodedDataBuffer->pkt_dts;

		avformat_codec_context* codecContext
			= static_cast<avformat_codec_context*>(fRawDecodedAudio->opaque);
		codecContext->channels = fCodecContext->channels;
		codecContext->sample_rate = fCodecContext->sample_rate;
	}

	fRawDecodedAudio->data[0] += decodedSize;
	fRawDecodedAudio->linesize[0] += decodedSize;
	fRawDecodedAudio->nb_samples += outFrames;

	fDecodedDataBufferOffset += remainingSize;

	// Update start times accordingly
	bigtime_t framesTimeInterval = static_cast<bigtime_t>(
		(1000000LL * frames) / fOutputFrameRate);
	fDecodedDataBuffer->pkt_dts += framesTimeInterval;
	// Start time of buffer is updated in case that it contains
	// more audio frames to move.
	fTempPacket.dts += framesTimeInterval;
	// Start time of fTempPacket is updated in case the fTempPacket
	// contains more audio frames to decode.
}


/*!	\brief Decodes next chunk of audio frames.

	This method handles all the details of loading the input buffer
	(fChunkBuffer) at the right time and of calling FFMPEG often engouh until
	some audio frames have been decoded.

	FFMPEG decides how much audio frames belong to a chunk. Because of that
	it is very likely that _DecodeNextAudioFrameChunk has to be called several
	times to decode enough audio frames to please the caller of
	BMediaDecoder::Decode().

	This function assumes to be called only when the following assumptions
	hold true:
		1. fDecodedDataBufferSize equals zero.

	After this function returns successfully the caller can safely make the
	following assumptions:
		1. fDecodedDataBufferSize is greater than zero.
		2. fDecodedDataBufferOffset is set to zero.
		3. fDecodedDataBuffer contains audio frames.


	\returns B_OK on successfully decoding one audio frame chunk.
	\returns B_LAST_BUFFER_ERROR No more audio frame chunks available. From
		this point on further calls will return this same error.
	\returns B_ERROR Decoding failed
*/
status_t
AVCodecDecoder::_DecodeNextAudioFrameChunk()
{
	assert(fDecodedDataBufferSize == 0);

	while (fDecodedDataBufferSize == 0) {
		status_t loadingChunkStatus
			= _LoadNextChunkIfNeededAndAssignStartTime();
		if (loadingChunkStatus != B_OK)
			return loadingChunkStatus;

		status_t decodingStatus
			= _DecodeSomeAudioFramesIntoEmptyDecodedDataBuffer();
		if (decodingStatus != B_OK) {
			// Assume the audio decoded until now is broken so replace it with
			// some silence.
			memset(fDecodedData, 0, fRawDecodedAudio->linesize[0]);

			if (!fAudioDecodeError) {
				// Report failure if not done already
				int32 chunkBufferOffset = fTempPacket.data - fChunkBuffer;
				printf("########### audio decode error, "
					"fTempPacket.size %d, fChunkBuffer data offset %" B_PRId32
					"\n", fTempPacket.size, chunkBufferOffset);
				fAudioDecodeError = true;
			}

			// Assume that next audio chunk can be decoded so keep decoding.
			continue;
		}

		fAudioDecodeError = false;
	}

	return B_OK;
}


/*!	\brief Tries to decode at least one audio frame and store it in the
		fDecodedDataBuffer.

	This function assumes to be called only when the following assumptions
	hold true:
		1. fDecodedDataBufferSize equals zero.
		2. fTempPacket.size is greater than zero.

	After this function returns successfully the caller can safely make the
	following assumptions:
		1. fDecodedDataBufferSize is greater than zero in the common case.
		   Also see "Note" below.
		2. fTempPacket was updated to exclude the data chunk that was consumed
		   by avcodec_send_packet().
		3. fDecodedDataBufferOffset is set to zero.

	When this function failed to decode at least one audio frame due to a
	decoding error the caller can safely make the following assumptions:
		1. fDecodedDataBufferSize equals zero.
		2. fTempPacket.size equals zero.

	Note: It is possible that there wasn't any audio frame decoded into
	fDecodedDataBuffer after calling this function. This is normal and can
	happen when there was either a decoding error or there is some decoding
	delay in FFMPEGs audio decoder. Another call to this method is totally
	safe and is even expected as long as the calling assumptions hold true.

	\returns B_OK Decoding successful. fDecodedDataBuffer contains decoded
		audio frames only when fDecodedDataBufferSize is greater than zero.
		fDecodedDataBuffer is empty, when avcodec_receive_frame() didn't return
		audio frames due to delayed decoding or incomplete audio frames.
	\returns B_ERROR Decoding failed thus fDecodedDataBuffer contains no audio
		frames.
*/
status_t
AVCodecDecoder::_DecodeSomeAudioFramesIntoEmptyDecodedDataBuffer()
{
	assert(fDecodedDataBufferSize == 0);

	av_frame_unref(fDecodedDataBuffer);
	fDecodedDataBufferOffset = 0;

	int error = avcodec_receive_frame(fCodecContext, fDecodedDataBuffer);
	if (error == AVERROR_EOF)
		return B_LAST_BUFFER_ERROR;

	if (error == AVERROR(EAGAIN)) {
		// We need to feed more data into the decoder
		avcodec_send_packet(fCodecContext, &fTempPacket);

		// All the data is always consumed by avcodec_send_packet
		fTempPacket.size = 0;

		// Try again to see if we can get some decoded audio out now
		error = avcodec_receive_frame(fCodecContext, fDecodedDataBuffer);
	}

	fDecodedDataBufferSize = fDecodedDataBuffer->nb_samples;
	if (fDecodedDataBufferSize < 0)
		fDecodedDataBufferSize = 0;

	if (error == 0)
		return B_OK;
	else
		return B_ERROR;
}


/*! \brief Updates relevant fields of the class member fHeader with the
		properties of the most recently decoded audio frame.

	The following fields of fHeader are updated:
		- fHeader.type
		- fHeader.file_pos
		- fHeader.orig_size
		- fHeader.start_time
		- fHeader.size_used
		- fHeader.u.raw_audio.frame_rate
		- fHeader.u.raw_audio.channel_count

	It is assumed that this function is called only	when the following asserts
	hold true:
		1. We actually got a new audio frame decoded by the audio decoder.
		2. fHeader wasn't updated for the new audio frame yet. You MUST call
		   this method only once per decoded audio frame.
		3. fRawDecodedAudio's fields relate to the first audio frame contained
		   in fDecodedData. Especially the following fields are of importance:
		       - fRawDecodedAudio->pkt_dts: Start time of first audio frame
		       - fRawDecodedAudio->opaque: Contains the following fields for
		         the first audio frame:
			         - channels: Channel count of first audio frame
			         - sample_rate: Frame rate of first audio frame
*/
void
AVCodecDecoder::_UpdateMediaHeaderForAudioFrame()
{
	fHeader.type = B_MEDIA_RAW_AUDIO;
	fHeader.file_pos = 0;
	fHeader.orig_size = 0;
	fHeader.start_time = fRawDecodedAudio->pkt_dts;
	fHeader.size_used = fRawDecodedAudio->linesize[0];

	avformat_codec_context* codecContext
		= static_cast<avformat_codec_context*>(fRawDecodedAudio->opaque);
	fHeader.u.raw_audio.channel_count = codecContext->channels;
	fHeader.u.raw_audio.frame_rate = codecContext->sample_rate;
}


/*! \brief Decodes next video frame.

    We decode exactly one video frame into fDecodedData. To achieve this goal,
    we might need to request several chunks of encoded data resulting in a
    variable execution time of this function.

    The length of the decoded video frame is stored in
    fDecodedDataSizeInBytes. If this variable is greater than zero, you can
    assert that there is a valid video frame available in fDecodedData.

    The decoded video frame in fDecodedData has color space conversion and
    deinterlacing already applied.

    To every decoded video frame there is a media_header populated in
    fHeader, containing the corresponding video frame properties.

	Normally every decoded video frame has a start_time field populated in the
	associated fHeader, that determines the presentation time of the frame.
	This relationship will only hold true, when each data chunk that is
	provided via GetNextChunk() contains data for exactly one encoded video
	frame (one complete frame) - not more and not less.

	We can decode data chunks that contain partial video frame data, too. In
	that case, you cannot trust the value of the start_time field in fHeader.
	We simply have no logic in place to establish a meaningful relationship
	between an incomplete frame and the start time it should be presented.
	Though this	might change in the future.

	We can decode data chunks that contain more than one video frame, too. In
	that case, you cannot trust the value of the start_time field in fHeader.
	We simply have no logic in place to track the start_time across multiple
	video frames. So a meaningful relationship between the 2nd, 3rd, ... frame
	and the start time it should be presented isn't established at the moment.
	Though this	might change in the future.

	More over the fOutputFrameRate variable is updated for every decoded video
	frame.

	On first call the member variables fSwsContext / fFormatConversionFunc	are
	initialized.

	\returns B_OK when we successfully decoded one video frame
	\returns B_LAST_BUFFER_ERROR when there are no more video frames available.
	\returns B_NO_MEMORY when we have no memory left for correct operation.
	\returns Other Errors
*/
status_t
AVCodecDecoder::_DecodeNextVideoFrame()
{
	int error;
	int send_error;

#if DO_PROFILING
	bigtime_t startTime = system_time();
#endif

	error = avcodec_receive_frame(fCodecContext, fRawDecodedPicture);

	if (error == AVERROR_EOF)
		return B_LAST_BUFFER_ERROR;

	if (error == AVERROR(EAGAIN)) {
		do {
			status_t loadingChunkStatus
				= _LoadNextChunkIfNeededAndAssignStartTime();
			if (loadingChunkStatus == B_LAST_BUFFER_ERROR)
				return _FlushOneVideoFrameFromDecoderBuffer();
			if (loadingChunkStatus != B_OK) {
				TRACE("[v] AVCodecDecoder::_DecodeNextVideoFrame(): error from "
					"GetNextChunk(): %s\n", strerror(loadingChunkStatus));
				return loadingChunkStatus;
			}

			char timestamp[AV_TS_MAX_STRING_SIZE];
			av_ts_make_time_string(timestamp,
				fTempPacket.dts, &fCodecContext->time_base);
			TRACE("[v] Feed %d more bytes (dts %s)\n", fTempPacket.size,
				timestamp);

			send_error = avcodec_send_packet(fCodecContext, &fTempPacket);
			if (send_error < 0 && send_error != AVERROR(EAGAIN)) {
				TRACE("[v] AVCodecDecoder: ignoring error in decoding frame "
				"%lld: %d\n", fFrame, error);
			}

			// Packet is consumed, clear it
			fTempPacket.data = NULL;
			fTempPacket.size = 0;

			error = avcodec_receive_frame(fCodecContext, fRawDecodedPicture);
			if (error != 0 && error != AVERROR(EAGAIN)) {
				TRACE("[v] frame %lld - decoding error, error code: %d, "
					"chunk size: %ld\n", fFrame, error, fChunkBufferSize);
			}

		} while (error != 0);
	}

#if DO_PROFILING
	bigtime_t formatConversionStart = system_time();
#endif

	status_t handleStatus = _HandleNewVideoFrameAndUpdateSystemState();
	if (handleStatus != B_OK)
		return handleStatus;

#if DO_PROFILING
	bigtime_t doneTime = system_time();
	decodingTime += formatConversionStart - startTime;
	conversionTime += doneTime - formatConversionStart;
	profileCounter++;
	if (!(fFrame % 5)) {
		printf("[v] profile: d1 = %lld, d2 = %lld (%lld) required %lld\n",
			decodingTime / profileCounter, conversionTime / profileCounter,
			fFrame, bigtime_t(1000000LL / fOutputFrameRate));
		decodingTime = 0;
		conversionTime = 0;
		profileCounter = 0;
	}
#endif
	return error;
}


/*!	\brief Applies all essential video input properties to fCodecContext that were
		passed to AVCodecDecoder when Setup() was called.

	Note: This function must be called before the AVCodec is opened via
	avcodec_open2(). Otherwise the behaviour of FFMPEG's video decoding
	function avcodec_decode_video2() is undefined.

	Essential properties applied from fInputFormat.u.encoded_video.output:
		- display.line_width copied to fCodecContext->width
		- display.line_count copied to fCodecContext->height
		- pixel_width_aspect and pixel_height_aspect converted to
		  fCodecContext->sample_aspect_ratio
		- field_rate converted to fCodecContext->time_base and
		  fCodecContext->ticks_per_frame

	Other essential properties being applied:
		- fExtraData to fCodecContext->extradata
		- fExtraDataSize to fCodecContext->extradata_size
*/
void
AVCodecDecoder::_ApplyEssentialVideoContainerPropertiesToContext()
{
	media_raw_video_format containerProperties
		= fInputFormat.u.encoded_video.output;

	fCodecContext->width = containerProperties.display.line_width;
	fCodecContext->height = containerProperties.display.line_count;

	if (containerProperties.pixel_width_aspect > 0
		&& containerProperties.pixel_height_aspect > 0) {
		ConvertVideoAspectWidthAndHeightToAVCodecContext(
			containerProperties.pixel_width_aspect,
			containerProperties.pixel_height_aspect, *fCodecContext);
	}

	if (containerProperties.field_rate > 0.0) {
		ConvertVideoFrameRateToAVCodecContext(containerProperties.field_rate,
			*fCodecContext);
	}

	fCodecContext->extradata = reinterpret_cast<uint8_t*>(fExtraData);
	fCodecContext->extradata_size = fExtraDataSize;
}


/*! \brief Loads the next  chunk into fChunkBuffer and assigns it (including
		the start time) to fTempPacket but only if fTempPacket is empty.

	\returns B_OK
		1. meaning: Next chunk is loaded.
		2. meaning: No need to load and assign anything. Proceed as usual.
	\returns B_LAST_BUFFER_ERROR No more chunks available. fChunkBuffer	and
		fTempPacket are left untouched.
	\returns Other errors Caller should bail out because fChunkBuffer and
		fTempPacket are in unknown states. Normal operation cannot be
		guaranteed.
*/
status_t
AVCodecDecoder::_LoadNextChunkIfNeededAndAssignStartTime()
{
	if (fTempPacket.size > 0)
		return B_OK;

	const void* chunkBuffer = NULL;
	size_t chunkBufferSize = 0;
		// In the case that GetNextChunk() returns an error fChunkBufferSize
		// should be left untouched.
	media_header chunkMediaHeader;

	status_t getNextChunkStatus = GetNextChunk(&chunkBuffer, &chunkBufferSize,
		&chunkMediaHeader);
	if (getNextChunkStatus != B_OK)
		return getNextChunkStatus;

	status_t chunkBufferPaddingStatus
		= _CopyChunkToChunkBufferAndAddPadding(chunkBuffer, chunkBufferSize);
	if (chunkBufferPaddingStatus != B_OK)
		return chunkBufferPaddingStatus;

	fTempPacket.data = fChunkBuffer;
	fTempPacket.size = fChunkBufferSize;
	fTempPacket.dts = chunkMediaHeader.start_time;
		// Let FFMPEG handle the correct relationship between start_time and
		// decoded a/v frame. By doing so we are simply copying the way how it
		// is implemented in ffplay.c for video frames (for audio frames it
		// works, too, but isn't used by ffplay.c).
		// \see http://git.videolan.org/?p=ffmpeg.git;a=blob;f=ffplay.c;h=09623db374e5289ed20b7cc28c262c4375a8b2e4;hb=9153b33a742c4e2a85ff6230aea0e75f5a8b26c2#l1502
		//
		// FIXME: Research how to establish a meaningful relationship between
		// start_time and decoded a/v frame when the received chunk buffer
		// contains partial a/v frames. Maybe some data formats do contain time
		// stamps (ake pts / dts fields) that can be evaluated by FFMPEG. But
		// as long as I don't have such video data to test it, it makes no
		// sense trying to implement it.
		//
		// FIXME: Implement tracking start_time of video frames originating in
		// data chunks that encode more than one video frame at a time. In that
		// case on would increment the start_time for each consecutive frame of
		// such a data chunk (like it is done for audio frame decoding). But as
		// long as I don't have such video data to test it, it makes no sense
		// to implement it.

#ifdef LOG_STREAM_TO_FILE
	BFile* logFile = fIsAudio ? &sAudioStreamLogFile : &sVideoStreamLogFile;
	if (sDumpedPackets < 100) {
		logFile->Write(chunkBuffer, fChunkBufferSize);
		printf("wrote %ld bytes\n", fChunkBufferSize);
		sDumpedPackets++;
	} else if (sDumpedPackets == 100)
		logFile->Unset();
#endif

	return B_OK;
}


/*! \brief Copies a chunk into fChunkBuffer and adds a "safety net" of
		additional memory as required by FFMPEG for input buffers to video
		decoders.

	This is needed so that some decoders can read safely a predefined number of
	bytes at a time for performance optimization purposes.

	The additional memory has a size of AV_INPUT_BUFFER_PADDING_SIZE as defined
	in avcodec.h.

	Ownership of fChunkBuffer memory is with the class so it needs to be freed
	at the right times (on destruction, on seeking).

	Also update fChunkBufferSize to reflect the size of the contained data
	(leaving out the padding).

	\param chunk The chunk to copy.
	\param chunkSize Size of the chunk in bytes

	\returns B_OK Padding was successful. You are responsible for releasing the
		allocated memory. fChunkBufferSize is set to chunkSize.
	\returns B_NO_MEMORY Padding failed.
		fChunkBuffer is set to NULL making it safe to call free() on it.
		fChunkBufferSize is set to 0 to reflect the size of fChunkBuffer.
*/
status_t
AVCodecDecoder::_CopyChunkToChunkBufferAndAddPadding(const void* chunk,
	size_t chunkSize)
{
	uint8_t* tmpBuffer = static_cast<uint8_t*>(realloc(fChunkBuffer,
		chunkSize + AV_INPUT_BUFFER_PADDING_SIZE));
	if (tmpBuffer == NULL) {
		free(fChunkBuffer);
		fChunkBuffer = NULL;
		fChunkBufferSize = 0;
		return B_NO_MEMORY;
	} else {
		fChunkBuffer = tmpBuffer;
	}

	memcpy(fChunkBuffer, chunk, chunkSize);
	memset(fChunkBuffer + chunkSize, 0, AV_INPUT_BUFFER_PADDING_SIZE);
		// Establish safety net, by zero'ing the padding area.

	fChunkBufferSize = chunkSize;

	return B_OK;
}


/*! \brief Executes all steps needed for a freshly decoded video frame.

	\see _UpdateMediaHeaderForVideoFrame() and
	\see _DeinterlaceAndColorConvertVideoFrame() for when you are allowed to
	call this method.

	\returns B_OK when video frame was handled successfully
	\returnb B_NO_MEMORY when no memory is left for correct operation.
*/
status_t
AVCodecDecoder::_HandleNewVideoFrameAndUpdateSystemState()
{
	_UpdateMediaHeaderForVideoFrame();
	status_t postProcessStatus = _DeinterlaceAndColorConvertVideoFrame();
	if (postProcessStatus != B_OK)
		return postProcessStatus;

	ConvertAVCodecContextToVideoFrameRate(*fCodecContext, fOutputFrameRate);

#ifdef DEBUG
	dump_ffframe_video(fRawDecodedPicture, "ffpict");
#endif

	fFrame++;

	return B_OK;
}


/*! \brief Flushes one video frame - if any - still buffered by the decoder.

	Some FFMPEG decoder are buffering video frames. To retrieve those buffered
	frames the decoder needs to be told so.

	The intended use of this method is to call it, once there are no more data
	chunks for decoding left. Reframed in other words: Once GetNextChunk()
	returns with status B_LAST_BUFFER_ERROR it is time to start flushing.

	\returns B_OK Retrieved one video frame, handled it accordingly and updated
		the system state accordingly.
		There maybe more video frames left. So it is valid for the client of
		AVCodecDecoder to call it one more time.

	\returns B_LAST_BUFFER_ERROR No video frame left.
		The client of the AVCodecDecoder should stop calling it now.

	\returns B_NO_MEMORY No memory left for correct operation.
*/
status_t
AVCodecDecoder::_FlushOneVideoFrameFromDecoderBuffer()
{
	// Tell the decoder there is nothing to send anymore
	avcodec_send_packet(fCodecContext, NULL);

	// Get any remaining frame
	int error = avcodec_receive_frame(fCodecContext, fRawDecodedPicture);

	if (error != 0 && error != AVERROR(EAGAIN)) {
		// video buffer is flushed successfully
		// (or there is an error, not much we can do about it)
		return B_LAST_BUFFER_ERROR;
	}

	return _HandleNewVideoFrameAndUpdateSystemState();
}


/*! \brief Updates relevant fields of the class member fHeader with the
		properties of the most recently decoded video frame.

	It is assumed that this function is called only	when the following asserts
	hold true:
		1. We actually got a new picture decoded by the video decoder.
		2. fHeader wasn't updated for the new picture yet. You MUST call this
		   method only once per decoded video frame.
		3. This function MUST be called after
		   _DeinterlaceAndColorConvertVideoFrame() as it relys on an updated
		    fDecodedDataSizeInBytes.
		4. There will be at maximumn only one decoded video frame in our cache
		   at any single point in time. Otherwise you couldn't tell to which
		   cached decoded video frame the properties in fHeader relate to.
		5. AVCodecContext is still valid for this video frame (This is the case
		   when this function is called after avcodec_decode_video2() and
		   before the next call to avcodec_decode_video2().
*/
void
AVCodecDecoder::_UpdateMediaHeaderForVideoFrame()
{
	fHeader.type = B_MEDIA_RAW_VIDEO;
	fHeader.file_pos = 0;
	fHeader.orig_size = 0;
	fHeader.start_time = fRawDecodedPicture->pkt_dts;
		// The pkt_dts is already in microseconds, even if ffmpeg docs says
		// 'in codec time_base units'
	fHeader.size_used = av_image_get_buffer_size(
		colorspace_to_pixfmt(fOutputColorSpace), fRawDecodedPicture->width,
		fRawDecodedPicture->height, 1);
	fHeader.u.raw_video.display_line_width = fRawDecodedPicture->width;
	fHeader.u.raw_video.display_line_count = fRawDecodedPicture->height;
	fHeader.u.raw_video.bytes_per_row
		= CalculateBytesPerRowWithColorSpaceAndVideoWidth(fOutputColorSpace,
			fRawDecodedPicture->width);
	fHeader.u.raw_video.field_gamma = 1.0;
	fHeader.u.raw_video.field_sequence = fFrame;
	fHeader.u.raw_video.field_number = 0;
	fHeader.u.raw_video.pulldown_number = 0;
	fHeader.u.raw_video.first_active_line = 1;
	fHeader.u.raw_video.line_count = fRawDecodedPicture->height;

	ConvertAVCodecContextToVideoAspectWidthAndHeight(*fCodecContext,
		fHeader.u.raw_video.pixel_width_aspect,
		fHeader.u.raw_video.pixel_height_aspect);

	char timestamp[AV_TS_MAX_STRING_SIZE];
	av_ts_make_time_string(timestamp,
		fRawDecodedPicture->best_effort_timestamp, &fCodecContext->time_base);

	TRACE("[v] start_time=%s field_sequence=%lu\n",
		timestamp, fHeader.u.raw_video.field_sequence);
}


/*! \brief This function applies deinterlacing (only if needed) and color
	conversion to the video frame in fRawDecodedPicture.

	It is assumed that fRawDecodedPicture wasn't deinterlaced and color
	converted yet (otherwise this function behaves in unknown manners).

	This function MUST be called after _UpdateMediaHeaderForVideoFrame() as it
	relys on the fHeader.size_used and fHeader.u.raw_video.bytes_per_row fields
	for correct operation

	You should only call this function when you	got a new picture decoded by
	the video decoder.

	When this function finishes the postprocessed video frame will be available
	in fPostProcessedDecodedPicture and fDecodedData (fDecodedDataSizeInBytes
	will be set accordingly).

	\returns B_OK video frame successfully deinterlaced and color converted.
	\returns B_NO_MEMORY Not enough memory available for correct operation.
*/
status_t
AVCodecDecoder::_DeinterlaceAndColorConvertVideoFrame()
{
	int displayWidth = fRawDecodedPicture->width;
	int displayHeight = fRawDecodedPicture->height;
	AVFrame deinterlacedPicture;
	bool useDeinterlacedPicture = false;

	if (fRawDecodedPicture->interlaced_frame) {
		AVFrame rawPicture;
		rawPicture.data[0] = fRawDecodedPicture->data[0];
		rawPicture.data[1] = fRawDecodedPicture->data[1];
		rawPicture.data[2] = fRawDecodedPicture->data[2];
		rawPicture.data[3] = fRawDecodedPicture->data[3];
		rawPicture.linesize[0] = fRawDecodedPicture->linesize[0];
		rawPicture.linesize[1] = fRawDecodedPicture->linesize[1];
		rawPicture.linesize[2] = fRawDecodedPicture->linesize[2];
		rawPicture.linesize[3] = fRawDecodedPicture->linesize[3];

		if (av_image_alloc(deinterlacedPicture.data,
				deinterlacedPicture.linesize, displayWidth, displayHeight,
				fCodecContext->pix_fmt, 1) < 0)
			return B_NO_MEMORY;

		// deinterlace implemented using avfilter
		_ProcessFilterGraph(&deinterlacedPicture, &rawPicture,
			fCodecContext->pix_fmt, displayWidth, displayHeight);
		useDeinterlacedPicture = true;
	}

	// Some decoders do not set pix_fmt until they have decoded 1 frame
#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
	if (fSwsContext == NULL) {
		fSwsContext = sws_getContext(displayWidth, displayHeight,
			fCodecContext->pix_fmt, displayWidth, displayHeight,
			colorspace_to_pixfmt(fOutputColorSpace),
			SWS_FAST_BILINEAR, NULL, NULL, NULL);
	}
#else
	if (fFormatConversionFunc == NULL) {
		fFormatConversionFunc = resolve_colorspace(fOutputColorSpace,
			fCodecContext->pix_fmt, displayWidth, displayHeight);
	}
#endif

	fDecodedDataSizeInBytes = fHeader.size_used;

	if (fDecodedData == NULL) {
		const size_t kOptimalAlignmentForColorConversion = 32;
		posix_memalign(reinterpret_cast<void**>(&fDecodedData),
			kOptimalAlignmentForColorConversion, fDecodedDataSizeInBytes);
	}
	if (fDecodedData == NULL)
		return B_NO_MEMORY;

	fPostProcessedDecodedPicture->data[0] = fDecodedData;
	fPostProcessedDecodedPicture->linesize[0]
		= fHeader.u.raw_video.bytes_per_row;

#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
	if (fSwsContext != NULL) {
#else
	if (fFormatConversionFunc != NULL) {
#endif
		if (useDeinterlacedPicture) {
			AVFrame deinterlacedFrame;
			deinterlacedFrame.data[0] = deinterlacedPicture.data[0];
			deinterlacedFrame.data[1] = deinterlacedPicture.data[1];
			deinterlacedFrame.data[2] = deinterlacedPicture.data[2];
			deinterlacedFrame.data[3] = deinterlacedPicture.data[3];
			deinterlacedFrame.linesize[0]
				= deinterlacedPicture.linesize[0];
			deinterlacedFrame.linesize[1]
				= deinterlacedPicture.linesize[1];
			deinterlacedFrame.linesize[2]
				= deinterlacedPicture.linesize[2];
			deinterlacedFrame.linesize[3]
				= deinterlacedPicture.linesize[3];

#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
			sws_scale(fSwsContext, deinterlacedFrame.data,
				deinterlacedFrame.linesize, 0, displayHeight,
				fPostProcessedDecodedPicture->data,
				fPostProcessedDecodedPicture->linesize);
#else
			(*fFormatConversionFunc)(&deinterlacedFrame,
				fPostProcessedDecodedPicture, displayWidth, displayHeight);
#endif
		} else {
#if USE_SWS_FOR_COLOR_SPACE_CONVERSION
			sws_scale(fSwsContext, fRawDecodedPicture->data,
				fRawDecodedPicture->linesize, 0, displayHeight,
				fPostProcessedDecodedPicture->data,
				fPostProcessedDecodedPicture->linesize);
#else
			(*fFormatConversionFunc)(fRawDecodedPicture,
				fPostProcessedDecodedPicture, displayWidth, displayHeight);
#endif
		}
	}

	if (fRawDecodedPicture->interlaced_frame)
		av_freep(&deinterlacedPicture.data[0]);

	return B_OK;
}


/*! \brief Init the deinterlace filter graph.

	\returns B_OK the filter graph could be built.
	\returns B_BAD_VALUE something was wrong with building the graph.
*/
status_t
AVCodecDecoder::_InitFilterGraph(enum AVPixelFormat pixfmt, int32 width,
	int32 height)
{
	if (fFilterGraph != NULL) {
		av_frame_free(&fFilterFrame);
		avfilter_graph_free(&fFilterGraph);
	}

	fFilterGraph = avfilter_graph_alloc();

	BString arguments;
	arguments.SetToFormat("buffer=video_size=%" B_PRId32 "x%" B_PRId32
		":pix_fmt=%d:time_base=1/1:pixel_aspect=0/1[in];[in]yadif[out];"
		"[out]buffersink", width, height,
		pixfmt);
	AVFilterInOut* inputs = NULL;
	AVFilterInOut* outputs = NULL;
	TRACE("[v] _InitFilterGraph(): %s\n", arguments.String());
	int ret = avfilter_graph_parse2(fFilterGraph, arguments.String(), &inputs,
		&outputs);
	if (ret < 0) {
		fprintf(stderr, "avfilter_graph_parse2() failed\n");
		return B_BAD_VALUE;
	}

	ret = avfilter_graph_config(fFilterGraph, NULL);
	if (ret < 0) {
		fprintf(stderr, "avfilter_graph_config() failed\n");
		return B_BAD_VALUE;
	}

	fBufferSourceContext = avfilter_graph_get_filter(fFilterGraph,
		"Parsed_buffer_0");
	fBufferSinkContext = avfilter_graph_get_filter(fFilterGraph,
		"Parsed_buffersink_2");
	if (fBufferSourceContext == NULL || fBufferSinkContext == NULL) {
		fprintf(stderr, "avfilter_graph_get_filter() failed\n");
		return B_BAD_VALUE;
	}
	fFilterFrame = av_frame_alloc();
	fLastWidth = width;
	fLastHeight = height;
	fLastPixfmt = pixfmt;

	return B_OK;
}


/*! \brief Process an AVPicture with the deinterlace filter graph.

    We decode exactly one video frame into dst.
	Equivalent function for avpicture_deinterlace() from version 2.x.

	\returns B_OK video frame successfully deinterlaced.
	\returns B_BAD_DATA No frame could be output.
	\returns B_NO_MEMORY Not enough memory available for correct operation.
*/
status_t
AVCodecDecoder::_ProcessFilterGraph(AVFrame *dst, const AVFrame *src,
	enum AVPixelFormat pixfmt, int32 width, int32 height)
{
	if (fFilterGraph == NULL || width != fLastWidth
		|| height != fLastHeight || pixfmt != fLastPixfmt) {

		status_t err = _InitFilterGraph(pixfmt, width, height);
		if (err != B_OK)
			return err;
	}

	memcpy(fFilterFrame->data, src->data, sizeof(src->data));
	memcpy(fFilterFrame->linesize, src->linesize, sizeof(src->linesize));
	fFilterFrame->width = width;
	fFilterFrame->height = height;
	fFilterFrame->format = pixfmt;

	int ret = av_buffersrc_add_frame(fBufferSourceContext, fFilterFrame);
	if (ret < 0)
		return B_NO_MEMORY;

	ret = av_buffersink_get_frame(fBufferSinkContext, fFilterFrame);
	if (ret < 0)
		return B_BAD_DATA;

	av_image_copy(dst->data, dst->linesize, (const uint8**)fFilterFrame->data,
		fFilterFrame->linesize, pixfmt, width, height);
	av_frame_unref(fFilterFrame);
	return B_OK;
}
