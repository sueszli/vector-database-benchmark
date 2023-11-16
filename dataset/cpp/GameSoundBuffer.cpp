//------------------------------------------------------------------------------
//	Copyright (c) 2001-2002, Haiku
//
//	Permission is hereby granted, free of charge, to any person obtaining a
//	copy of this software and associated documentation files (the "Software"),
//	to deal in the Software without restriction, including without limitation
//	the rights to use, copy, modify, merge, publish, distribute, sublicense,
//	and/or sell copies of the Software, and to permit persons to whom the
//	Software is furnished to do so, subject to the following conditions:
//
//	The above copyright notice and this permission notice shall be included in
//	all copies or substantial portions of the Software.
//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//	FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//	DEALINGS IN THE SOFTWARE.
//
//	File Name:		GameSoundBuffer.h
//	Author:			Christopher ML Zumwalt May (zummy@users.sf.net)
//	Description:	Interface to a single sound, managed by the GameSoundDevice.
//------------------------------------------------------------------------------


#include "GameSoundBuffer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <MediaRoster.h>
#include <MediaAddOn.h>
#include <MediaTheme.h>
#include <TimeSource.h>
#include <BufferGroup.h>

#include "GameProducer.h"
#include "GameSoundDevice.h"
#include "StreamingGameSound.h"
#include "GSUtility.h"

// Sound Buffer Utility functions ----------------------------------------
template<typename T, int32 min, int32 middle, int32 max>
static inline void
ApplyMod(T* data, int64 index, float* pan)
{
	data[index * 2] = clamp<T, min, max>(float(data[index * 2] - middle)
		* pan[0] + middle);
	data[index * 2 + 1] = clamp<T, min, max>(float(data[index * 2 + 1] - middle)
		* pan[1] + middle);
}


// GameSoundBuffer -------------------------------------------------------
GameSoundBuffer::GameSoundBuffer(const gs_audio_format * format)
	:
	fLooping(false),
	fIsConnected(false),
	fIsPlaying(false),
	fGain(1.0),
	fPan(0.0),
	fPanLeft(1.0),
	fPanRight(1.0),
	fGainRamp(NULL),
	fPanRamp(NULL)
{
	fConnection = new Connection;
	fNode = new GameProducer(this, format);

	fFrameSize = get_sample_size(format->format) * format->channel_count;

	fFormat = *format;
}


// Play must stop before the distructor is called; otherwise, a fatal
// error occures if the playback is in a subclass.
GameSoundBuffer::~GameSoundBuffer()
{
	BMediaRoster* roster = BMediaRoster::Roster();

	if (fIsConnected) {
		// Ordinarily we'd stop *all* of the nodes in the chain at this point.
		// However, one of the nodes is the System Mixer, and stopping the Mixer
		// is a Bad Idea (tm). So, we just disconnect from it, and release our
		// references to the nodes that we're using.  We *are* supposed to do
		// that even for global nodes like the Mixer.
		roster->Disconnect(fConnection->producer.node, fConnection->source,
			fConnection->consumer.node, fConnection->destination);

		roster->ReleaseNode(fConnection->producer);
		roster->ReleaseNode(fConnection->consumer);
	}

	delete fGainRamp;
	delete fPanRamp;

	delete fConnection;
	delete fNode;
}


const gs_audio_format &
GameSoundBuffer::Format() const
{
	return fFormat;
}


bool
GameSoundBuffer::IsLooping() const
{
	return fLooping;
}


void
GameSoundBuffer::SetLooping(bool looping)
{
	fLooping = looping;
}


float
GameSoundBuffer::Gain() const
{
	return fGain;
}


status_t
GameSoundBuffer::SetGain(float gain, bigtime_t duration)
{
	if (gain < 0.0 || gain > 1.0)
		return B_BAD_VALUE;

	delete fGainRamp;
	fGainRamp = NULL;

	if (duration > 100000)
		fGainRamp  = InitRamp(&fGain, gain, fFormat.frame_rate, duration);
	else
		fGain = gain;

	return B_OK;
}


float
GameSoundBuffer::Pan() const
{
	return fPan;
}


status_t
GameSoundBuffer::SetPan(float pan, bigtime_t duration)
{
	if (pan < -1.0 || pan > 1.0)
		return B_BAD_VALUE;

	delete fPanRamp;
	fPanRamp = NULL;

	if (duration < 100000) {
		fPan = pan;

		if (fPan < 0.0) {
			fPanLeft = 1.0;
			fPanRight = 1.0 + fPan;
		} else {
			fPanRight = 1.0;
			fPanLeft = 1.0 - fPan;
		}
	} else
		fPanRamp = InitRamp(&fPan, pan, fFormat.frame_rate, duration);

	return B_OK;
}


status_t
GameSoundBuffer::GetAttributes(gs_attribute * attributes,
	size_t attributeCount)
{
	for (size_t i = 0; i < attributeCount; i++) {
		switch (attributes[i].attribute) {
			case B_GS_GAIN:
				attributes[i].value = fGain;
				if (fGainRamp)
					attributes[i].duration = fGainRamp->duration;
				break;

			case B_GS_PAN:
				attributes[i].value = fPan;
				if (fPanRamp)
					attributes[i].duration = fPanRamp->duration;
				break;

			case B_GS_LOOPING:
				attributes[i].value = (fLooping) ? -1.0 : 0.0;
				attributes[i].duration = bigtime_t(0);
				break;

			default:
				attributes[i].value = 0.0;
				attributes[i].duration = bigtime_t(0);
				break;
		}
	}

	return B_OK;
}


status_t
GameSoundBuffer::SetAttributes(gs_attribute * attributes,
	size_t attributeCount)
{
	status_t error = B_OK;

	for (size_t i = 0; i < attributeCount; i++) {
		switch (attributes[i].attribute) {
			case B_GS_GAIN:
				error = SetGain(attributes[i].value, attributes[i].duration);
				break;

			case B_GS_PAN:
				error = SetPan(attributes[i].value, attributes[i].duration);
				break;

			case B_GS_LOOPING:
				fLooping = bool(attributes[i].value);
				break;

			default:
				break;
		}
	}

	return error;
}


void
GameSoundBuffer::Play(void * data, int64 frames)
{
	// Mh... should we add some locking?
	if (!fIsPlaying)
		return;

	if (fFormat.channel_count == 2) {
		float pan[2];
		pan[0] = fPanRight * fGain;
		pan[1] = fPanLeft * fGain;

		FillBuffer(data, frames);

		switch (fFormat.format) {
			case gs_audio_format::B_GS_U8:
			{
				for (int64 i = 0; i < frames; i++) {
					ApplyMod<uint8, 0, 128, UINT8_MAX>((uint8*)data, i, pan);
					UpdateMods();
				}

				break;
			}

			case gs_audio_format::B_GS_S16:
			{
				for (int64 i = 0; i < frames; i++) {
					ApplyMod<int16, INT16_MIN, 0, INT16_MAX>((int16*)data, i,
						pan);
					UpdateMods();
				}

				break;
			}

			case gs_audio_format::B_GS_S32:
			{
				for (int64 i = 0; i < frames; i++) {
					ApplyMod<int32, INT32_MIN, 0, INT32_MAX>((int32*)data, i,
						pan);
					UpdateMods();
				}

				break;
			}

			case gs_audio_format::B_GS_F:
			{
				for (int64 i = 0; i < frames; i++) {
					ApplyMod<float, -1, 0, 1>((float*)data, i, pan);
					UpdateMods();
				}

				break;
			}
		}
	} else if (fFormat.channel_count == 1) {
		// FIXME the output should be stereo, and we could pan mono sounds
		// here. But currently the output has the same number of channels as
		// the sound and we can't do this.
		// FIXME also, we don't handle the gain here.
		FillBuffer(data, frames);
	} else
		debugger("Invalid number of channels.");

}


void
GameSoundBuffer::UpdateMods()
{
	// adjust the gain if needed
	if (fGainRamp) {
		if (ChangeRamp(fGainRamp)) {
			delete fGainRamp;
			fGainRamp = NULL;
		}
	}

	// adjust the ramp if needed
	if (fPanRamp) {
		if (ChangeRamp(fPanRamp)) {
			delete fPanRamp;
			fPanRamp = NULL;
		} else {
			if (fPan < 0.0) {
				fPanLeft = 1.0;
				fPanRight = 1.0 + fPan;
			} else {
				fPanRight = 1.0;
				fPanLeft = 1.0 - fPan;
			}
		}
	}
}


void
GameSoundBuffer::Reset()
{
	fGain = 1.0;
	delete fGainRamp;
	fGainRamp = NULL;

	fPan = 0.0;
	fPanLeft = 1.0;
	fPanRight = 1.0;

	delete fPanRamp;
	fPanRamp = NULL;

	fLooping = false;
}


status_t
GameSoundBuffer::Connect(media_node * consumer)
{
	BMediaRoster* roster = BMediaRoster::Roster();
	status_t err = roster->RegisterNode(fNode);

	if (err != B_OK)
		return err;

	// make sure the Media Roster knows that we're using the node
	err = roster->GetNodeFor(fNode->Node().node, &fConnection->producer);

	if (err != B_OK)
		return err;

	// connect to the mixer
	fConnection->consumer = *consumer;

	// set the producer's time source to be the "default" time source, which
	// the Mixer uses too.
	err = roster->GetTimeSource(&fConnection->timeSource);
	if (err != B_OK)
		return err;

	err = roster->SetTimeSourceFor(fConnection->producer.node,
		fConnection->timeSource.node);
	if (err != B_OK)
		return err;
	// got the nodes; now we find the endpoints of the connection
	media_input mixerInput;
	media_output soundOutput;
	int32 count = 1;
	err = roster->GetFreeOutputsFor(fConnection->producer, &soundOutput, 1,
		&count);

	if (err != B_OK)
		return err;
	count = 1;
	err = roster->GetFreeInputsFor(fConnection->consumer, &mixerInput, 1,
		&count);
	if (err != B_OK)
		return err;

	// got the endpoints; now we connect it!
	media_format format;
	format.type = B_MEDIA_RAW_AUDIO;
	format.u.raw_audio = media_raw_audio_format::wildcard;
	err = roster->Connect(soundOutput.source, mixerInput.destination, &format,
		&soundOutput, &mixerInput);
	if (err != B_OK)
		return err;

	// the inputs and outputs might have been reassigned during the
	// nodes' negotiation of the Connect().  That's why we wait until
	// after Connect() finishes to save their contents.
	fConnection->format = format;
	fConnection->source = soundOutput.source;
	fConnection->destination = mixerInput.destination;

	fIsConnected = true;
	return B_OK;
}


status_t
GameSoundBuffer::StartPlaying()
{
	if (fIsPlaying)
		return EALREADY;

	BMediaRoster* roster = BMediaRoster::Roster();
	BTimeSource* source = roster->MakeTimeSourceFor(fConnection->producer);

	// make sure we give the producer enough time to run buffers through
	// the node chain, otherwise it'll start up already late
	bigtime_t latency = 0;
	status_t status = roster->GetLatencyFor(fConnection->producer, &latency);
	if (status == B_OK) {
		status = roster->StartNode(fConnection->producer,
			source->Now() + latency);
	}
	source->Release();

	fIsPlaying = true;

	return status;
}


status_t
GameSoundBuffer::StopPlaying()
{
	if (!fIsPlaying)
		return EALREADY;

	BMediaRoster* roster = BMediaRoster::Roster();
	roster->StopNode(fConnection->producer, 0, true);
		// synchronous stop

	Reset();
	fIsPlaying = false;

	return B_OK;
}


bool
GameSoundBuffer::IsPlaying()
{
	return fIsPlaying;
}


// SimpleSoundBuffer ------------------------------------------------------
SimpleSoundBuffer::SimpleSoundBuffer(const gs_audio_format * format,
	const void * data, int64 frames)
	:
	GameSoundBuffer(format),
	fPosition(0)
{
	fBufferSize = frames * fFrameSize;
	fBuffer = (char*)data;
}


SimpleSoundBuffer::~SimpleSoundBuffer()
{
	delete [] fBuffer;
}


void
SimpleSoundBuffer::Reset()
{
	GameSoundBuffer::Reset();
	fPosition = 0;
}


void
SimpleSoundBuffer::FillBuffer(void * data, int64 frames)
{
	char * buffer = (char*)data;
	size_t bytes = fFrameSize * frames;

	if (fPosition + bytes >= fBufferSize) {
		if (fPosition < fBufferSize) {
			// copy the remaining frames
			size_t remainder = fBufferSize - fPosition;
			memcpy(buffer, &fBuffer[fPosition], remainder);

			bytes -= remainder;
			buffer += remainder;
		}

		if (fLooping) {
			// restart the sound from the beginning
			memcpy(buffer, fBuffer, bytes);
			fPosition = bytes;
			bytes = 0;
		} else {
			fPosition = fBufferSize;
		}

		if (bytes > 0) {
			// Fill the rest with silence
			int middle = 0;
			if (fFormat.format == gs_audio_format::B_GS_U8)
				middle = 128;
			memset(buffer, middle, bytes);
		}
	} else {
		memcpy(buffer, &fBuffer[fPosition], bytes);
		fPosition += bytes;
	}
}


// StreamingSoundBuffer ------------------------------------------------------
StreamingSoundBuffer::StreamingSoundBuffer(const gs_audio_format * format,
	const void * streamHook, size_t inBufferFrameCount, size_t inBufferCount)
	:
	GameSoundBuffer(format),
	fStreamHook(const_cast<void *>(streamHook))
{
	if (inBufferFrameCount != 0 && inBufferCount  != 0) {
		BBufferGroup *bufferGroup
			= new BBufferGroup(inBufferFrameCount * fFrameSize, inBufferCount);
		fNode->SetBufferGroup(fConnection->source, bufferGroup);
	}
}


StreamingSoundBuffer::~StreamingSoundBuffer()
{
}


void
StreamingSoundBuffer::FillBuffer(void * buffer, int64 frames)
{
	BStreamingGameSound* object = (BStreamingGameSound*)fStreamHook;

	size_t bytes = fFrameSize * frames;
	object->FillBuffer(buffer, bytes);
}
