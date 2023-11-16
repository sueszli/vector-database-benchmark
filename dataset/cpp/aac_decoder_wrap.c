//
//  aac_decoder_wrap.c
//  DeaDBeeF
//
//  Created by Oleksiy Yakovenko on 4/4/20.
//  Copyright © 2020 Oleksiy Yakovenko. All rights reserved.
//

#include "aac_decoder_protocol.h"

void
aacDecoderClose (aacDecoderHandle_t *dec) {
    dec->callbacks->close (dec);
}

int
aacDecoderInit (aacDecoderHandle_t *dec, uint8_t *buff, size_t buffSize, unsigned *samplerate, unsigned *channels) {
    return dec->callbacks->init (dec, buff, buffSize, samplerate, channels);
}

int
aacDecoderInitRaw (aacDecoderHandle_t *dec, uint8_t *buff, size_t buffSize, unsigned *samplerate, unsigned *channels) {
    return dec->callbacks->initRaw (dec, buff, buffSize, samplerate, channels);
}


uint8_t *
aacDecoderDecodeFrame (aacDecoderHandle_t *dec, aacDecoderFrameInfo_t *frameInfo, const uint8_t *buffer, size_t bufferSize) {
    return dec->callbacks->decodeFrame (dec, frameInfo, buffer, bufferSize);
}
