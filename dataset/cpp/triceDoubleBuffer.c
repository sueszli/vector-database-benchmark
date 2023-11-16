//! \file triceDoubleBuffer.c
//! \author Thomas.Hoehenleitner [at] seerose.net
//! //////////////////////////////////////////////////////////////////////////
#include "trice.h"

#if TRICE_BUFFER == TRICE_DOUBLE_BUFFER

static void TriceOut( uint32_t* tb, size_t tLen );

//! triceBuffer is a double buffer for better write speed.
static uint32_t triceBuffer[2][TRICE_DEFERRED_BUFFER_SIZE/8] = {0}; 

//! triceSwap is the index of the active write buffer. !triceSwap is the active read buffer index.
static int triceSwap = 0;

//! TriceBufferWritePosition is the active write position.
uint32_t* TriceBufferWritePosition = &triceBuffer[0][TRICE_DATA_OFFSET>>2];

//! TriceBufferWritePosition is used by TRICE_PUT macros.
uint32_t* TriceBufferLastWritePosition;

//! triceBufferWriteLimit is the triceBuffer written limit. 
static uint32_t* triceBufferWriteLimit = &triceBuffer[1][TRICE_DATA_OFFSET>>2];

#if TRICE_DIAGNOSTICS == 1

//! TriceSingleMaxWordCount is a diagnostics value usable to optimize buffer size.
unsigned TriceSingleMaxWordCount = 0;

//! TriceHalfBufferDepthMax is a diagnostics value usable to optimize buffer size.
unsigned TriceHalfBufferDepthMax = 0; 

#endif

//! triceBufferSwap swaps the trice double buffer and returns the read buffer address.
static uint32_t* triceBufferSwap( void ){
    TRICE_ENTER_CRITICAL_SECTION
    triceBufferWriteLimit = TriceBufferWritePosition; // keep end position
    triceSwap = !triceSwap; // exchange the 2 buffers
    TriceBufferWritePosition = &triceBuffer[triceSwap][TRICE_DATA_OFFSET>>2]; // set write position for next TRICE
    TRICE_LEAVE_CRITICAL_SECTION
    return &triceBuffer[!triceSwap][0]; //lint !e514
}

//! triceDepth returns the total trice byte count ready for transfer.
//! The trice data start at tb + TRICE_DATA_OFFSET.
//! The returned depth is without the TRICE_DATA_OFFSET offset.
static size_t triceDepth( uint32_t const* tb ){
    size_t depth = (triceBufferWriteLimit - tb)<<2; //lint !e701 // 32-bit write width 
    return depth - TRICE_DATA_OFFSET;
}

//! TriceTransfer, if possible, swaps the double buffer and initiates a write.
//! It is the resposibility of the app to call this function once every 10-100 milliseconds.
void TriceTransfer( void ){
    if( 0 == TriceOutDepth() ){ // transmission done for slowest output channel, so a swap is possible
        uint32_t* tb = triceBufferSwap(); 
        size_t tLen = triceDepth(tb); // tlen is always a multiple of 4
        if( tLen ){
            TriceOut( tb, tLen );
        }
    } // else: transmission not done yet
}

//! TriceOut encodes trices and writes them in one step to the output.
//! This function is called only, when the slowest deferred output device has finished its last buffer.
//! At the half buffer start tb,ls -l are TRICE_DATA_OFFSET bytes space followed by a number of trice messages which all contain
//! 0-3 padding bytes and therefor have a length of a multiple of 4. There is no additional space between these trice messages.
//! When XTEA enabled, only (TRICE_TRANSFER_MODE == TRICE_PACK_MULTI_MODE) is allowed, because the 4 bytes behind a trice messages
//! are changed, when the trice length is not a multiple of 8, but only of 4. (XTEA can encrypt only multiple of 8 lenth packages.)
//! \param tb is start of uint32_t* trice buffer. The space TRICE_DATA_OFFSET at
//! the tb start is for in-buffer encoding of the trice data.
//! \param tLen is length of trice data. tlen is always a multiple of 4 because
//! of 32-bit alignment and padding bytes.
static void TriceOut( uint32_t* tb, size_t tLen ){
    uint8_t* enc = (uint8_t*)tb; // encoded data starting address
    size_t encLen = 0;
    uint8_t* buf = enc + TRICE_DATA_OFFSET; // start of 32-bit aligned trices
    size_t len = tLen; // (byte count)
    int triceID;
    #if TRICE_DIAGNOSTICS == 1
    tLen += TRICE_DATA_OFFSET; 
    TriceHalfBufferDepthMax = tLen < TriceHalfBufferDepthMax ? TriceHalfBufferDepthMax : tLen;
    #endif
    // do it
    while(len){
        uint8_t* triceStart;
        size_t triceLen; // This is the trice netto length (without padding bytes).
        triceID = TriceNext( &buf, &len, &triceStart, &triceLen );
        if( triceID <= 0 ){ // on data error
            break;   // ignore following data
        }
        #if TRICE_TRANSFER_MODE == TRICE_SAFE_SINGLE_MODE
            #ifdef XTEA_ENCRYPT_KEY
                // Behind the trice brutto length (with padding bytes), 4 bytes could be used as scratch pad when XTEA is active.
                // Therefore, when XTEA is used, the single trice must be moved first by 4 bytes in lower address direction if its length is not a multiple of 4.
                #error not implemented
            #endif
        encLen += TriceDeferredEncode( enc+encLen, triceStart, triceLen );
        #endif
        #if  TRICE_TRANSFER_MODE == TRICE_PACK_MULTI_MODE
        // This action removes all padding bytes of the trices, compacting their sequence this way
        memmove(enc + TRICE_DATA_OFFSET + encLen, triceStart, triceLen );
        encLen += triceLen;
        #endif
    }
    #if TRICE_TRANSFER_MODE == TRICE_PACK_MULTI_MODE
    // At this point the compacted trice messages start TRICE_DATA_OFFSET bytes after tb (now enc) and the encLen is their total netto length.
    // Behind this up to 7 bytes can be used as scratch pad when XTEA is active. That is ok, because the half buffer should not get totally filled.
    encLen = TriceDeferredEncode( enc, enc + TRICE_DATA_OFFSET, encLen);
    #endif

    // Reaching here means all trice data in the current double buffer are encoded
    // into a single continuous buffer having 0-delimiters between them or not but at the ent is a 0-delimiter.
    //
    // output
    TriceNonBlockingDeferredWrite( triceID, enc, encLen ); //lint !e771 Info 771: Symbol 'triceID' conceivably not initialized. Comment: tLen is always > 0.
}

#endif // #if TRICE_BUFFER == TRICE_DOUBLE_BUFFER
