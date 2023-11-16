// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "bitvectorfile.h"
#include <vespa/searchlib/common/bitvector.h>
#include <vespa/searchlib/common/fileheadercontext.h>
#include <vespa/searchlib/index/bitvectorkeys.h>
#include <vespa/searchlib/util/file_settings.h>
#include <vespa/vespalib/data/fileheader.h>
#include <vespa/vespalib/util/size_literals.h>
#include <vespa/fastlib/io/bufferedfile.h>
#include <cassert>

namespace search::diskindex {

using search::index::BitVectorWordSingleKey;
using search::common::FileHeaderContext;

namespace {

void
readHeader(vespalib::FileHeader &h,
           const vespalib::string &name)
{
    Fast_BufferedFile file(32_Ki);
    file.ReadOpenExisting(name.c_str());
    h.readFile(file);
}

}

BitVectorFileWrite::BitVectorFileWrite(BitVectorKeyScope scope)
    : BitVectorIdxFileWrite(scope),
      _datFile(),
      _datHeaderLen(0)
{
}


BitVectorFileWrite::~BitVectorFileWrite() = default;


void
BitVectorFileWrite::open(const vespalib::string &name,
                         uint32_t docIdLimit,
                         const TuneFileSeqWrite &tuneFileWrite,
                         const FileHeaderContext &fileHeaderContext)
{
    vespalib::string datname = name + ".bdat";

    assert( ! _datFile);

    Parent::open(name, docIdLimit, tuneFileWrite, fileHeaderContext);

    _datFile = std::make_unique<Fast_BufferedFile>(new FastOS_File);
    if (tuneFileWrite.getWantSyncWrites()) {
        _datFile->EnableSyncWrites();
    }
    if (tuneFileWrite.getWantDirectIO()) {
        _datFile->EnableDirectIO();
    }
    _datFile->WriteOpen(datname.c_str());

    if (_datHeaderLen == 0) {
        assert(_numKeys == 0);
        makeDatHeader(fileHeaderContext);
    }

    int64_t pos;
    size_t bitmapbytes;

    bitmapbytes = BitVector::getFileBytes(_docIdLimit);

    pos = static_cast<int64_t>(_numKeys) *
          static_cast<int64_t>(bitmapbytes) + _datHeaderLen;

    int64_t olddatsize = _datFile->getSize();
    assert(olddatsize >= pos);
    (void) olddatsize;

    _datFile->SetSize(pos);

    assert(pos == _datFile->getPosition());
}


void
BitVectorFileWrite::makeDatHeader(const FileHeaderContext &fileHeaderContext)
{
    vespalib::FileHeader h(FileSettings::DIRECTIO_ALIGNMENT);
    using Tag = vespalib::GenericHeader::Tag;
    fileHeaderContext.addTags(h, _datFile->GetFileName());
    h.putTag(Tag("docIdLimit", _docIdLimit));
    h.putTag(Tag("numKeys", _numKeys));
    h.putTag(Tag("frozen", 0));
    h.putTag(Tag("fileBitSize", 0));
    h.putTag(Tag("desc", "Bitvector data file"));
    _datFile->SetPosition(0);
    _datHeaderLen = h.writeFile(*_datFile);
    _datFile->Flush();
}


void
BitVectorFileWrite::updateDatHeader(uint64_t fileBitSize)
{
    vespalib::FileHeader h(FileSettings::DIRECTIO_ALIGNMENT);
    using Tag = vespalib::GenericHeader::Tag;
    readHeader(h, _datFile->GetFileName());
    FileHeaderContext::setFreezeTime(h);
    h.putTag(Tag("numKeys", _numKeys));
    h.putTag(Tag("frozen", 1));
    h.putTag(Tag("fileBitSize", fileBitSize));
    bool sync_ok = _datFile->Sync();
    assert(sync_ok);
    assert(h.getSize() == _datHeaderLen);
    _datFile->SetPosition(0);
    h.writeFile(*_datFile);
    sync_ok = _datFile->Sync();
    assert(sync_ok);
}


void
BitVectorFileWrite::addWordSingle(uint64_t wordNum,
                                  const BitVector &bitVector)
{
    assert(bitVector.size() == _docIdLimit);
    bitVector.invalidateCachedCount();
    Parent::addWordSingle(wordNum, bitVector.countTrueBits());
    _datFile->WriteBuf(bitVector.getStart(),
                       bitVector.getFileBytes());
}


void
BitVectorFileWrite::flush()
{
    Parent::flush();
    _datFile->Flush();
}


void
BitVectorFileWrite::sync()
{
    flush();
    Parent::syncCommon();
    bool sync_ok = _datFile->Sync();
    assert(sync_ok);
}


void
BitVectorFileWrite::close()
{
    size_t bitmapbytes = BitVector::getFileBytes(_docIdLimit);

    if (_datFile != nullptr) {
        if (_datFile->IsOpened()) {
            uint64_t pos = _datFile->getPosition();
            assert(pos == static_cast<uint64_t>(_numKeys) *
                   static_cast<uint64_t>(bitmapbytes) + _datHeaderLen);
            (void) bitmapbytes;
            _datFile->alignEndForDirectIO();
            updateDatHeader(pos * 8);
            bool close_ok = _datFile->Close();
            assert(close_ok);
        }
        _datFile.reset();
    }
    Parent::close();
}

BitVectorCandidate::~BitVectorCandidate() = default;

}
