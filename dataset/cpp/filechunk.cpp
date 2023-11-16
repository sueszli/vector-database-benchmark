// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "filechunk.h"
#include "data_store_file_chunk_stats.h"
#include "summaryexceptions.h"
#include "randreaders.h"
#include <vespa/searchlib/util/filekit.h>
#include <vespa/vespalib/util/lambdatask.h>
#include <vespa/vespalib/data/fileheader.h>
#include <vespa/vespalib/data/databuffer.h>
#include <vespa/vespalib/stllike/asciistream.h>
#include <vespa/vespalib/objects/nbostream.h>
#include <vespa/vespalib/util/executor.h>
#include <vespa/vespalib/util/arrayqueue.hpp>
#include <vespa/fastos/file.h>
#include <filesystem>
#include <future>

#include <vespa/log/log.h>
LOG_SETUP(".search.filechunk");

using vespalib::CpuUsage;
using vespalib::GenericHeader;
using vespalib::getErrorString;

namespace search {

namespace {

constexpr size_t ALIGNMENT=0x1000;
constexpr size_t ENTRY_BIAS_SIZE=8;
const vespalib::string DOC_ID_LIMIT_KEY("docIdLimit");

}

using vespalib::make_string;

FileChunk::ChunkInfo::ChunkInfo(uint64_t offset, uint32_t size, uint64_t lastSerial) noexcept
    : _lastSerial(lastSerial),
      _offset(offset),
      _size(size)
{
    assert(valid());
}

vespalib::string
FileChunk::NameId::createName(const vespalib::string &baseName) const {
    vespalib::asciistream os;
    os << baseName << '/' << vespalib::setfill('0') << vespalib::setw(19) << getId();
    return os.str();
}

vespalib::string
FileChunk::createIdxFileName(const vespalib::string & name) {
    return name + ".idx";
}

vespalib::string
FileChunk::createDatFileName(const vespalib::string & name) {
    return name + ".dat";
}

FileChunk::FileChunk(FileId fileId, NameId nameId, const vespalib::string & baseName,
                     const TuneFileSummary & tune, const IBucketizer * bucketizer)
    : _fileId(fileId),
      _nameId(nameId),
      _name(nameId.createName(baseName)),
      _erasedCount(0),
      _erasedBytes(0),
      _diskFootprint(0),
      _sumNumBuckets(0),
      _numChunksWithBuckets(0),
      _numUniqueBuckets(0),
      _file(),
      _bucketizer(bucketizer),
      _addedBytes(0),
      _tune(tune),
      _dataFileName(createDatFileName(_name)),
      _idxFileName(createIdxFileName(_name)),
      _chunkInfo(),
      _lastPersistedSerialNum(0),
      _dataHeaderLen(0u),
      _idxHeaderLen(0u),
      _numLids(0),
      _docIdLimit(std::numeric_limits<uint32_t>::max()),
      _modificationTime()
{
    FastOS_File dataFile(_dataFileName.c_str());
    if (dataFile.OpenReadOnly()) {
        if (!dataFile.Sync()) {
            throw SummaryException("Failed syncing dat file", dataFile, VESPA_STRLOC);
        }
        _diskFootprint.fetch_add(dataFile.getSize(), std::memory_order_relaxed);
        FastOS_File idxFile(_idxFileName.c_str());
        if (idxFile.OpenReadOnly()) {
            if (!idxFile.Sync()) {
                throw SummaryException("Failed syncing idx file", idxFile, VESPA_STRLOC);
            }
            _diskFootprint.fetch_add(idxFile.getSize(), std::memory_order_relaxed);
            _modificationTime = FileKit::getModificationTime(_idxFileName);
        } else {
            throw SummaryException("Failed opening idx file", idxFile, VESPA_STRLOC);
        }
    }
}

FileChunk::~FileChunk() = default;

void
FileChunk::addNumBuckets(size_t numBucketsInChunk)
{
    _sumNumBuckets += numBucketsInChunk;
    if (numBucketsInChunk != 0) {
        ++_numChunksWithBuckets;
    }
}

void
FileChunk::TmpChunkMeta::fill(vespalib::nbostream & is) {
    resize(getNumEntries());
    for (LidMeta & lm : *this) {
        lm.deserialize(is);
    }
}

void
FileChunk::erase()
{
    _file.reset();
    std::filesystem::remove(std::filesystem::path(_idxFileName));
    std::filesystem::remove(std::filesystem::path(_dataFileName));
}

void
FileChunk::updateLidMap(const unique_lock &guard, ISetLid &ds, uint64_t serialNum, uint32_t docIdLimit)
{
    assert(_chunkInfo.empty());

    FastOS_File idxFile(_idxFileName.c_str());
    idxFile.enableMemoryMap(0);
    if ( ! idxFile.OpenReadOnly()) {
        LOG_ABORT("should not reach here");
    }
    if ( ! idxFile.IsMemoryMapped()) {
        assert(idxFile.getSize() == 0);
        return;
    }
    const int64_t fileSize = idxFile.getSize();
    if (_idxHeaderLen == 0) {
        _idxHeaderLen = readIdxHeader(idxFile, _docIdLimit);
    }
    BucketDensityComputer globalBucketMap(_bucketizer);
    // Guard comes from the same bucketizer so the same guard can be used
    // for both local and global BucketDensityComputer
    vespalib::GenerationHandler::Guard bucketizerGuard = globalBucketMap.getGuard();
    vespalib::nbostream is(static_cast<const char *>(idxFile.MemoryMapPtr(0)) + _idxHeaderLen,
                           fileSize - _idxHeaderLen);
    for (size_t count=0; ! is.empty() && is.good(); count++) {
        const int64_t lastKnownGoodPos = _idxHeaderLen + is.rp();
        TmpChunkMeta chunkMeta;
        try {
            chunkMeta.deserialize(is);
            chunkMeta.fill(is);
            if ((count == 0) && (chunkMeta.getLastSerial() < serialNum)) {
                LOG(warning, "last serial num(%" PRIu64 ") from previous file is bigger than my first(%" PRIu64
                             "). That is odd.Current filename is '%s'",
                    serialNum, chunkMeta.getLastSerial(), _idxFileName.c_str());
                serialNum = chunkMeta.getLastSerial();
            }
            assert(serialNum <= chunkMeta.getLastSerial());
            serialNum = handleChunk(guard, ds, docIdLimit, bucketizerGuard, globalBucketMap, chunkMeta);
            assert(serialNum >= _lastPersistedSerialNum.load(std::memory_order_relaxed));
            _lastPersistedSerialNum.store(serialNum, std::memory_order_relaxed);
        } catch (const vespalib::IllegalStateException & e) {
            LOG(warning, "Exception deserializing idx file : %s", e.what());
            LOG(warning, "File '%s' seems to be partially truncated. Will truncate from size=%" PRId64 " to %" PRId64,
                         _idxFileName.c_str(), fileSize, lastKnownGoodPos);
            FastOS_File toTruncate(_idxFileName.c_str());
            if ( toTruncate.OpenReadWrite()) {
                if (toTruncate.SetSize(lastKnownGoodPos)) {
                } else {
                    throw SummaryException("SetSize() failed.", toTruncate, VESPA_STRLOC);
                }
            } else {
                throw SummaryException("Open for truncation failed.", toTruncate, VESPA_STRLOC);
            }
            break;
        }
    }
    _numUniqueBuckets = globalBucketMap.getNumBuckets();
}

uint64_t
FileChunk::handleChunk(const unique_lock &guard, ISetLid &ds, uint32_t docIdLimit,
                       const vespalib::GenerationHandler::Guard & bucketizerGuard, BucketDensityComputer &globalBucketMap,
                       const TmpChunkMeta & chunkMeta) {
    BucketDensityComputer bucketMap(_bucketizer);
    for (size_t i(0), m(chunkMeta.getNumEntries()); i < m; i++) {
        const LidMeta & lidMeta(chunkMeta[i]);
        if (lidMeta.getLid() < docIdLimit) {
            if (_bucketizer && (lidMeta.size() > 0)) {
                document::BucketId bucketId = _bucketizer->getBucketOf(bucketizerGuard, lidMeta.getLid());
                bucketMap.recordLid(bucketId);
                globalBucketMap.recordLid(bucketId);
            }
            ds.setLid(guard, lidMeta.getLid(), LidInfo(getFileId().getId(), _chunkInfo.size(), lidMeta.size()));
            _numLids++;
        } else {
            remove(lidMeta.getLid(), lidMeta.size());
        }
        _addedBytes.store(getAddedBytes() + adjustSize(lidMeta.size()), std::memory_order_relaxed);
    }
    uint64_t serialNum = chunkMeta.getLastSerial();
    addNumBuckets(bucketMap.getNumBuckets());
    _chunkInfo.emplace_back(chunkMeta.getOffset(), chunkMeta.getSize(), chunkMeta.getLastSerial());
    return serialNum;
}


void
FileChunk::enableRead()
{
    if (_tune._randRead.getWantDirectIO()) {
        LOG(debug, "enableRead(): DirectIORandRead: file='%s'", _dataFileName.c_str());
        _file = std::make_unique<DirectIORandRead>(_dataFileName);
    } else if (_tune._randRead.getWantMemoryMap()) {
        const int mmapFlags(_tune._randRead.getMemoryMapFlags());
        const int fadviseOptions(_tune._randRead.getAdvise());
        if (frozen()) {
            LOG(debug, "enableRead(): MMapRandRead: file='%s'", _dataFileName.c_str());
            _file = std::make_unique<MMapRandRead>(_dataFileName, mmapFlags, fadviseOptions);
        } else {
            LOG(debug, "enableRead(): MMapRandReadDynamic: file='%s'", _dataFileName.c_str());
            _file = std::make_unique<MMapRandReadDynamic>(_dataFileName, mmapFlags, fadviseOptions);
        }
    } else {
        LOG(debug, "enableRead(): NormalRandRead: file='%s'", _dataFileName.c_str());
        _file = std::make_unique<NormalRandRead>(_dataFileName);
    }
    _dataHeaderLen = readDataHeader(*_file);
    if (_dataHeaderLen == 0u) {
        throw std::runtime_error(make_string("bad file header: %s", _dataFileName.c_str()));
    }
}

size_t FileChunk::adjustSize(size_t sz) {
    return sz + ENTRY_BIAS_SIZE;
}
void
FileChunk::remove(uint32_t lid, uint32_t size)
{
     (void) lid;
     _erasedCount.store(getErasedCount() + 1, std::memory_order_relaxed);
     _erasedBytes.store(getErasedBytes() + adjustSize(size), std::memory_order_relaxed);
}

uint64_t
FileChunk::getLastPersistedSerialNum() const
{
    return _lastPersistedSerialNum.load(std::memory_order_relaxed);
}

vespalib::system_time
FileChunk::getModificationTime() const
{
    return _modificationTime;
}

namespace {

struct FixedParams {
    const IGetLid & db;
    IWriteData & dest;
    const vespalib::GenerationHandler::Guard & lidReadGuard;
    uint32_t fileId;
    IFileChunkVisitorProgress *visitorProgress;
};

void
appendChunks(FixedParams * args, Chunk::UP chunk)
{
    const Chunk::LidList ll(chunk->getUniqueLids());
    for (const Chunk::Entry & e : ll) {
        LidInfo lidInfo(args->fileId, chunk->getId(), e.netSize());
        if (args->db.getLid(args->lidReadGuard, e.getLid()) == lidInfo) {
            auto guard(args->db.getLidGuard(e.getLid()));
            if (args->db.getLid(args->lidReadGuard, e.getLid()) == lidInfo) {
                // I am still in use, so I need to be taken care of.
                vespalib::ConstBufferRef data(chunk->getLid(e.getLid()));
                args->dest.write(std::move(guard), chunk->getId(), e.getLid(), data);
            }
        }
    }
    if (args->visitorProgress != nullptr) {
        args->visitorProgress->updateProgress();
    }
}

}

void
FileChunk::appendTo(vespalib::Executor & executor, const IGetLid & db, IWriteData & dest,
                    uint32_t numChunks, IFileChunkVisitorProgress *visitorProgress,
                    vespalib::CpuUsage::Category cpu_category)
{
    assert(frozen() || visitorProgress);
    vespalib::GenerationHandler::Guard lidReadGuard(db.getLidReadGuard());
    assert(numChunks <= getNumChunks());
    FixedParams fixedParams = {db, dest, lidReadGuard, getFileId().getId(), visitorProgress};
    size_t limit = std::thread::hardware_concurrency();
    vespalib::ArrayQueue<std::future<Chunk::UP>> queue;
    for (size_t chunkId(0); chunkId < numChunks; chunkId++) {
        std::promise<Chunk::UP> promisedChunk;
        std::future<Chunk::UP> futureChunk = promisedChunk.get_future();
        auto task = vespalib::makeLambdaTask([promise = std::move(promisedChunk), chunkId, this]() mutable {
            const ChunkInfo & cInfo(_chunkInfo[chunkId]);
            vespalib::DataBuffer whole(0ul, ALIGNMENT);
            FileRandRead::FSP keepAlive(_file->read(cInfo.getOffset(), whole, cInfo.getSize()));
            promise.set_value(std::make_unique<Chunk>(chunkId, whole.getData(), whole.getDataLen()));
        });
        executor.execute(CpuUsage::wrap(std::move(task), cpu_category));

        while (queue.size() >= limit) {
            appendChunks(&fixedParams, queue.front().get());
            queue.pop();
        }

        queue.push(std::move(futureChunk));
    }
    while ( ! queue.empty() ) {
        appendChunks(&fixedParams, queue.front().get());
        queue.pop();
    }
    dest.close();
}

void
FileChunk::read(LidInfoWithLidV::const_iterator begin, size_t count, IBufferVisitor & visitor) const
{
    if (count == 0) { return; }
    uint32_t prevChunk = begin->getChunkId();
    uint32_t start(0);
    for (size_t i(0); i < count; i++) {
        const LidInfoWithLid & li = *(begin + i);
        if (li.getChunkId() != prevChunk) {
            ChunkInfo ci = _chunkInfo[prevChunk];
            read(begin + start, i - start, ci, visitor);
            prevChunk = li.getChunkId();
            start = i;
        }
    }
    ChunkInfo ci = _chunkInfo[prevChunk];
    read(begin + start, count - start, ci, visitor);
}

void
FileChunk::read(LidInfoWithLidV::const_iterator begin, size_t count, ChunkInfo ci, IBufferVisitor & visitor) const
{
    vespalib::DataBuffer whole(0ul, ALIGNMENT);
    FileRandRead::FSP keepAlive = _file->read(ci.getOffset(), whole, ci.getSize());
    Chunk chunk(begin->getChunkId(), whole.getData(), whole.getDataLen());
    for (size_t i(0); i < count; i++) {
        const LidInfoWithLid & li = *(begin + i);
        vespalib::ConstBufferRef buf = chunk.getLid(li.getLid());
        if (buf.size() != 0) {
            visitor.visit(li.getLid(), buf);
        }
    }
}

ssize_t
FileChunk::read(uint32_t lid, SubChunkId chunkId,
                vespalib::DataBuffer & buffer) const
{
    return (chunkId < _chunkInfo.size())
        ? read(lid, chunkId, _chunkInfo[chunkId], buffer)
        : -1;
}

ssize_t
FileChunk::read(uint32_t lid, SubChunkId chunkId, const ChunkInfo & chunkInfo,
                vespalib::DataBuffer & buffer) const
{
    vespalib::DataBuffer whole(0ul, ALIGNMENT);
    FileRandRead::FSP keepAlive(_file->read(chunkInfo.getOffset(), whole, chunkInfo.getSize()));
    Chunk chunk(chunkId, whole.getData(), whole.getDataLen());
    return chunk.read(lid, buffer);
}

uint64_t
FileChunk::readDataHeader(FileRandRead &datFile)
{
    uint64_t dataHeaderLen(0);
    int64_t fileSize = datFile.getSize();
    uint32_t hl = GenericHeader::getMinSize();
    if (fileSize >= hl) {
        vespalib::DataBuffer h(hl, ALIGNMENT);
        datFile.read(0, h, hl);
        GenericHeader::BufferReader rd(h);
        uint32_t headerLen = GenericHeader::readSize(rd);
        if (headerLen <= fileSize) {
            dataHeaderLen = headerLen;
        }
    }
    return dataHeaderLen;
}


uint64_t
FileChunk::readIdxHeader(FastOS_FileInterface &idxFile, uint32_t &docIdLimit)
{
    int64_t fileSize = idxFile.getSize();
    uint32_t hl = GenericHeader::getMinSize();
    uint64_t idxHeaderLen = 0;
    if (fileSize >= hl) {
        GenericHeader::MMapReader rd(static_cast<const char *> (idxFile.MemoryMapPtr(0)), hl);
        uint32_t headerLen = GenericHeader::readSize(rd);
        if (headerLen <= fileSize) {
            idxHeaderLen = headerLen;
        }
    }
    if (idxHeaderLen == 0u) {
        throw SummaryException("bad file header", idxFile, VESPA_STRLOC);
    }
    GenericHeader::MMapReader reader(static_cast<const char *> (idxFile.MemoryMapPtr(0)), idxHeaderLen);
    GenericHeader header;
    header.read(reader);
    docIdLimit = readDocIdLimit(header);
    return idxHeaderLen;
}

uint32_t
FileChunk::readDocIdLimit(vespalib::GenericHeader &header)
{
    if (header.hasTag(DOC_ID_LIMIT_KEY)) {
        return header.getTag(DOC_ID_LIMIT_KEY).asInteger();
    } else {
        return std::numeric_limits<uint32_t>::max();
    }
}

void
FileChunk::writeDocIdLimit(vespalib::GenericHeader &header, uint32_t docIdLimit)
{
    header.putTag(vespalib::GenericHeader::Tag(DOC_ID_LIMIT_KEY, docIdLimit));
}

void
FileChunk::verify(bool reportOnly) const
{
    (void) reportOnly;
    LOG(info,
        "Verifying file '%s' with fileid '%u'. erased-count='%zu' and erased-bytes='%zu'. diskFootprint='%zu'",
        _name.c_str(), _fileId.getId(), getErasedCount(), getErasedBytes(), _diskFootprint.load(std::memory_order_relaxed));
    uint64_t lastSerial(0);
    size_t chunkId(0);
    bool errorInPrev(false);
    for (const ChunkInfo & ci : _chunkInfo) {
        vespalib::DataBuffer whole(0ul, ALIGNMENT);
        FileRandRead::FSP keepAlive(_file->read(ci.getOffset(), whole, ci.getSize()));
        try {
            Chunk chunk(chunkId++, whole.getData(), whole.getDataLen());
            assert(chunk.getLastSerial() >= lastSerial);
            lastSerial = chunk.getLastSerial();
            if (errorInPrev) {
                LOG(error, "Last serial number in first good chunk is %" PRIu64, chunk.getLastSerial());
                errorInPrev = false;
            }
        } catch (const std::exception & e) {
            LOG(error,
                "Errors in chunk number %zu/%zu at file offset %" PRIu64 " and size %u."
                " Last known good serial number = %" PRIu64 "\n.Got Exception : %s",
                chunkId, _chunkInfo.size(), ci.getOffset(), ci.getSize(), lastSerial, e.what());
            errorInPrev = true;
        }
    }
}

uint32_t
FileChunk::getNumChunks() const
{
    return _chunkInfo.size();
}

size_t
FileChunk::getMemoryFootprint() const
{
    // The memory footprint does not vary before or after flush
    // Once frozen, there is no variable component.
    // It is all captured by getMemoryMetaFootprint()
    return 0;
}   
    
size_t
FileChunk::getMemoryMetaFootprint() const
{
    return sizeof(*this) + _chunkInfo.capacity()*sizeof(ChunkInfoVector::value_type);
}

vespalib::MemoryUsage
FileChunk::getMemoryUsage() const
{
    vespalib::MemoryUsage result;
    result.incAllocatedBytes(sizeof(*this));
    result.incUsedBytes(sizeof(*this));
    result.incAllocatedBytes(_chunkInfo.capacity()*sizeof(ChunkInfoVector::value_type));
    result.incUsedBytes(_chunkInfo.size()*sizeof(ChunkInfoVector::value_type));
    return result;
}

bool
FileChunk::isIdxFileEmpty(const vespalib::string & name)
{
    vespalib::string fileName(name + ".idx");
    FastOS_File idxFile(fileName.c_str());
    idxFile.enableMemoryMap(0);
    if (idxFile.OpenReadOnly()) {
        if (idxFile.IsMemoryMapped()) {
            int64_t fileSize = idxFile.getSize();
            uint32_t docIdLimit = std::numeric_limits<uint32_t>::max();
            int64_t idxHeaderLen = FileChunk::readIdxHeader(idxFile, docIdLimit);
            return fileSize <= idxHeaderLen;
        } else if ( idxFile.getSize() == 0u) {
            return true;
        } else {
            throw SummaryException("Failed opening idx file for memorymapping", idxFile, VESPA_STRLOC);
        }
    } else {
        throw SummaryException("Failed opening idx file readonly ", idxFile, VESPA_STRLOC);
    }
}

void
FileChunk::eraseIdxFile(const vespalib::string & name)
{
    vespalib::string fileName(createIdxFileName(name));
    std::filesystem::remove(std::filesystem::path(fileName));
}

void
FileChunk::eraseDatFile(const vespalib::string & name)
{
    vespalib::string fileName(createDatFileName(name));
    std::filesystem::remove(std::filesystem::path(fileName));
}


DataStoreFileChunkStats
FileChunk::getStats() const
{
    uint64_t diskFootprint = getDiskFootprint();
    uint64_t diskBloat = getDiskBloat();
    double bucketSpread = getBucketSpread();
    uint64_t serialNum = getLastPersistedSerialNum();
    uint32_t docIdLimit = getDocIdLimit();
    uint64_t nameId = getNameId().getId();
    return {diskFootprint, diskBloat, bucketSpread, serialNum, serialNum, docIdLimit, nameId};
}

} // namespace search
