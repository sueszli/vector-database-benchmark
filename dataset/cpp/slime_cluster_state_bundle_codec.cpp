// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "slime_cluster_state_bundle_codec.h"
#include <vespa/document/bucket/fixed_bucket_spaces.h>
#include <vespa/vdslib/state/clusterstate.h>
#include <vespa/vdslib/state/cluster_state_bundle.h>
#include <vespa/vespalib/data/slime/slime.h>
#include <vespa/vespalib/stllike/asciistream.h>
#include <vespa/vespalib/util/size_literals.h>

using document::FixedBucketSpaces;
using vespalib::slime::Cursor;
using vespalib::slime::BinaryFormat;
using vespalib::DataBuffer;
using vespalib::ConstBufferRef;
using vespalib::compression::CompressionConfig;
using vespalib::compression::decompress;
using vespalib::compression::compress;
using vespalib::Memory;
using namespace vespalib::slime;

namespace storage::rpc {

// TODO find a suitable home for this class to avoid dupes with rpcsendv2.cpp
namespace {
class OutputBuf : public vespalib::Output {
public:
    explicit OutputBuf(size_t estimatedSize) : _buf(estimatedSize) { }
    ~OutputBuf() override;
    vespalib::DataBuffer & getBuf() { return _buf; }
private:
    vespalib::WritableMemory reserve(size_t bytes) override {
        _buf.ensureFree(bytes);
        return {_buf.getFree(), _buf.getFreeLen()};
    }
    Output &commit(size_t bytes) override {
        _buf.moveFreeToData(bytes);
        return *this;
    }
    vespalib::DataBuffer _buf;
};

OutputBuf::~OutputBuf() = default;

vespalib::string serialize_state(const lib::ClusterState& state) {
    vespalib::asciistream as;
    state.serialize(as);
    return as.str();
}

const Memory StatesField("states");
const Memory BaselineField("baseline");
const Memory SpacesField("spaces");
const Memory DeferredActivationField("deferred-activation");
const Memory FeedBlockField("feed-block");
const Memory BlockFeedInClusterField("block-feed-in-cluster");
const Memory DescriptionField("description");

}

// Only used from unit tests; the cluster controller encodes all bundles
// we decode in practice.
EncodedClusterStateBundle SlimeClusterStateBundleCodec::encode(
        const lib::ClusterStateBundle& bundle) const
{
    vespalib::Slime slime;
    Cursor& root = slime.setObject();
    if (bundle.deferredActivation()) {
        root.setBool(DeferredActivationField, bundle.deferredActivation());
    }
    Cursor& states = root.setObject(StatesField);
    states.setString(BaselineField, serialize_state(*bundle.getBaselineClusterState()));
    Cursor& spaces = states.setObject(SpacesField);
    for (const auto& sp : bundle.getDerivedClusterStates()) {
        spaces.setString(FixedBucketSpaces::to_string(sp.first), serialize_state(*sp.second));
    }
    // We only encode feed block state if the cluster is actually blocked.
    if (bundle.block_feed_in_cluster()) {
        Cursor& feed_block = root.setObject(FeedBlockField);
        feed_block.setBool(BlockFeedInClusterField, true);
        feed_block.setString(DescriptionField, bundle.feed_block()->description());
    }

    OutputBuf out_buf(4_Ki);
    BinaryFormat::encode(slime, out_buf);
    ConstBufferRef to_compress(out_buf.getBuf().getData(), out_buf.getBuf().getDataLen());
    auto buf = std::make_unique<DataBuffer>(vespalib::roundUp2inN(out_buf.getBuf().getDataLen()));
    auto actual_type = compress(CompressionConfig::LZ4, to_compress, *buf, false);

    EncodedClusterStateBundle encoded_bundle;
    encoded_bundle._compression_type = actual_type;
    assert(to_compress.size() <= INT32_MAX);
    encoded_bundle._uncompressed_length = to_compress.size();
    encoded_bundle._buffer = std::move(buf);
    return encoded_bundle;
}

namespace {


struct StateInserter : vespalib::slime::ObjectTraverser {
    lib::ClusterStateBundle::BucketSpaceStateMapping& _space_states;

    explicit StateInserter(lib::ClusterStateBundle::BucketSpaceStateMapping& space_states)
        : _space_states(space_states) {}

    void field(const Memory& symbol, const Inspector& inspector) override {
        _space_states.emplace(FixedBucketSpaces::from_string(symbol.make_stringref()),
                              std::make_shared<const lib::ClusterState>(inspector.asString().make_string()));
    }
};

}

std::shared_ptr<const lib::ClusterStateBundle> SlimeClusterStateBundleCodec::decode(
        const EncodedClusterStateBundle& encoded_bundle) const
{
    ConstBufferRef blob(encoded_bundle._buffer->getData(), encoded_bundle._buffer->getDataLen());
    DataBuffer uncompressed;
    decompress(encoded_bundle._compression_type, encoded_bundle._uncompressed_length,
               blob, uncompressed, false);
    if (encoded_bundle._uncompressed_length != uncompressed.getDataLen()) {
        throw std::range_error(vespalib::make_string("ClusterStateBundle indicated uncompressed size (%u) is "
                                                     "not equal to actual uncompressed size (%zu)",
                                                     encoded_bundle._uncompressed_length,
                                                     uncompressed.getDataLen()));
    }

    vespalib::Slime slime;
    BinaryFormat::decode(Memory(uncompressed.getData(), uncompressed.getDataLen()), slime);
    Inspector& root = slime.get();
    Inspector& states = root[StatesField];
    lib::ClusterState baseline(states[BaselineField].asString().make_string());

    Inspector& spaces = states[SpacesField];
    lib::ClusterStateBundle::BucketSpaceStateMapping space_states;
    StateInserter inserter(space_states);
    spaces.traverse(inserter);

    const bool deferred_activation = root[DeferredActivationField].asBool(); // Defaults to false if not set.

    Inspector& fb = root[FeedBlockField];
    if (fb.valid()) {
        lib::ClusterStateBundle::FeedBlock feed_block(fb[BlockFeedInClusterField].asBool(),
                                                      fb[DescriptionField].asString().make_string());
        return std::make_shared<lib::ClusterStateBundle>(baseline, std::move(space_states), feed_block, deferred_activation);
    }

    // TODO add shared_ptr constructor for baseline?
    return std::make_shared<lib::ClusterStateBundle>(baseline, std::move(space_states), deferred_activation);
}

}
