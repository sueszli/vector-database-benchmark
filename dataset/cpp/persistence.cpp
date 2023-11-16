// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "persistence.h"
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/update/documentupdate.h>
#include <vespa/vespalib/util/exceptions.h>
#include <vespa/vespalib/stllike/asciistream.h>
#include <ostream>

namespace storage::api {

IMPLEMENT_COMMAND(PutCommand, PutReply)
IMPLEMENT_REPLY(PutReply)
IMPLEMENT_COMMAND(UpdateCommand, UpdateReply)
IMPLEMENT_REPLY(UpdateReply)
IMPLEMENT_COMMAND(GetCommand, GetReply)
IMPLEMENT_REPLY(GetReply)
IMPLEMENT_COMMAND(RemoveCommand, RemoveReply)
IMPLEMENT_REPLY(RemoveReply)

TestAndSetCommand::TestAndSetCommand(const MessageType & messageType, const document::Bucket &bucket)
    : BucketInfoCommand(messageType, bucket)
{}
TestAndSetCommand::~TestAndSetCommand() = default;

PutCommand::PutCommand(const document::Bucket &bucket, const DocumentSP& doc, Timestamp time)
    : TestAndSetCommand(MessageType::PUT, bucket),
      _doc(doc),
      _timestamp(time),
      _updateTimestamp(0)
{
    if ( !_doc ) {
        throw vespalib::IllegalArgumentException("Cannot put a null document", VESPA_STRLOC);
    }
}

PutCommand::~PutCommand() = default;

const document::DocumentId&
PutCommand::getDocumentId() const {
    return _doc->getId();
}

const document::DocumentType *
PutCommand::getDocumentType() const {
    return &_doc->getType();
}

vespalib::string
PutCommand::getSummary() const
{
    vespalib::asciistream stream;
    stream << "Put(BucketId(0x" << vespalib::hex << getBucketId().getId() << "), "
           << _doc->getId().toString()
           << ", timestamp " << vespalib::dec << _timestamp
           << ')';

    return stream.str();
}

void
PutCommand::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "Put(" << getBucketId() << ", " << _doc->getId()
        << ", timestamp " << _timestamp << ", size "
        << _doc->serialize().size() << ")";
    if (verbose) {
        out << " {\n" << indent << "  ";
        _doc->print(out, verbose, indent + "  ");
        out << "\n" << indent << "}" << " : ";
        BucketInfoCommand::print(out, verbose, indent);
    }
}

PutReply::PutReply(const PutCommand& cmd, bool wasFoundFlag)
    : BucketInfoReply(cmd),
      _docId(cmd.getDocumentId()),
      _document(cmd.getDocument()),
      _timestamp(cmd.getTimestamp()),
      _updateTimestamp(cmd.getUpdateTimestamp()),
      _wasFound(wasFoundFlag)
{
}

PutReply::~PutReply() = default;

void
PutReply::print(std::ostream& out, bool verbose,
                const std::string& indent) const
{
    out << "PutReply(" << _docId << ", " << getBucketId() << ", timestamp " << _timestamp;

    if (hasBeenRemapped()) {
        out << " (was remapped)";
    }

    out << ")";
    if (verbose) {
        out << " : ";
        BucketInfoReply::print(out, verbose, indent);
    }
}

UpdateCommand::UpdateCommand(const document::Bucket &bucket, const document::DocumentUpdate::SP& update, Timestamp time)
    : TestAndSetCommand(MessageType::UPDATE, bucket),
      _update(update),
      _timestamp(time),
      _oldTimestamp(0)
{
    if ( ! _update) {
        throw vespalib::IllegalArgumentException("Cannot update a null update", VESPA_STRLOC);
    }
}

const document::DocumentType *
UpdateCommand::getDocumentType() const {
    return &_update->getType();
}

UpdateCommand::~UpdateCommand() = default;

const document::DocumentId&
UpdateCommand::getDocumentId() const {
    return _update->getId();
}

vespalib::string
UpdateCommand::getSummary() const {
    vespalib::asciistream stream;
    stream << "Update(BucketId(0x" << vespalib::hex << getBucketId().getId() << "), "
           << _update->getId().toString() << ", timestamp " << vespalib::dec << _timestamp;
    if (_oldTimestamp != 0) {
        stream << ", old timestamp " << _oldTimestamp;
    }
    stream << ')';

    return stream.str();
}

void
UpdateCommand::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "Update(" << getBucketId() << ", " << _update->getId() << ", timestamp " << _timestamp;
    if (_oldTimestamp != 0) {
        out << ", old timestamp " << _oldTimestamp;
    }
    out << ")";
    if (verbose) {
        out << " {\n" << indent << "  ";
        _update->print(out, verbose, indent + "  ");
        out << "\n" << indent << "} : ";
        BucketInfoCommand::print(out, verbose, indent);
    }
}

UpdateReply::UpdateReply(const UpdateCommand& cmd, Timestamp oldTimestamp)
    : BucketInfoReply(cmd),
      _docId(cmd.getDocumentId()),
      _timestamp(cmd.getTimestamp()),
      _oldTimestamp(oldTimestamp),
      _consistentNode((uint16_t)-1)
{
}

UpdateReply::~UpdateReply() = default;

void
UpdateReply::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "UpdateReply("
        << _docId << ", " << getBucketId() << ", timestamp "
        << _timestamp << ", timestamp of updated doc: " << _oldTimestamp;

    if (_consistentNode != (uint16_t)-1) {
        out << " Was inconsistent (best node " << _consistentNode << ")";
    }

    out << ")";

    if (verbose) {
        out << " : ";
        BucketInfoReply::print(out, verbose, indent);
    }
}

GetCommand::GetCommand(const document::Bucket &bucket, const document::DocumentId& docId,
                       vespalib::stringref fieldSet, Timestamp before)
    : BucketInfoCommand(MessageType::GET, bucket),
      _docId(docId),
      _beforeTimestamp(before),
      _fieldSet(fieldSet),
      _internal_read_consistency(InternalReadConsistency::Strong)
{
}

GetCommand::~GetCommand() = default;

vespalib::string
GetCommand::getSummary() const
{
    vespalib::asciistream stream;
    stream << "Get(BucketId(" << vespalib::hex << getBucketId().getId() << "), " << _docId.toString()
           << ", beforetimestamp " << vespalib::dec << _beforeTimestamp;
    if (has_condition()) {
        stream << ", condition " << condition().getSelection();
    }
    stream << ')';

    return stream.str();
}


void
GetCommand::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "Get(" << getBucketId() << ", " << _docId;
    if (has_condition()) {
        out << ", condition " << condition().getSelection();
    }
    out << ")";
    if (verbose) {
        out << " : ";
        BucketCommand::print(out, verbose, indent);
    }
}

GetReply::GetReply(const GetCommand& cmd,
                   const DocumentSP& doc,
                   Timestamp lastModified,
                   bool had_consistent_replicas,
                   bool is_tombstone,
                   bool condition_matched)
    : BucketInfoReply(cmd),
      _docId(cmd.getDocumentId()),
      _fieldSet(cmd.getFieldSet()),
      _doc(doc),
      _beforeTimestamp(cmd.getBeforeTimestamp()),
      _lastModifiedTime(lastModified),
      _had_consistent_replicas(had_consistent_replicas),
      _is_tombstone(is_tombstone),
      _condition_matched(condition_matched)
{
}

GetReply::~GetReply() = default;

void
GetReply::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "GetReply(" << getBucketId() << ", " << _docId << ", timestamp " << _lastModifiedTime << ")";
    if (verbose) {
        out << " : ";
        BucketReply::print(out, verbose, indent);
    }
}

RemoveCommand::RemoveCommand(const document::Bucket &bucket, const document::DocumentId& docId, Timestamp timestamp)
    : TestAndSetCommand(MessageType::REMOVE, bucket),
      _docId(docId),
      _timestamp(timestamp)
{
}

RemoveCommand::~RemoveCommand() = default;

vespalib::string
RemoveCommand::getSummary() const {
    vespalib::asciistream stream;
    stream << "Remove(BucketId(0x" << vespalib::hex << getBucketId().getId() << "), "
           << _docId.toString() << ", timestamp " << vespalib::dec << _timestamp << ')';

    return stream.str();
}
void
RemoveCommand::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "Remove(" << getBucketId() << ", " << _docId << ", timestamp " << _timestamp << ")";
    if (verbose) {
        out << " : ";
        BucketInfoCommand::print(out, verbose, indent);
    }
}

RemoveReply::RemoveReply(const RemoveCommand& cmd, Timestamp oldTimestamp)
    : BucketInfoReply(cmd),
      _docId(cmd.getDocumentId()),
      _timestamp(cmd.getTimestamp()),
      _oldTimestamp(oldTimestamp)
{
}

RemoveReply::~RemoveReply() = default;

void
RemoveReply::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "RemoveReply(" << getBucketId() << ", " << _docId << ", timestamp " << _timestamp;
    if (_oldTimestamp != 0) {
        out << ", removed doc from " << _oldTimestamp;
    } else {
        out << ", not found";
    }
    out << ")";
    if (verbose) {
        out << " : ";
        BucketInfoReply::print(out, verbose, indent);
    }
}

}
