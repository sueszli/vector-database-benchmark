# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: legacy_claim.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import stream_pb2 as stream__pb2
from . import certificate_pb2 as certificate__pb2
from . import signature_pb2 as signature__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='legacy_claim.proto',
  package='legacy_pb',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x12legacy_claim.proto\x12\tlegacy_pb\x1a\x0cstream.proto\x1a\x11\x63\x65rtificate.proto\x1a\x0fsignature.proto\"\xd9\x02\n\x05\x43laim\x12)\n\x07version\x18\x01 \x02(\x0e\x32\x18.legacy_pb.Claim.Version\x12-\n\tclaimType\x18\x02 \x02(\x0e\x32\x1a.legacy_pb.Claim.ClaimType\x12!\n\x06stream\x18\x03 \x01(\x0b\x32\x11.legacy_pb.Stream\x12+\n\x0b\x63\x65rtificate\x18\x04 \x01(\x0b\x32\x16.legacy_pb.Certificate\x12\x30\n\x12publisherSignature\x18\x05 \x01(\x0b\x32\x14.legacy_pb.Signature\"*\n\x07Version\x12\x13\n\x0fUNKNOWN_VERSION\x10\x00\x12\n\n\x06_0_0_1\x10\x01\"H\n\tClaimType\x12\x16\n\x12UNKNOWN_CLAIM_TYPE\x10\x00\x12\x0e\n\nstreamType\x10\x01\x12\x13\n\x0f\x63\x65rtificateType\x10\x02')
  ,
  dependencies=[stream__pb2.DESCRIPTOR,certificate__pb2.DESCRIPTOR,signature__pb2.DESCRIPTOR,])



_CLAIM_VERSION = _descriptor.EnumDescriptor(
  name='Version',
  full_name='legacy_pb.Claim.Version',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_VERSION', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='_0_0_1', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=313,
  serialized_end=355,
)
_sym_db.RegisterEnumDescriptor(_CLAIM_VERSION)

_CLAIM_CLAIMTYPE = _descriptor.EnumDescriptor(
  name='ClaimType',
  full_name='legacy_pb.Claim.ClaimType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_CLAIM_TYPE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='streamType', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='certificateType', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=357,
  serialized_end=429,
)
_sym_db.RegisterEnumDescriptor(_CLAIM_CLAIMTYPE)


_CLAIM = _descriptor.Descriptor(
  name='Claim',
  full_name='legacy_pb.Claim',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='legacy_pb.Claim.version', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='claimType', full_name='legacy_pb.Claim.claimType', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stream', full_name='legacy_pb.Claim.stream', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='certificate', full_name='legacy_pb.Claim.certificate', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='publisherSignature', full_name='legacy_pb.Claim.publisherSignature', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _CLAIM_VERSION,
    _CLAIM_CLAIMTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=84,
  serialized_end=429,
)

_CLAIM.fields_by_name['version'].enum_type = _CLAIM_VERSION
_CLAIM.fields_by_name['claimType'].enum_type = _CLAIM_CLAIMTYPE
_CLAIM.fields_by_name['stream'].message_type = stream__pb2._STREAM
_CLAIM.fields_by_name['certificate'].message_type = certificate__pb2._CERTIFICATE
_CLAIM.fields_by_name['publisherSignature'].message_type = signature__pb2._SIGNATURE
_CLAIM_VERSION.containing_type = _CLAIM
_CLAIM_CLAIMTYPE.containing_type = _CLAIM
DESCRIPTOR.message_types_by_name['Claim'] = _CLAIM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Claim = _reflection.GeneratedProtocolMessageType('Claim', (_message.Message,), dict(
  DESCRIPTOR = _CLAIM,
  __module__ = 'legacy_claim_pb2'
  # @@protoc_insertion_point(class_scope:legacy_pb.Claim)
  ))
_sym_db.RegisterMessage(Claim)


# @@protoc_insertion_point(module_scope)
