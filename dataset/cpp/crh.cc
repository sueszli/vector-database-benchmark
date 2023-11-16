// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2018 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

//
// This file anchors the abstract ircd::crh::hash vtable and default
// functionalities. Various implementations of crh::hash will be contained
// within other units where the third-party dependency which implements it
// is included (ex. openssl.cc). This is so we don't include and mix
// everything here just for hash functions.
//

ircd::crh::hash::~hash()
noexcept
{
}

ircd::crh::hash &
ircd::crh::hash::operator+=(const const_buffer &buf)
{
	update(buf);
	return *this;
}

void
ircd::crh::hash::operator()(const mutable_buffer &out,
                            const const_buffer &in)
{
	update(in);
	finalize(out);
}

void
ircd::crh::hash::finalize(const mutable_buffer &b)
{
	digest(b);
}
