//
// UUIDHelper.cpp
//
// Copyright (c) 2004-2010 The OpenVanilla Project (http://openvanilla.org)
// All rights reserved.
// 
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//

#include <rpc.h>

#if defined(__APPLE__)
    #include <OpenVanilla/OpenVanilla.h>
#else
    #include "OpenVanilla.h"
#endif

#include "UUIDHelper.h"

namespace CareService
{

using namespace OpenVanilla;

const string UUIDHelper::CreateUUID()
{
    UUID uuid;
    UuidCreateSequential(&uuid);
    RPC_WSTR rpcString;
    UuidToString(&uuid, &rpcString);    
	string result = OVUTF8::FromUTF16((wchar_t*)rpcString);
    RpcStringFree(&rpcString);
    return result;
}
    
};
