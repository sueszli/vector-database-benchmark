//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/crypto/srp_math.h"

#include "base/logging.h"
#include "base/crypto/generic_hash.h"
#include "base/strings/string_util.h"
#include "base/strings/unicode.h"

#include <openssl/opensslv.h>
#include <openssl/bn.h>

namespace base {

namespace {

//--------------------------------------------------------------------------------------------------
// xy = BLAKE2b512(PAD(x) || PAD(y))
BigNum calc_xy(const BigNum& x, const BigNum& y, const BigNum& N)
{
    if (x.get() != N.get() && BN_ucmp(x, N) >= 0)
        return BigNum();

    if (y.get() != N.get() && BN_ucmp(y, N) >= 0)
        return BigNum();

    int N_bytes = BN_num_bytes(N);
    if (N_bytes <= 0)
        return BigNum();

    const size_t xy_size = static_cast<size_t>(N_bytes) + static_cast<size_t>(N_bytes);
    ByteArray xy(xy_size);

    if (BN_bn2binpad(x, xy.data(), N_bytes) < 0)
        return BigNum();

    if (BN_bn2binpad(y, xy.data() + N_bytes, N_bytes) < 0)
        return BigNum();

    return BigNum::fromByteArray(
        GenericHash::hash(GenericHash::BLAKE2b512, xy.data(), xy_size));
}

//--------------------------------------------------------------------------------------------------
// k = BLAKE2b512(N | PAD(g))
BigNum calc_k(const BigNum& N, const BigNum& g)
{
    return calc_xy(N, g, N);
}

} // namespace

//--------------------------------------------------------------------------------------------------
// static
// u = BLAKE2b512(PAD(A) | PAD(B))
BigNum SrpMath::calc_u(const BigNum& A, const BigNum& B, const BigNum& N)
{
    return calc_xy(A, B, N);
}

//--------------------------------------------------------------------------------------------------
// static
// B = k*v + g^b % N
BigNum SrpMath::calc_B(const BigNum& b, const BigNum& N, const BigNum& g, const BigNum& v)
{
    if (!b.isValid() || !N.isValid() || !g.isValid() || !v.isValid())
    {
        LOG(LS_ERROR) << "Invalid params";
        return BigNum();
    }

    BigNum::Context ctx = BigNum::Context::create();
    if (!ctx.isValid())
    {
        LOG(LS_ERROR) << "BigNum::Context::create failed";
        return BigNum();
    }

    BigNum gb = BigNum::create();
    if (!gb.isValid())
    {
        LOG(LS_ERROR) << "BigNum::create failed";
        return BigNum();
    }

    if (!BN_mod_exp(gb, g, b, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_exp failed";
        return BigNum();
    }

    BigNum k = calc_k(N, g);
    if (!k.isValid())
    {
        LOG(LS_ERROR) << "Invalid k";
        return BigNum();
    }

    BigNum kv = BigNum::create();
    if (!kv.isValid())
    {
        LOG(LS_ERROR) << "BigNum::create failed";
        return BigNum();
    }

    if (!BN_mod_mul(kv, v, k, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_mul failed";
        return BigNum();
    }

    BigNum B = BigNum::create();
    if (!B.isValid())
    {
        LOG(LS_ERROR) << "BigNum::create failed";
        return BigNum();
    }

    if (!BN_mod_add(B, gb, kv, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_add failed";
        return BigNum();
    }

    return B;
}

//--------------------------------------------------------------------------------------------------
// static
// x = BLAKE2b512(s | BLAKE2b512(I | ":" | p))
BigNum SrpMath::calc_x(const BigNum& s, std::u16string_view I, std::u16string_view p)
{
    return calc_x(s, I, fromStdString(utf8FromUtf16(p)));
}

//--------------------------------------------------------------------------------------------------
// static
BigNum SrpMath::calc_x(const BigNum& s, std::u16string_view I, const ByteArray& p)
{
    if (!s.isValid() || I.empty() || p.empty())
    {
        LOG(LS_ERROR) << "Invalid params";
        return BigNum();
    }

    GenericHash hash(GenericHash::BLAKE2b512);

    hash.addData(utf8FromUtf16(toLower(I)));
    hash.addData(":");
    hash.addData(p);

    ByteArray temp = hash.result();
    ByteArray salt = s.toByteArray();

    hash.reset();

    hash.addData(salt);
    hash.addData(temp);

    return BigNum::fromByteArray(hash.result());
}

//--------------------------------------------------------------------------------------------------
// static
// A = g^a % N
BigNum SrpMath::calc_A(const BigNum& a, const BigNum& N, const BigNum& g)
{
    if (!a.isValid() || !N.isValid() || !g.isValid())
    {
        LOG(LS_ERROR) << "Invalid params";
        return BigNum();
    }

    BigNum::Context ctx = BigNum::Context::create();
    BigNum A = BigNum::create();

    if (!A.isValid() || !ctx.isValid())
    {
        LOG(LS_ERROR) << "BigNum::Context::create or BigNum::create failed";
        return BigNum();
    }

    if (!BN_mod_exp(A, g, a, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_exp failed";
        return BigNum();
    }

    return A;
}

//--------------------------------------------------------------------------------------------------
// static
// S = (A * v^u) ^ b % N
BigNum SrpMath::calcServerKey(const BigNum& A, const BigNum& v, const BigNum& u, const BigNum& b,
                              const BigNum& N)
{
    if (!A.isValid() || !v.isValid() || !u.isValid() || !b.isValid() || !N.isValid())
    {
        LOG(LS_ERROR) << "Invalid parameters";
        return BigNum();
    }

    BigNum::Context ctx = BigNum::Context::create();
    BigNum tmp = BigNum::create();

    if (!ctx.isValid() || !tmp.isValid())
    {
        LOG(LS_ERROR) << "BigNum::Context::create or BigNum::create failed";
        return BigNum();
    }

    if (!BN_mod_exp(tmp, v, u, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_exp failed";
        return BigNum();
    }

    if (!BN_mod_mul(tmp, A, tmp, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_mul failed";
        return BigNum();
    }

    BigNum S = BigNum::create();
    if (!S.isValid())
    {
        LOG(LS_ERROR) << "BigNum::create failed";
        return BigNum();
    }

    if (!BN_mod_exp(S, tmp, b, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_exp failed";
        return BigNum();
    }

    return S;
}

//--------------------------------------------------------------------------------------------------
// static
// K = (B - (k * g^x)) ^ (a + (u * x)) % N
BigNum SrpMath::calcClientKey(const BigNum& N, const BigNum& B, const BigNum& g, const BigNum& x,
                              const BigNum& a, const BigNum& u)
{
    if (!N.isValid() || !B.isValid() || !g.isValid() || !x.isValid() || !a.isValid() || !u.isValid())
    {
        LOG(LS_ERROR) << "Invalid params";
        return BigNum();
    }

    BigNum::Context ctx = BigNum::Context::create();
    if (!ctx.isValid())
    {
        LOG(LS_ERROR) << "BigNum::Context::create failed";
        return BigNum();
    }

    BigNum tmp = BigNum::create();
    BigNum tmp2 = BigNum::create();
    BigNum tmp3 = BigNum::create();

    if (!tmp.isValid() || !tmp2.isValid() || !tmp3.isValid())
    {
        LOG(LS_ERROR) << "BigNum::create failed";
        return BigNum();
    }

    if (!BN_mod_exp(tmp, g, x, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_exp failed";
        return BigNum();
    }

    BigNum k = calc_k(N, g);
    if (!k.isValid())
    {
        LOG(LS_ERROR) << "calc_k failed";
        return BigNum();
    }

    if (!BN_mod_mul(tmp2, tmp, k, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_mul failed";
        return BigNum();
    }

    if (!BN_mod_sub(tmp, B, tmp2, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_sub failed";
        return BigNum();
    }

    if (!BN_mul(tmp3, u, x, ctx))
    {
        LOG(LS_ERROR) << "BN_mul failed";
        return BigNum();
    }

    if (!BN_add(tmp2, a, tmp3))
    {
        LOG(LS_ERROR) << "BN_add failed";
        return BigNum();
    }

    BigNum K = BigNum::create();
    if (!K.isValid())
    {
        LOG(LS_ERROR) << "BigNum::create failed";
        return BigNum();
    }

    if (!BN_mod_exp(K, tmp, tmp2, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_exp failed";
        return BigNum();
    }

    return K;
}

//--------------------------------------------------------------------------------------------------
// static
bool SrpMath::verify_B_mod_N(const BigNum& B, const BigNum& N)
{
    if (!B.isValid() || !N.isValid())
    {
        LOG(LS_ERROR) << "Invalid params";
        return false;
    }

    BigNum::Context ctx = BigNum::Context::create();
    BigNum result = BigNum::create();

    if (!ctx.isValid() || !result.isValid())
    {
        LOG(LS_ERROR) << "BigNum::Context::create or BigNum::create failed";
        return false;
    }

    if (!BN_nnmod(result, B, N, ctx))
    {
        LOG(LS_ERROR) << "BN_nnmod failed";
        return false;
    }

    return !BN_is_zero(result);
}

//--------------------------------------------------------------------------------------------------
// static
bool SrpMath::verify_A_mod_N(const BigNum& A, const BigNum& N)
{
    return verify_B_mod_N(A, N);
}

//--------------------------------------------------------------------------------------------------
// static
BigNum SrpMath::calc_v(std::u16string_view I, std::u16string_view p, const BigNum& s,
                       const BigNum& N, const BigNum& g)
{
    if (I.empty() || p.empty() || !N.isValid() || !g.isValid() || !s.isValid())
    {
        LOG(LS_ERROR) << "Invalid params";
        return BigNum();
    }

    BigNum::Context ctx = BigNum::Context::create();
    BigNum v = BigNum::create();

    if (!ctx.isValid() || !v.isValid())
    {
        LOG(LS_ERROR) << "BigNum::Context::create or BigNum::create failed";
        return BigNum();
    }

    BigNum x = calc_x(s, I, p);

    if (!BN_mod_exp(v, g, x, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_exp failed";
        return BigNum();
    }

    return v;
}

//--------------------------------------------------------------------------------------------------
// static
BigNum SrpMath::calc_v(std::u16string_view I, const ByteArray& p, const BigNum& s,
                       const BigNum& N, const BigNum& g)
{
    if (I.empty() || p.empty() || !N.isValid() || !g.isValid() || !s.isValid())
    {
        LOG(LS_ERROR) << "Invalid params";
        return BigNum();
    }

    BigNum::Context ctx = BigNum::Context::create();
    BigNum v = BigNum::create();

    if (!ctx.isValid() || !v.isValid())
    {
        LOG(LS_ERROR) << "BigNum::Context::create or BigNum::create failed";
        return BigNum();
    }

    BigNum x = calc_x(s, I, p);

    if (!BN_mod_exp(v, g, x, N, ctx))
    {
        LOG(LS_ERROR) << "BN_mod_exp failed";
        return BigNum();
    }

    return v;
}

} // namespace base
