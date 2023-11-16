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

#include "base/crypto/data_cryptor_chacha20_poly1305.h"

#include "base/logging.h"
#include "base/crypto/openssl_util.h"
#include "base/crypto/random.h"
#include "base/crypto/secure_memory.h"

#include <openssl/evp.h>

namespace base {

namespace {

static const size_t kKeySize = 32; // 256 bits, 32 bytes.
static const size_t kIVSize = 12; // 96 bits, 12 bytes.
static const size_t kTagSize = 16; // 128 bits, 16 bytes.
static const size_t kHeaderSize = kIVSize + kTagSize;

//--------------------------------------------------------------------------------------------------
EVP_CIPHER_CTX_ptr createCipher(std::string_view key, const char* iv, int type)
{
    if (key.size() != kKeySize)
    {
        LOG(LS_ERROR) << "Wrong key size: " << key.size();
        return nullptr;
    }

    EVP_CIPHER_CTX_ptr ctx(EVP_CIPHER_CTX_new());
    if (!ctx)
    {
        LOG(LS_ERROR) << "EVP_CIPHER_CTX_new failed";
        return nullptr;
    }

    if (EVP_CipherInit_ex(ctx.get(), EVP_chacha20_poly1305(),
                          nullptr, nullptr, nullptr, type) != 1)
    {
        LOG(LS_ERROR) << "EVP_EncryptInit_ex failed";
        return nullptr;
    }

    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_SET_IVLEN, kIVSize, nullptr) != 1)
    {
        LOG(LS_ERROR) << "EVP_CIPHER_CTX_ctrl failed";
        return nullptr;
    }

    if (EVP_CIPHER_CTX_set_key_length(ctx.get(), kKeySize) != 1)
    {
        LOG(LS_ERROR) << "EVP_CIPHER_CTX_set_key_length failed";
        return nullptr;
    }

    if (EVP_CipherInit_ex(ctx.get(), nullptr, nullptr,
                          reinterpret_cast<const uint8_t*>(key.data()),
                          reinterpret_cast<const uint8_t*>(iv),
                          type) != 1)
    {
        LOG(LS_ERROR) << "EVP_CIPHER_CTX_ctrl failed";
        return nullptr;
    }

    return ctx;
}

} // namespace

//--------------------------------------------------------------------------------------------------
DataCryptorChaCha20Poly1305::DataCryptorChaCha20Poly1305(std::string_view key)
    : key_(key)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
DataCryptorChaCha20Poly1305::~DataCryptorChaCha20Poly1305()
{
    memZero(&key_);
}

//--------------------------------------------------------------------------------------------------
bool DataCryptorChaCha20Poly1305::encrypt(std::string_view in, std::string* out)
{
    if (in.empty())
    {
        LOG(LS_ERROR) << "Empty buffer passed";
        return false;
    }

    out->resize(in.size() + kHeaderSize);

    if (!Random::fillBuffer(out->data(), kIVSize))
    {
        LOG(LS_ERROR) << "Random::fillBuffer failed";
        return false;
    }

    EVP_CIPHER_CTX_ptr cipher = createCipher(key_, out->data(), 1);
    if (!cipher)
    {
        LOG(LS_ERROR) << "Unable to create cipher";
        return false;
    }

    int length;

    if (EVP_EncryptUpdate(cipher.get(),
                          reinterpret_cast<uint8_t*>(out->data()) + kHeaderSize,
                          &length,
                          reinterpret_cast<const uint8_t*>(in.data()),
                          static_cast<int>(in.size())) != 1)
    {
        LOG(LS_ERROR) << "EVP_EncryptUpdate failed";
        return false;
    }

    if (EVP_EncryptFinal_ex(cipher.get(),
                            reinterpret_cast<uint8_t*>(out->data()) + kHeaderSize + length,
                            &length) != 1)
    {
        LOG(LS_ERROR) << "EVP_EncryptFinal_ex failed";
        return false;
    }

    if (EVP_CIPHER_CTX_ctrl(cipher.get(),
                            EVP_CTRL_AEAD_GET_TAG,
                            kTagSize,
                            reinterpret_cast<uint8_t*>(out->data()) + kIVSize) != 1)
    {
        LOG(LS_ERROR) << "EVP_CIPHER_CTX_ctrl failed";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
bool DataCryptorChaCha20Poly1305::decrypt(std::string_view in, std::string* out)
{
    if (in.size() <= kHeaderSize)
    {
        LOG(LS_ERROR) << "Header missed";
        return false;
    }

    EVP_CIPHER_CTX_ptr cipher = createCipher(key_, in.data(), 0);
    if (!cipher)
    {
        LOG(LS_ERROR) << "Unable to create cipher";
        return false;
    }

    out->resize(in.size() - kHeaderSize);

    int length;

    if (EVP_DecryptUpdate(cipher.get(),
                          reinterpret_cast<uint8_t*>(out->data()),
                          &length,
                          reinterpret_cast<const uint8_t*>(in.data()) + kHeaderSize,
                          static_cast<int>(in.size() - kHeaderSize)) != 1)
    {
        LOG(LS_ERROR) << "EVP_DecryptUpdate failed";
        return false;
    }

    if (EVP_CIPHER_CTX_ctrl(cipher.get(),
                            EVP_CTRL_AEAD_SET_TAG,
                            kTagSize,
                            reinterpret_cast<uint8_t*>(
                                const_cast<char*>(in.data())) + kIVSize) != 1)
    {
        LOG(LS_ERROR) << "EVP_CIPHER_CTX_ctrl failed";
        return false;
    }

    if (EVP_DecryptFinal_ex(cipher.get(),
                            reinterpret_cast<uint8_t*>(out->data()) + length,
                            &length) <= 0)
    {
        LOG(LS_ERROR) << "EVP_DecryptFinal_ex failed";
        return false;
    }

    return true;
}

} // namespace base
