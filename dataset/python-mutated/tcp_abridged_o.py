import logging
import os
from typing import Optional
import pyrogram
from pyrogram.crypto import aes
from .tcp import TCP
log = logging.getLogger(__name__)

class TCPAbridgedO(TCP):
    RESERVED = (b'HEAD', b'POST', b'GET ', b'OPTI', b'\xee' * 4)

    def __init__(self, ipv6: bool, proxy: dict):
        if False:
            i = 10
            return i + 15
        super().__init__(ipv6, proxy)
        self.encrypt = None
        self.decrypt = None

    async def connect(self, address: tuple):
        await super().connect(address)
        while True:
            nonce = bytearray(os.urandom(64))
            if bytes([nonce[0]]) != b'\xef' and nonce[:4] not in self.RESERVED and (nonce[4:8] != b'\x00' * 4):
                nonce[56] = nonce[57] = nonce[58] = nonce[59] = 239
                break
        temp = bytearray(nonce[55:7:-1])
        self.encrypt = (nonce[8:40], nonce[40:56], bytearray(1))
        self.decrypt = (temp[0:32], temp[32:48], bytearray(1))
        nonce[56:64] = aes.ctr256_encrypt(nonce, *self.encrypt)[56:64]
        await super().send(nonce)

    async def send(self, data: bytes, *args):
        length = len(data) // 4
        data = (bytes([length]) if length <= 126 else b'\x7f' + length.to_bytes(3, 'little')) + data
        payload = await self.loop.run_in_executor(pyrogram.crypto_executor, aes.ctr256_encrypt, data, *self.encrypt)
        await super().send(payload)

    async def recv(self, length: int=0) -> Optional[bytes]:
        length = await super().recv(1)
        if length is None:
            return None
        length = aes.ctr256_decrypt(length, *self.decrypt)
        if length == b'\x7f':
            length = await super().recv(3)
            if length is None:
                return None
            length = aes.ctr256_decrypt(length, *self.decrypt)
        data = await super().recv(int.from_bytes(length, 'little') * 4)
        if data is None:
            return None
        return await self.loop.run_in_executor(pyrogram.crypto_executor, aes.ctr256_decrypt, data, *self.decrypt)