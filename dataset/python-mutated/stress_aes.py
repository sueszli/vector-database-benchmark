aes_s_box_table = bytes((99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22))

def aes_gf8_mul_2(x):
    if False:
        for i in range(10):
            print('nop')
    if x & 128:
        return x << 1 ^ 283
    else:
        return x << 1

def aes_gf8_mul_3(x):
    if False:
        print('Hello World!')
    return x ^ aes_gf8_mul_2(x)

def aes_s_box(a):
    if False:
        i = 10
        return i + 15
    return aes_s_box_table[a & 255]

def aes_r_con(a):
    if False:
        while True:
            i = 10
    ans = 1
    while a > 1:
        ans <<= 1
        if ans & 256:
            ans ^= 283
        a -= 1
    return ans

def aes_add_round_key(state, w):
    if False:
        for i in range(10):
            print('nop')
    for i in range(16):
        state[i] ^= w[i]

def aes_sb_sr_mc_ark(state, w, w_idx, temp):
    if False:
        i = 10
        return i + 15
    temp_idx = 0
    for i in range(4):
        x0 = aes_s_box_table[state[i * 4]]
        x1 = aes_s_box_table[state[1 + (i + 1 & 3) * 4]]
        x2 = aes_s_box_table[state[2 + (i + 2 & 3) * 4]]
        x3 = aes_s_box_table[state[3 + (i + 3 & 3) * 4]]
        temp[temp_idx] = aes_gf8_mul_2(x0) ^ aes_gf8_mul_3(x1) ^ x2 ^ x3 ^ w[w_idx]
        temp[temp_idx + 1] = x0 ^ aes_gf8_mul_2(x1) ^ aes_gf8_mul_3(x2) ^ x3 ^ w[w_idx + 1]
        temp[temp_idx + 2] = x0 ^ x1 ^ aes_gf8_mul_2(x2) ^ aes_gf8_mul_3(x3) ^ w[w_idx + 2]
        temp[temp_idx + 3] = aes_gf8_mul_3(x0) ^ x1 ^ x2 ^ aes_gf8_mul_2(x3) ^ w[w_idx + 3]
        w_idx += 4
        temp_idx += 4
    for i in range(16):
        state[i] = temp[i]

def aes_sb_sr_ark(state, w, w_idx, temp):
    if False:
        i = 10
        return i + 15
    temp_idx = 0
    for i in range(4):
        x0 = aes_s_box_table[state[i * 4]]
        x1 = aes_s_box_table[state[1 + (i + 1 & 3) * 4]]
        x2 = aes_s_box_table[state[2 + (i + 2 & 3) * 4]]
        x3 = aes_s_box_table[state[3 + (i + 3 & 3) * 4]]
        temp[temp_idx] = x0 ^ w[w_idx]
        temp[temp_idx + 1] = x1 ^ w[w_idx + 1]
        temp[temp_idx + 2] = x2 ^ w[w_idx + 2]
        temp[temp_idx + 3] = x3 ^ w[w_idx + 3]
        w_idx += 4
        temp_idx += 4
    for i in range(16):
        state[i] = temp[i]

def aes_state(state, w, temp, nr):
    if False:
        print('Hello World!')
    aes_add_round_key(state, w)
    w_idx = 16
    for i in range(nr - 1):
        aes_sb_sr_mc_ark(state, w, w_idx, temp)
        w_idx += 16
    aes_sb_sr_ark(state, w, w_idx, temp)

def aes_key_expansion(key, w, temp, nk, nr):
    if False:
        i = 10
        return i + 15
    for i in range(4 * nk):
        w[i] = key[i]
    w_idx = 4 * nk - 4
    for i in range(nk, 4 * (nr + 1)):
        t = temp
        t_idx = 0
        if i % nk == 0:
            t[0] = aes_s_box(w[w_idx + 1]) ^ aes_r_con(i // nk)
            for j in range(1, 4):
                t[j] = aes_s_box(w[w_idx + (j + 1) % 4])
        elif nk > 6 and i % nk == 4:
            for j in range(0, 4):
                t[j] = aes_s_box(w[w_idx + j])
        else:
            t = w
            t_idx = w_idx
        w_idx += 4
        for j in range(4):
            w[w_idx + j] = w[w_idx + j - 4 * nk] ^ t[t_idx + j]

class AES:

    def __init__(self, keysize):
        if False:
            print('Hello World!')
        if keysize == 128:
            self.nk = 4
            self.nr = 10
        elif keysize == 192:
            self.nk = 6
            self.nr = 12
        else:
            assert keysize == 256
            self.nk = 8
            self.nr = 14
        self.state = bytearray(16)
        self.w = bytearray(16 * (self.nr + 1))
        self.temp = bytearray(16)
        self.state_pos = 16

    def set_key(self, key):
        if False:
            while True:
                i = 10
        aes_key_expansion(key, self.w, self.temp, self.nk, self.nr)
        self.state_pos = 16

    def set_iv(self, iv):
        if False:
            for i in range(10):
                print('nop')
        for i in range(16):
            self.state[i] = iv[i]
        self.state_pos = 16

    def get_some_state(self, n_needed):
        if False:
            for i in range(10):
                print('nop')
        if self.state_pos >= 16:
            aes_state(self.state, self.w, self.temp, self.nr)
            self.state_pos = 0
        n = 16 - self.state_pos
        if n > n_needed:
            n = n_needed
        return n

    def apply_to(self, data):
        if False:
            for i in range(10):
                print('nop')
        idx = 0
        n = len(data)
        while n > 0:
            ln = self.get_some_state(n)
            n -= ln
            for i in range(ln):
                data[idx + i] ^= self.state[self.state_pos + i]
            idx += ln
            self.state_pos += n
import time
import _thread

class LockedCounter:

    def __init__(self):
        if False:
            return 10
        self.lock = _thread.allocate_lock()
        self.value = 0

    def add(self, val):
        if False:
            for i in range(10):
                print('nop')
        self.lock.acquire()
        self.value += val
        self.lock.release()
count = LockedCounter()

def thread_entry(n_loop):
    if False:
        while True:
            i = 10
    global count
    aes = AES(256)
    key = bytearray(256 // 8)
    iv = bytearray(16)
    data = bytearray(128)
    for loop in range(n_loop):
        aes.set_key(key)
        aes.set_iv(iv)
        for i in range(8):
            aes.apply_to(data)
        aes.set_key(key)
        aes.set_iv(iv)
        for i in range(8):
            aes.apply_to(data)
        for i in range(len(data)):
            assert data[i] == 0
    count.add(1)
if __name__ == '__main__':
    import sys
    if hasattr(sys, 'settrace'):
        n_thread = 2
        n_loop = 2
    elif sys.platform == 'rp2':
        n_thread = 1
        n_loop = 2
    elif sys.platform in ('esp32', 'pyboard'):
        n_thread = 2
        n_loop = 2
    else:
        n_thread = 20
        n_loop = 5
    for i in range(n_thread):
        _thread.start_new_thread(thread_entry, (n_loop,))
    thread_entry(n_loop)
    while count.value < n_thread:
        time.sleep(1)
    print('done')