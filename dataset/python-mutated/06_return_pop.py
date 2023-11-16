def write(self, s, spos):
    if False:
        return 10
    if not s:
        return
    if not isinstance(s, basestring):
        s = str(s)
    slen = self.len
    if spos == slen:
        self.len = self.pos = spos + len(s)
        return
    if spos > slen:
        slen = spos
    newpos = spos + len(s)
    if spos < slen:
        if self.buflist:
            self.buf += ''.join(self.buflist)
        self.buflist = [self.buf[:spos], s, self.buf[newpos:]]
        if newpos > slen:
            slen = newpos
    else:
        self.buflist.append(s)