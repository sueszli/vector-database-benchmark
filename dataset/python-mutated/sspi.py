"""
Helper classes for SSPI authentication via the win32security module.

SSPI authentication involves a token-exchange "dance", the exact details
of which depends on the authentication provider used.  There are also
a number of complex flags and constants that need to be used - in most
cases, there are reasonable defaults.

These classes attempt to hide these details from you until you really need
to know.  They are not designed to handle all cases, just the common ones.
If you need finer control than offered here, just use the win32security
functions directly.
"""
import sspicon
import win32security
error = win32security.error

class _BaseAuth:

    def __init__(self):
        if False:
            return 10
        self.reset()

    def reset(self):
        if False:
            print('Hello World!')
        'Reset everything to an unauthorized state'
        self.ctxt = None
        self.authenticated = False
        self.initiator_name = None
        self.service_name = None
        self.next_seq_num = 0

    def _get_next_seq_num(self):
        if False:
            i = 10
            return i + 15
        'Get the next sequence number for a transmission.  Default\n        implementation is to increment a counter\n        '
        ret = self.next_seq_num
        self.next_seq_num = self.next_seq_num + 1
        return ret

    def encrypt(self, data):
        if False:
            i = 10
            return i + 15
        'Encrypt a string, returning a tuple of (encrypted_data, trailer).\n        These can be passed to decrypt to get back the original string.\n        '
        pkg_size_info = self.ctxt.QueryContextAttributes(sspicon.SECPKG_ATTR_SIZES)
        trailersize = pkg_size_info['SecurityTrailer']
        encbuf = win32security.PySecBufferDescType()
        encbuf.append(win32security.PySecBufferType(len(data), sspicon.SECBUFFER_DATA))
        encbuf.append(win32security.PySecBufferType(trailersize, sspicon.SECBUFFER_TOKEN))
        encbuf[0].Buffer = data
        self.ctxt.EncryptMessage(0, encbuf, self._get_next_seq_num())
        return (encbuf[0].Buffer, encbuf[1].Buffer)

    def decrypt(self, data, trailer):
        if False:
            for i in range(10):
                print('nop')
        'Decrypt a previously encrypted string, returning the orignal data'
        encbuf = win32security.PySecBufferDescType()
        encbuf.append(win32security.PySecBufferType(len(data), sspicon.SECBUFFER_DATA))
        encbuf.append(win32security.PySecBufferType(len(trailer), sspicon.SECBUFFER_TOKEN))
        encbuf[0].Buffer = data
        encbuf[1].Buffer = trailer
        self.ctxt.DecryptMessage(encbuf, self._get_next_seq_num())
        return encbuf[0].Buffer

    def sign(self, data):
        if False:
            i = 10
            return i + 15
        'sign a string suitable for transmission, returning the signature.\n        Passing the data and signature to verify will determine if the data\n        is unchanged.\n        '
        pkg_size_info = self.ctxt.QueryContextAttributes(sspicon.SECPKG_ATTR_SIZES)
        sigsize = pkg_size_info['MaxSignature']
        sigbuf = win32security.PySecBufferDescType()
        sigbuf.append(win32security.PySecBufferType(len(data), sspicon.SECBUFFER_DATA))
        sigbuf.append(win32security.PySecBufferType(sigsize, sspicon.SECBUFFER_TOKEN))
        sigbuf[0].Buffer = data
        self.ctxt.MakeSignature(0, sigbuf, self._get_next_seq_num())
        return sigbuf[1].Buffer

    def verify(self, data, sig):
        if False:
            i = 10
            return i + 15
        'Verifies data and its signature.  If verification fails, an sspi.error\n        will be raised.\n        '
        sigbuf = win32security.PySecBufferDescType()
        sigbuf.append(win32security.PySecBufferType(len(data), sspicon.SECBUFFER_DATA))
        sigbuf.append(win32security.PySecBufferType(len(sig), sspicon.SECBUFFER_TOKEN))
        sigbuf[0].Buffer = data
        sigbuf[1].Buffer = sig
        self.ctxt.VerifySignature(sigbuf, self._get_next_seq_num())

    def unwrap(self, token):
        if False:
            print('Hello World!')
        "\n        GSSAPI's unwrap with SSPI.\n        https://docs.microsoft.com/en-us/windows/win32/secauthn/sspi-kerberos-interoperability-with-gssapi\n\n        Usable mainly with Kerberos SSPI package, but this is not enforced.\n\n        Return the clear text, and a boolean that is True if the token was encrypted.\n        "
        buffer = win32security.PySecBufferDescType()
        buffer.append(win32security.PySecBufferType(len(token), sspicon.SECBUFFER_STREAM))
        buffer[0].Buffer = token
        buffer.append(win32security.PySecBufferType(0, sspicon.SECBUFFER_DATA))
        pfQOP = self.ctxt.DecryptMessage(buffer, self._get_next_seq_num())
        r = buffer[1].Buffer
        return (r, not pfQOP == sspicon.SECQOP_WRAP_NO_ENCRYPT)

    def wrap(self, msg, encrypt=False):
        if False:
            return 10
        "\n        GSSAPI's wrap with SSPI.\n        https://docs.microsoft.com/en-us/windows/win32/secauthn/sspi-kerberos-interoperability-with-gssapi\n\n        Usable mainly with Kerberos SSPI package, but this is not enforced.\n\n        Wrap a message to be sent to the other side. Encrypted if encrypt is True.\n        "
        size_info = self.ctxt.QueryContextAttributes(sspicon.SECPKG_ATTR_SIZES)
        trailer_size = size_info['SecurityTrailer']
        block_size = size_info['BlockSize']
        buffer = win32security.PySecBufferDescType()
        buffer.append(win32security.PySecBufferType(len(msg), sspicon.SECBUFFER_DATA))
        buffer[0].Buffer = msg
        buffer.append(win32security.PySecBufferType(trailer_size, sspicon.SECBUFFER_TOKEN))
        buffer.append(win32security.PySecBufferType(block_size, sspicon.SECBUFFER_PADDING))
        fQOP = 0 if encrypt else sspicon.SECQOP_WRAP_NO_ENCRYPT
        self.ctxt.EncryptMessage(fQOP, buffer, self._get_next_seq_num())
        r = buffer[1].Buffer + buffer[0].Buffer + buffer[2].Buffer
        return r

    def _amend_ctx_name(self):
        if False:
            for i in range(10):
                print('nop')
        'Adds initiator and service names in the security context for ease of use'
        if not self.authenticated:
            raise ValueError('Sec context is not completely authenticated')
        try:
            names = self.ctxt.QueryContextAttributes(sspicon.SECPKG_ATTR_NATIVE_NAMES)
        except error:
            pass
        else:
            (self.initiator_name, self.service_name) = names

class ClientAuth(_BaseAuth):
    """Manages the client side of an SSPI authentication handshake"""

    def __init__(self, pkg_name, client_name=None, auth_info=None, targetspn=None, scflags=None, datarep=sspicon.SECURITY_NETWORK_DREP):
        if False:
            i = 10
            return i + 15
        if scflags is None:
            scflags = sspicon.ISC_REQ_INTEGRITY | sspicon.ISC_REQ_SEQUENCE_DETECT | sspicon.ISC_REQ_REPLAY_DETECT | sspicon.ISC_REQ_CONFIDENTIALITY
        self.scflags = scflags
        self.datarep = datarep
        self.targetspn = targetspn
        self.pkg_info = win32security.QuerySecurityPackageInfo(pkg_name)
        (self.credentials, self.credentials_expiry) = win32security.AcquireCredentialsHandle(client_name, self.pkg_info['Name'], sspicon.SECPKG_CRED_OUTBOUND, None, auth_info)
        _BaseAuth.__init__(self)

    def authorize(self, sec_buffer_in):
        if False:
            while True:
                i = 10
        'Perform *one* step of the client authentication process. Pass None for the first round'
        if sec_buffer_in is not None and (not isinstance(sec_buffer_in, win32security.PySecBufferDescType)):
            sec_buffer_new = win32security.PySecBufferDescType()
            tokenbuf = win32security.PySecBufferType(self.pkg_info['MaxToken'], sspicon.SECBUFFER_TOKEN)
            tokenbuf.Buffer = sec_buffer_in
            sec_buffer_new.append(tokenbuf)
            sec_buffer_in = sec_buffer_new
        sec_buffer_out = win32security.PySecBufferDescType()
        tokenbuf = win32security.PySecBufferType(self.pkg_info['MaxToken'], sspicon.SECBUFFER_TOKEN)
        sec_buffer_out.append(tokenbuf)
        ctxtin = self.ctxt
        if self.ctxt is None:
            self.ctxt = win32security.PyCtxtHandleType()
        (err, attr, exp) = win32security.InitializeSecurityContext(self.credentials, ctxtin, self.targetspn, self.scflags, self.datarep, sec_buffer_in, self.ctxt, sec_buffer_out)
        self.ctxt_attr = attr
        self.ctxt_expiry = exp
        if err in (sspicon.SEC_I_COMPLETE_NEEDED, sspicon.SEC_I_COMPLETE_AND_CONTINUE):
            self.ctxt.CompleteAuthToken(sec_buffer_out)
        self.authenticated = err == 0
        if self.authenticated:
            self._amend_ctx_name()
        return (err, sec_buffer_out)

class ServerAuth(_BaseAuth):
    """Manages the server side of an SSPI authentication handshake"""

    def __init__(self, pkg_name, spn=None, scflags=None, datarep=sspicon.SECURITY_NETWORK_DREP):
        if False:
            i = 10
            return i + 15
        self.spn = spn
        self.datarep = datarep
        if scflags is None:
            scflags = sspicon.ASC_REQ_INTEGRITY | sspicon.ASC_REQ_SEQUENCE_DETECT | sspicon.ASC_REQ_REPLAY_DETECT | sspicon.ASC_REQ_CONFIDENTIALITY
        self.scflags = scflags
        self.pkg_info = win32security.QuerySecurityPackageInfo(pkg_name)
        (self.credentials, self.credentials_expiry) = win32security.AcquireCredentialsHandle(spn, self.pkg_info['Name'], sspicon.SECPKG_CRED_INBOUND, None, None)
        _BaseAuth.__init__(self)

    def authorize(self, sec_buffer_in):
        if False:
            print('Hello World!')
        'Perform *one* step of the server authentication process.'
        if sec_buffer_in is not None and (not isinstance(sec_buffer_in, win32security.PySecBufferDescType)):
            sec_buffer_new = win32security.PySecBufferDescType()
            tokenbuf = win32security.PySecBufferType(self.pkg_info['MaxToken'], sspicon.SECBUFFER_TOKEN)
            tokenbuf.Buffer = sec_buffer_in
            sec_buffer_new.append(tokenbuf)
            sec_buffer_in = sec_buffer_new
        sec_buffer_out = win32security.PySecBufferDescType()
        tokenbuf = win32security.PySecBufferType(self.pkg_info['MaxToken'], sspicon.SECBUFFER_TOKEN)
        sec_buffer_out.append(tokenbuf)
        ctxtin = self.ctxt
        if self.ctxt is None:
            self.ctxt = win32security.PyCtxtHandleType()
        (err, attr, exp) = win32security.AcceptSecurityContext(self.credentials, ctxtin, sec_buffer_in, self.scflags, self.datarep, self.ctxt, sec_buffer_out)
        self.ctxt_attr = attr
        self.ctxt_expiry = exp
        if err in (sspicon.SEC_I_COMPLETE_NEEDED, sspicon.SEC_I_COMPLETE_AND_CONTINUE):
            self.ctxt.CompleteAuthToken(sec_buffer_out)
        self.authenticated = err == 0
        if self.authenticated:
            self._amend_ctx_name()
        return (err, sec_buffer_out)
if __name__ == '__main__':
    ssp = 'Kerberos'
    flags = sspicon.ISC_REQ_MUTUAL_AUTH | sspicon.ISC_REQ_INTEGRITY | sspicon.ISC_REQ_SEQUENCE_DETECT | sspicon.ISC_REQ_CONFIDENTIALITY | sspicon.ISC_REQ_REPLAY_DETECT
    (cred_handle, exp) = win32security.AcquireCredentialsHandle(None, ssp, sspicon.SECPKG_CRED_INBOUND, None, None)
    cred = cred_handle.QueryCredentialsAttributes(sspicon.SECPKG_CRED_ATTR_NAMES)
    print('We are:', cred)
    sspiclient = ClientAuth(ssp, scflags=flags, targetspn=cred)
    sspiserver = ServerAuth(ssp, scflags=flags)
    print('SSP : {} ({})'.format(sspiclient.pkg_info['Name'], sspiclient.pkg_info['Comment']))
    sec_buffer = None
    client_step = 0
    server_step = 0
    while not sspiclient.authenticated or len(sec_buffer[0].Buffer):
        client_step += 1
        (err, sec_buffer) = sspiclient.authorize(sec_buffer)
        print('Client step %s' % client_step)
        if sspiserver.authenticated and len(sec_buffer[0].Buffer) == 0:
            break
        server_step += 1
        (err, sec_buffer) = sspiserver.authorize(sec_buffer)
        print('Server step %s' % server_step)
    print('Initiator name from the service side:', sspiserver.initiator_name)
    print('Service name from the client side:   ', sspiclient.service_name)
    data = b'hello'
    sig = sspiclient.sign(data)
    sspiserver.verify(data, sig)
    (encrypted, sig) = sspiclient.encrypt(data)
    decrypted = sspiserver.decrypt(encrypted, sig)
    assert decrypted == data
    wrapped = sspiclient.wrap(data)
    (unwrapped, was_encrypted) = sspiserver.unwrap(wrapped)
    print('encrypted ?', was_encrypted)
    assert data == unwrapped
    wrapped = sspiserver.wrap(data, encrypt=True)
    (unwrapped, was_encrypted) = sspiclient.unwrap(wrapped)
    print('encrypted ?', was_encrypted)
    assert data == unwrapped
    print('cool!')