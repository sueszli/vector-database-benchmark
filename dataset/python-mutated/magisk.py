import hashlib
import os
from zipfile import ZipFile
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from mitmproxy import certs
from mitmproxy import ctx
from mitmproxy.options import CONF_BASENAME
MODULE_PROP_TEXT = 'id=mitmproxycert\nname=MITMProxy cert\nversion=v1\nversionCode=1\nauthor=mitmproxy\ndescription=Adds the mitmproxy certificate to the system store\ntemplate=3'
CONFIG_SH_TEXT = '\nMODID=mitmproxycert\nAUTOMOUNT=true\nPROPFILE=false\nPOSTFSDATA=false\nLATESTARTSERVICE=false\n\nprint_modname() {\n  ui_print "*******************************"\n  ui_print "    MITMProxy cert installer   "\n  ui_print "*******************************"\n}\n\nREPLACE="\n"\n\nset_permissions() {\n  set_perm_recursive  $MODPATH  0  0  0755  0644\n}\n'
UPDATE_BINARY_TEXT = '\n#!/sbin/sh\n\n#################\n# Initialization\n#################\n\numask 022\n\n# echo before loading util_functions\nui_print() { echo "$1"; }\n\nrequire_new_magisk() {\n  ui_print "*******************************"\n  ui_print " Please install Magisk v20.4+! "\n  ui_print "*******************************"\n  exit 1\n}\n\nOUTFD=$2\nZIPFILE=$3\n\nmount /data 2>/dev/null\n[ -f /data/adb/magisk/util_functions.sh ] || require_new_magisk\n. /data/adb/magisk/util_functions.sh\n[ $MAGISK_VER_CODE -lt 20400 ] && require_new_magisk\n\ninstall_module\nexit 0\n'

def get_ca_from_files() -> x509.Certificate:
    if False:
        for i in range(10):
            print('nop')
    certstore_path = os.path.expanduser(ctx.options.confdir)
    certstore = certs.CertStore.from_store(path=certstore_path, basename=CONF_BASENAME, key_size=ctx.options.key_size, passphrase=ctx.options.cert_passphrase.encode('utf8') if ctx.options.cert_passphrase else None)
    return certstore.default_ca._cert

def subject_hash_old(ca: x509.Certificate) -> str:
    if False:
        return 10
    full_hash = hashlib.md5(ca.subject.public_bytes()).digest()
    sho = full_hash[0] | full_hash[1] << 8 | full_hash[2] << 16 | full_hash[3] << 24
    return hex(sho)[2:]

def write_magisk_module(path: str):
    if False:
        i = 10
        return i + 15
    ca = get_ca_from_files()
    der_cert = ca.public_bytes(serialization.Encoding.DER)
    with ZipFile(path, 'w') as zipp:
        zipp.writestr(f'system/etc/security/cacerts/{subject_hash_old(ca)}.0', der_cert)
        zipp.writestr('module.prop', MODULE_PROP_TEXT)
        zipp.writestr('config.sh', CONFIG_SH_TEXT)
        zipp.writestr('META-INF/com/google/android/updater-script', '#MAGISK')
        zipp.writestr('META-INF/com/google/android/update-binary', UPDATE_BINARY_TEXT)
        zipp.writestr('common/file_contexts_image', '/magisk(/.*)? u:object_r:system_file:s0')
        zipp.writestr('common/post-fs-data.sh', 'MODDIR=${0%/*}')
        zipp.writestr('common/service.sh', 'MODDIR=${0%/*}')
        zipp.writestr('common/system.prop', '')