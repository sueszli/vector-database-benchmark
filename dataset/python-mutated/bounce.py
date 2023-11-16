"""
Support for bounce message generation.
"""
import email.utils
import os
import time
from io import SEEK_END, SEEK_SET, StringIO
from twisted.mail import smtp
BOUNCE_FORMAT = 'From: postmaster@{failedDomain}\nTo: {failedFrom}\nSubject: Returned Mail: see transcript for details\nMessage-ID: {messageID}\nContent-Type: multipart/report; report-type=delivery-status;\n    boundary="{boundary}"\n\n--{boundary}\n\n{transcript}\n\n--{boundary}\nContent-Type: message/delivery-status\nArrival-Date: {ctime}\nFinal-Recipient: RFC822; {failedTo}\n'

def generateBounce(message, failedFrom, failedTo, transcript='', encoding='utf-8'):
    if False:
        i = 10
        return i + 15
    '\n    Generate a bounce message for an undeliverable email message.\n\n    @type message: a file-like object\n    @param message: The undeliverable message.\n\n    @type failedFrom: L{bytes} or L{unicode}\n    @param failedFrom: The originator of the undeliverable message.\n\n    @type failedTo: L{bytes} or L{unicode}\n    @param failedTo: The destination of the undeliverable message.\n\n    @type transcript: L{bytes} or L{unicode}\n    @param transcript: An error message to include in the bounce message.\n\n    @type encoding: L{str} or L{unicode}\n    @param encoding: Encoding to use, default: utf-8\n\n    @rtype: 3-L{tuple} of (E{1}) L{bytes}, (E{2}) L{bytes}, (E{3}) L{bytes}\n    @return: The originator, the destination and the contents of the bounce\n        message.  The destination of the bounce message is the originator of\n        the undeliverable message.\n    '
    if isinstance(failedFrom, bytes):
        failedFrom = failedFrom.decode(encoding)
    if isinstance(failedTo, bytes):
        failedTo = failedTo.decode(encoding)
    if not transcript:
        transcript = "I'm sorry, the following address has permanent errors: {failedTo}.\nI've given up, and I will not retry the message again.\n".format(failedTo=failedTo)
    failedAddress = email.utils.parseaddr(failedTo)[1]
    data = {'boundary': '{}_{}_{}'.format(time.time(), os.getpid(), 'XXXXX'), 'ctime': time.ctime(time.time()), 'failedAddress': failedAddress, 'failedDomain': failedAddress.split('@', 1)[1], 'failedFrom': failedFrom, 'failedTo': failedTo, 'messageID': smtp.messageid(uniq='bounce'), 'message': message, 'transcript': transcript}
    fp = StringIO()
    fp.write(BOUNCE_FORMAT.format(**data))
    orig = message.tell()
    message.seek(0, SEEK_END)
    sz = message.tell()
    message.seek(orig, SEEK_SET)
    if sz > 10000:
        while 1:
            line = message.readline()
            if isinstance(line, bytes):
                line = line.decode(encoding)
            if len(line) <= 0:
                break
            fp.write(line)
    else:
        messageContent = message.read()
        if isinstance(messageContent, bytes):
            messageContent = messageContent.decode(encoding)
        fp.write(messageContent)
    return (b'', failedFrom.encode(encoding), fp.getvalue().encode(encoding))