import socket
import select
import pickle
import errno
import sys
import consts as c

class MyMessage(object):

    def __init__(self, my_type, message):
        if False:
            while True:
                i = 10
        self.type = my_type
        self.message = message

class Error(Exception):
    pass

class CommError(Error):

    def __init__(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.message = message

def send_message(socket, message_data, caller):
    if False:
        for i in range(10):
            print('nop')
    '\n\tFormat message into fixed length header and encoded message data\n\t\tutf-8(Fixed length header) -> b(message data)\n\n\tReturn: True if message sent successfully\n\t'
    message = pickle.dumps(message_data)
    message_len = len(message)
    message_header = f'{message_len:<{c.HEADER_LENGTH}}'.encode('utf-8')
    message = bytes(message_header) + message
    total_sent = 0
    try:
        while total_sent < len(message):
            sent = socket.send(message[total_sent:])
            if sent == 0:
                raise RuntimeError('Socket connection broken')
            total_sent = total_sent + sent
        return True
    except ConnectionResetError as e:
        if caller == c.CLIENT:
            raise CommError('Error: Server disconnected') from e
        elif caller == c.SERVER:
            return None
        return None
    except IOError as e:
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print(f'Send Error:{str(e)}')
            if caller == c.CLIENT:
                raise CommError('Error: Server disconnected') from e
            elif caller == c.SERVER:
                return None
        print(e)
        return None
    except Exception as e:
        print('Send Error', e)
        raise CommError('Send Error') from e

def receive_message(client_socket, caller):
    if False:
        while True:
            i = 10
    '\n\tReceives a message from the given socket and decodes it based on the format\n\t\tutf-8(Fixed length header) -> b(message data)\n\n\tReturn:\tmessage-> {header: message header, "data": message data}\n\t'
    try:
        message_header = client_socket.recv(c.HEADER_LENGTH)
        if len(message_header) < c.HEADER_LENGTH:
            message_header += client_socket.recv(c.HEADER_LENGTH - len(message_header))
        if len(message_header) is not c.HEADER_LENGTH:
            raise CommError('Connection closed unexpectedly')
        message_length = int(message_header.decode('utf-8'))
        bytes_received = 0
        chunks = []
        while bytes_received < message_length:
            chunk = client_socket.recv(min(message_length - bytes_received, c.CHUNK_SIZE))
            if chunk == b'':
                raise RuntimeError('Socket connection broken')
            chunks.append(chunk)
            bytes_received += len(chunk)
        message_b = b''.join(chunks)
        message = pickle.loads(message_b)
        return {'header': message_header, 'data': message}
    except IOError as e:
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            c.print_debug(f'Read Error:{str(e)}')
            if caller == c.CLIENT:
                raise CommError('Error: Server disconnected') from e
            elif caller == c.SERVER:
                raise CommError('Error: Client disconnected')
        return None
    except ConnectionResetError as e:
        if caller == c.CLIENT:
            print('Error: Server has disconnected, closing client')
            raise CommError('Recv Error: Server disconnected') from e
        elif caller == c.SERVER:
            return None
    except CommError as e:
        raise CommError('Connection closed unexpectedly')
    except Exception as e:
        print('Recv Error', e)
        raise CommError('Recv Error') from e
        return None