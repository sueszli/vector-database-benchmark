import socket
import ssl
import threading
(client_sock, server_sock) = socket.socketpair()
client_done = threading.Event()

def server_thread_fn():
    if False:
        for i in range(10):
            print('nop')
    server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_ctx.load_cert_chain('trio-test-1.pem')
    server = server_ctx.wrap_socket(server_sock, server_side=True, suppress_ragged_eofs=False)
    while True:
        data = server.recv(4096)
        print('server got:', data)
        if not data:
            print('server waiting for client to finish everything')
            client_done.wait()
            print('server attempting to send back close-notify')
            server.unwrap()
            print('server ok')
            break
        server.sendall(data)
server_thread = threading.Thread(target=server_thread_fn)
server_thread.start()
client_ctx = ssl.create_default_context(cafile='trio-test-CA.pem')
client = client_ctx.wrap_socket(client_sock, server_hostname='trio-test-1.example.org')
assert client.getpeercert() is not None
client.sendall(b'x')
assert client.recv(10) == b'x'
print('client sending close_notify')
client.setblocking(False)
try:
    client.unwrap()
except ssl.SSLWantReadError:
    print('client got SSLWantReadError as expected')
else:
    raise AssertionError()
client.close()
client_done.set()