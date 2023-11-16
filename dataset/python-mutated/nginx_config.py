"""General purpose nginx test configuration generator."""
import atexit
import getpass
import sys
from contextlib import ExitStack
from typing import Optional
if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources

def construct_nginx_config(nginx_root: str, nginx_webroot: str, http_port: int, https_port: int, other_port: int, default_server: bool, key_path: Optional[str]=None, cert_path: Optional[str]=None, wtf_prefix: str='le') -> str:
    if False:
        print('Hello World!')
    '\n    This method returns a full nginx configuration suitable for integration tests.\n    :param str nginx_root: nginx root configuration path\n    :param str nginx_webroot: nginx webroot path\n    :param int http_port: HTTP port to listen on\n    :param int https_port: HTTPS port to listen on\n    :param int other_port: other HTTP port to listen on\n    :param bool default_server: True to set a default server in nginx config, False otherwise\n    :param str key_path: the path to a SSL key\n    :param str cert_path: the path to a SSL certificate\n    :param str wtf_prefix: the prefix to use in all domains handled by this nginx config\n    :return: a string containing the full nginx configuration\n    :rtype: str\n    '
    if not key_path:
        file_manager = ExitStack()
        atexit.register(file_manager.close)
        ref = importlib_resources.files('certbot_integration_tests').joinpath('assets', 'key.pem')
        key_path = str(file_manager.enter_context(importlib_resources.as_file(ref)))
    if not cert_path:
        file_manager = ExitStack()
        atexit.register(file_manager.close)
        ref = importlib_resources.files('certbot_integration_tests').joinpath('assets', 'cert.pem')
        cert_path = str(file_manager.enter_context(importlib_resources.as_file(ref)))
    return '# This error log will be written regardless of server scope error_log\n# definitions, so we have to set this here in the main scope.\n#\n# Even doing this, Nginx will still try to create the default error file, and\n# log a non-fatal error when it fails. After that things will work, however.\nerror_log {nginx_root}/error.log;\n\n# The pidfile will be written to /var/run unless this is set.\npid {nginx_root}/nginx.pid;\n\nuser {user};\nworker_processes 1;\n\nevents {{\n  worker_connections 1024;\n}}\n\n# “This comment contains valid Unicode”.\n\nhttp {{\n  # Set an array of temp, cache and log file options that will otherwise default to\n  # restricted locations accessible only to root.\n  client_body_temp_path {nginx_root}/client_body;\n  fastcgi_temp_path {nginx_root}/fastcgi_temp;\n  proxy_temp_path {nginx_root}/proxy_temp;\n  #scgi_temp_path {nginx_root}/scgi_temp;\n  #uwsgi_temp_path {nginx_root}/uwsgi_temp;\n  access_log {nginx_root}/error.log;\n\n  # This should be turned off in a Virtualbox VM, as it can cause some\n  # interesting issues with data corruption in delivered files.\n  sendfile off;\n\n  tcp_nopush on;\n  tcp_nodelay on;\n  keepalive_timeout 65;\n  types_hash_max_size 2048;\n\n  #include /etc/nginx/mime.types;\n  index index.html index.htm index.php;\n\n  log_format   main \'$remote_addr - $remote_user [$time_local] $status \'\n    \'"$request" $body_bytes_sent "$http_referer" \'\n    \'"$http_user_agent" "$http_x_forwarded_for"\';\n\n  default_type application/octet-stream;\n\n  server {{\n    # IPv4.\n    listen {http_port} {default_server};\n    # IPv6.\n    listen [::]:{http_port} {default_server};\n    server_name nginx.{wtf_prefix}.wtf nginx2.{wtf_prefix}.wtf;\n\n    root {nginx_webroot};\n\n    location / {{\n      # First attempt to serve request as file, then as directory, then fall\n      # back to index.html.\n      try_files $uri $uri/ /index.html;\n    }}\n  }}\n\n  server {{\n    listen {http_port};\n    listen [::]:{http_port};\n    server_name nginx3.{wtf_prefix}.wtf;\n\n    root {nginx_webroot};\n\n    location /.well-known/ {{\n      return 404;\n    }}\n\n    return 301 https://$host$request_uri;\n  }}\n\n  server {{\n    listen {other_port};\n    listen [::]:{other_port};\n    server_name nginx4.{wtf_prefix}.wtf nginx5.{wtf_prefix}.wtf;\n  }}\n\n  server {{\n    listen {http_port};\n    listen [::]:{http_port};\n    listen {https_port} ssl;\n    listen [::]:{https_port} ssl;\n    if ($scheme != "https") {{\n      return 301 https://$host$request_uri;\n    }}\n    server_name nginx6.{wtf_prefix}.wtf nginx7.{wtf_prefix}.wtf;\n\n    ssl_certificate {cert_path};\n    ssl_certificate_key {key_path};\n  }}\n}}\n'.format(nginx_root=nginx_root, nginx_webroot=nginx_webroot, user=getpass.getuser(), http_port=http_port, https_port=https_port, other_port=other_port, default_server='default_server' if default_server else '', wtf_prefix=wtf_prefix, key_path=key_path, cert_path=cert_path)