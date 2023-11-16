"""Production settings and globals."""
from urllib.parse import urlparse
import ssl
import certifi

import dj_database_url

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.redis import RedisIntegration

from .common import *  # noqa

# Database
DEBUG = int(os.environ.get("DEBUG", 1)) == 1
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("PGUSER", "plane"),
        "USER": "",
        "PASSWORD": "",
        "HOST": os.environ.get("PGHOST", "localhost"),
    }
}

# CORS WHITELIST ON PROD
CORS_ORIGIN_WHITELIST = [
    # "https://example.com",
    # "https://sub.example.com",
    # "http://localhost:8080",
    # "http://127.0.0.1:9000"
]
# Parse database configuration from $DATABASE_URL
DATABASES["default"] = dj_database_url.config()
SITE_ID = 1

# Enable Connection Pooling (if desired)
# DATABASES['default']['ENGINE'] = 'django_postgrespool'

# Honor the 'X-Forwarded-Proto' header for request.is_secure()
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Allow all host headers
ALLOWED_HOSTS = ["*"]

# TODO: Make it FALSE and LIST DOMAINS IN FULL PROD.
CORS_ALLOW_ALL_ORIGINS = True

STORAGES = {
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}


# Make true if running in a docker environment
DOCKERIZED = int(os.environ.get("DOCKERIZED", 0)) == 1
FILE_SIZE_LIMIT = int(os.environ.get("FILE_SIZE_LIMIT", 5242880))
USE_MINIO = int(os.environ.get("USE_MINIO", 0)) == 1

sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    integrations=[DjangoIntegration(), RedisIntegration()],
    # If you wish to associate users to errors (assuming you are using
    # django.contrib.auth) you may enable sending PII data.
    traces_sample_rate=1,
    send_default_pii=True,
    environment="staging",
    profiles_sample_rate=1.0,
)

# The AWS region to connect to.
AWS_REGION = os.environ.get("AWS_REGION")

# The AWS access key to use.
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")

# The AWS secret access key to use.
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# The optional AWS session token to use.
# AWS_SESSION_TOKEN = ""


# The name of the bucket to store files in.
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")

# How to construct S3 URLs ("auto", "path", "virtual").
AWS_S3_ADDRESSING_STYLE = "auto"

# The full URL to the S3 endpoint. Leave blank to use the default region URL.
AWS_S3_ENDPOINT_URL = os.environ.get("AWS_S3_ENDPOINT_URL", "")

# A prefix to be applied to every stored file. This will be joined to every filename using the "/" separator.
AWS_S3_KEY_PREFIX = ""

# Whether to enable authentication for stored files. If True, then generated URLs will include an authentication
# token valid for `AWS_S3_MAX_AGE_SECONDS`. If False, then generated URLs will not include an authentication token,
# and their permissions will be set to "public-read".
AWS_S3_BUCKET_AUTH = False

# How long generated URLs are valid for. This affects the expiry of authentication tokens if `AWS_S3_BUCKET_AUTH`
# is True. It also affects the "Cache-Control" header of the files.
# Important: Changing this setting will not affect existing files.
AWS_S3_MAX_AGE_SECONDS = 60 * 60  # 1 hours.

# A URL prefix to be used for generated URLs. This is useful if your bucket is served through a CDN. This setting
# cannot be used with `AWS_S3_BUCKET_AUTH`.
AWS_S3_PUBLIC_URL = ""

# If True, then files will be stored with reduced redundancy. Check the S3 documentation and make sure you
# understand the consequences before enabling.
# Important: Changing this setting will not affect existing files.
AWS_S3_REDUCED_REDUNDANCY = False

# The Content-Disposition header used when the file is downloaded. This can be a string, or a function taking a
# single `name` argument.
# Important: Changing this setting will not affect existing files.
AWS_S3_CONTENT_DISPOSITION = ""

# The Content-Language header used when the file is downloaded. This can be a string, or a function taking a
# single `name` argument.
# Important: Changing this setting will not affect existing files.
AWS_S3_CONTENT_LANGUAGE = ""

# A mapping of custom metadata for each file. Each value can be a string, or a function taking a
# single `name` argument.
# Important: Changing this setting will not affect existing files.
AWS_S3_METADATA = {}

# If True, then files will be stored using AES256 server-side encryption.
# If this is a string value (e.g., "aws:kms"), that encryption type will be used.
# Otherwise, server-side encryption is not be enabled.
# Important: Changing this setting will not affect existing files.
AWS_S3_ENCRYPT_KEY = False

# The AWS S3 KMS encryption key ID (the `SSEKMSKeyId` parameter) is set from this string if present.
# This is only relevant if AWS S3 KMS server-side encryption is enabled (above).
# AWS_S3_KMS_ENCRYPTION_KEY_ID = ""

# If True, then text files will be stored using gzip content encoding. Files will only be gzipped if their
# compressed size is smaller than their uncompressed size.
# Important: Changing this setting will not affect existing files.
AWS_S3_GZIP = True

# The signature version to use for S3 requests.
AWS_S3_SIGNATURE_VERSION = None

# If True, then files with the same name will overwrite each other. By default it's set to False to have
# extra characters appended.
AWS_S3_FILE_OVERWRITE = False

# AWS Settings End
STORAGES["default"] = {
        "BACKEND": "django_s3_storage.storage.S3Storage",
}

# Enable Connection Pooling (if desired)
# DATABASES['default']['ENGINE'] = 'django_postgrespool'

# Honor the 'X-Forwarded-Proto' header for request.is_secure()
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Allow all host headers
ALLOWED_HOSTS = [
    "*",
]

SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True


REDIS_URL = os.environ.get("REDIS_URL")

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL,
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "CONNECTION_POOL_KWARGS": {"ssl_cert_reqs": False},
        },
    }
}

RQ_QUEUES = {
    "default": {
        "USE_REDIS_CACHE": "default",
    }
}


WEB_URL = os.environ.get("WEB_URL")

PROXY_BASE_URL = os.environ.get("PROXY_BASE_URL", False)

ANALYTICS_SECRET_KEY = os.environ.get("ANALYTICS_SECRET_KEY", False)
ANALYTICS_BASE_API = os.environ.get("ANALYTICS_BASE_API", False)


OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", False)
GPT_ENGINE = os.environ.get("GPT_ENGINE", "gpt-3.5-turbo")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", False)

LOGGER_BASE_URL = os.environ.get("LOGGER_BASE_URL", False)

redis_url = os.environ.get("REDIS_URL")
broker_url = (
    f"{redis_url}?ssl_cert_reqs={ssl.CERT_NONE.name}&ssl_ca_certs={certifi.where()}"
)

CELERY_RESULT_BACKEND = broker_url
CELERY_BROKER_URL = broker_url

GITHUB_ACCESS_TOKEN = os.environ.get("GITHUB_ACCESS_TOKEN", False)

ENABLE_SIGNUP = os.environ.get("ENABLE_SIGNUP", "1") == "1"


# Unsplash Access key
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
