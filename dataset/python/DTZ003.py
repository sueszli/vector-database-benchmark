import datetime

# qualified
datetime.datetime.utcnow()

from datetime import datetime

# unqualified
datetime.utcnow()

# uses `astimezone` method
datetime.utcnow().astimezone()
