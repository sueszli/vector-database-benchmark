"""Class for backend status."""
import html
from qiskit.exceptions import QiskitError

class BackendStatus:
    """Class representing Backend Status."""

    def __init__(self, backend_name: str, backend_version: str, operational: bool, pending_jobs: int, status_msg: str):
        if False:
            i = 10
            return i + 15
        "Initialize a BackendStatus object\n\n        Args:\n            backend_name: The backend's name\n            backend_version: The backend's version of the form X.Y.Z\n            operational: True if the backend is operational\n            pending_jobs: The number of pending jobs on the backend\n            status_msg: The status msg for the backend\n\n        Raises:\n            QiskitError: If the backend version is in an invalid format\n        "
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.operational = operational
        if pending_jobs < 0:
            raise QiskitError('Pending jobs must be >=0')
        self.pending_jobs = pending_jobs
        self.status_msg = status_msg

    @classmethod
    def from_dict(cls, data):
        if False:
            i = 10
            return i + 15
        'Create a new BackendStatus object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the BaseBakend to create.\n                         It will be in the same format as output by\n                         :func:`to_dict`.\n\n        Returns:\n            BackendStatus: The BackendStatus from the input dictionary.\n        '
        return cls(**data)

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary format representation of the BackendStatus.\n\n        Returns:\n            dict: The dictionary form of the QobjHeader.\n        '
        return self.__dict__

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, BackendStatus):
            if self.__dict__ == other.__dict__:
                return True
        return False

    def _repr_html_(self) -> str:
        if False:
            return 10
        "Return html representation of the object\n\n        Returns:\n            Representation used in Jupyter notebook and other IDE's that call the method\n\n        "
        rpr = self.__repr__()
        html_code = f'<pre>{html.escape(rpr)}</pre><b>name</b>: {self.backend_name}<br/><b>version</b>: {self.backend_version}, <b>pending jobs</b>: {self.pending_jobs}<br/><b>status</b>: {self.status_msg}<br/>'
        return html_code