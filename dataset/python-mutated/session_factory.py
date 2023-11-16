import os
from typing import Type
import onnxruntime as ort
from .sessions import sessions_class
from .sessions.base import BaseSession
from .sessions.u2net import U2netSession

def new_session(model_name: str='u2net', providers=None, *args, **kwargs) -> BaseSession:
    if False:
        i = 10
        return i + 15
    "\n    Create a new session object based on the specified model name.\n\n    This function searches for the session class based on the model name in the 'sessions_class' list.\n    It then creates an instance of the session class with the provided arguments.\n    The 'sess_opts' object is created using the 'ort.SessionOptions()' constructor.\n    If the 'OMP_NUM_THREADS' environment variable is set, the 'inter_op_num_threads' option of 'sess_opts' is set to its value.\n\n    Parameters:\n        model_name (str): The name of the model.\n        providers: The providers for the session.\n        *args: Additional positional arguments.\n        **kwargs: Additional keyword arguments.\n\n    Returns:\n        BaseSession: The created session object.\n    "
    session_class: Type[BaseSession] = U2netSession
    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break
    sess_opts = ort.SessionOptions()
    if 'OMP_NUM_THREADS' in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ['OMP_NUM_THREADS'])
        sess_opts.intra_op_num_threads = int(os.environ['OMP_NUM_THREADS'])
    return session_class(model_name, sess_opts, providers, *args, **kwargs)