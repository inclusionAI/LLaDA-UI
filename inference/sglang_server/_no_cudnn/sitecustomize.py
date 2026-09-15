"""Optional startup hook for systems where vision Conv2d cannot initialize cuDNN.

Place this file in a dedicated directory and prepend that directory to
``PYTHONPATH``. Do not enable the hook on systems with a healthy cuDNN setup.
"""

try:
    import torch as _t

    _t.backends.cudnn.enabled = False
    _t.backends.cudnn.benchmark = False
except Exception:
    pass
