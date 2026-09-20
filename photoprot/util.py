"""Small shared helpers."""
from __future__ import annotations

import sys
from typing import Any

from tqdm import tqdm


def progress(*args: Any, **kwargs: Any) -> tqdm:
    """tqdm that stays readable when stderr is a log file rather than a terminal.

    Redirected tqdm output writes one line per refresh, which turns a long job's
    log into megabytes of progress bar. When stderr is not a TTY we throttle hard
    so the log gets a handful of lines instead.
    """
    kwargs.setdefault("file", sys.stderr)
    if not sys.stderr.isatty():
        kwargs.setdefault("mininterval", 15.0)
        kwargs.setdefault("ascii", True)
    else:
        kwargs.setdefault("miniters", 1)
    return tqdm(*args, **kwargs)
