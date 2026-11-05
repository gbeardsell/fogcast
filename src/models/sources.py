"""NOAA source discovery, bounded downloads, calibration and spatial registration.

GOES supplies current B2/CSM observations; NBM supplies weather conditioning.
Fitted B2 processing and input packing live in dataset.py. Heavy decoding and
projection dependencies are imported where needed, so the GRIB subprocess never
imports pyproj. Both HTTP block and working-file caches remain bounded.
"""
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class DownloadBudget:
    """Shared byte allowance and deadline for one run.
    
    All source readers use the same instance so many small requests cannot each 
    claim a fresh new alloance. Checks occur at request boundaries (before and after)."""

    def __init__(self, *, seconds=600, maximum_bytes=128 * 1024**2):
        """Start the elapsed-time allowance now and initialize HTTP counters"""

        # Monotonicity means the clock never goes backward: later calls to time.monotonic() return values greater than or equal to earlier calls.
        # Unlike the system clock, it isn’t affected by manual time changes or clock corrections, making it reliable for measuring elapsed time and checking deadlines.
        self.deadline = time.monotonic() + seconds
        self.maximum_bytes = maximum_bytes

        # Counter initialization
        self.response_bytes = 0
        self.requests = 0

    def remaining(self):
        """Return seconds left, raising ``TimeoutError`` when the deadline passed."""
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("live forecast deadline reached")
        return remaining

    def request(self, url, *, method="GET", headers=None, maximum=1024**2):
        """Read one response within per-call and shared byte allowances.
        
        Return ``(body, lowercase_headers, status)``. HEAD requests return 
        an empty body. Other requests read at most allowance + 1 bytes to 
        detect oversized responses and count those bytes before raising. 
        Socket timeout is the smaller of 30 seconds and the remianing run time.
        HTTP errors propagate; there is no automatic retry or Range validation here.
        """
        allowance = min(maximum, self.maximum_bytes - self.response_bytes)
        if allowance <= 0:
            raise RuntimeError("live download byte limit reached")

        request = Request(url, method=method, headers=headers or {})
        with urlopen(request, timeout=min(30, self.remaining())) as response:
            status, info = response.status, dict(response.headers)
            self.requests += 1




