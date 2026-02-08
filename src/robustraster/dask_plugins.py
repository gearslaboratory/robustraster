import os
import json
import ee
import time
import random
import logging
from distributed import WorkerPlugin

def make_robust_computePixels(original_computePixels):
    """
    Wraps ee.data.computePixels with exponential backoff for
    'Too Many Requests' (429) or other retryable errors.
    """
    # Sentinel to avoid double-wrapping if the plugin is re-initialized
    if getattr(original_computePixels, "_is_robust_wrapper", False):
        return original_computePixels

    def wrapper(*args, **kwargs):
        # Configuration for backoff
        max_retries = 10
        base_delay = 1.0
        max_delay = 60.0
        
        # Signals that indicate we should retry
        retry_signals = [
            "too many requests",
            "request was rejected",
            "request rate",
            "concurrency limit",
            "rate or concurrency",
            "rate limit",
            "quota exceeded",
            "resource exhausted",
            "backend error",
            "internal error",
            "timed out",
            "timeout",
            "throttl",
        ]

        for attempt in range(max_retries):
            try:
                return original_computePixels(*args, **kwargs)
            except getattr(ee, 'EEException', Exception) as e:
                msg = str(e).lower()
                # Check if this is a retryable error
                if not any(s in msg for s in retry_signals):
                    raise  # Not retryable

                # It is retryable; calculate delay
                delay = min(max_delay, base_delay * (2 ** attempt))
                # Add jitter: random factor between 0.5 and 1.5
                delay *= (0.5 + random.random())

                # Log the retry (this appears in Dask worker logs)
                logging.warning(
                    f"[robustraster] EE rate limit hit (attempt {attempt+1}/{max_retries}). "
                    f"Retrying in {delay:.2f}s. Error: {e}"
                )
                
                time.sleep(delay)
        
        # If we exit the loop, we ran out of retries
        raise RuntimeError(f"Earth Engine computePixels failed after {max_retries} retries due to rate limiting.")

    wrapper._is_robust_wrapper = True
    return wrapper

class EEPlugin(WorkerPlugin):
    def __init__(self):
        pass

    def setup(self, worker):
        self.worker = worker
        try:
            # Assume credentials already exist at default location
            # Note: The user typically runs ee.Initialize() on the client, which
            # sets up credentials. On the worker, we might need to rely on
            # environment variables or pre-existing auth if not explicitly passed.
            # However, xee often handles some of this if credentials are picklable.
            # We'll just ensure the library is initialized if possible.
            try:
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            except Exception:
                # If already initialized or detailed auth needed, this might fail or be skipped.
                # Proceeding to patch anyway because xee might be using a session established elsewhere.
                pass
            
            # --- PATCH: Monkeypatch ee.data.computePixels ---
            # This affects all calls made by xee or invalidation logic within this worker.
            if hasattr(ee, "data") and hasattr(ee.data, "computePixels"):
                logging.info("[robustraster] Patching ee.data.computePixels with backoff logic.")
                ee.data.computePixels = make_robust_computePixels(ee.data.computePixels)
            else:
                logging.warning("[robustraster] Could not match ee.data.computePixels to patch it.")

        except Exception as e:
            # Don't crash the worker, just log the failure to setup EE plugin
            logging.error(f"[robustraster] Failed to setup EEPlugin: {e}")
