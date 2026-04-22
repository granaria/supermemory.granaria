"""Dashboard subpackage — configuration UI + server-side integration.

This package is self-contained: nothing outside `phase1/` imports from it,
so the dashboard is strictly opt-in. `local_supermemory/server.py` calls
exactly one function from here (`integration.apply_save_policy`); if
dashboard is ever removed, only that one call site needs a stub.
"""
