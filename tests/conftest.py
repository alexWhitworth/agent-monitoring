"""Shared pytest configuration for agent_monitor tests.

Mutation-run guard (F-013): mutmut drives multiple pytest sessions inside a
single process. Each session constructs a fresh ``unittest.TestCase`` instance,
so ``@given``-decorated test methods are invoked with different ``self``
objects on one thread. That trips hypothesis's ``differing_executors`` health
check, whose real purpose is to catch a test being run by genuinely different
executors (e.g. pytest vs. Django test runners). Under mutmut it is a false
positive, so it is suppressed for mutation runs only; normal pytest runs keep
every health check active.
"""

import sys

from hypothesis import HealthCheck, settings

if "mutmut" in sys.modules:
    settings.register_profile(
        "mutmut",
        suppress_health_check=[HealthCheck.differing_executors],
    )
    settings.load_profile("mutmut")
