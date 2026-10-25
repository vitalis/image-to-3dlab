"""The sphere remesh is opt-in. The default command must not run it."""

import inspect

from scripts.diagnose_remesh_kernel import WATCHDOG_NOTE, main, sphere_enabled


def test_the_sphere_is_off_unless_asked():
    assert sphere_enabled(sphere=False, skip_sphere=False) is False
    assert sphere_enabled(sphere=True, skip_sphere=False) is True
    assert sphere_enabled(sphere=True, skip_sphere=True) is False


def test_the_watchdog_note_is_printed_before_any_metal_work():
    assert "MTL_CAPTURE_ENABLED" in WATCHDOG_NOTE
    source = inspect.getsource(main)
    assert source.index("WATCHDOG_NOTE") < source.index("test_sphere")
    assert "sphere_enabled" in source
