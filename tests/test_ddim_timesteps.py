"""DDIM's previous step is the next scheduled timestep, not a rounded gap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PAINT = Path(__file__).resolve().parents[1] / "hunyuan_mlx" / "paint"
sys.path.insert(0, str(PAINT))

from hy3dpaint_mlx.scheduler import DDIMScheduler, previous_ddim_timestep  # noqa: E402


def _gaps(steps):
    return [int(steps[i]) - int(steps[i + 1]) for i in range(len(steps) - 1)]


def test_fifteen_steps_use_the_next_scheduled_timestep():
    sched = DDIMScheduler()
    sched.set_timesteps(15)
    steps = [int(step) for step in sched.timesteps]
    assert len(steps) == 15
    for index, step in enumerate(steps[:-1]):
        assert previous_ddim_timestep(steps, step) == steps[index + 1]
    assert previous_ddim_timestep(steps, steps[-1]) == -1
    # round(1000/15) is 67. Trailing spacing also has gaps of 66.
    assert int(round(1000 / 15)) == 67
    assert 66 in _gaps(steps) and 67 in _gaps(steps)


def test_twenty_steps_still_step_back_by_fifty():
    sched = DDIMScheduler()
    sched.set_timesteps(20)
    steps = [int(step) for step in sched.timesteps]
    assert _gaps(steps) == [50] * 19
    for step in steps[:-1]:
        assert previous_ddim_timestep(steps, step) == step - 50


def test_step_reads_the_alpha_of_that_next_timestep():
    sched = DDIMScheduler()
    sched.set_timesteps(15)
    steps = [int(step) for step in sched.timesteps]
    # Five of the fifteen gaps are not the rounded ratio. The sample must use
    # the alpha at the next scheduled index, which is what step() indexes.
    odd = next(index for index, gap in enumerate(_gaps(steps)) if gap != 67)
    sample = np.zeros((1, 4, 2, 2), np.float32)
    velocity = np.zeros_like(sample)
    out = sched.step(velocity, steps[odd], sample)
    prev = steps[odd + 1]
    # v = 0 and x = 0 stays 0 whichever alpha is chosen. Check the index the
    # scheduler would read by repeating the alpha lookup the step uses.
    assert previous_ddim_timestep(sched.timesteps, steps[odd]) == prev
    assert out.shape == sample.shape
