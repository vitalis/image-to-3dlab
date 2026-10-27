"""Focused tests for dependency-free browser modules that can run under Node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
THREE_URL = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()


def _with_three(body: str) -> str:
    """Resolve the bare import `three` the way studio.html does, for this test only.

    The hook has to be in place before the modules load, so the body imports with
    `await import`, not a static import (those run first).
    """
    return (
        "import { registerHooks } from 'node:module';\n"
        "registerHooks({ resolve(specifier, context, next) {\n"
        f"  if (specifier === 'three') return {{ url: {json.dumps(THREE_URL)}, shortCircuit: true }};\n"
        "  return next(specifier, context);\n"
        "} });\n"
        f"{body}"
    )


def _node(program: str):
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_rig_inspection_deduplicates_bones_and_resets_each_skeleton():
    module_url = (REPO / "viewer" / "animation" / "rig-inspection.js").as_uri()
    program = f"""
      import {{ describeBone, inspectRig, resetRigPose }} from {json.dumps(module_url)};
      const parent = {{ isBone: true, name: 'root' }};
      const child = {{ isBone: true, name: 'knee' }};
      const bone = {{
        isBone: true, name: 'hip', parent, children: [child],
        position: {{ x: 1, y: 2, z: 3 }},
        quaternion: {{ x: 0, y: 0, z: 0, w: 1 }},
        scale: {{ x: 1, y: 1, z: 1 }},
      }};
      const skeleton = {{ bones: [bone], resets: 0, pose() {{ this.resets += 1; }} }};
      const mesh = {{ isSkinnedMesh: true, skeleton }};
      const root = {{ traverse(callback) {{ callback(bone); callback(mesh); callback(mesh); }} }};
      const rig = inspectRig(root, [{{ name: 'idle' }}]);
      bone.position.x = 4;
      const description = describeBone(bone, rig);
      resetRigPose(rig);
      console.log(JSON.stringify({{
        bones: rig.bones.map((item) => item.name),
        skeletons: rig.skeletons.length,
        meshes: rig.skinnedMeshes.length,
        clips: rig.animations.map((item) => item.name),
        resets: skeleton.resets,
        description,
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "bones": ["hip"],
        "skeletons": 1,
        "meshes": 2,
        "clips": ["idle"],
        "resets": 1,
        "description": {
            "name": "hip",
            "parent": "root",
            "children": ["knee"],
            "bind": {
                "position": [1, 2, 3],
                "quaternion": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            },
            "pose": {
                "position": [4, 2, 3],
                "quaternion": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            },
        },
    }


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_animation_player_owns_transport_without_owning_threejs():
    module_url = (REPO / "viewer" / "animation" / "player.js").as_uri()
    program = f"""
      import {{ AnimationPlayer }} from {json.dumps(module_url)};
      let queued = null;
      const action = {{
        time: 0, paused: false,
        reset() {{ this.time = 0; return this; }},
        play() {{ return this; }}
      }};
      const mixer = {{
        clipAction() {{ return action; }},
        stopAllAction() {{}},
        update(delta) {{ action.time += delta; }},
        addEventListener() {{}},
        getRoot() {{ return {{}}; }},
        uncacheRoot() {{}},
      }};
      const frames = [];
      const states = [];
      const player = new AnimationPlayer(mixer, {{
        requestFrame(callback) {{ queued = callback; return 7; }},
        cancelFrame() {{ queued = null; }},
        onFrame(time, duration) {{ frames.push([time, duration]); }},
        onStateChange(playing) {{ states.push(playing); }},
      }});
      player.select({{ name: 'walk', duration: 2 }});
      player.play();
      queued(1000);
      queued(1500);
      player.pause();
      player.seek(9);
      console.log(JSON.stringify({{
        time: player.time,
        duration: player.duration,
        paused: action.paused,
        states,
        lastFrame: frames.at(-1),
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "time": 2,
        "duration": 2,
        "paused": True,
        "states": [False, False, True, False],
        "lastFrame": [2, 2],
    }


def _near(got, want, eps=1e-5):
    return len(got) == len(want) and all(abs(a - b) <= eps for a, b in zip(got, want))


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_camera_view_controls_snap_and_reset_without_touching_model_state():
    """Top used to sit on the pole and set up to (0, 0, -1). A sideways drag then did not move.
    The real orbit control has to run, not a stub update."""
    controls_url = (REPO / "viewer" / "components" / "camera-view-controls.js").as_uri()
    orbit_url = (REPO / "viewer" / "vendor" / "controls" / "OrbitControls.js").as_uri()
    program = _with_three(f"""
      const THREE = await import({json.dumps(THREE_URL)});
      const {{ OrbitControls }} = await import({json.dumps(orbit_url)});
      const {{ setCameraView }} = await import({json.dumps(controls_url)});
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      camera.position.set(0, 0, 4);
      const controls = new OrbitControls(camera, null);
      const modelState = {{ rigEdits: 3, pose: 'idle' }};
      const view = {{ camera, controls, modelState }};
      const distanceOf = () => camera.position.distanceTo(controls.target);
      const snap = (name) => {{
        const before = distanceOf();
        const ok = setCameraView(view, name);
        return {{ ok, distance: distanceOf(), kept: Math.abs(distanceOf() - before) < 1e-5,
          position: camera.position.toArray(), up: camera.up.toArray() }};
      }};
      const front = snap('front');
      const back = snap('back');
      const left = snap('left');
      const right = snap('right');
      const top = snap('top');
      const beforeDrag = camera.position.clone();
      controls._rotateLeft(0.8);
      controls.update();
      const dragged = beforeDrag.distanceTo(camera.position);
      const bottom = snap('bottom');
      const reset = snap('reset');
      console.log(JSON.stringify({{
        front, back, left, right, top, bottom, reset, dragged,
        modelState,
      }}));
    """)
    out = _node(program)

    assert out["modelState"] == {"rigEdits": 3, "pose": "idle"}
    for name in ("front", "back", "left", "right", "top", "bottom", "reset"):
        assert out[name]["ok"] is True
        assert _near(out[name]["up"], [0, 1, 0])
    for name in ("front", "back", "left", "right"):
        assert out[name]["kept"] is True
        assert abs(out[name]["distance"] - 4) < 1e-5
    assert _near(out["left"]["position"], [-4, 0, 0])
    assert _near(out["right"]["position"], [4, 0, 0])
    assert _near(out["front"]["position"], [0, 0, 4])
    assert _near(out["back"]["position"], [0, 0, -4])
    assert out["top"]["position"][1] > 3.9
    assert out["bottom"]["position"][1] < -3.9
    assert out["dragged"] > 0.01
    assert _near(out["reset"]["position"], [1.45, 0.6, 2.45], eps=1e-4)
    assert _near(out["reset"]["up"], [0, 1, 0])


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_before_and_after_puts_the_up_direction_back():
    controls_url = (REPO / "viewer" / "components" / "camera-view-controls.js").as_uri()
    orbit_url = (REPO / "viewer" / "vendor" / "controls" / "OrbitControls.js").as_uri()
    program = _with_three(f"""
      const THREE = await import({json.dumps(THREE_URL)});
      const {{ OrbitControls }} = await import({json.dumps(orbit_url)});
      const {{ setCameraView, rememberCamera, recallCamera }} = await import({json.dumps(controls_url)});
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      camera.position.set(0, 0, 4);
      const controls = new OrbitControls(camera, null);
      const view = {{ camera, controls }};
      setCameraView(view, 'top');
      const shot = rememberCamera(view);
      camera.position.set(2, 0, 0);
      camera.up.set(0, 0, -1);
      controls.target.set(1, 1, 1);
      recallCamera(view, shot);
      console.log(JSON.stringify({{
        up: camera.up.toArray(),
        position: camera.position.toArray(),
        target: controls.target.toArray(),
        savedUp: shot.up.toArray(),
      }}));
    """)
    out = _node(program)
    assert _near(out["savedUp"], [0, 1, 0])
    assert _near(out["up"], [0, 1, 0])
    assert _near(out["target"], [0, 0, 0])
    assert abs(out["position"][1] - 4) < 0.2


def test_rig_sidecar_schema_is_valid_json_and_versioned():
    schema = json.loads((REPO / "rigs" / "rig-sidecar.schema.json").read_text())

    assert schema["properties"]["schemaVersion"]["const"] == 1
    assert schema["properties"]["coordinateSpace"]["const"] == "armature-local"
    assert "binding" in schema["properties"]


def _run_panel(script: str) -> object:
    """Drive the shipped JobProgressPanel against a minimal DOM and read the rows back.

    The panel is the thing that actually renders "estimating…", so it is the thing under
    test — a re-implementation of its rules here would pass while the shipped file stayed
    broken, which is exactly how this bug survived (2026-09-21).
    """
    module_url = (REPO / "viewer" / "components" / "job-progress.js").as_uri()
    program = f"""
      import {{ JobProgressPanel }} from {json.dumps(module_url)};

      // Enough DOM for the panel: rows it creates, and the three text elements it writes.
      const make = () => {{
        const node = {{
          className: '', dataset: {{}}, innerHTML: '', textContent: '', style: {{}},
          children: [],
          classList: {{
            add: (c) => {{ node.className += ' ' + c; }},
            contains: (c) => node.className.split(/\\s+/).includes(c),
          }},
          appendChild: (child) => {{ node.children.push(child); return child; }},
          querySelector: (selector) => {{
            const key = selector.replace('.', '');
            node._parts = node._parts || {{}};
            node._parts[key] = node._parts[key] || make();
            return node._parts[key];
          }},
        }};
        return node;
      }};
      globalThis.document = {{ createElement: make }};

      const host = make(), bar = make(), label = make(), eta = make();
      const panel = new JobProgressPanel({{ stages: host, bar, label, eta }});
      const rowState = () => host.children.map((row) => [
        row.className.trim(), row.querySelector('.stage-detail').textContent,
      ]);
      {script}
    """
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_welcome(expr: str):
    module_url = (REPO / "viewer" / "components" / "welcome-content.js").as_uri()
    program = f"import * as w from {json.dumps(module_url)}; console.log(JSON.stringify({expr}));"
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


WELCOME = {
    "brand": {"name": "AssetFurnace"},
    "version": "0.3.0",
    "routes": [
        {"id": "pixal3d", "state": "missing"},
        {"id": "qwen-image", "state": "ready"},
    ],
    "news": [
        {
            "version": "0.3.0",
            "sections": {
                "Fixed": ["f1"],
                "Added": ["a1", "a2", "a3"],
                "Changed": ["c1"],
            },
        }
    ],
}


def _run_banner(expr: str):
    module_url = (REPO / "viewer" / "components" / "update-banner.js").as_uri()
    program = f"import * as b from {json.dumps(module_url)}; console.log(JSON.stringify({expr}));"
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


CHECK = {
    "enabled": True,
    "newer": True,
    "current": "0.2.0",
    "latest": "0.3.0",
    "url": "https://github.com/x/releases/tag/v0.3.0",
    "command": "curl … | bash",
}


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_a_gated_warning_goes_once_access_is_confirmed_and_no_sooner():
    module_url = (REPO / "viewer" / "components" / "gated-caveat.js").as_uri()
    program = f"""
      import {{ showsCaveat }} from {json.dumps(module_url)};
      const gated = {{ caveat: 'gated', gated_repo: 'org/model' }};
      const other = {{ caveat: 'not licensed in the EU', gated_repo: null }};
      console.log(JSON.stringify([
        showsCaveat(gated, {{ 'org/model': 'yes' }}),
        showsCaveat(gated, {{ 'org/model': 'no' }}),
        showsCaveat(gated, {{ 'org/model': 'unknown' }}),
        showsCaveat(gated, {{}}),
        showsCaveat(other, {{ 'org/model': 'yes' }}),
        showsCaveat({{ caveat: null }}, {{}}),
      ]));
    """
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [False, True, True, True, True, False]


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_the_setup_panel_says_a_run_finished_and_looks_like_it():
    """The title used to be written once, at the start, and never again: a 16 GB fetch that
    ended perfectly sat there saying "Downloading", at 100%, looking like a page that gave up
    halfway. The panel is now also *painted* with how the run ended, not just titled."""
    module_url = (REPO / "viewer" / "components" / "setup-progress.js").as_uri()
    program = f"""
      import {{ runTitle, setupVerb, panelState }} from {json.dumps(module_url)};
      const weights = {{ label: 'TRELLIS.2 (AMD)', build_present: true,
                         setup_fetches_weights: true }};
      const unbuilt = {{ label: 'TRELLIS.2 (AMD)', build_present: false,
                         setup_fetches_weights: true }};
      const logs = {{ label: 'Pixal3D', build_present: true, setup_fetches_weights: false }};
      const inFlight = ['queued', 'log', 'downloading', 'building'];
      const endings = ['done', 'error', 'cancelled'];
      console.log(JSON.stringify({{
        verbs: [setupVerb(unbuilt, 'download'), setupVerb(weights, 'download'),
                setupVerb(logs, 'download'), setupVerb(weights, 'rebuild')],
        whileRunning: inFlight.map((phase) => runTitle(weights, 'download', {{ phase }})),
        states: inFlight.map((phase) => panelState({{ phase }})),
        end: endings.map((phase) => runTitle(weights, 'download', {{ phase }})),
        endStates: endings.map((phase) => panelState({{ phase }})),
        // A resumed run whose backend the catalogue no longer lists still names something.
        unknown: runTitle({{ label: 'trellis-rocm' }}, 'download', {{ phase: 'done' }}),
      }}));
    """
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(result.stdout)

    assert out["verbs"] == ["Setting up", "Downloading", "Building", "Rebuilding"]
    assert out["whileRunning"] == ["Downloading TRELLIS.2 (AMD)"] * 4, (
        "a run in flight must not claim an outcome it has not reached"
    )
    assert out["states"] == ["running"] * 4
    assert out["end"] == [
        "✓ Installed · TRELLIS.2 (AMD)",
        "! Setup failed · TRELLIS.2 (AMD)",
        "■ Setup cancelled · TRELLIS.2 (AMD)",
    ]
    assert out["endStates"] == ["done", "error", "cancelled"]
    assert out["unknown"] == "✓ Installed · trellis-rocm"


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_the_finished_marks_are_the_studios_own():
    """One convention: the Activity list marks a finished thing '✓', '!' and '■', and the Setup
    panel borrows them rather than inventing a second set."""
    pages_url = (REPO / "viewer" / "studio" / "pages.js").as_uri()
    setup_url = (REPO / "viewer" / "components" / "setup-progress.js").as_uri()
    program = f"""
      import {{ MARK, activityParts }} from {json.dumps(pages_url)};
      import {{ runTitle }} from {json.dumps(setup_url)};
      const backend = {{ label: 'Stable Fast 3D', build_present: true,
                         setup_fetches_weights: true }};
      console.log(JSON.stringify({{
        done: runTitle(backend, 'download', {{ phase: 'done' }}).split(' ')[0],
        error: runTitle(backend, 'download', {{ phase: 'error' }}).split(' ')[0],
        cancelled: runTitle(backend, 'download', {{ phase: 'cancelled' }}).split(' ')[0],
        activity: activityParts({{ result: 'done', step: 'rig', title: 'x', time: '' }}),
      }}));
    """
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(result.stdout)

    assert (out["done"], out["error"], out["cancelled"]) == ("✓", "!", "■")
    assert out["activity"][0].split(" ")[0] == out["done"]


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_a_replaced_load_does_not_write_the_pane_and_an_empty_model_is_not_placed():
    module_url = (REPO / "viewer" / "components" / "model-viewport.js").as_uri()
    program = _with_three(f"""
      const THREE = await import({json.dumps(THREE_URL)});
      const {{
        placeLoadedModel, reportLoadProgress, reportLoadFailure, releaseViewResources,
      }} = await import({json.dumps(module_url)});
      const stats = {{ textContent: 'keep' }};
      const added = [];
      const scene = {{ add(object) {{ added.push(object); }} }};
      const box = new THREE.Mesh(new THREE.BoxGeometry(2, 4, 1));
      const live = {{ disposed: false, stats, scene, materials: [] }};
      const placed = placeLoadedModel(live, box, {{}});
      const staleGeometry = new THREE.BoxGeometry(1, 1, 1);
      let staleDisposed = 0;
      staleGeometry.addEventListener('dispose', () => {{ staleDisposed += 1; }});
      const staleRoot = new THREE.Mesh(staleGeometry);
      const staleStats = {{ textContent: 'keep' }};
      const stale = {{ disposed: false, stats: staleStats, scene, materials: [] }};
      let staleLoaded = false;
      const ignored = placeLoadedModel(stale, staleRoot, {{
        isCurrent: () => false,
        onLoaded() {{ staleLoaded = true; }},
      }});
      const progressStats = {{ textContent: 'keep' }};
      const progressView = {{ disposed: false, stats: progressStats }};
      const progressIgnored = reportLoadProgress(progressView,
        {{ lengthComputable: true, loaded: 1, total: 4 }}, () => false);
      reportLoadProgress({{ disposed: true, stats: progressStats }},
        {{ lengthComputable: true, loaded: 1, total: 2 }}, () => true);
      const progressOk = reportLoadProgress(progressView,
        {{ lengthComputable: true, loaded: 1, total: 4 }}, () => true);
      const failStats = {{ textContent: 'keep' }};
      let failCalled = false;
      const failIgnored = reportLoadFailure({{ disposed: true, stats: failStats }},
        new Error('late'), () => {{ failCalled = true; }}, () => true);
      const emptyErrors = [];
      const emptyStats = {{ textContent: '' }};
      const emptyAdded = [];
      const empty = {{ disposed: false, stats: emptyStats, scene: {{ add(o) {{ emptyAdded.push(o); }} }}, materials: [] }};
      const emptyPlaced = placeLoadedModel(empty, new THREE.Group(), {{
        onError(error) {{ emptyErrors.push(error.message); }},
      }});
      const point = new THREE.Mesh(new THREE.BufferGeometry().setAttribute(
        'position', new THREE.BufferAttribute(new Float32Array([1, 2, 3]), 3)));
      const pointErrors = [];
      const pointPlaced = placeLoadedModel(
        {{ disposed: false, stats: {{ textContent: '' }}, scene: {{ add() {{}} }}, materials: [] }},
        point, {{ onError(error) {{ pointErrors.push(error.message); }} }});
      let envDisposed = false;
      let aborted = false;
      const going = {{
        root: new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)),
        scene: {{ environment: {{ dispose() {{ envDisposed = true; }} }} }},
        loader: {{ abort() {{ aborted = true; }} }},
        materials: [],
      }};
      let rootDisposed = 0;
      going.root.geometry.addEventListener('dispose', () => {{ rootDisposed += 1; }});
      releaseViewResources(going);
      const noAbort = {{ root: null, scene: {{}}, loader: {{}}, materials: [] }};
      releaseViewResources(noAbort);
      console.log(JSON.stringify({{
        placed, scale: live.root.scale.x, added: added.length, faces: live.stats.textContent,
        ignored, staleDisposed, staleLoaded, staleStats: staleStats.textContent,
        progressIgnored, progressOk, progressText: progressStats.textContent,
        failIgnored, failCalled, failText: failStats.textContent,
        emptyPlaced, emptyErrors, emptyAdded: emptyAdded.length, emptyText: emptyStats.textContent,
        pointPlaced, pointErrors,
        envDisposed, aborted, rootDisposed, goingDisposed: going.disposed,
        noAbortDisposed: noAbort.disposed,
      }}));
    """)
    out = _node(program)
    assert out["placed"] is True
    assert abs(out["scale"] - 0.25) < 1e-6
    assert out["added"] == 1
    assert "faces" in out["faces"]
    assert out["ignored"] is False
    assert out["staleDisposed"] >= 1
    assert out["staleLoaded"] is False
    assert out["staleStats"] == "keep"
    assert out["progressIgnored"] is False
    assert out["progressOk"] is True
    assert out["progressText"] == "loading… 25%"
    assert out["failIgnored"] is False
    assert out["failCalled"] is False
    assert out["failText"] == "keep"
    assert out["emptyPlaced"] is False
    assert out["emptyAdded"] == 0
    assert out["emptyErrors"] == ["This model has no size."]
    assert out["emptyText"] == "FAILED"
    assert out["pointPlaced"] is False
    assert out["pointErrors"] == ["This model has no size."]
    assert out["envDisposed"] is True
    assert out["aborted"] is True
    assert out["rootDisposed"] >= 1
    assert out["goingDisposed"] is True
    assert out["noAbortDisposed"] is True


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_reset_pose_returns_the_playhead_and_the_bind_pose():
    player_url = (REPO / "viewer" / "animation" / "player.js").as_uri()
    rig_url = (REPO / "viewer" / "animation" / "rig-inspection.js").as_uri()
    program = f"""
      import {{ AnimationPlayer }} from {json.dumps(player_url)};
      import {{ restCharacter }} from {json.dumps(rig_url)};
      const action = {{ time: 0, paused: true, reset() {{ this.time = 0; return this; }}, play() {{ return this; }} }};
      const mixer = {{
        clipAction() {{ return action; }},
        stopAllAction() {{}},
        update() {{}},
        addEventListener() {{}},
        getRoot() {{ return {{}}; }},
        uncacheRoot() {{}},
      }};
      const frames = [];
      const player = new AnimationPlayer(mixer, {{
        requestFrame() {{ return 1; }},
        cancelFrame() {{}},
        onFrame(time, duration) {{ frames.push([time, duration]); }},
      }});
      player.select({{ name: 'walk', duration: 2 }});
      player.play();
      player.seek(1.5);
      const skeleton = {{ poses: 0, pose() {{ this.poses += 1; }} }};
      restCharacter(player, {{ skeletons: [skeleton] }});
      console.log(JSON.stringify({{
        time: player.time, paused: action.paused, poses: skeleton.poses, last: frames.at(-1),
      }}));
    """
    out = _node(program)
    assert out["time"] == 0
    assert out["paused"] is True
    assert out["poses"] == 1
    assert out["last"] == [0, 2]


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_leaving_an_unkept_patch_asks_before_the_session_is_cleared():
    viewer_url = (REPO / "viewer" / "studio" / "viewer.js").as_uri()
    question = "Close the brush and throw away the patch you have not kept?"
    program = _with_three(f"""
      const {{ choosePatchLeave }} = await import({json.dumps(viewer_url)});
      const asked = [];
      const confirm = (text) => {{ asked.push(text); return answers.shift(); }};
      let answers = [false];
      const stay = choosePatchLeave('preview-1', false, confirm);
      answers = [true];
      const discard = choosePatchLeave('preview-1', false, confirm);
      const noPreview = choosePatchLeave(null, false, confirm);
      const held = choosePatchLeave('preview-1', true, confirm);
      console.log(JSON.stringify({{ stay, discard, noPreview, held, asked }}));
    """)
    out = _node(program)
    assert out["stay"] == "stay"
    assert out["discard"] == "discard"
    assert out["noPreview"] == "go"
    assert out["held"] == "go"
    assert out["asked"] == [question, question]


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_a_patch_stays_on_the_asset_it_was_drawn_on():
    """Switching assets while a preview is in flight, or cancelling that switch,
    must not file the patch onto the other mesh."""
    viewer_url = (REPO / "viewer" / "studio" / "viewer.js").as_uri()
    program = _with_three(f"""
      const {{ previewBody, keepBody, samePatchAsset, openAsset }} = await import({json.dumps(viewer_url)});
      const source = {{ id: 'a', finished: 'animate/a/finished.glb' }};
      const other = {{ id: 'b', finished: 'animate/b/finished.glb' }};
      console.log(JSON.stringify({{
        preview: previewBody(source, null, [{{ point: [0, 1, 0] }}]),
        keep: keepBody(source, 'preview-1'),
        dropped: samePatchAsset(source, other),
        stillHere: samePatchAsset(source, source),
        cancelled: openAsset(source, other, 'stay'),
        accepted: openAsset(source, other, 'go'),
      }}));
    """)
    out = _node(program)
    assert out["preview"]["from"] == "animate/a/finished.glb"
    assert out["keep"] == {"from": "animate/a/finished.glb", "preview": "preview-1"}
    assert out["dropped"] is False
    assert out["stillHere"] is True
    assert out["cancelled"]["id"] == "a"
    assert out["accepted"]["id"] == "b"


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_indexed_obj_skips_a_bad_face_and_does_not_keep_spare_room():
    module_url = (REPO / "viewer" / "IndexedOBJLoader.js").as_uri()
    program = _with_three(f"""
      const {{ IndexedOBJLoader }} = await import({json.dumps(module_url)});
      const text = [
        'v 0 0 0', 'v 1 0 0', 'v 0 1 0', 'v 0 0 1',
        'f 1 2 3', 'f 1 2 99', 'f 1 2 0', 'f 1 2 3 99', 'f -1 -2 -3',
      ].join('\\n');
      const bytes = new TextEncoder().encode(text);
      globalThis.fetch = async () => ({{
        ok: true,
        headers: {{ get: () => String(bytes.byteLength) }},
        body: new ReadableStream({{ start(controller) {{ controller.enqueue(bytes); controller.close(); }} }}),
      }});
      let mesh = null;
      await new IndexedOBJLoader().load('/model.obj', (root) => {{ mesh = root; }});
      const position = mesh.geometry.attributes.position.array;
      const index = mesh.geometry.index.array;
      console.log(JSON.stringify({{
        index: Array.from(index),
        positionTight: position.buffer.byteLength === position.byteLength,
        indexTight: index.buffer.byteLength === index.byteLength,
        positionLength: position.length,
      }}));
    """)
    out = _node(program)
    assert out["index"] == [0, 1, 2, 3, 2, 1]
    assert out["positionTight"] is True
    assert out["indexTight"] is True
    assert out["positionLength"] == 12


@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
@pytest.mark.skipif(
    NODE is None, reason="Node is required to execute browser ES modules"
)
def test_a_missing_setup_stops_the_poll():
    module_url = (REPO / "viewer" / "components" / "setup-progress.js").as_uri()
    program = f"""
      import {{ openSetupWatch }} from {json.dumps(module_url)};
      const phases = [];
      const urls = [];
      let timers = 0;
      openSetupWatch({{ status_url: '/api/setup/qwen-image/status', events_url: '/api/setup/qwen-image/events' }}, {{
        fetchImpl: async (url) => {{
          urls.push(url);
          return {{ status: 404, ok: false, json: async () => ({{
            error: 'This setup is no longer running.', status: 'error',
          }}) }};
        }},
        EventSourceImpl: class {{ constructor() {{}} close() {{}} }},
        onEvent: (event) => phases.push(event.phase),
        setTimer: () => {{ timers += 1; return 1; }},
        clearTimer() {{}},
        interval: 5,
      }});
      await new Promise((resolve) => setTimeout(resolve, 0));
      console.log(JSON.stringify({{ phases, urls, timers }}));
    """
    out = _node(program)
    assert out["urls"] == ["/api/setup/qwen-image/status"]
    assert out["phases"] == ["error"]
    assert out["timers"] == 0


def test_setup_polls_after_the_stream_drops_and_does_not_repeat_the_log():
    module_url = (REPO / "viewer" / "components" / "setup-progress.js").as_uri()
    program = f"""
      import {{ openSetupWatch }} from {json.dumps(module_url)};
      const logs = [];
      const phases = [];
      const drops = [];
      const urls = [];
      let n = 0;
      const fetchImpl = async (url) => {{
        urls.push(url);
        n += 1;
        if (n === 1) throw new Error('reset');
        if (n === 2) return {{ ok: true, json: async () => ({{
          status: 'running', last_event: {{ phase: 'downloading', detail: '10%' }}, log_tail: 'fetch a',
        }}) }};
        return {{ ok: true, json: async () => ({{
          status: 'done', last_event: {{ phase: 'done', detail: 'done' }}, log_tail: 'fetch a\\nfetch b',
        }}) }};
      }};
      const timers = [];
      const watch = openSetupWatch({{ status_url: '/api/setup/pixal3d/status', events_url: '/events' }}, {{
        fetchImpl,
        EventSourceImpl: class {{ constructor() {{}} close() {{}} }},
        onLog: (text) => logs.push(text),
        onEvent: (event) => phases.push(event.phase),
        onDrop: (message) => drops.push(message),
        interval: 5,
        now: () => 0,
        setTimer: (fn) => {{ timers.push(fn); return timers.length; }},
        clearTimer() {{}},
      }});
      await new Promise((resolve) => setTimeout(resolve, 0));
      timers.shift()();
      await new Promise((resolve) => setTimeout(resolve, 0));
      timers.shift()();
      await new Promise((resolve) => setTimeout(resolve, 0));
      const sources = [];
      class Stream {{
        constructor(url) {{ this.url = url; sources.push(this); }}
        close() {{ this.closed = true; }}
      }}
      const toolMessages = [];
      const toolDrops = [];
      const toolTimers = [];
      openSetupWatch({{ events_url: '/api/setup/run/abc/events', setup_run_id: 'abc' }}, {{
        EventSourceImpl: Stream,
        fetchImpl: async () => {{ throw new Error('no status for this install'); }},
        onEvent: (event) => toolMessages.push(event.message),
        onDrop: (message) => toolDrops.push(message),
        setTimer: (fn) => {{ toolTimers.push(fn); return toolTimers.length; }},
        clearTimer() {{}},
        interval: 5,
      }});
      sources[0].onmessage({{ data: JSON.stringify({{ phase: 'setup', message: 'line 1' }}) }});
      sources[0].onmessage({{ data: JSON.stringify({{ phase: 'setup', message: 'line 2' }}) }});
      sources[0].onerror();
      toolTimers.shift()();
      sources[1].onmessage({{ data: JSON.stringify({{ phase: 'setup', message: 'line 1' }}) }});
      sources[1].onmessage({{ data: JSON.stringify({{ phase: 'setup', message: 'line 2' }}) }});
      sources[1].onmessage({{ data: JSON.stringify({{ phase: 'setup_done', message: 'all done', status: 'done' }}) }});
      let release;
      let updates = 0;
      const stopped = openSetupWatch({{ status_url: '/api/setup/pixal3d/status' }}, {{
        fetchImpl: () => new Promise((resolve) => {{ release = resolve; }}),
        EventSourceImpl: class {{ constructor() {{}} close() {{}} }},
        onEvent: () => {{ updates += 1; }},
        setTimer: () => {{ throw new Error('scheduled after stop'); }},
        interval: 5,
      }});
      await Promise.resolve();
      stopped.stop();
      release({{ ok: true, json: async () => ({{ status: 'running', last_event: {{ phase: 'downloading' }}, log_tail: 'x' }}) }});
      await new Promise((resolve) => setTimeout(resolve, 0));
      const resumed = [];
      openSetupWatch({{ backend: 'pixal3d', events_url: '/api/setup/pixal3d/events' }}, {{
        fetchImpl: async (url) => {{ resumed.push(url); return {{ ok: true, json: async () => ({{
          status: 'running', last_event: {{ phase: 'building' }}, log_tail: '',
        }}) }}; }},
        EventSourceImpl: class {{ constructor() {{}} close() {{}} }},
        setTimer: () => 1,
        clearTimer() {{}},
        interval: 5,
      }});
      await new Promise((resolve) => setTimeout(resolve, 0));
      watch.stop();
      console.log(JSON.stringify({{ logs, phases, drops, urls, toolMessages, toolDrops, updates, resumed }}));
    """
    out = _node(program)
    assert out["drops"] == ["The connection dropped. Still checking…"]
    assert out["logs"] == ["fetch a", "fetch a\nfetch b"]
    assert out["phases"] == ["downloading", "done"]
    assert out["urls"] == ["/api/setup/pixal3d/status"] * 3
    assert out["toolDrops"] == ["The connection dropped. Still checking…"]
    assert out["toolMessages"] == ["line 1", "line 2", "all done"]
    assert out["updates"] == 0
    assert out["resumed"] == ["/api/setup/pixal3d/status"]


def viewer_scripts() -> list[Path]:
    """Every first-party viewer script; vendored libraries are someone else's to parse."""
    viewer = REPO / "viewer"
    return sorted(
        p
        for p in viewer.rglob("*.js")
        if "vendor" not in p.relative_to(viewer).parts and "node_modules" not in p.parts
    )


@pytest.mark.skipif(NODE is None, reason="Node is required to parse browser ES modules")
@pytest.mark.parametrize(
    "script", viewer_scripts(), ids=lambda p: str(p.relative_to(REPO))
)
def test_every_viewer_script_parses(script):
    # One syntax error in any module stops the whole app loading: no tabs, no drop, no
    # browse, and dropped files download instead. 0.3.5 shipped exactly that.
    result = subprocess.run(
        [NODE, "--input-type=module", "--check"],
        stdin=script.open("rb"),
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
