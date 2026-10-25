import * as THREE from '../vendor/three.module.js';

// Just off straight up and straight down. The orbit control refuses those two
// angles, and a camera parked there will not turn sideways.
const POLE_GAP = 0.05;

// phi is how far from straight up (0) toward straight down (π). theta turns around up.
const VIEWS = {
  front: { phi: Math.PI / 2, theta: 0 },
  back: { phi: Math.PI / 2, theta: Math.PI },
  left: { phi: Math.PI / 2, theta: -Math.PI / 2 },
  right: { phi: Math.PI / 2, theta: Math.PI / 2 },
  top: { phi: POLE_GAP, theta: 0 },
  bottom: { phi: Math.PI - POLE_GAP, theta: 0 },
};

const DEFAULT_POSITION = new THREE.Vector3(1.45, 0.6, 2.45);
const DEFAULT_TARGET = new THREE.Vector3(0, 0, 0);

/** Snap a model viewport camera without changing model, pose, or rig-edit state.
 *  Up stays (0, 1, 0). Top and bottom sit just inside the pole so a sideways drag still turns. */
export function setCameraView(view, name) {
  if (!view?.camera || !view?.controls) return false;
  const preset = VIEWS[name];
  if (name !== 'reset' && !preset) return false;

  const target = name === 'reset' ? DEFAULT_TARGET : view.controls.target;
  const currentDistance = view.camera.position.distanceTo(view.controls.target);
  const distance = Number.isFinite(currentDistance) && currentDistance > 0.1
    ? currentDistance
    : DEFAULT_POSITION.length();

  view.camera.up.set(0, 1, 0);
  if (name === 'reset') {
    view.controls.target.copy(DEFAULT_TARGET);
    view.camera.position.copy(DEFAULT_POSITION);
  } else {
    const offset = new THREE.Vector3().setFromSphericalCoords(distance, preset.phi, preset.theta);
    view.camera.position.copy(target).add(offset);
  }
  view.camera.lookAt(view.controls.target);
  view.camera.updateProjectionMatrix();
  view.controls.update();
  return true;
}

/** Camera position, target, and up, so Before / After can put the same view back. */
export function rememberCamera(view) {
  if (!view?.camera || !view?.controls) return null;
  return {
    position: view.camera.position.clone(),
    target: view.controls.target.clone(),
    up: view.camera.up.clone(),
  };
}

/** Put a saved camera back, including which way is up. */
export function recallCamera(view, shot) {
  if (!view?.camera || !view?.controls || !shot?.position || !shot?.target || !shot?.up) return false;
  view.camera.up.copy(shot.up);
  view.controls.target.copy(shot.target);
  view.camera.position.copy(shot.position);
  view.camera.lookAt(view.controls.target);
  view.camera.updateProjectionMatrix();
  view.controls.update();
  return true;
}

/** Create the standard six-view and camera-reset toolbar used by 3D workshop rooms. */
export function createCameraViewControls({ container, getView, onChange }) {
  const toolbar = document.createElement('div');
  toolbar.className = 'viewport-camera-controls';
  toolbar.setAttribute('aria-label', 'Camera views');

  const buttons = [
    ['front', 'Front', 'F'],
    ['back', 'Back', 'B'],
    ['left', 'Left', 'L'],
    ['right', 'Right', 'R'],
    ['top', 'Top', 'T'],
    ['bottom', 'Bottom', 'D'],
    ['reset', 'Reset View', 'Reset'],
  ].map(([name, label, shortLabel]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.cameraView = name;
    button.textContent = shortLabel;
    button.title = label;
    button.setAttribute('aria-label', label);
    button.disabled = true;
    button.onclick = () => {
      if (setCameraView(getView(), name)) onChange?.(name);
    };
    toolbar.appendChild(button);
    return button;
  });

  container.appendChild(toolbar);
  return {
    element: toolbar,
    setEnabled(enabled) { buttons.forEach((button) => { button.disabled = !enabled; }); },
    dispose() { toolbar.remove(); },
  };
}
