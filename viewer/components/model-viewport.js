import * as THREE from 'three';
import { GLTFLoader } from '../vendor/loaders/GLTFLoader.js';
import { OrbitControls } from '../vendor/controls/OrbitControls.js';
import { IndexedOBJLoader } from '../IndexedOBJLoader.js';
import { RoomEnvironment } from '../vendor/environments/RoomEnvironment.js';
import { inspectRig, resetRigPose } from '../animation/rig-inspection.js';

// Pure geometry shared by every viewport. Each renderer still creates its own PMREM
// texture because GPU resources cannot be shared across WebGL contexts.
const roomEnvironment = new RoomEnvironment();

/**
 * Create and begin loading one reusable model viewport.
 *
 * Modes own the surrounding pane chrome and decide what happens after a model loads. The
 * component owns the Three.js scene, camera, renderer, controls, model normalization, and
 * resource lifecycle.
 */
export function createModelViewport({ pane, spec, slotIndex = null, onChange, onLoaded, onError, isCurrent }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  pane.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x100d0c);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  camera.position.set(0, 0.3, 3);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);
  if (onChange) controls.addEventListener('change', onChange);

  // Studio HDRI environment, matching Blender's Material Preview/LookDev lighting.
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromScene(roomEnvironment, 0.04).texture;
  pmremGenerator.dispose();
  scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x30302a, 0.15));

  const view = {
    kind: 'model', spec, slotIndex, pane, renderer, scene, camera, controls,
    root: null, modelRoot: null, stats: pane.querySelector('.stats'), materials: [],
    overlay: null, rig: inspectRig(null), disposed: false, loader: null,
  };
  view.resetPose = () => resetRigPose(view.rig);
  loadModel(view, onLoaded, onError, isCurrent);
  return view;
}

/** True while this view is still the one on screen. */
export function loadIsCurrent(view, isCurrent) {
  if (!view || view.disposed) return false;
  if (typeof isCurrent === 'function' && !isCurrent()) return false;
  return true;
}

/** Drop one object's geometry and materials. A stale load uses this and does not join the scene. */
export function releaseObject(root) {
  root?.traverse?.((object) => {
    object.geometry?.dispose?.();
    const list = Array.isArray(object.material) ? object.material : (object.material ? [object.material] : []);
    for (const material of list) material.dispose?.();
  });
}

/** Mark the view dead, cancel a loader that can cancel, and free its geometry,
 *  materials, and lighting texture. The vendored GLTF loader has no cancel.
 *  Callers still ignore its late callbacks. */
export function releaseViewResources(view) {
  if (!view) return;
  view.disposed = true;
  if (typeof view.loader?.abort === 'function') {
    try { view.loader.abort(); } catch { /* the fetch already ended */ }
  }
  releaseObject(view.root);
  view.root = null;
  view.modelRoot = null;
  const env = view.scene?.environment;
  if (env?.dispose) env.dispose();
  if (view.scene) view.scene.environment = null;
}

export function disposeModelViewport(view) {
  releaseViewResources(view);
  view.controls.dispose();
  view.renderer.dispose();
  view.renderer.forceContextLoss();
}

/** 1 / longest side, or null when the model has nothing we can frame. */
export function unitFitScale(box) {
  const size = box.getSize(new THREE.Vector3());
  const longest = Math.max(size.x, size.y, size.z);
  if (box.isEmpty() || !Number.isFinite(longest) || longest <= 0) return null;
  return 1 / longest;
}

/** Progress text. Does nothing once the view is gone or a newer load has started. */
export function reportLoadProgress(view, event, isCurrent) {
  if (!loadIsCurrent(view, isCurrent)) return false;
  if (!event?.lengthComputable || !view.stats) return false;
  view.stats.textContent = `loading… ${Math.round(event.loaded / event.total * 100)}%`;
  return true;
}

/** Failure text, then the caller's error. Ignored when this load is no longer current. */
export function reportLoadFailure(view, error, onError, isCurrent) {
  if (!loadIsCurrent(view, isCurrent)) return false;
  if (view.stats) view.stats.textContent = 'FAILED';
  if (onError) onError(error);
  return true;
}

/** Fit `root` into the view. Returns false without touching the pane when the load is stale
 *  or the model has no size. */
export function placeLoadedModel(view, root, { animations = [], onLoaded, onError, isCurrent } = {}) {
  if (!loadIsCurrent(view, isCurrent)) {
    releaseObject(root);
    return false;
  }
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const scale = unitFitScale(box);
  if (scale == null) {
    releaseObject(root);
    reportLoadFailure(view, new Error('This model has no size.'), onError, isCurrent);
    return false;
  }

  // Normalize to a unit box centered on the origin so differently scaled exports remain
  // directly comparable and every room can use the same camera framing contract.
  const pivot = new THREE.Group();
  const centre = box.getCenter(new THREE.Vector3());
  root.position.sub(centre);
  pivot.scale.setScalar(scale);
  pivot.add(root);
  pivot.updateMatrixWorld(true);

  let faces = 0;
  let verts = 0;
  root.traverse((object) => {
    if (!object.isMesh) return;
    const geometry = object.geometry;
    // Shape-stage exports may omit NORMAL. Inspection materials and wireframe need a
    // real attribute even when the original flat-shaded material appears acceptable.
    if (geometry && !geometry.attributes.normal) geometry.computeVertexNormals();
    verts += geometry?.attributes?.position?.count || 0;
    faces += geometry?.index ? geometry.index.count / 3 : (geometry?.attributes?.position?.count || 0) / 3;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material) continue;
      view.materials.push({
        mesh: object,
        original: material,
        flat: null,
        normal: null,
        origFlatShading: !!material.flatShading,
      });
    }
  });

  view.root = pivot;
  view.modelRoot = root;
  view.rig = inspectRig(root, animations);
  view.scene.add(pivot);
  if (view.stats) view.stats.textContent = `${faces.toLocaleString()} faces\n${verts.toLocaleString()} verts (as stored)`;
  if (onLoaded) onLoaded(view);
  return true;
}

function loadModel(view, onLoaded, onError, isCurrent) {
  const { spec } = view;
  const loaded = (root, animations = []) => {
    placeLoadedModel(view, root, { animations, onLoaded, onError, isCurrent });
  };
  const progress = (event) => reportLoadProgress(view, event, isCurrent);
  const failed = (error) => reportLoadFailure(view, error, onError, isCurrent);

  if (spec.ext === 'obj') {
    const loader = new IndexedOBJLoader();
    view.loader = loader;
    loader.load(spec.url, (root) => {
      // Match o_voxel's native TRELLIS -> GLB coordinate transform for raw OBJ inputs.
      root.rotation.x = -Math.PI / 2;
      loaded(root);
    }, progress, failed);
    return;
  }

  // The vendored GLTF loader cannot cancel a fetch. Late callbacks are ignored above.
  const loader = spec.manager ? new GLTFLoader(spec.manager) : new GLTFLoader();
  view.loader = loader;
  loader.load(spec.url, (gltf) => loaded(gltf.scene, gltf.animations || []), progress, failed);
}
