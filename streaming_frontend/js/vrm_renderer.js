/**
 * VRM Renderer — Three.js + @pixiv/three-vrm
 *
 * Reusable renderer for loading a VRM model and applying ARKit-style
 * blendshape weights with EMA smoothing.
 *
 * Dependencies (loaded via importmap in HTML):
 *   - three
 *   - three/addons/loaders/GLTFLoader
 *   - @pixiv/three-vrm
 */

class VRMRenderer {
    /**
     * @param {string} canvasId  id of the <canvas> element
     */
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) throw new Error(`Canvas #${canvasId} not found`);

        // Three.js basics
        this.scene = new THREE.Scene();
        this.clock = new THREE.Clock();

        // Camera — facing head from the front
        this.camera = new THREE.PerspectiveCamera(20, 1, 0.1, 20);
        this.camera.position.set(0, 1.35, 1.8);
        this.camera.lookAt(0, 1.35, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
        });
        this.renderer.setSize(512, 512);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;

        // Lighting
        const ambient = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambient);
        const dir = new THREE.DirectionalLight(0xffffff, 0.8);
        dir.position.set(0.5, 1.5, 1);
        this.scene.add(dir);

        // Model state
        this.vrm = null;          // VRM instance (if loaded)
        this.glbScene = null;     // raw GLTF scene (if GLB loaded)
        this.morphMeshes = [];    // [{ mesh, nameToIndex }] for GLB morph targets
        this.emaState = {};       // name → smoothed value
        this.emaAlpha = 0.7;      // default EMA factor
        this._headPoseEmaAlpha = 0.4;  // head pose EMA — more responsive than blendshapes
        this.headPoseScale = 1.0;      // head pose scale (1.0 = raw radians from JSON)
        this._headPoseEma = { pitch: 0, yaw: 0, roll: 0 };  // head pose EMA state
        this._pendingHeadPose = null;  // applied in _tick() after vrm.update()
        this.headPoseEnabled = true;   // toggle head rotation on/off
        this._glbHeadNode = null;      // cached GLB head bone

        // Animation loop
        this._animId = null;
        this._running = false;
    }

    /* ------------------------------------------------------------------ */
    /*  Model loading (VRM / GLB)                                         */
    /* ------------------------------------------------------------------ */

    /**
     * Load a model file. Detects VRM vs plain GLB by extension.
     * @param {string} url       object URL or path
     * @param {string} filename  original filename (used for extension check)
     * @returns {Promise<void>}
     */
    async loadModel(url, filename) {
        const ext = (filename || '').split('.').pop().toLowerCase();
        if (ext === 'vrm') {
            return this.loadVRM(url);
        }
        return this.loadGLB(url);
    }

    /**
     * Load a VRM from a URL / object URL.
     * @param {string} url
     * @returns {Promise<void>}
     */
    async loadVRM(url) {
        const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
        const { VRMLoaderPlugin } = await import('@pixiv/three-vrm');

        this._removeCurrentModel();

        const loader = new GLTFLoader();
        loader.register((parser) => new VRMLoaderPlugin(parser));

        return new Promise((resolve, reject) => {
            loader.load(
                url,
                (gltf) => {
                    const vrm = gltf.userData.vrm;
                    if (!vrm) {
                        reject(new Error('Not a valid VRM file'));
                        return;
                    }

                    vrm.scene.rotation.y = Math.PI;
                    this.scene.add(vrm.scene);
                    this.vrm = vrm;
                    console.log('[VRM]', vrm);
                    const mgr = vrm.expressionManager;
                    if (mgr) {
                        const mapKeys = Object.keys(mgr.expressionMap);
                        const exprNames = mgr.expressions.map(e => e.expressionName);
                        console.log('[VRM] expressionMap keys:', mapKeys);
                        console.log('[VRM] expression names:', exprNames);
                        // Test specific lookup
                        console.log('[VRM] getExpression("jawOpen"):', mgr.getExpression('jawOpen'));
                        console.log('[VRM] getExpression("Fcl_MTH_Open"):', mgr.getExpression('Fcl_MTH_Open'));
                    }
                    this._autoFrameVRM(vrm);
                    this.emaState = {};
                    resolve();
                },
                undefined,
                (err) => reject(err),
            );
        });
    }

    /**
     * Load a plain GLB/GLTF and index its morph targets.
     * @param {string} url
     * @returns {Promise<void>}
     */
    async loadGLB(url) {
        const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');

        this._removeCurrentModel();

        const loader = new GLTFLoader();

        return new Promise((resolve, reject) => {
            loader.load(
                url,
                (gltf) => {
                    const root = gltf.scene;
                    root.rotation.y = Math.PI;
                    this.scene.add(root);
                    this.glbScene = root;

                    // Index morph targets across all meshes
                    this.morphMeshes = [];
                    root.traverse((obj) => {
                        if (obj.isMesh && obj.morphTargetDictionary) {
                            this.morphMeshes.push({
                                mesh: obj,
                                nameToIndex: obj.morphTargetDictionary,
                            });
                        }
                    });

                    this._autoFrameScene(root);
                    this.emaState = {};
                    resolve();
                },
                undefined,
                (err) => reject(err),
            );
        });
    }

    /**
     * Remove whatever model is currently loaded.
     */
    _removeCurrentModel() {
        if (this.vrm) {
            this.scene.remove(this.vrm.scene);
            this.vrm = null;
        }
        if (this.glbScene) {
            this.scene.remove(this.glbScene);
            this.glbScene = null;
        }
        this.morphMeshes = [];
        this._glbHeadNode = null;
    }

    /**
     * Adjust camera to frame the head (VRM — uses humanoid bone).
     */
    _autoFrameVRM(vrm) {
        const head = vrm.humanoid?.getNormalizedBoneNode('head');
        if (!head) return;
        const pos = new THREE.Vector3();
        head.getWorldPosition(pos);
        this.camera.position.set(pos.x, pos.y + 0.02, pos.z + 1.0);
        this.camera.lookAt(pos.x, pos.y + 0.02, pos.z);
    }

    /**
     * Adjust camera to frame a generic GLTF scene (bounding box centre).
     */
    _autoFrameScene(root) {
        const box = new THREE.Box3().setFromObject(root);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const dist = maxDim * 1.5;
        // Focus on upper portion (head area)
        this.camera.position.set(center.x, center.y + size.y * 0.25, center.z + dist);
        this.camera.lookAt(center.x, center.y + size.y * 0.25, center.z);
    }

    /* ------------------------------------------------------------------ */
    /*  Blendshape application                                            */
    /* ------------------------------------------------------------------ */

    /**
     * Apply a dict of blendshape weights (ARKit names) to the VRM.
     * Values are EMA-smoothed before applying.
     *
     * @param {Object.<string,number>} weightsDict  e.g. { jawOpen: 0.4, ... }
     */
    applyBlendshapes(weightsDict) {
        if (!this.vrm && this.morphMeshes.length === 0) return;

        // Debug: log mouth blendshapes ~1/sec
        if (!this._bsLogCounter) this._bsLogCounter = 0;
        this._bsLogCounter++;
        const shouldLog = this._bsLogCounter % 60 === 1;
        const mouthLog = {};
        const rejected = [];

        for (const [name, raw] of Object.entries(weightsDict)) {
            if (name === '_neutral') continue;

            // EMA smoothing
            const prev = this.emaState[name] ?? raw;
            let alpha = this.emaAlpha;

            // Asymmetric blink: close fast (low alpha), open slow (high alpha)
            if (name === 'eyeBlinkLeft' || name === 'eyeBlinkRight') {
                alpha = raw > prev ? 0.3 : 0.85;
            }

            const smoothed = alpha * prev + (1 - alpha) * raw;
            this.emaState[name] = smoothed;

            if (this.vrm) {
                // VRM path: expressionManager
                const mgr = this.vrm.expressionManager;
                if (mgr) {
                    // Try original name, then UpperCamelCase fallback
                    let key = name;
                    if (!mgr.getExpression(key)) {
                        key = name.charAt(0).toUpperCase() + name.slice(1);
                    }
                    if (shouldLog && (name.startsWith('mouth') || name.startsWith('jaw'))) {
                        if (mgr.getExpression(key)) {
                            mouthLog[name] = +smoothed.toFixed(3);
                        } else {
                            rejected.push(name);
                        }
                    }
                    mgr.setValue(key, smoothed);
                }
            } else {
                // GLB path: morphTargetInfluences
                for (const { mesh, nameToIndex } of this.morphMeshes) {
                    const idx = nameToIndex[name];
                    if (idx !== undefined) {
                        mesh.morphTargetInfluences[idx] = smoothed;
                    }
                }
            }
        }

        if (shouldLog && this.vrm) {
            console.log('[Blendshape] applied:', mouthLog, '| rejected:', rejected);
        }
    }

    /**
     * Reset EMA state (e.g. on seek / stop).
     */
    resetSmoothing() {
        this.emaState = {};
        this._headPoseEma = { pitch: 0, yaw: 0, roll: 0 };
        this._pendingHeadPose = null;
    }

    /**
     * Apply head rotation (pitch, yaw, roll in radians) to the head bone.
     * Values are scaled by headPoseScale and EMA-smoothed.
     *
     * @param {number} pitch  rotation around X axis (radians)
     * @param {number} yaw    rotation around Y axis (radians)
     * @param {number} roll   rotation around Z axis (radians)
     */
    applyHeadPose(pitch, yaw, roll) {
        const s = this.headPoseScale;
        pitch *= s; yaw *= s; roll *= s;

        const a = this._headPoseEmaAlpha;
        this._headPoseEma.pitch = a * this._headPoseEma.pitch + (1 - a) * pitch;
        this._headPoseEma.yaw   = a * this._headPoseEma.yaw   + (1 - a) * yaw;
        this._headPoseEma.roll  = a * this._headPoseEma.roll  + (1 - a) * roll;

        // Defer actual application to _tick(), after vrm.update() which
        // would otherwise overwrite head bone rotation via lookAt system.
        this._pendingHeadPose = {
            pitch: this._headPoseEma.pitch,
            yaw:   this._headPoseEma.yaw,
            roll:  this._headPoseEma.roll,
        };
    }

    /**
     * Apply pending head pose to the head bone.
     * Called from _tick() AFTER vrm.update() so we write to the raw bone
     * that is actually rendered (normalized bone gets overwritten by
     * humanoid/expression/lookAt updates inside vrm.update()).
     */
    _applyPendingHeadPose() {
        if (!this.headPoseEnabled) return;
        const hp = this._pendingHeadPose;
        if (!hp) return;

        const q = new THREE.Quaternion().setFromEuler(
            new THREE.Euler(hp.pitch, hp.yaw, hp.roll, 'YXZ'),
        );

        if (this.vrm) {
            const head = this.vrm.humanoid?.getRawBoneNode('head');
            if (head) {
                head.quaternion.multiply(q);
            }
        } else if (this.glbScene) {
            if (!this._glbHeadNode) {
                this._glbHeadNode = this._findNodeByName(this.glbScene, 'head');
            }
            if (this._glbHeadNode) {
                this._glbHeadNode.quaternion.multiply(q);
            }
        }
    }

    /**
     * Case-insensitive recursive search for a node by name.
     * @param {THREE.Object3D} root
     * @param {string} name
     * @returns {THREE.Object3D|null}
     */
    _findNodeByName(root, name) {
        const lower = name.toLowerCase();
        let found = null;
        root.traverse((obj) => {
            if (!found && obj.name && obj.name.toLowerCase() === lower) {
                found = obj;
            }
        });
        return found;
    }

    /* ------------------------------------------------------------------ */
    /*  Render loop                                                       */
    /* ------------------------------------------------------------------ */

    start() {
        if (this._running) return;
        this._running = true;
        this._tick();
    }

    stop() {
        this._running = false;
        if (this._animId != null) {
            cancelAnimationFrame(this._animId);
            this._animId = null;
        }
    }

    _tick() {
        if (!this._running) return;
        this._animId = requestAnimationFrame(() => this._tick());

        const dt = this.clock.getDelta();

        if (this.vrm) {
            this.vrm.update(dt);
        }

        // Apply head pose AFTER vrm.update() on the RAW bone directly.
        // vrm.update() runs humanoid→expression→lookAt which can all
        // touch raw bones; writing to raw after guarantees our rotation sticks.
        this._applyPendingHeadPose();

        this.renderer.render(this.scene, this.camera);
    }

    /* ------------------------------------------------------------------ */
    /*  Cleanup                                                           */
    /* ------------------------------------------------------------------ */

    cleanup() {
        this.stop();
        this._removeCurrentModel();
        this.renderer.dispose();
    }
}

// Export for module-less usage
window.VRMRenderer = VRMRenderer;
