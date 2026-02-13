/**
 * BlendshapePlayer — JSON + WebAudio synchronised playback engine.
 *
 * Reads the FaceMesh JSON format produced by streaming_offline:
 *   { fps, num_frames, blendshape_names, frames: [{ t, blendshapes }] }
 *
 * Uses audioContext.currentTime as master clock.
 * Binary-search + linear interpolation for smooth frame lookup.
 */

class BlendshapePlayer {
    /**
     * @param {VRMRenderer} vrmRenderer
     */
    constructor(vrmRenderer) {
        /** @type {VRMRenderer} */
        this.vrmRenderer = vrmRenderer;

        // JSON data
        this.names = [];       // blendshape name list
        this.frames = [];      // [{ t, blendshapes }]
        this.duration = 0;     // seconds

        // Audio state
        /** @type {AudioContext|null} */
        this.audioCtx = null;
        /** @type {AudioBuffer|null} */
        this.audioBuffer = null;
        /** @type {AudioBufferSourceNode|null} */
        this.sourceNode = null;

        // Playback state
        this.playing = false;
        this.paused = false;
        this._t0 = 0;         // audioCtx.currentTime at play()
        this._pauseOffset = 0; // accumulated time when paused
        this._animId = null;

        // Callbacks
        this.onTimeUpdate = null;  // (currentTime, duration) => void
        this.onEnded = null;       // () => void
        this.onStateChange = null; // (state: string) => void
    }

    /* ------------------------------------------------------------------ */
    /*  Data loading                                                      */
    /* ------------------------------------------------------------------ */

    /**
     * Parse FaceMesh JSON data.
     * @param {object} jsonData  parsed JSON object
     */
    loadJSON(jsonData) {
        this.names = jsonData.blendshape_names || [];
        this.frames = jsonData.frames || [];
        if (this.frames.length > 0) {
            this.duration = this.frames[this.frames.length - 1].t;
        }
    }

    /**
     * @returns {boolean} true if JSON has been loaded
     */
    get hasJSON() {
        return this.frames.length > 0;
    }

    /**
     * Decode an audio file (wav, mp3, ogg).
     * @param {File} audioFile
     * @returns {Promise<void>}
     */
    async loadAudio(audioFile) {
        if (!this.audioCtx) {
            this.audioCtx = new AudioContext();
        }
        const arrayBuf = await audioFile.arrayBuffer();
        this.audioBuffer = await this.audioCtx.decodeAudioData(arrayBuf);
        // Use audio length as authoritative duration
        this.duration = this.audioBuffer.duration;
    }

    /**
     * @returns {boolean} true if audio has been loaded
     */
    get hasAudio() {
        return this.audioBuffer != null;
    }

    /* ------------------------------------------------------------------ */
    /*  Transport controls                                                */
    /* ------------------------------------------------------------------ */

    play() {
        if (this.playing && !this.paused) return;

        if (!this.audioCtx) {
            this.audioCtx = new AudioContext();
        }

        // Resume AudioContext if suspended (browser autoplay policy)
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }

        if (this.paused) {
            // Resume from pause offset
            this._startAudio(this._pauseOffset);
            this.paused = false;
        } else {
            // Fresh start
            this._pauseOffset = 0;
            this._startAudio(0);
        }

        this.playing = true;
        this._emitState('playing');
        this._scheduleFrame();
    }

    pause() {
        if (!this.playing || this.paused) return;
        this._pauseOffset = this._currentTime();
        this._stopAudio();
        this.paused = true;
        this._cancelFrame();
        this._emitState('paused');
    }

    stop() {
        this._stopAudio();
        this.playing = false;
        this.paused = false;
        this._pauseOffset = 0;
        this._cancelFrame();
        this.vrmRenderer.resetSmoothing();
        this._emitState('stopped');
        if (this.onTimeUpdate) this.onTimeUpdate(0, this.duration);
    }

    /**
     * Seek to a given time (seconds). Works while playing or paused.
     * @param {number} time
     */
    seek(time) {
        time = Math.max(0, Math.min(time, this.duration));

        if (this.playing && !this.paused) {
            // Restart audio from new position
            this._stopAudio();
            this._startAudio(time);
        } else {
            this._pauseOffset = time;
        }
        this.vrmRenderer.resetSmoothing();
        if (this.onTimeUpdate) this.onTimeUpdate(time, this.duration);
    }

    /* ------------------------------------------------------------------ */
    /*  Audio helpers                                                     */
    /* ------------------------------------------------------------------ */

    _startAudio(offset) {
        if (!this.audioBuffer || !this.audioCtx) {
            // No audio — use wall-clock fallback
            this._t0 = performance.now() / 1000 - offset;
            return;
        }

        this.sourceNode = this.audioCtx.createBufferSource();
        this.sourceNode.buffer = this.audioBuffer;
        this.sourceNode.connect(this.audioCtx.destination);
        this.sourceNode.onended = () => {
            if (this.playing && !this.paused) {
                this.stop();
                if (this.onEnded) this.onEnded();
            }
        };
        this.sourceNode.start(0, offset);
        this._t0 = this.audioCtx.currentTime - offset;
    }

    _stopAudio() {
        if (this.sourceNode) {
            try { this.sourceNode.stop(); } catch (_) { /* already stopped */ }
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }
    }

    /**
     * Current playback time in seconds.
     */
    _currentTime() {
        if (this.audioCtx && this.audioBuffer) {
            return this.audioCtx.currentTime - this._t0;
        }
        // Fallback: wall-clock
        return performance.now() / 1000 - this._t0;
    }

    /* ------------------------------------------------------------------ */
    /*  Frame loop                                                        */
    /* ------------------------------------------------------------------ */

    _scheduleFrame() {
        this._animId = requestAnimationFrame(() => this._onFrame());
    }

    _cancelFrame() {
        if (this._animId != null) {
            cancelAnimationFrame(this._animId);
            this._animId = null;
        }
    }

    _onFrame() {
        if (!this.playing || this.paused) return;

        const t = this._currentTime();

        if (t >= this.duration) {
            this.stop();
            if (this.onEnded) this.onEnded();
            return;
        }

        // Apply blendshapes + head pose if JSON loaded
        if (this.frames.length > 0) {
            const { blendshapes, headPose } = this._interpolate(t);
            this.vrmRenderer.applyBlendshapes(blendshapes);
            if (headPose) {
                this.vrmRenderer.applyHeadPose(headPose.pitch, headPose.yaw, headPose.roll);
            }
        }

        if (this.onTimeUpdate) this.onTimeUpdate(t, this.duration);

        this._scheduleFrame();
    }

    /* ------------------------------------------------------------------ */
    /*  Binary search + linear interpolation                              */
    /* ------------------------------------------------------------------ */

    /**
     * Find the index k such that frames[k].t <= t < frames[k+1].t.
     * Uses bisect-right then steps back.
     * @param {number} t
     * @returns {number}
     */
    _findFrame(t) {
        let lo = 0;
        let hi = this.frames.length;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (this.frames[mid].t <= t) lo = mid + 1;
            else hi = mid;
        }
        return Math.max(0, lo - 1);
    }

    /**
     * Linearly interpolate blendshapes and head_pose at time t.
     * @param {number} t
     * @returns {{ blendshapes: Object.<string,number>, headPose: {pitch:number,yaw:number,roll:number}|null }}
     */
    _interpolate(t) {
        const k = this._findFrame(t);
        const f0 = this.frames[k];

        // Past last frame — return last
        if (k + 1 >= this.frames.length) {
            return { blendshapes: f0.blendshapes, headPose: f0.head_pose || null };
        }

        const f1 = this.frames[k + 1];
        const dt = f1.t - f0.t;
        if (dt <= 0) {
            return { blendshapes: f0.blendshapes, headPose: f0.head_pose || null };
        }

        const alpha = (t - f0.t) / dt;

        // Interpolate blendshapes
        const blendshapes = {};
        for (const name of this.names) {
            const v0 = f0.blendshapes[name] ?? 0;
            const v1 = f1.blendshapes[name] ?? 0;
            blendshapes[name] = v0 * (1 - alpha) + v1 * alpha;
        }

        // Interpolate head_pose
        let headPose = null;
        const hp0 = f0.head_pose;
        const hp1 = f1.head_pose;
        if (hp0 && hp1) {
            headPose = {
                pitch: hp0.pitch * (1 - alpha) + hp1.pitch * alpha,
                yaw:   hp0.yaw   * (1 - alpha) + hp1.yaw   * alpha,
                roll:  hp0.roll  * (1 - alpha) + hp1.roll  * alpha,
            };
        } else if (hp0) {
            headPose = hp0;
        }

        return { blendshapes, headPose };
    }

    /* ------------------------------------------------------------------ */
    /*  State helpers                                                     */
    /* ------------------------------------------------------------------ */

    _emitState(state) {
        if (this.onStateChange) this.onStateChange(state);
    }
}

window.BlendshapePlayer = BlendshapePlayer;
