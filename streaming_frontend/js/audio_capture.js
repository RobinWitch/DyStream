/**
 * Audio capture manager for WebRTC mode.
 * No AudioWorklet needed — WebRTC handles audio encoding/transport natively.
 * This class only provides audio level metering from the local stream.
 */

class AudioCaptureManager {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.source = null;

        // Callbacks
        this.onAudioLevelCallback = null;

        // Level polling
        this._levelInterval = null;
    }

    /**
     * Initialize from an existing MediaStream (from getUserMedia via WebRTC).
     * @param {MediaStream} stream - The local audio stream.
     */
    initFromStream(stream) {
        this.audioContext = new AudioContext();
        this.source = this.audioContext.createMediaStreamSource(stream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.source.connect(this.analyser);
    }

    /**
     * Start polling audio level.
     */
    start() {
        if (this._levelInterval) return;

        this._levelInterval = setInterval(() => {
            const level = this.getAudioLevel();
            if (this.onAudioLevelCallback) {
                this.onAudioLevelCallback(level);
            }
        }, 50); // 20 Hz polling
    }

    /**
     * Stop polling.
     */
    stop() {
        if (this._levelInterval) {
            clearInterval(this._levelInterval);
            this._levelInterval = null;
        }
    }

    /**
     * Get current audio level (0-1).
     */
    getAudioLevel() {
        if (!this.analyser) return 0;

        const data = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteTimeDomainData(data);

        let sum = 0;
        for (let i = 0; i < data.length; i++) {
            const v = (data[i] - 128) / 128;
            sum += v * v;
        }

        return Math.min(1, Math.sqrt(sum / data.length) * 5);
    }

    /**
     * Cleanup all resources.
     */
    cleanup() {
        this.stop();

        if (this.source) {
            this.source.disconnect();
            this.source = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.analyser = null;
    }

    /**
     * Set audio level callback.
     * @param {function} callback - Called with level (0-1).
     */
    onAudioLevel(callback) {
        this.onAudioLevelCallback = callback;
    }
}
