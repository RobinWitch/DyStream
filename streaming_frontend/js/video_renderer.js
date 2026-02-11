/**
 * Video renderer using a <video> element with WebRTC MediaStream.
 * Replaces the canvas-based JPEG renderer.
 */

class VideoRenderer {
    constructor(videoElementId) {
        this.video = document.getElementById(videoElementId);

        // Stats
        this.framesRendered = 0;
        this.fps = 0;
        this._frameCount = 0;
        this._lastTime = 0;
        this._rvfcActive = false;
    }

    /**
     * Attach a WebRTC MediaStream to the video element.
     * @param {MediaStream} stream - Remote video stream.
     */
    setStream(stream) {
        this.video.srcObject = stream;
        // Ensure playback starts (autoplay may be blocked by browser policy)
        this.video.play().catch(err => {
            console.warn('Video autoplay blocked, will play on user interaction:', err.message);
        });
    }

    /**
     * Start FPS measurement.
     * Uses requestVideoFrameCallback (Chrome 83+) for accurate frame counting.
     */
    start() {
        this.framesRendered = 0;
        this.fps = 0;
        this._frameCount = 0;
        this._lastTime = performance.now();

        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            this._rvfcActive = true;
            this._rvfcLoop();
        } else {
            // Fallback: use getVideoPlaybackQuality() API (if available) or timeupdate
            this._lastFallbackFrames = 0;
            this._fallbackInterval = setInterval(() => {
                this._updateFpsFallback();
            }, 1000);
        }
    }

    /**
     * Stop rendering / measurement.
     */
    stop() {
        this._rvfcActive = false;
        this.video.srcObject = null;

        if (this._fallbackInterval) {
            clearInterval(this._fallbackInterval);
            this._fallbackInterval = null;
        }
    }

    /**
     * requestVideoFrameCallback loop for accurate frame counting.
     */
    _rvfcLoop() {
        if (!this._rvfcActive) return;

        this.video.requestVideoFrameCallback((now, metadata) => {
            this.framesRendered++;
            this._frameCount++;

            if (now - this._lastTime >= 1000) {
                this.fps = this._frameCount;
                this._frameCount = 0;
                this._lastTime = now;
            }

            this._rvfcLoop();
        });
    }

    /**
     * Fallback FPS estimation using getVideoPlaybackQuality().
     */
    _updateFpsFallback() {
        if (this.video.getVideoPlaybackQuality) {
            const quality = this.video.getVideoPlaybackQuality();
            const totalFrames = quality.totalVideoFrames;
            this.fps = totalFrames - this._lastFallbackFrames;
            this.framesRendered = totalFrames;
            this._lastFallbackFrames = totalFrames;
        }
    }

    /**
     * Get current stats.
     */
    getStats() {
        return {
            framesRendered: this.framesRendered,
            fps: this.fps,
        };
    }

    /**
     * Cleanup.
     */
    cleanup() {
        this.stop();
    }
}
