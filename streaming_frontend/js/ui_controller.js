/**
 * UI Controller — orchestrates WebRTC client, audio metering, and video display.
 */

class UIController {
    constructor() {
        // Components
        this.rtcClient = new WebRTCClient();
        this.audioCapture = new AudioCaptureManager();
        this.videoRenderer = new VideoRenderer('videoElement');

        // State
        this.currentMode = 'speaker';
        this.isStreaming = false;

        // UI elements
        this.elements = {
            // Status
            statusText: document.getElementById('statusText'),
            connectionStatus: document.getElementById('connectionStatus'),

            // Mode buttons
            speakerModeBtn: document.getElementById('speakerModeBtn'),
            listenerModeBtn: document.getElementById('listenerModeBtn'),
            modeDescription: document.getElementById('modeDescription'),

            // Avatar selection
            avatarSelect: document.getElementById('avatarSelect'),

            // Control buttons
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),

            // Stats
            fpsValue: document.getElementById('fpsValue'),
            latencyValue: document.getElementById('latencyValue'),
            framesValue: document.getElementById('framesValue'),

            // Audio level
            audioLevelFill: document.getElementById('audioLevelFill'),

            // Settings
            denoisingSteps: document.getElementById('denoisingSteps'),
            denoisingValue: document.getElementById('denoisingValue'),
            cfgAudio: document.getElementById('cfgAudio'),
            cfgAudioValue: document.getElementById('cfgAudioValue'),
            cfgAudioOther: document.getElementById('cfgAudioOther'),
            cfgAudioOtherValue: document.getElementById('cfgAudioOtherValue'),
            cfgAll: document.getElementById('cfgAll'),
            cfgAllValue: document.getElementById('cfgAllValue'),
            applySettingsBtn: document.getElementById('applySettingsBtn'),
        };

        this._setupEventListeners();
        this._setupRTCCallbacks();
        this._startStatsUpdate();
    }

    // ── Event Listeners ──

    _setupEventListeners() {
        // Mode buttons
        this.elements.speakerModeBtn.addEventListener('click', () => this._switchMode('speaker'));
        this.elements.listenerModeBtn.addEventListener('click', () => this._switchMode('listener'));

        // Control buttons
        this.elements.startBtn.addEventListener('click', () => this._startStreaming());
        this.elements.stopBtn.addEventListener('click', () => this._stopStreaming());

        // Settings sliders
        this.elements.denoisingSteps.addEventListener('input', (e) => {
            this.elements.denoisingValue.textContent = e.target.value;
        });
        this.elements.cfgAudio.addEventListener('input', (e) => {
            this.elements.cfgAudioValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        this.elements.cfgAudioOther.addEventListener('input', (e) => {
            this.elements.cfgAudioOtherValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        this.elements.cfgAll.addEventListener('input', (e) => {
            this.elements.cfgAllValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        this.elements.applySettingsBtn.addEventListener('click', () => this._applySettings());
    }

    // ── WebRTC Callbacks ──

    _setupRTCCallbacks() {
        this.rtcClient.onConnected((info) => {
            console.log('WebRTC connected, session:', info.session_id);
            this._updateStatus('Connected', 'connected');
            this.elements.startBtn.disabled = false;

            // Initialize audio level metering from the local stream
            const localStream = this.rtcClient.getLocalStream();
            if (localStream) {
                this.audioCapture.initFromStream(localStream);
                this.audioCapture.onAudioLevel((level) => {
                    this.elements.audioLevelFill.style.width = `${level * 100}%`;
                });
                this.audioCapture.start();
            }
        });

        this.rtcClient.onDisconnected(() => {
            console.log('WebRTC disconnected');
            this._updateStatus('Disconnected', 'disconnected');
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = true;

            if (this.isStreaming) {
                this._stopStreaming();
            }
        });

        this.rtcClient.onVideoTrack((stream) => {
            console.log('Received remote video stream');
            this.videoRenderer.setStream(stream);
            this.videoRenderer.start();
        });

        this.rtcClient.onStatus((msg) => {
            console.log('Status from server:', msg);
            this._handleStatusUpdate(msg);
        });

        this.rtcClient.onError((error) => {
            console.error('WebRTC error:', error);
            this._showError(error);
        });
    }

    // ── Actions ──

    async _startStreaming() {
        if (this.isStreaming) return;

        console.log('Starting streaming...');
        this._updateStatus('Connecting...', 'connecting');

        try {
            // Connect WebRTC (gets microphone, negotiates SDP)
            await this.rtcClient.connect();

            // Tell server to start inference
            this.rtcClient.startStreaming();

            this.isStreaming = true;
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this._updateStatus('Streaming', 'streaming');

            console.log('Streaming started');

        } catch (error) {
            console.error('Failed to start streaming:', error);
            this._showError('Failed to start streaming: ' + error.message);
        }
    }

    _stopStreaming() {
        if (!this.isStreaming) return;

        console.log('Stopping streaming...');

        this.rtcClient.stopStreaming();
        this.rtcClient.disconnect();
        this.videoRenderer.stop();
        this.audioCapture.cleanup();

        this.isStreaming = false;
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this._updateStatus('Disconnected', 'disconnected');

        console.log('Streaming stopped');
    }

    _switchMode(mode) {
        if (mode === this.currentMode) return;

        this.currentMode = mode;

        if (mode === 'speaker') {
            this.elements.speakerModeBtn.classList.add('active');
            this.elements.listenerModeBtn.classList.remove('active');
            this.elements.modeDescription.textContent =
                'AI behaves as speaker (your voice drives speaking motion)';
        } else {
            this.elements.speakerModeBtn.classList.remove('active');
            this.elements.listenerModeBtn.classList.add('active');
            this.elements.modeDescription.textContent =
                'AI behaves as listener (your voice elicits reactive listening motion)';
        }

        this.rtcClient.switchMode(mode);
        console.log('Switched to mode:', mode);
    }

    _applySettings() {
        const config = {
            denoising_steps: parseInt(this.elements.denoisingSteps.value),
            cfg_audio: parseFloat(this.elements.cfgAudio.value),
            cfg_audio_other: parseFloat(this.elements.cfgAudioOther.value),
            cfg_all: parseFloat(this.elements.cfgAll.value),
        };

        console.log('Applying settings:', config);
        this.rtcClient.updateConfig(config);

        this._updateStatus('Settings applied', 'success');
        setTimeout(() => {
            if (this.isStreaming) {
                this._updateStatus('Streaming', 'streaming');
            } else {
                this._updateStatus('Connected', 'connected');
            }
        }, 2000);
    }

    // ── Status handling ──

    _handleStatusUpdate(message) {
        const status = message.status;

        if (status === 'start_ok') {
            this._updateStatus('Streaming', 'streaming');
        } else if (status === 'stop_ok') {
            this._updateStatus('Connected', 'connected');
        } else if (status === 'mode_switch_ok') {
            console.log('Mode switch acknowledged');
        } else if (status === 'config_update_ok') {
            console.log('Config update acknowledged');
        }
    }

    _updateStatus(text, className = '') {
        this.elements.statusText.textContent = text;
        this.elements.connectionStatus.textContent = text;
        this.elements.connectionStatus.className = className;
    }

    _showError(message) {
        alert('Error: ' + message);
    }

    // ── Stats update loop ──

    _startStatsUpdate() {
        setInterval(async () => {
            if (!this.isStreaming) return;

            // Video renderer stats (FPS via requestVideoFrameCallback)
            const renderStats = this.videoRenderer.getStats();
            this.elements.fpsValue.textContent = renderStats.fps;
            this.elements.framesValue.textContent = renderStats.framesRendered;

            // WebRTC stats (jitter as proxy for latency)
            const rtcStats = await this.rtcClient.getStats();
            if (rtcStats) {
                const jitterMs = Math.round((rtcStats.jitter || 0) * 1000);
                this.elements.latencyValue.textContent = jitterMs + 'ms';
            }
        }, 1000);
    }
}
