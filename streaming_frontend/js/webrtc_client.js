/**
 * WebRTC client for DyStream streaming.
 * Handles audio sending (getUserMedia → Opus → server),
 * video receiving (server VP8 → <video> element),
 * and control messages via DataChannel.
 */

class WebRTCClient {
    constructor() {
        this.pc = null;
        this.controlChannel = null;
        this.localStream = null;
        this.sessionId = null;

        // Pending commands queued before DataChannel opens
        this._pendingMessages = [];

        // Callbacks
        this.onConnectedCallback = null;
        this.onDisconnectedCallback = null;
        this.onVideoTrackCallback = null;
        this.onStatusCallback = null;
        this.onErrorCallback = null;
    }

    /**
     * Establish WebRTC connection with the server.
     */
    async connect() {
        try {
            this.pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });

            // Create DataChannel for control messages (must be before offer)
            this.controlChannel = this.pc.createDataChannel('control');
            this.controlChannel.onopen = () => {
                console.log('DataChannel open');
                // Flush any commands queued before channel opened
                for (const msg of this._pendingMessages) {
                    this.controlChannel.send(JSON.stringify(msg));
                }
                this._pendingMessages = [];
            };
            this.controlChannel.onmessage = (e) => {
                try {
                    const msg = JSON.parse(e.data);
                    if (this.onStatusCallback) {
                        this.onStatusCallback(msg);
                    }
                } catch (err) {
                    console.error('Error parsing DataChannel message:', err);
                }
            };

            // Handle server → client video track
            this.pc.ontrack = (event) => {
                if (event.track.kind === 'video') {
                    console.log('Received video track from server');
                    if (this.onVideoTrackCallback) {
                        this.onVideoTrackCallback(event.streams[0]);
                    }
                }
            };

            // Connection state changes
            this.pc.onconnectionstatechange = () => {
                console.log('WebRTC connection state:', this.pc.connectionState);
                if (this.pc.connectionState === 'disconnected' ||
                    this.pc.connectionState === 'failed' ||
                    this.pc.connectionState === 'closed') {
                    if (this.onDisconnectedCallback) {
                        this.onDisconnectedCallback();
                    }
                }
            };

            // Get user audio (browser handles Opus encoding)
            this.localStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Add audio track to peer connection
            this.localStream.getAudioTracks().forEach(track => {
                this.pc.addTrack(track, this.localStream);
            });

            // Add recvonly video transceiver so the SDP offer includes a video m-line
            // for the server to send generated frames back
            this.pc.addTransceiver('video', { direction: 'recvonly' });

            // Create and send offer
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);

            // Wait for ICE gathering to complete
            await this._waitIceGathering();

            // Send offer to server, get answer
            const resp = await fetch('/api/webrtc/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: this.pc.localDescription.sdp,
                    type: this.pc.localDescription.type
                })
            });

            if (!resp.ok) {
                throw new Error(`Server returned ${resp.status}: ${resp.statusText}`);
            }

            const answer = await resp.json();
            this.sessionId = answer.session_id;

            await this.pc.setRemoteDescription(new RTCSessionDescription({
                sdp: answer.sdp,
                type: answer.type
            }));

            console.log('WebRTC connection established, session:', this.sessionId);

            if (this.onConnectedCallback) {
                this.onConnectedCallback({ session_id: this.sessionId });
            }

        } catch (error) {
            console.error('WebRTC connection failed:', error);
            if (this.onErrorCallback) {
                this.onErrorCallback('WebRTC connection failed: ' + error.message);
            }
        }
    }

    /**
     * Disconnect and release all resources.
     */
    disconnect() {
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
        if (this.localStream) {
            this.localStream.getTracks().forEach(t => t.stop());
            this.localStream = null;
        }
        this.controlChannel = null;
        this.sessionId = null;
        console.log('WebRTC disconnected');
    }

    /**
     * Wait for ICE gathering to complete (or timeout).
     */
    _waitIceGathering() {
        return new Promise((resolve) => {
            if (this.pc.iceGatheringState === 'complete') {
                resolve();
                return;
            }

            const timeout = setTimeout(() => {
                // Proceed even if not all candidates gathered
                resolve();
            }, 3000);

            this.pc.onicegatheringstatechange = () => {
                if (this.pc.iceGatheringState === 'complete') {
                    clearTimeout(timeout);
                    resolve();
                }
            };
        });
    }

    // ── Control messages via DataChannel ──

    sendControl(msg) {
        if (this.controlChannel && this.controlChannel.readyState === 'open') {
            this.controlChannel.send(JSON.stringify(msg));
        } else {
            // Queue for when DataChannel opens
            this._pendingMessages.push(msg);
        }
    }

    startStreaming() {
        this.sendControl({ type: 'start' });
    }

    stopStreaming() {
        this.sendControl({ type: 'stop' });
    }

    switchMode(mode) {
        this.sendControl({ type: 'mode_switch', mode: mode });
    }

    updateConfig(config) {
        this.sendControl({ type: 'config_update', config: config });
    }

    // ── Accessors ──

    getLocalStream() {
        return this.localStream;
    }

    /**
     * Get WebRTC stats for the inbound video track.
     * Returns { fps, bytesReceived, packetsLost, jitter } or null.
     */
    async getStats() {
        if (!this.pc) return null;

        try {
            const stats = await this.pc.getStats();
            let result = null;

            stats.forEach(report => {
                if (report.type === 'inbound-rtp' && report.kind === 'video') {
                    result = {
                        framesDecoded: report.framesDecoded || 0,
                        framesPerSecond: report.framesPerSecond || 0,
                        bytesReceived: report.bytesReceived || 0,
                        packetsLost: report.packetsLost || 0,
                        jitter: report.jitter || 0,
                    };
                }
            });

            return result;
        } catch {
            return null;
        }
    }

    // ── Callback setters ──

    onConnected(cb)    { this.onConnectedCallback = cb; }
    onDisconnected(cb) { this.onDisconnectedCallback = cb; }
    onVideoTrack(cb)   { this.onVideoTrackCallback = cb; }
    onStatus(cb)       { this.onStatusCallback = cb; }
    onError(cb)        { this.onErrorCallback = cb; }
}
