/**
 * WebSocket client for DyStream streaming.
 * Handles bidirectional communication with the server using JSON protocol.
 */

class WebSocketClient {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;

        // Callbacks
        this.onConnectedCallback = null;
        this.onDisconnectedCallback = null;
        this.onFrameCallback = null;
        this.onStatusCallback = null;
        this.onErrorCallback = null;
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log(`Connecting to WebSocket: ${wsUrl}`);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = this._handleOpen.bind(this);
        this.ws.onclose = this._handleClose.bind(this);
        this.ws.onerror = this._handleError.bind(this);
        this.ws.onmessage = this._handleMessage.bind(this);
    }

    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        if (this.ws) {
            console.log('Disconnecting WebSocket');
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
        this.sessionId = null;
    }

    /**
     * Send JSON control message
     */
    sendControl(message) {
        if (!this.isConnected) {
            console.error('Cannot send message - not connected');
            return;
        }

        try {
            this.ws.send(JSON.stringify(message));
        } catch (error) {
            console.error('Error sending control message:', error);
        }
    }

    /**
     * Send audio chunk as base64 JSON
     * @param {Float32Array} audioData - Audio samples
     */
    sendAudioChunk(audioData) {
        if (!this.isConnected) {
            return;
        }

        try {
            // Float32Array -> base64
            const bytes = new Uint8Array(audioData.buffer, audioData.byteOffset, audioData.byteLength);
            let binary = '';
            for (let i = 0; i < bytes.length; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64Data = btoa(binary);

            this.sendControl({ type: 'audio', data: base64Data });
        } catch (error) {
            console.error('Error sending audio chunk:', error);
        }
    }

    /**
     * Send reference image as base64 JSON
     * @param {ArrayBuffer} imageData - Image bytes (JPEG/PNG)
     */
    sendReferenceImage(imageData) {
        if (!this.isConnected) {
            console.error('Cannot send image - not connected');
            return;
        }

        try {
            const bytes = new Uint8Array(imageData);
            let binary = '';
            for (let i = 0; i < bytes.length; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            this.sendControl({ type: 'reference_image', data: btoa(binary) });
            console.log('Sent reference image');
        } catch (error) {
            console.error('Error sending reference image:', error);
        }
    }

    /**
     * Start streaming
     */
    startStreaming() {
        this.sendControl({ type: 'start' });
    }

    /**
     * Stop streaming
     */
    stopStreaming() {
        this.sendControl({ type: 'stop' });
    }

    /**
     * Switch mode
     * @param {string} mode - 'speaker' or 'listener'
     */
    switchMode(mode) {
        this.sendControl({ type: 'mode_switch', mode: mode });
    }

    /**
     * Update configuration
     * @param {object} config - Configuration parameters
     */
    updateConfig(config) {
        this.sendControl({ type: 'config_update', config: config });
    }

    /**
     * Handle WebSocket open
     */
    _handleOpen() {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
    }

    /**
     * Handle WebSocket close
     */
    _handleClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);
        this.isConnected = false;

        if (this.onDisconnectedCallback) {
            this.onDisconnectedCallback();
        }

        // Attempt reconnection
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
            if (this.onErrorCallback) {
                this.onErrorCallback('Connection lost - max reconnection attempts reached');
            }
        }
    }

    /**
     * Handle WebSocket error
     */
    _handleError(error) {
        console.error('WebSocket error:', error);

        if (this.onErrorCallback) {
            this.onErrorCallback('WebSocket connection error');
        }
    }

    /**
     * Handle WebSocket message (JSON-only)
     */
    _handleMessage(event) {
        if (typeof event.data === 'string') {
            try {
                const message = JSON.parse(event.data);
                this._handleTextMessage(message);
            } catch (error) {
                console.error('Error parsing JSON message:', error);
            }
        }
    }

    /**
     * Handle JSON text message
     */
    _handleTextMessage(message) {
        const msgType = message.type;

        switch (msgType) {
            case 'init':
                this.sessionId = message.session_id;
                console.log('Session initialized:', this.sessionId);
                console.log('Initial config:', message.config);

                if (this.onConnectedCallback) {
                    this.onConnectedCallback(message);
                }
                break;

            case 'frame':
                if (this.onFrameCallback) {
                    this.onFrameCallback({
                        type: 'jpeg_base64',
                        data: message.data,
                        timestamp: message.timestamp
                    });
                }
                break;

            case 'status':
                console.log('Status update:', message.status);

                if (this.onStatusCallback) {
                    this.onStatusCallback(message);
                }
                break;

            case 'error':
                console.error('Server error:', message.message);

                if (this.onErrorCallback) {
                    this.onErrorCallback(message.message);
                }
                break;

            default:
                console.log('Unknown message type:', msgType);
        }
    }

    /**
     * Set callbacks
     */
    onConnected(callback) {
        this.onConnectedCallback = callback;
    }

    onDisconnected(callback) {
        this.onDisconnectedCallback = callback;
    }

    onFrame(callback) {
        this.onFrameCallback = callback;
    }

    onStatus(callback) {
        this.onStatusCallback = callback;
    }

    onError(callback) {
        this.onErrorCallback = callback;
    }
}
