/**
 * AudioWorklet processor for low-latency audio capture.
 * Accumulates samples and sends chunks to main thread.
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        // Get chunk size from options
        this.chunkSize = options.processorOptions?.chunkSize || 640;

        // Buffer to accumulate samples
        this.buffer = [];

        // Audio level tracking (RMS)
        this.levelSmoothingFactor = 0.8;
        this.currentLevel = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];

        // If no input, return
        if (!input || input.length === 0) {
            return true;
        }

        // Get first channel
        const inputChannel = input[0];

        // Accumulate samples
        for (let i = 0; i < inputChannel.length; i++) {
            this.buffer.push(inputChannel[i]);

            // If buffer reaches chunk size, send it
            if (this.buffer.length >= this.chunkSize) {
                // Extract chunk
                const chunk = this.buffer.slice(0, this.chunkSize);
                this.buffer = this.buffer.slice(this.chunkSize);

                // Calculate audio level (RMS)
                const level = this._calculateRMS(chunk);
                this.currentLevel = this.levelSmoothingFactor * this.currentLevel +
                                   (1 - this.levelSmoothingFactor) * level;

                // Send chunk to main thread
                this.port.postMessage({
                    type: 'audio-chunk',
                    chunk: chunk
                });

                // Send audio level
                this.port.postMessage({
                    type: 'audio-level',
                    level: this.currentLevel
                });
            }
        }

        return true; // Keep processor alive
    }

    /**
     * Calculate RMS (Root Mean Square) for audio level
     */
    _calculateRMS(samples) {
        let sum = 0;
        for (let i = 0; i < samples.length; i++) {
            sum += samples[i] * samples[i];
        }
        const rms = Math.sqrt(sum / samples.length);

        // Normalize to 0-1 range (assuming input is -1 to 1)
        return Math.min(rms * 5, 1.0); // Multiply by 5 for better visibility
    }
}

registerProcessor('audio-processor', AudioProcessor);
