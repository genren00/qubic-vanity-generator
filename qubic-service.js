// qubic-service.js
const { QubicHelper } = require('@qubic-lib/qubic-ts-library');

class QubicService {
    constructor() {
        this.helper = new QubicHelper();
        this.isProcessing = false;
    }

    async createIdPackage(seed) {
        try {
            const idPackage = await this.helper.createIdPackage(seed);
            return {
                publicId: idPackage.publicId,
                publicKeyB64: idPackage.publicKeyB64,
                privateKeyB64: idPackage.privateKeyB64,
                status: 'ok'
            };
        } catch (error) {
            return {
                status: 'error',
                error: error.message
            };
        }
    }

    start() {
        process.stdin.setEncoding('utf8');
        
        process.stdin.on('data', async (data) => {
            if (this.isProcessing) {
                return; // Skip if already processing
            }
            
            this.isProcessing = true;
            
            try {
                const input = data.trim();
                if (input === 'exit') {
                    process.exit(0);
                }
                
                if (input === 'ping') {
                    console.log(JSON.stringify({ status: 'ok', message: 'pong' }));
                } else {
                    const result = await this.createIdPackage(input);
                    console.log(JSON.stringify(result));
                }
            } catch (error) {
                console.log(JSON.stringify({ 
                    status: 'error', 
                    error: 'Internal service error' 
                }));
            } finally {
                this.isProcessing = false;
            }
        });

        // Signal that we're ready
        console.log(JSON.stringify({ status: 'ready', message: 'Qubic service started' }));
    }
}

// Start the service
const service = new QubicService();
service.start();
