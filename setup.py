#!/usr/bin/env python3
"""
Setup script for Qubic Vanity Address Generator
"""

import subprocess
import sys
import os

def check_nodejs():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js not found")
    print("Please install Node.js from https://nodejs.org/")
    return False

def check_npm():
    """Check if npm is installed"""
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ npm found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ npm not found")
    return False

def install_dependencies():
    """Install Node.js dependencies"""
    print("📦 Installing Node.js dependencies...")
    try:
        result = subprocess.run(['npm', 'install'], check=True, capture_output=True, text=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_service_file():
    """Create the Node.js service file"""
    service_js = '''const { QubicHelper } = require('@qubic-lib/qubic-ts-library');

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
                return;
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

        console.log(JSON.stringify({ status: 'ready', message: 'Qubic service started' }));
    }
}

const service = new QubicService();
service.start();'''
    
    with open('qubic-service.js', 'w') as f:
        f.write(service_js)
    
    print("✓ Service file created")

def create_package_json():
    """Create package.json file"""
    package_json = '''{
  "name": "qubic-service",
  "version": "1.0.0",
  "description": "Persistent Qubic service for fast address generation",
  "main": "qubic-service.js",
  "dependencies": {
    "@qubic-lib/qubic-ts-library": "^1.0.0"
  }
}'''
    
    with open('package.json', 'w') as f:
        f.write(package_json)
    
    print("✓ package.json created")

def main():
    print("🚀 Setting up Qubic Vanity Address Generator")
    print("=" * 50)
    
    # Check prerequisites
    if not check_nodejs():
        sys.exit(1)
    
    if not check_npm():
        sys.exit(1)
    
    # Create necessary files
    create_package_json()
    create_service_file()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("You can now run the generator with:")
    print("python qubic_vanity_generator.py")

if __name__ == "__main__":
    main()
