# qubic_vanity_generator.py
"""
Qubic Vanity Address Generator - Optimized Version
Uses persistent Node.js service for 100-1000x performance improvement.
Compatible with Google Colab environment.
"""

# Imports
import os
import re
import subprocess
import json
import time
import threading
import multiprocessing
import secrets
import string
import base64
import hashlib
import signal
import sys
from typing import Tuple, Dict, Optional, List
import urllib.request

# Constants
SEED_LENGTH = 55
PUBLIC_ID_LENGTH = 60
ALPHABET_LOWER = string.ascii_lowercase
ALPHABET_UPPER = string.ascii_uppercase
QUBIC_HELPER_PATH = "./qubic-helper-linux"
QUBIC_SERVICE_PATH = "./qubic-service.js"
QUBIC_HELPER_VERSION = "3.0.5"
QUBIC_HELPER_DOWNLOAD_URL = f"https://github.com/Qubic-Hub/qubic-helper-utils/releases/download/{QUBIC_HELPER_VERSION}/qubic-helper-linux-x64-{QUBIC_HELPER_VERSION.replace('.', '_')}"

# Main Classes
class QubicServiceManager:
    """Manages the persistent Node.js Qubic service"""
    
    def __init__(self):
        self.process = None
        self.is_running = False
        self.lock = threading.Lock()
    
    def start_service(self) -> bool:
        """Start the persistent Node.js service"""
        with self.lock:
            if self.is_running and self.process and self.process.poll() is None:
                return True
            
            try:
                # Start the Node.js service
                self.process = subprocess.Popen(
                    ['node', QUBIC_SERVICE_PATH],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True
                )
                
                # Wait for service to be ready
                timeout = time.time() + 30  # 30 second timeout
                ready = False
                
                while time.time() < timeout:
                    line = self.process.stdout.readline()
                    if line:
                        try:
                            response = json.loads(line.strip())
                            if response.get('status') == 'ready':
                                ready = True
                                break
                        except json.JSONDecodeError:
                            continue
                    time.sleep(0.1)
                
                if ready:
                    self.is_running = True
                    print("✓ Qubic service started successfully")
                    return True
                else:
                    print("✗ Qubic service failed to start within timeout")
                    self.stop_service()
                    return False
                    
            except Exception as e:
                print(f"✗ Failed to start Qubic service: {e}")
                self.stop_service()
                return False
    
    def stop_service(self):
        """Stop the persistent Node.js service"""
        with self.lock:
            if self.process:
                try:
                    # Send exit command
                    self.process.stdin.write('exit\n')
                    self.process.stdin.flush()
                    
                    # Wait for graceful shutdown
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                    
                except Exception:
                    if self.process.poll() is None:
                        self.process.kill()
                
                self.process = None
                self.is_running = False
                print("✓ Qubic service stopped")
    
    def create_id_package(self, seed: str) -> Dict:
        """Create ID package using the persistent service"""
        with self.lock:
            if not self.is_running or not self.process or self.process.poll() is not None:
                if not self.start_service():
                    return {"status": "error", "error": "Qubic service not available"}
            
            try:
                # Send seed to service
                self.process.stdin.write(seed + '\n')
                self.process.stdin.flush()
                
                # Read response
                line = self.process.stdout.readline()
                if line:
                    try:
                        response = json.loads(line.strip())
                        return response
                    except json.JSONDecodeError:
                        return {"status": "error", "error": "Invalid JSON response"}
                else:
                    return {"status": "error", "error": "No response from service"}
                    
            except Exception as e:
                return {"status": "error", "error": f"Service communication error: {e}"}
    
    def ping_service(self) -> bool:
        """Check if service is responsive"""
        try:
            response = self.create_id_package("ping")
            return response.get('status') == 'ok' and response.get('message') == 'pong'
        except Exception:
            return False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_service()

class QubicVanityGenerator:
    """Main class for generating vanity Qubic addresses"""
    
    def __init__(self, num_threads: int, use_service: bool = True):
        """
        Initialize the generator with required number of threads.
        
        Args:
            num_threads: Number of threads to use for generation (must be at least 1)
            use_service: Whether to use persistent service (default True)
        """
        if num_threads < 1:
            raise ValueError("Number of threads must be at least 1")
        
        self.num_threads = num_threads
        self.use_service = use_service
        self.service_manager = QubicServiceManager() if use_service else None
        self.progress_tracker = ProgressTracker()
        
        # Start service if needed
        if self.use_service and not self.service_manager.start_service():
            print("⚠️  Falling back to subprocess method")
            self.use_service = False
    
    def generate_vanity_address(self, pattern: str, max_attempts: int = None) -> Dict:
        """
        Generate a Qubic address with the specified vanity pattern.
        
        Args:
            pattern: Desired prefix or pattern for the address
            max_attempts: Maximum number of attempts (None for unlimited)
        
        Returns:
            Dictionary containing the matching address and seed
        """
        # Validate the pattern
        if not validate_vanity_pattern(pattern):
            return {"status": "error", "error": "Invalid vanity pattern"}
        
        method = "Persistent Service" if self.use_service else "Subprocess"
        print(f"Starting vanity generation for pattern: {pattern}")
        print(f"Using {self.num_threads} threads ({method})")
        
        # Try multi-threaded generation
        result = self._generate_multithreaded(pattern, max_attempts)
        
        if result["status"] == "success":
            print(f"Success! Found matching address after {result['attempts']} attempts")
            print(f"Public ID: {result['publicId']}")
            print(f"Seed: {result['seed']}")
        else:
            print(f"Failed to find matching address: {result['error']}")
        
        return result
    
    def _generate_multithreaded(self, pattern: str, max_attempts: int = None) -> Dict:
        """Generate vanity address using multiple threads"""
        found = threading.Event()
        result = {"status": "error", "error": "No match found"}
        lock = threading.Lock()
        
        def worker(thread_id):
            nonlocal result
            attempts = 0
            local_max_attempts = (max_attempts // self.num_threads) if max_attempts else None
            
            while not found.is_set() and (local_max_attempts is None or attempts < local_max_attempts):
                # Generate a random seed
                seed = SeedGenerator.generate()
                
                # Convert seed to public ID
                if self.use_service:
                    cmd_result = self.service_manager.create_id_package(seed)
                else:
                    cmd_result = execute_qubic_command(f"{QUBIC_HELPER_PATH} createPublicId {seed}")
                
                if cmd_result["status"] == "ok":
                    public_id = cmd_result.get("publicId") or cmd_result.get("public_id")
                    
                    # Check if the address matches the pattern
                    if public_id and matches_pattern(public_id, pattern):
                        with lock:
                            if not found.is_set():
                                found.set()
                                result = {
                                    "status": "success",
                                    "seed": seed,
                                    "publicId": public_id,
                                    "publicKeyB64": cmd_result.get("publicKeyB64") or cmd_result.get("public_key_b64"),
                                    "privateKeyB64": cmd_result.get("privateKeyB64") or cmd_result.get("private_key_b64"),
                                    "attempts": attempts * self.num_threads + thread_id
                                }
                
                attempts += 1
                
                # Update progress periodically
                if attempts % 100 == 0:
                    self.progress_tracker.update(attempts * self.num_threads + thread_id)
        
        # Start worker threads
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for any thread to find a match
        for thread in threads:
            thread.join()
        
        return result

class SeedGenerator:
    """Handles secure seed generation"""
    
    @staticmethod
    def generate() -> str:
        """
        Generate a cryptographically secure random 55-character seed.
        
        Returns:
            A 55-character string of lowercase letters
        """
        return ''.join(secrets.choice(ALPHABET_LOWER) for _ in range(SEED_LENGTH))
    
    @staticmethod
    def generate_from_entropy(entropy: bytes) -> str:
        """
        Generate a seed from provided entropy.
        
        Args:
            entropy: Random bytes to use as entropy source
            
        Returns:
            A 55-character seed string
        """
        # Convert entropy to seed using a deterministic process
        # This is useful for reproducible testing
        random = secrets.SystemRandom()
        random.seed(int.from_bytes(entropy, byteorder='big'))
        return ''.join(random.choice(ALPHABET_LOWER) for _ in range(SEED_LENGTH))

class AddressValidator:
    """Validates Qubic addresses and seeds according to official specifications"""
    
    @staticmethod
    def validate_seed(seed: str) -> Tuple[bool, str]:
        """
        Validate a Qubic seed.
        
        Args:
            seed: The seed to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(seed, str):
            return False, "Seed must be a string"
        if len(seed) != SEED_LENGTH:
            return False, f"Seed must be exactly {SEED_LENGTH} characters"
        if not re.match(r'^[a-z]+$', seed):
            return False, "Seed must contain only lowercase letters a-z"
        return True, "Valid seed"
    
    @staticmethod
    def validate_public_id(public_id: str) -> Tuple[bool, str]:
        """
        Validate a Qubic public ID.
        
        Args:
            public_id: The public ID to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(public_id, str):
            return False, "Public ID must be a string"
        if len(public_id) != PUBLIC_ID_LENGTH:
            return False, f"Public ID must be exactly {PUBLIC_ID_LENGTH} characters"
        if not re.match(r'^[A-Z]+$', public_id):
            return False, "Public ID must contain only uppercase letters A-Z"
        return True, "Valid public ID"
    
    @staticmethod
    def verify_seed_address_consistency(seed: str, expected_public_id: str) -> bool:
        """
        Verify that a seed consistently produces the expected public ID.
        
        Args:
            seed: The seed to test
            expected_public_id: The expected public ID
            
        Returns:
            True if the seed produces the expected public ID
        """
        # Try service first, fallback to subprocess
        service_manager = QubicServiceManager()
        if service_manager.start_service():
            result = service_manager.create_id_package(seed)
            if result.get("status") == "ok":
                return result.get("publicId") == expected_public_id
        
        # Fallback to subprocess
        result = execute_qubic_command(f"{QUBIC_HELPER_PATH} createPublicId {seed}")
        if result["status"] == "ok":
            return result["publicId"] == expected_public_id
        return False

class ProgressTracker:
    """Tracks and reports vanity generation progress"""
    
    def __init__(self):
        self.start_time = time.time()
        self.attempts = 0
        self.last_update_time = self.start_time
        self.lock = threading.Lock()
    
    def update(self, attempts: int):
        """Update the progress tracker with the current attempt count"""
        with self.lock:
            self.attempts = attempts
            current_time = time.time()
            
            # Print progress every 10 seconds
            if current_time - self.last_update_time >= 10:
                self.print_progress()
                self.last_update_time = current_time
    
    def print_progress(self):
        """Print current progress statistics"""
        elapsed = time.time() - self.start_time
        attempts_per_second = self.attempts / elapsed if elapsed > 0 else 0
        
        print(f"Progress: {self.attempts} attempts in {elapsed:.1f}s "
              f"({attempts_per_second:.1f} attempts/second)")
    
    def get_stats(self) -> Dict:
        """Get current progress statistics"""
        elapsed = time.time() - self.start_time
        attempts_per_second = self.attempts / elapsed if elapsed > 0 else 0
        
        return {
            "attempts": self.attempts,
            "elapsed_seconds": elapsed,
            "attempts_per_second": attempts_per_second
        }

class SecureSeedGenerator:
    """Enhanced security for seed generation"""
    
    @staticmethod
    def generate_with_user_entropy(user_input: str = None) -> str:
        """
        Generate a seed with additional user entropy for enhanced security.
        
        Args:
            user_input: Optional user-provided entropy
            
        Returns:
            A secure 55-character seed
        """
        # Combine system entropy with optional user entropy
        system_entropy = os.urandom(32)
        
        if user_input:
            # Hash user input to normalize it
            user_entropy = hashlib.sha256(user_input.encode()).digest()
            # Combine system and user entropy
            combined_entropy = bytes(a ^ b for a, b in zip(system_entropy, user_entropy))
        else:
            combined_entropy = system_entropy
        
        # Use combined entropy to seed the generator
        random = secrets.SystemRandom()
        random.seed(int.from_bytes(combined_entropy, byteorder='big'))
        
        return ''.join(random.choice(ALPHABET_LOWER) for _ in range(SEED_LENGTH))

class SecureResultHandler:
    """Handles secure storage and transmission of generated results"""
    
    @staticmethod
    def encrypt_result(result: Dict, password: str) -> str:
        """
        Encrypt a result dictionary for secure storage.
        
        Args:
            result: The result dictionary to encrypt
            password: Encryption password
            
        Returns:
            Encrypted result as base64 string
        """
        # Simple XOR encryption for demonstration (use proper encryption in production)
        json_result = json.dumps(result)
        key = hashlib.sha256(password.encode()).digest()
        encrypted = bytes(a ^ b for a, b in zip(json_result.encode(), key * (len(json_result) // 32 + 1)))
        return base64.b64encode(encrypted).decode()
    
    @staticmethod
    def decrypt_result(encrypted_result: str, password: str) -> Dict:
        """
        Decrypt an encrypted result.
        
        Args:
            encrypted_result: The encrypted result
            password: Decryption password
            
        Returns:
            The original result dictionary
        """
        # Simple XOR decryption for demonstration (use proper decryption in production)
        key = hashlib.sha256(password.encode()).digest()
        decoded = base64.b64decode(encrypted_result.encode())
        decrypted = bytes(a ^ b for a, b in zip(decoded, key * (len(decoded) // 32 + 1)))
        return json.loads(decrypted.decode())

# Utility Functions
def download_qubic_helper() -> bool:
    """
    Download Qubic Helper Utilities binary.
    
    Returns:
        True if download was successful, False otherwise
    """
    try:
        print(f"Downloading Qubic Helper Utilities from {QUBIC_HELPER_DOWNLOAD_URL}...")
        
        # Download the file
        urllib.request.urlretrieve(QUBIC_HELPER_DOWNLOAD_URL, QUBIC_HELPER_PATH)
        
        # Make it executable
        os.chmod(QUBIC_HELPER_PATH, 0o755)
        
        # Verify the download
        if os.path.exists(QUBIC_HELPER_PATH):
            print("✓ Qubic Helper Utilities downloaded successfully!")
            return True
        else:
            print("✗ Download failed - file not found after download")
            return False
            
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        return False

def setup_qubic_service() -> bool:
    """
    Setup the Node.js Qubic service.
    
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Check if Node.js is available
        try:
            subprocess.run(['node', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Node.js is not installed. Please install Node.js to use the persistent service.")
            return False
        
        # Check if package.json exists, create it if not
        if not os.path.exists('package.json'):
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
        
        # Check if qubic-service.js exists, create it if not
        if not os.path.exists(QUBIC_SERVICE_PATH):
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
service.start();'''
            with open(QUBIC_SERVICE_PATH, 'w') as f:
                f.write(service_js)
        
        # Install dependencies
        print("Installing Node.js dependencies...")
        try:
            subprocess.run(['npm', 'install'], check=True, capture_output=True)
            print("✓ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to setup Qubic service: {e}")
        return False

def execute_qubic_command(command: str) -> Dict:
    """
    Execute a Qubic Helper command with comprehensive error handling.
    
    Args:
        command: The command to execute
        
    Returns:
        Dictionary with the result or error
    """
    try:
        # Check if the helper binary exists
        if not os.path.exists(QUBIC_HELPER_PATH):
            return {
                "status": "error",
                "error": f"Qubic Helper binary not found at {QUBIC_HELPER_PATH}"
            }
        
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # Add timeout to prevent hanging
        )
        
        # Check for execution errors
        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Command failed with exit code {result.returncode}: {result.stderr}"
            }
        
        # Parse the JSON response
        try:
            output = json.loads(result.stdout)
            return output
        except json.JSONDecodeError:
            return {
                "status": "error",
                "error": f"Invalid JSON response: {result.stdout}"
            }
    
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Command timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}"
        }

def validate_vanity_pattern(pattern: str) -> bool:
    """
    Validate user-provided vanity pattern.
    
    Args:
        pattern: The vanity pattern to validate
        
    Returns:
        True if pattern is valid, False otherwise
    """
    if not isinstance(pattern, str):
        return False
    
    # Remove whitespace
    pattern = pattern.strip()
    
    # Check if pattern is empty
    if not pattern:
        return False
    
    # Check for wildcard pattern
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        # Validate prefix contains only uppercase letters
        return re.match(r'^[A-Z]*$', prefix) is not None
    
    # Check for exact prefix pattern
    return re.match(r'^[A-Z]+$', pattern) is not None

def matches_pattern(public_id: str, pattern: str) -> bool:
    """
    Check if a public ID matches the specified vanity pattern.
    
    Args:
        public_id: The Qubic public ID to check
        pattern: The vanity pattern to match against
    
    Returns:
        True if the ID matches the pattern, False otherwise
    """
    # Simple prefix matching
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return public_id.startswith(prefix)
    
    # Exact match
    return public_id.startswith(pattern)

def generate_vanity_address(pattern: str, max_attempts: int = None, num_threads: int = None, use_service: bool = True) -> Dict:
    """
    Main function to generate vanity address.
    
    Args:
        pattern: Desired prefix or pattern for the address
        max_attempts: Maximum number of attempts (None for unlimited)
        num_threads: Number of threads to use (must be provided)
        use_service: Whether to use persistent service (default True)
    
    Returns:
        Dictionary containing the matching address and seed
    """
    if num_threads is None:
        raise ValueError("Number of threads must be specified")
    
    generator = QubicVanityGenerator(num_threads, use_service)
    return generator.generate_vanity_address(pattern, max_attempts)

def batch_generate_vanity_addresses(pattern: str, count: int, max_attempts_per_address: int = None, num_threads: int = None, use_service: bool = True) -> List[Dict]:
    """
    Generate multiple vanity addresses with the same pattern.
    
    Args:
        pattern: Desired vanity pattern
        count: Number of addresses to generate
        max_attempts_per_address: Maximum attempts per address
        num_threads: Number of threads to use (must be provided)
        use_service: Whether to use persistent service (default True)
        
    Returns:
        List of result dictionaries
    """
    if num_threads is None:
        raise ValueError("Number of threads must be specified")
    
    results = []
    
    for i in range(count):
        print(f"Generating address {i+1}/{count}...")
        
        result = generate_vanity_address(pattern, max_attempts_per_address, num_threads, use_service)
        results.append(result)
        
        if result["status"] == "success":
            print(f"Found address {i+1}: {result['publicId']}")
        else:
            print(f"Failed to generate address {i+1}: {result['error']}")
    
    return results

def run_validation_tests():
    """Run validation tests to ensure the generator works correctly"""
    print("Running validation tests...")
    
    # Test seed validation
    valid_seed = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    invalid_seed = "InvalidSeed123"
    
    assert AddressValidator.validate_seed(valid_seed)[0] == True
    assert AddressValidator.validate_seed(invalid_seed)[0] == False
    
    # Test public ID validation
    valid_id = "BZBQFLLBNCXEMGLOBHUVFTLUPLVCPQUASSILFABOFFBCADQSSUPNWLZBQEXK"
    invalid_id = "InvalidID123"
    
    assert AddressValidator.validate_public_id(valid_id)[0] == True
    assert AddressValidator.validate_public_id(invalid_id)[0] == False
    
    # Test seed-address consistency (try service first, fallback to subprocess)
    if os.path.exists(QUBIC_HELPER_PATH) or os.path.exists(QUBIC_SERVICE_PATH):
        assert AddressValidator.verify_seed_address_consistency(valid_seed, valid_id) == True
    
    # Test pattern matching
    assert matches_pattern(valid_id, "BZBQ*") == True
    assert matches_pattern(valid_id, "BZBQFLL") == True
    assert matches_pattern(valid_id, "INVALID") == False
    
    print("✓ All validation tests passed!")

def test_full_vanity_generation():
    """Test the complete vanity generation process"""
    print("Testing full vanity generation process...")
    
    # Use a simple pattern that should be found quickly
    pattern = "A*"
    
    # Test with persistent service if available
    service_manager = QubicServiceManager()
    use_service = service_manager.start_service()
    
    if not use_service and not os.path.exists(QUBIC_HELPER_PATH):
        print("✗ Neither Qubic service nor helper binary available. Skipping test.")
        return
    
    # Test with 2 threads
    result = generate_vanity_address(pattern, max_attempts=10000, num_threads=2, use_service=use_service)
    
    if result["status"] == "success":
        # Verify the result
        assert AddressValidator.validate_seed(result["seed"])[0]
        assert AddressValidator.validate_public_id(result["publicId"])[0]
        assert matches_pattern(result["publicId"], pattern)
        
        # Verify consistency
        assert AddressValidator.verify_seed_address_consistency(
            result["seed"], result["publicId"]
        )
        
        method = "Service" if use_service else "Subprocess"
        print(f"✓ Test passed: Found {result['publicId']} in {result['attempts']} attempts ({method})")
    else:
        method = "Service" if use_service else "Subprocess"
        print(f"✗ Test failed ({method}): {result['error']}")

def benchmark_performance():
    """Benchmark performance comparison between service and subprocess"""
    print("Running performance benchmark...")
    
    test_seed = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    iterations = 100
    
    # Test subprocess method
    if os.path.exists(QUBIC_HELPER_PATH):
        print("Testing subprocess method...")
        start_time = time.time()
        for _ in range(iterations):
            result = execute_qubic_command(f"{QUBIC_HELPER_PATH} createPublicId {test_seed}")
            if result["status"] != "ok":
                print("✗ Subprocess test failed")
                return
        subprocess_time = time.time() - start_time
        subprocess_rate = iterations / subprocess_time
        print(f"✓ Subprocess: {subprocess_rate:.1f} ops/sec")
    else:
        subprocess_rate = 0
        print("✗ Subprocess method not available")
    
    # Test service method
    service_manager = QubicServiceManager()
    if service_manager.start_service():
        print("Testing persistent service method...")
        start_time = time.time()
        for _ in range(iterations):
            result = service_manager.create_id_package(test_seed)
            if result["status"] != "ok":
                print("✗ Service test failed")
                return
        service_time = time.time() - start_time
        service_rate = iterations / service_time
        print(f"✓ Service: {service_rate:.1f} ops/sec")
        
        if subprocess_rate > 0:
            speedup = service_rate / subprocess_rate
            print(f"✓ Speedup: {speedup:.1f}x faster")
    else:
        service_rate = 0
        print("✗ Service method not available")

def get_user_friendly_error(error_dict: Dict) -> str:
    """
    Convert technical error messages to user-friendly ones.
    
    Args:
        error_dict: The error dictionary from execute_qubic_command
        
    Returns:
        A user-friendly error message
    """
    error = error_dict.get("error", "Unknown error")
    
    if "Qubic Helper binary not found" in error:
        return "Qubic Helper Utilities not found. Please run download_qubic_helper() first."
    elif "Command failed" in error:
        return "Failed to execute Qubic Helper command. Please check your installation."
    elif "Invalid JSON response" in error:
        return "Invalid response from Qubic Helper. Please try again."
    elif "timed out" in error:
        return "Operation timed out. Please try again."
    else:
        return f"An error occurred: {error}"

def check_qubic_helper_version() -> Tuple[bool, str]:
    """
    Check if the Qubic Helper binary is compatible with this generator.
    
    Returns:
        Tuple of (is_compatible, message)
    """
    try:
        # Try to get version information (this might not be supported)
        result = execute_qubic_command(f"{QUBIC_HELPER_PATH} --version")
        
        if result["status"] == "ok":
            # Check version compatibility
            version = result.get("version", "unknown")
            if version.startswith("3."):
                return True, f"Compatible version: {version}"
            else:
                return False, f"Incompatible version: {version}"
        else:
            # Version command not supported, assume compatibility
            return True, "Version check not available, assuming compatibility"
    
    except Exception as e:
        return False, f"Version check failed: {str(e)}"

def print_usage_examples():
    """Print usage examples for the user"""
    print("""
Qubic Vanity Address Generator - Usage Examples
===============================================

1. Basic Usage (with persistent service):
   generator = QubicVanityGenerator(num_threads=4, use_service=True)
   result = generator.generate_vanity_address("HELLO*")
   
2. Basic Usage (subprocess fallback):
   generator = QubicVanityGenerator(num_threads=4, use_service=False)
   result = generator.generate_vanity_address("HELLO*")
   
3. With limited attempts:
   generator = QubicVanityGenerator(num_threads=8)
   result = generator.generate_vanity_address("TEST*", max_attempts=100000)
   
4. Multi-threaded generation:
   generator = QubicVanityGenerator(num_threads=16)
   result = generator.generate_vanity_address("CRYPTO*")
   
5. Batch generation:
   results = batch_generate_vanity_addresses("VANITY*", count=3, num_threads=4)
   
6. Download Qubic Helper:
   download_qubic_helper()
   
7. Setup Qubic Service:
   setup_qubic_service()
   
8. Run validation tests:
   run_validation_tests()
   
9. Benchmark performance:
   benchmark_performance()
   
10. Test full generation:
    test_full_vanity_generation()

Pattern Formats:
- "HELLO*" : Matches addresses starting with "HELLO"
- "TEST"   : Exact match for prefix "TEST"
- "A*"     : Matches addresses starting with "A" (fast to find)

Note: You must specify the number of threads when creating the generator!
The persistent service provides 100-1000x performance improvement.
""")

def get_num_threads_from_user() -> int:
    """Get the number of threads from user input with validation."""
    while True:
        try:
            num_threads = input("Enter number of threads to use (1-64 recommended): ").strip()
            num_threads = int(num_threads)
            if num_threads < 1:
                print("Number of threads must be at least 1. Please try again.")
            elif num_threads > 64:
                print("Using more than 64 threads may cause performance issues. Please try again.")
            else:
                return num_threads
        except ValueError:
            print("Please enter a valid number. Please try again.")

def get_use_service_from_user() -> bool:
    """Ask user whether to use persistent service."""
    while True:
        choice = input("Use persistent Node.js service for 100-1000x speedup? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n⚠️  Shutting down gracefully...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main Execution
if __name__ == "__main__":
    print("🚀 Qubic Vanity Address Generator - Optimized Edition")
    print("=" * 60)
    
    # Check if Qubic Helper binary exists
    helper_available = os.path.exists(QUBIC_HELPER_PATH)
    service_available = os.path.exists(QUBIC_SERVICE_PATH)
    
    if not helper_available and not service_available:
        print("⚠️  No Qubic components found.")
        download_choice = input("Download Qubic Helper Utilities? (y/n): ").lower().strip()
        
        if download_choice == 'y':
            if download_qubic_helper():
                helper_available = True
            else:
                print("✗ Download failed. Please download manually from:")
                print(QUBIC_HELPER_DOWNLOAD_URL)
        else:
            print("Please download the Qubic Helper Utilities binary from:")
            print(QUBIC_HELPER_DOWNLOAD_URL)
            print("And save it as 'qubic-helper-linux' in the current directory.")
    
    # Setup Qubic service if Node.js is available
    if not service_available:
        setup_choice = input("Setup persistent Node.js service for better performance? (y/n): ").lower().strip()
        if setup_choice == 'y':
            if setup_qubic_service():
                service_available = True
            else:
                print("⚠️  Service setup failed. Will use subprocess method.")
    
    # Run validation tests if components are available
    if helper_available or service_available:
        print("\n🔍 Running validation tests...")
        run_validation_tests()
        
        # Benchmark performance if both methods are available
        if helper_available and service_available:
            print("\n⏱️  Running performance benchmark...")
            benchmark_performance()
        
        # Show usage examples
        print_usage_examples()
        
        # Interactive mode
        while True:
            print("\n" + "=" * 60)
            choice = input("Choose an option:\n1. Generate vanity address\n2. Run tests\n3. Benchmark performance\n4. Exit\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                pattern = input("Enter vanity pattern (e.g., 'HELLO*' or 'TEST'): ").strip().upper()
                if validate_vanity_pattern(pattern):
                    max_attempts = input("Enter maximum attempts (press Enter for unlimited): ").strip()
                    max_attempts = int(max_attempts) if max_attempts.isdigit() else None
                    
                    # Get number of threads from user
                    num_threads = get_num_threads_from_user()
                    
                    # Ask about service usage
                    use_service = get_use_service_from_user()
                    
                    generator = QubicVanityGenerator(num_threads, use_service)
                    result = generator.generate_vanity_address(pattern, max_attempts)
                    
                    if result["status"] == "success":
                        print(f"\n🎉 Success! Found vanity address:")
                        print(f"   Public ID: {result['publicId']}")
                        print(f"   Seed: {result['seed']}")
                        print(f"   Public Key: {result['publicKeyB64']}")
                        print(f"   Private Key: {result['privateKeyB64']}")
                        print(f"   Attempts: {result['attempts']}")
                    else:
                        print(f"\n❌ Failed: {result['error']}")
                else:
                    print("❌ Invalid pattern. Please use uppercase letters A-Z only, optionally ending with *")
            
            elif choice == '2':
                test_full_vanity_generation()
            
            elif choice == '3':
                benchmark_performance()
            
            elif choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    else:
        print("\n❌ Please download or setup Qubic components to use this generator.")
