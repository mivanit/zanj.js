// test-helpers.js
// Shared testing utilities

const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

// Polyfill browser globals for Node.js
global.atob = (str) => Buffer.from(str, 'base64').toString('binary');
global.btoa = (str) => Buffer.from(str, 'binary').toString('base64');

/**
 * Fetch polyfill for file:// URLs and local paths
 * Returns a Response-like object compatible with the browser fetch API
 */
function fetchPolyfill(url, options) {
	// Handle file:// URLs and local paths
	let filePath = url;

	// Remove file:// prefix if present
	if (filePath.startsWith('file://')) {
		filePath = filePath.slice(7);
	}

	// Check if this is a local file path
	if (!filePath.startsWith('http://') && !filePath.startsWith('https://')) {
		try {
			const content = fs.readFileSync(filePath);

			// Create a Response-like object
			return Promise.resolve({
				ok: true,
				status: 200,
				statusText: 'OK',
				headers: new Map(),

				async arrayBuffer() {
					return content.buffer.slice(
						content.byteOffset,
						content.byteOffset + content.byteLength
					);
				},

				async text() {
					return content.toString('utf8');
				},

				async json() {
					return JSON.parse(content.toString('utf8'));
				},

				async blob() {
					throw new Error('blob() not implemented in fetch polyfill');
				},
			});
		} catch (err) {
			// Return a failed response
			return Promise.resolve({
				ok: false,
				status: 404,
				statusText: 'Not Found',
				headers: new Map(),
				error: err,
			});
		}
	}

	// Fall back to native fetch for HTTP(S) URLs
	if (global.fetch) {
		return global.fetch(url, options);
	}

	throw new Error('HTTP fetch not available and URL is not a file path');
}

// Load source files using vm to create a shared context
const srcPath = path.join(__dirname, '..');
const arrayCode = fs.readFileSync(path.join(srcPath, 'src/array.js'), 'utf8');
const zanjCode = fs.readFileSync(path.join(srcPath, 'src/zanj.js'), 'utf8');

// Create sandbox with Node.js globals
const sandbox = {
	atob: global.atob,
	btoa: global.btoa,
	console,
	Buffer,
	fetch: fetchPolyfill,  // Use our polyfill that supports file paths
	TextDecoder,
	Uint8Array,
	Uint16Array,
	Uint32Array,
	Int8Array,
	Int16Array,
	Int32Array,
	Float32Array,
	Float64Array,
	BigInt64Array,
	BigUint64Array,
	DataView,
	Proxy,
	Reflect,
	Array,
	Object,
	String,
	Number,
	Boolean,
	Math,
	Error,
	TypeError,
	ReferenceError,
	Map,
};

// Execute in vm context and export classes
vm.createContext(sandbox);
vm.runInContext(arrayCode + '\nthis.NDArray = NDArray;', sandbox);
vm.runInContext(zanjCode + '\nthis.ZanjLoader = ZanjLoader;', sandbox);

// Verify and extract classes
if (!sandbox.NDArray) throw new Error('NDArray not found in sandbox');
if (!sandbox.ZanjLoader) throw new Error('ZanjLoader not found in sandbox');

const { NDArray, ZanjLoader } = sandbox;

/**
 * Check if two arrays are equal within tolerance
 */
function arrayEqual(a, b, tolerance = 1e-6) {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) {
		if (Math.abs(a[i] - b[i]) > tolerance) return false;
	}
	return true;
}

/**
 * Assert arrays are equal
 */
function assertArrayEqual(a, b, message, tolerance = 1e-6) {
	assert.ok(arrayEqual(a, b, tolerance), message || `Arrays not equal: ${a} vs ${b}`);
}

/**
 * Load a JSON file
 */
function loadJSON(filePath) {
	return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

module.exports = {
	assert,
	arrayEqual,
	assertArrayEqual,
	loadJSON,
	NDArray,
	ZanjLoader,
};
