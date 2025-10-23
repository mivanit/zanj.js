// Node.js test for inline array formats
// Run with: node test-inline-arrays.js

// Polyfill browser globals
global.atob = (str) => Buffer.from(str, 'base64').toString('binary');
global.btoa = (str) => Buffer.from(str, 'binary').toString('base64');

// Load the modules
eval(require('fs').readFileSync('src/array.js', 'utf8'));
eval(require('fs').readFileSync('src/zanj.js', 'utf8'));

let passCount = 0;
let failCount = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`✓ ${message}`);
    passCount++;
  } else {
    console.log(`✗ ${message}`);
    failCount++;
    throw new Error("Assertion failed: " + message);
  }
}

function arrayEqual(a, b, tolerance = 1e-6) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tolerance) return false;
  }
  return true;
}

console.log("\n=== Test 1: array_list_meta ===");
const listMeta = {
  "__muutils_format__": "numpy.ndarray:array_list_meta",
  "shape": [2, 3],
  "dtype": "float32",
  "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  "n_elements": 6
};
const arr1 = NDArray.fromJSON(listMeta);
assert(arr1.shape[0] === 2 && arr1.shape[1] === 3, "shape is [2, 3]");
assert(arr1.dtype === "float32", "dtype is float32");
assert(arrayEqual(arr1.data, [1, 2, 3, 4, 5, 6]), "data is correct");

console.log("\n=== Test 2: array_b64_meta ===");
// Base64 encoded [1.0, 2.0, 3.0] as float32
const b64Meta = {
  "__muutils_format__": "numpy.ndarray:array_b64_meta",
  "shape": [3],
  "dtype": "float32",
  "data": "AACAPwAAAEAAAEBA", // base64 of [1.0, 2.0, 3.0] in float32
  "n_elements": 3
};
const arr2 = NDArray.fromJSON(b64Meta);
assert(arr2.shape[0] === 3, "shape is [3]");
assert(arr2.dtype === "float32", "dtype is float32");
assert(arrayEqual(arr2.data, [1.0, 2.0, 3.0]), "data is correct");

console.log("\n=== Test 3: array_hex_meta ===");
// Hex encoded [1, 2, 3] as uint8
const hexMeta = {
  "__muutils_format__": "numpy.ndarray:array_hex_meta",
  "shape": [3],
  "dtype": "uint8",
  "data": "010203",
  "n_elements": 3
};
const arr3 = NDArray.fromJSON(hexMeta);
assert(arr3.shape[0] === 3, "shape is [3]");
assert(arr3.dtype === "uint8", "dtype is uint8");
assert(arrayEqual(arr3.data, [1, 2, 3]), "data is correct");

console.log("\n=== Test 4: zero_dim ===");
const zeroDim = {
  "__muutils_format__": "numpy.ndarray:zero_dim",
  "shape": [],
  "dtype": "float32",
  "data": 42.0,
  "n_elements": 1
};
const arr4 = NDArray.fromJSON(zeroDim);
assert(arr4.shape.length === 0, "shape is []");
assert(arr4.dtype === "float32", "dtype is float32");
assert(arr4.data[0] === 42.0, "data is 42.0");

console.log("\n=== Test 5: ZanjLoader integration ===");
const loader = new ZanjLoader("/fake/path");
const mockData = {
  "inline_array": {
    "__muutils_format__": "numpy.ndarray:array_list_meta",
    "shape": [2],
    "dtype": "int32",
    "data": [10, 20],
    "n_elements": 2
  },
  "nested": {
    "value": 123,
    "another_array": {
      "__muutils_format__": "numpy.ndarray:array_b64_meta",
      "shape": [2],
      "dtype": "uint8",
      "data": "AQI=", // base64 of [1, 2] as uint8
      "n_elements": 2
    }
  }
};
const processed = loader._makeLazy(mockData);
assert(processed.inline_array instanceof NDArray, "inline_array is NDArray");
assert(processed.inline_array.shape[0] === 2, "inline_array shape is [2]");
assert(arrayEqual(processed.inline_array.data, [10, 20]), "inline_array data is correct");
assert(processed.nested.value === 123, "nested value preserved");
assert(processed.nested.another_array instanceof NDArray, "nested array is NDArray");
assert(arrayEqual(processed.nested.another_array.data, [1, 2]), "nested array data is correct");

console.log("\n=== Test 6: inferFormat ===");
assert(NDArray.inferFormat(listMeta) === "array_list_meta", "infer array_list_meta");
assert(NDArray.inferFormat(b64Meta) === "array_b64_meta", "infer array_b64_meta");
assert(NDArray.inferFormat(hexMeta) === "array_hex_meta", "infer array_hex_meta");
assert(NDArray.inferFormat(zeroDim) === "zero_dim", "infer zero_dim");
assert(NDArray.inferFormat([1, 2, 3]) === "list", "infer list");
assert(NDArray.inferFormat({}) === null, "infer null for plain object");

console.log(`\n=== Results: ${passCount} passed, ${failCount} failed ===`);
if (failCount === 0) {
  console.log("✓ All tests passed!");
  process.exit(0);
} else {
  console.log("✗ Some tests failed");
  process.exit(1);
}
