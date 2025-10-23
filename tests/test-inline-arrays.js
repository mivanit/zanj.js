// test-inline-arrays.js
// Tests for inline array format deserialization with explicit JSON construction

const { describe, it } = require('node:test');
const { assert, assertArrayEqual, NDArray, ZanjLoader } = require('./test-helpers.js');

describe('NDArray.fromJSON() - Inline Array Formats', () => {

	it('deserializes array_list_meta format', () => {
		const listMeta = {
			"__muutils_format__": "numpy.ndarray:array_list_meta",
			"shape": [2, 3],
			"dtype": "float32",
			"data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
			"n_elements": 6
		};
		const arr = NDArray.fromJSON(listMeta);

		assert.strictEqual(arr.shape[0], 2);
		assert.strictEqual(arr.shape[1], 3);
		assert.strictEqual(arr.dtype, "float32");
		assertArrayEqual(arr.data, [1, 2, 3, 4, 5, 6]);
	});

	it('deserializes array_b64_meta format', () => {
		const b64Meta = {
			"__muutils_format__": "numpy.ndarray:array_b64_meta",
			"shape": [3],
			"dtype": "float32",
			"data": "AACAPwAAAEAAAEBA", // [1.0, 2.0, 3.0]
			"n_elements": 3
		};
		const arr = NDArray.fromJSON(b64Meta);

		assert.strictEqual(arr.shape[0], 3);
		assert.strictEqual(arr.dtype, "float32");
		assertArrayEqual(arr.data, [1.0, 2.0, 3.0]);
	});

	it('deserializes array_hex_meta format', () => {
		const hexMeta = {
			"__muutils_format__": "numpy.ndarray:array_hex_meta",
			"shape": [3],
			"dtype": "uint8",
			"data": "010203",
			"n_elements": 3
		};
		const arr = NDArray.fromJSON(hexMeta);

		assert.strictEqual(arr.shape[0], 3);
		assert.strictEqual(arr.dtype, "uint8");
		assertArrayEqual(arr.data, [1, 2, 3]);
	});

	it('deserializes zero_dim format', () => {
		const zeroDim = {
			"__muutils_format__": "numpy.ndarray:zero_dim",
			"shape": [],
			"dtype": "float32",
			"data": 42.0,
			"n_elements": 1
		};
		const arr = NDArray.fromJSON(zeroDim);

		assert.strictEqual(arr.shape.length, 0);
		assert.strictEqual(arr.dtype, "float32");
		assert.strictEqual(arr.data[0], 42.0);
	});

	it('handles different dtypes correctly', () => {
		// int16
		const int16Array = {
			"__muutils_format__": "numpy.ndarray:array_b64_meta",
			"shape": [2, 3],
			"dtype": "int16",
			"data": "AQACAAMABAAFAAYA", // [[1,2,3], [4,5,6]]
			"n_elements": 6
		};
		const arr16 = NDArray.fromJSON(int16Array);
		assert.strictEqual(arr16.dtype, "int16");
		assertArrayEqual(arr16.data, [1, 2, 3, 4, 5, 6]);

		// int64 (BigInt)
		const int64Array = {
			"__muutils_format__": "numpy.ndarray:array_list_meta",
			"shape": [2],
			"dtype": "int64",
			"data": [100, 200],
			"n_elements": 2
		};
		const arr64 = NDArray.fromJSON(int64Array);
		assert.strictEqual(arr64.dtype, "int64");
		assert.strictEqual(arr64.data[0], 100n);
		assert.strictEqual(arr64.data[1], 200n);
	});
});

describe('NDArray.inferFormat()', () => {

	it('infers array_list_meta', () => {
		const obj = {
			"__muutils_format__": "numpy.ndarray:array_list_meta",
			"data": [[1, 2], [3, 4]]
		};
		assert.strictEqual(NDArray.inferFormat(obj), "array_list_meta");
	});

	it('infers array_b64_meta', () => {
		const obj = {
			"__muutils_format__": "torch.Tensor:array_b64_meta",
			"data": "ABC="
		};
		assert.strictEqual(NDArray.inferFormat(obj), "array_b64_meta");
	});

	it('infers array_hex_meta', () => {
		const obj = {
			"__muutils_format__": "numpy.ndarray:array_hex_meta",
			"data": "deadbeef"
		};
		assert.strictEqual(NDArray.inferFormat(obj), "array_hex_meta");
	});

	it('infers zero_dim', () => {
		const obj = {
			"__muutils_format__": "numpy.ndarray:zero_dim",
			"data": 3.14
		};
		assert.strictEqual(NDArray.inferFormat(obj), "zero_dim");
	});

	it('infers list for plain arrays', () => {
		assert.strictEqual(NDArray.inferFormat([1, 2, 3]), "list");
	});

	it('returns null for plain objects', () => {
		assert.strictEqual(NDArray.inferFormat({}), null);
		assert.strictEqual(NDArray.inferFormat({ foo: "bar" }), null);
	});
});

describe('ZanjLoader integration', () => {

	it('deserializes inline arrays in nested structures', () => {
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
					"data": "AQI=", // [1, 2]
					"n_elements": 2
				}
			}
		};

		const processed = loader._makeLazy(mockData);

		// Check inline_array
		assert.ok(processed.inline_array instanceof NDArray);
		assert.strictEqual(processed.inline_array.shape[0], 2);
		assertArrayEqual(processed.inline_array.data, [10, 20]);

		// Check nested structure
		assert.strictEqual(processed.nested.value, 123);
		assert.ok(processed.nested.another_array instanceof NDArray);
		assertArrayEqual(processed.nested.another_array.data, [1, 2]);
	});

	it('preserves non-array objects', () => {
		const loader = new ZanjLoader("/fake/path");
		const mockData = {
			"plain_object": { foo: "bar", num: 42 },
			"plain_array": [1, 2, 3],
			"string": "hello"
		};

		const processed = loader._makeLazy(mockData);

		assert.strictEqual(processed.plain_object.foo, "bar");
		assert.strictEqual(processed.plain_object.num, 42);
		assert.deepStrictEqual(processed.plain_array, [1, 2, 3]);
		assert.strictEqual(processed.string, "hello");
	});
});
