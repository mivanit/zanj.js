// test-python-generated.js
// Tests loading Python-generated ZANJ files

const { describe, it, before } = require('node:test');
const path = require('path');
const { assert, assertArrayEqual, loadJSON, ZanjLoader } = require('./test-helpers.js');

const TEMP_DIR = path.join(__dirname, '.temp');

describe('Python-generated ZANJ files', () => {

	describe('basic.zanj', () => {
		let data;

		before(async () => {
			const loader = new ZanjLoader(path.join(TEMP_DIR, 'basic'));
			data = await loader.read();
		});

		it('loads small_float array', () => {
			assert.ok(data.small_float);
			assert.strictEqual(data.small_float.dtype, 'float32');
			assertArrayEqual(data.small_float.data, [1.0, 2.0, 3.0]);
		});

		it('loads small_int array', () => {
			assert.ok(data.small_int);
			assert.strictEqual(data.small_int.dtype, 'int32');
			assertArrayEqual(data.small_int.data, [10, 20, 30]);
		});

		it('loads scalar array', () => {
			assert.ok(data.scalar);
			assert.strictEqual(data.scalar.dtype, 'float64');
			assert.strictEqual(data.scalar.data[0], 42.0);
		});
	});

	describe('all-formats.zanj', () => {
		let data;

		before(async () => {
			const loader = new ZanjLoader(path.join(TEMP_DIR, 'all-formats'));
			data = await loader.read();
		});

		it('loads list_meta format', () => {
			assert.ok(data.list_meta);
			assert.deepStrictEqual(data.list_meta.shape, [2, 2]);
			assertArrayEqual(data.list_meta.data, [1, 2, 3, 4]);
		});

		it('loads b64_meta format', () => {
			assert.ok(data.b64_meta);
			assertArrayEqual(data.b64_meta.data, [5.0, 6.0, 7.0, 8.0]);
		});

		it('loads hex_meta format', () => {
			assert.ok(data.hex_meta);
			assertArrayEqual(data.hex_meta.data, [0xDE, 0xAD, 0xBE, 0xEF]);
		});

		it('loads zero_dim format', () => {
			assert.ok(data.zero_dim);
			assert.strictEqual(data.zero_dim.shape.length, 0);
			assert.strictEqual(data.zero_dim.data[0], 3.14159);
		});
	});

	describe('mixed.zanj', () => {
		let data;

		before(async () => {
			const loader = new ZanjLoader(path.join(TEMP_DIR, 'mixed'));
			data = await loader.read();
		});

		it('loads inline small array', () => {
			assert.ok(data.inline_small);
			assertArrayEqual(data.inline_small.data, [1, 2, 3, 4, 5]);
		});

		it('loads external large array', async () => {
			assert.ok(data.external_big);
			const arr = await data.external_big;
			assert.strictEqual(arr.shape[0], 100);
			assert.strictEqual(arr.shape[1], 32);
			assert.strictEqual(arr.dtype, 'float32');
		});

		it('loads nested inline array', () => {
			assert.ok(data.nested.inline_nested);
			assertArrayEqual(data.nested.inline_nested.data, [0.1, 0.2, 0.3]);
		});

		it('preserves metadata', () => {
			assert.strictEqual(data.nested.metadata.name, 'test');
			assert.strictEqual(data.nested.metadata.version, 1);
		});
	});

	describe('edge-cases.zanj', () => {
		let data;

		before(async () => {
			const loader = new ZanjLoader(path.join(TEMP_DIR, 'edge-cases'));
			data = await loader.read();
		});

		it('handles empty 1D array', () => {
			assert.ok(data.empty_1d);
			assert.strictEqual(data.empty_1d.shape[0], 0);
			assert.strictEqual(data.empty_1d.data.length, 0);
		});

		it('handles single element array', () => {
			assert.ok(data.single_element);
			assert.strictEqual(data.single_element.data[0], 999);
		});

		it('handles high rank array', () => {
			assert.ok(data.high_rank);
			assert.strictEqual(data.high_rank.ndim, 4);
			assert.deepStrictEqual(data.high_rank.shape, [2, 2, 2, 2]);
		});

		it('handles uint64 large values', () => {
			assert.ok(data.uint64_max);
			assert.strictEqual(data.uint64_max.dtype, 'uint64');
			// BigInt comparison
			assert.strictEqual(data.uint64_max.data[0], 9223372036854775807n);
		});

		it('handles negative integers', () => {
			assert.ok(data.negative_int);
			assertArrayEqual(data.negative_int.data, [-1, -2, -3]);
		});
	});

	describe('dtypes.zanj', () => {
		let data;

		before(async () => {
			const loader = new ZanjLoader(path.join(TEMP_DIR, 'dtypes'));
			data = await loader.read();
		});

		const dtypeTests = [
			{ name: 'uint8', expected: [1, 2, 3] },
			{ name: 'uint16', expected: [100, 200, 300] },
			{ name: 'uint32', expected: [1000, 2000, 3000] },
			{ name: 'int8', expected: [-1, 0, 1] },
			{ name: 'int16', expected: [-100, 0, 100] },
			{ name: 'int32', expected: [-1000, 0, 1000] },
			{ name: 'float32', expected: [1.5, 2.5, 3.5] },
			{ name: 'float64', expected: [1.123456789, 2.123456789] },
		];

		for (const { name, expected } of dtypeTests) {
			it(`handles ${name} dtype`, () => {
				assert.ok(data[name], `Missing ${name}`);
				assert.strictEqual(data[name].dtype, name);
				assertArrayEqual(data[name].data, expected);
			});
		}

		// Special handling for 64-bit integers (BigInt)
		it('handles uint64 dtype', () => {
			assert.ok(data.uint64);
			assert.strictEqual(data.uint64.dtype, 'uint64');
			assert.strictEqual(data.uint64.data[0], 10000n);
			assert.strictEqual(data.uint64.data[1], 20000n);
			assert.strictEqual(data.uint64.data[2], 30000n);
		});

		it('handles int64 dtype', () => {
			assert.ok(data.int64);
			assert.strictEqual(data.int64.dtype, 'int64');
			assert.strictEqual(data.int64.data[0], -10000n);
			assert.strictEqual(data.int64.data[1], 0n);
			assert.strictEqual(data.int64.data[2], 10000n);
		});
	});
});
