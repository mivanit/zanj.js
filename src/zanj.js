// zanj.js
// Minimal ZANJ loader (frontend-only). Supports refs to npy or json.
// Requires array.js to be loaded first (NDArray, npyjs).
const REF_KEY = "$ref";

function joinUrl(base, rel) {
const b = base.replace(/\/+$/, "");
const r = String(rel || "").replace(/^\/+/, "");
return b + "/" + r;
}

class LazyValue {
constructor(loader) {
	this._loader = loader;  // () => Promise<any>
	this._promise = null;
}
load() {
	if (!this._promise) this._promise = this._loader();
	return this._promise;
}
}

class ZanjLoader {
	constructor(opts) {
		this.baseUrl = (opts && opts.baseUrl) ? opts.baseUrl.replace(/\/+$/, "") : "";
		this.fetchInit = (opts && opts.fetchInit) || undefined;
		this._cache = new Map();
	}

	async loadRoot() {
		const url = joinUrl(this.baseUrl, "__zanj__.json");
		const res = await fetch(url, this.fetchInit);
		if (!res.ok) throw new Error("Failed to fetch " + url + ": " + res.status);
		const root = await res.json();
		return this._makeLazy(root);
	}

	_makeLazy(node) {
		if (node == null) return node;

		if (Array.isArray(node)) {
		return node.map((v) => this._makeLazy(v));
		}

		if (typeof node === "object") {
		// reference?
		if (Object.prototype.hasOwnProperty.call(node, REF_KEY)) {
			const ref = node;
			const path = String(ref[REF_KEY]);
			const fmt = String(ref.format || this._inferFormat(path));
			if (fmt !== "npy" && fmt !== "json") {
			throw new Error("Unsupported ref format: " + fmt);
			}
			const key = fmt + ":" + path;
			return new LazyValue(() => this._loadCached(fmt, path, key));
		}

		// plain object: recurse
		const out = {};
		for (const k in node) {
			if (Object.prototype.hasOwnProperty.call(node, k)) {
			out[k] = this._makeLazy(node[k]);
			}
		}
		return out;
		}

		// primitive
		return node;
	}

	_inferFormat(path) {
		const p = String(path).toLowerCase();
		if (p.endsWith(".npy")) return "npy";
		if (p.endsWith(".json")) return "json";
		throw new Error("Cannot infer format from path: " + path);
	}

	_loadCached(fmt, path, key) {
		const hit = this._cache.get(key);
		if (hit) return hit;

		let p;
		if (fmt === "npy") {
		// NDArray.load(url, callback?, fetchArgs)
		// NDArray is defined in array.js (user-provided)
		p = global.NDArray.load(joinUrl(this.baseUrl, path), undefined, this.fetchInit);
		} else if (fmt === "json") {
		p = fetch(joinUrl(this.baseUrl, path), this.fetchInit).then((r) => {
			if (!r.ok) throw new Error("Failed to fetch " + path + ": " + r.status);
			return r.json();
		});
		} else {
		p = Promise.reject(new Error("Unsupported format: " + fmt));
		}

		this._cache.set(key, p);
		return p;
	}
}