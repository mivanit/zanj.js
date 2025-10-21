# zanj.js

JavaScript loader for [ZANJ](https://github.com/mivanit/ZANJ) files with transparent lazy loading.

[ZANJ](https://github.com/mivanit/ZANJ) is a Python format for storing large data structures with external references to NumPy arrays, JSON, and JSONL files. This library lets you load ZANJ files in JavaScript with automatic lazy loading - access data naturally and it loads when needed.

Its aim is to simplify building web dashboards (in my case, those that are the endpoint of some interpretability pipeline) that display a bunch of data created in python -- I found myself spending way too much time writing boilerplate for saving/loading the data in a reasonable structure that didn't overload the client with huge files.

## Creating ZANJ Files (Python)

```python
import numpy as np
from zanj import ZANJ

# Create your data
data = {
    "meta": {"title": "My Data"},
    "array": np.random.normal(size=(100, 32)).astype(np.float32), # big file
	"records": [f"item {i}" for i in range(1000)], # another big file
    "items": ["a", "b", "c", "d", "e"],
}

# Save and extract for web -- we don't want to load the whole zipfile
ZANJ().save(data, "data.zanj")

import zipfile
with zipfile.ZipFile("data.zanj") as zf:
    zf.extractall("web/data")
```

## Installation

```html
<script src="src/array.js"></script>
<script src="src/zanj.js"></script>
```

## Usage

```javascript
// Load the data
const loader = new ZanjLoader("web/data");
const data = await loader.read();

// Access inline data immediately
console.log(data.meta.title);  // "My Data"

// Large arrays load automatically with await
const arr = await data.array;
console.log(arr.shape);  // [100, 32]
console.log(arr.dtype);  // "float32"

// Access array elements
const value = arr.get(0, 5);
```

## Features

- Transparent lazy loading - data loads when accessed
- Supports NumPy arrays (.npy), JSON (.json), and JSONL (.jsonl)
- Float16 support with automatic conversion to Float32
- Multidimensional array indexing and slicing
- Caching to avoid redundant fetches
