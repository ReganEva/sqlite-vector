# @sqliteai/sqlite-vector

[![npm version](https://badge.fury.io/js/@sqliteai%2Fsqlite-vector.svg)](https://www.npmjs.com/package/@sqliteai/sqlite-vector)
[![License](https://img.shields.io/badge/license-Elastic%202.0-blue.svg)](../../../LICENSE.md)

> SQLite Vector extension for Node.js - Cross-platform vector embeddings and similarity search

**SQLite Vector** brings powerful vector search capabilities to SQLite in Node.js. Perfect for Edge AI, semantic search, RAG applications, and similarity matching - all running locally without external dependencies.

## Features

- ✅ **Cross-platform** - Works on macOS, Linux (glibc/musl), and Windows
- ✅ **Zero configuration** - Automatically detects and loads the correct binary for your platform
- ✅ **TypeScript native** - Full type definitions included
- ✅ **Modern ESM + CJS** - Works with both ES modules and CommonJS
- ✅ **Small footprint** - Only downloads binaries for your platform (~30MB)
- ✅ **Offline-ready** - No external services required

## Installation

```bash
npm install @sqliteai/sqlite-vector
```

The package automatically downloads the correct native extension for your platform during installation.

### Supported Platforms

| Platform | Architecture | Package |
|----------|-------------|---------|
| macOS | ARM64 (Apple Silicon) | `@sqliteai/sqlite-vector-darwin-arm64` |
| macOS | x86_64 (Intel) | `@sqliteai/sqlite-vector-darwin-x86_64` |
| Linux | ARM64 (glibc) | `@sqliteai/sqlite-vector-linux-arm64` |
| Linux | ARM64 (musl/Alpine) | `@sqliteai/sqlite-vector-linux-arm64-musl` |
| Linux | x86_64 (glibc) | `@sqliteai/sqlite-vector-linux-x86_64` |
| Linux | x86_64 (musl/Alpine) | `@sqliteai/sqlite-vector-linux-x86_64-musl` |
| Windows | x86_64 | `@sqliteai/sqlite-vector-win32-x86_64` |

## Usage

### Basic Usage

```typescript
import { getExtensionPath } from '@sqliteai/sqlite-vector';
import Database from 'better-sqlite3';

// Get the path to the vector extension
const extensionPath = getExtensionPath();

// Load it into your SQLite database
const db = new Database(':memory:');
db.loadExtension(extensionPath);

// Use vector functions
db.exec(`
  CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    vector BLOB,
    text TEXT
  );
`);

// Initialize vector search
db.prepare("SELECT vector_init('embeddings', 'vector', 'type=FLOAT32,dimension=384')").run();

// Insert vectors (using your embedding model)
const embedding = new Float32Array(384);
// ... fill embedding with your model's output

db.prepare('INSERT INTO embeddings (vector, text) VALUES (?, ?)').run(
  Buffer.from(embedding.buffer),
  'Sample text'
);

// Perform similarity search
const query = new Float32Array(384); // Your query embedding
const results = db.prepare(`
  SELECT e.id, e.text, v.distance
  FROM embeddings AS e
  JOIN vector_quantize_scan('embeddings', 'vector', ?, 10) AS v
  ON e.id = v.rowid
  ORDER BY v.distance ASC
`).all(Buffer.from(query.buffer));

console.log(results);
```

### CommonJS

```javascript
const { getExtensionPath } = require('@sqliteai/sqlite-vector');
const Database = require('better-sqlite3');

const db = new Database(':memory:');
db.loadExtension(getExtensionPath());

// Ready to use
const version = db.prepare('SELECT vector_version()').pluck().get();
console.log('Vector extension version:', version);
```

### Get Extension Information

```typescript
import { getExtensionInfo } from '@sqliteai/sqlite-vector';

const info = getExtensionInfo();
console.log(info);
// {
//   platform: 'darwin-arm64',
//   packageName: '@sqliteai/sqlite-vector-darwin-arm64',
//   binaryName: 'vector.dylib',
//   path: '/path/to/node_modules/@sqliteai/sqlite-vector-darwin-arm64/vector.dylib'
// }
```

## Examples

For complete, runnable examples, see the [sqlite-extensions-guide](https://github.com/sqliteai/sqlite-extensions-guide/tree/main/examples/node):

- **[basic-usage.js](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/examples/node/basic-usage.js)** - Generic extension loading for any @sqliteai extension
- **[semantic-search.js](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/examples/node/semantic-search.js)** - Complete semantic search with OpenAI or on-device embeddings
- **[with-multiple-extensions.js](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/examples/node/with-multiple-extensions.js)** - Loading multiple extensions together (vector + sync + AI + js)

These examples are generic and work with all SQLite extensions: `sqlite-vector`, `sqlite-sync`, `sqlite-js`, and `sqlite-ai`.

## API Reference

### `getExtensionPath(): string`

Returns the absolute path to the SQLite Vector extension binary for the current platform.

**Returns:** `string` - Absolute path to the extension file (`.so`, `.dylib`, or `.dll`)

**Throws:** `ExtensionNotFoundError` - If the extension binary cannot be found for the current platform

**Example:**
```typescript
import { getExtensionPath } from '@sqliteai/sqlite-vector';

const path = getExtensionPath();
// => '/path/to/node_modules/@sqliteai/sqlite-vector-darwin-arm64/vector.dylib'
```

---

### `getExtensionInfo(): ExtensionInfo`

Returns detailed information about the extension for the current platform.

**Returns:** `ExtensionInfo` object with the following properties:
- `platform: Platform` - Current platform identifier (e.g., `'darwin-arm64'`)
- `packageName: string` - Name of the platform-specific npm package
- `binaryName: string` - Filename of the binary (e.g., `'vector.dylib'`)
- `path: string` - Full path to the extension binary

**Throws:** `ExtensionNotFoundError` - If the extension binary cannot be found

**Example:**
```typescript
import { getExtensionInfo } from '@sqliteai/sqlite-vector';

const info = getExtensionInfo();
console.log(`Running on ${info.platform}`);
console.log(`Extension path: ${info.path}`);
```

---

### `getCurrentPlatform(): Platform`

Returns the current platform identifier.

**Returns:** `Platform` - One of:
- `'darwin-arm64'` - macOS ARM64
- `'darwin-x86_64'` - macOS x86_64
- `'linux-arm64'` - Linux ARM64 (glibc)
- `'linux-arm64-musl'` - Linux ARM64 (musl)
- `'linux-x86_64'` - Linux x86_64 (glibc)
- `'linux-x86_64-musl'` - Linux x86_64 (musl)
- `'win32-x86_64'` - Windows x86_64

**Throws:** `Error` - If the platform is unsupported

---

### `isMusl(): boolean`

Detects if the system uses musl libc (Alpine Linux, etc.).

**Returns:** `boolean` - `true` if musl is detected, `false` otherwise

---

### `class ExtensionNotFoundError extends Error`

Error thrown when the SQLite Vector extension cannot be found for the current platform.

## Vector Search Guide

For detailed information on how to use the vector search features, see the [main documentation](https://github.com/sqliteai/sqlite-vector/blob/main/README.md).

### Quick Reference

```sql
-- Initialize vector column
SELECT vector_init('table', 'column', 'type=FLOAT32,dimension=384');

-- Quantize vectors for faster search
SELECT vector_quantize('table', 'column');

-- Preload into memory for 4-5x speedup
SELECT vector_quantize_preload('table', 'column');

-- Search for similar vectors
SELECT * FROM table AS t
JOIN vector_quantize_scan('table', 'column', <query_vector>, <limit>) AS v
ON t.rowid = v.rowid
ORDER BY v.distance;
```

### Distance Metrics

Specify the distance metric during initialization:

```sql
-- L2 (Euclidean) - default
SELECT vector_init('table', 'column', 'type=FLOAT32,dimension=384,distance=L2');

-- Cosine similarity
SELECT vector_init('table', 'column', 'type=FLOAT32,dimension=384,distance=COSINE');

-- Dot product
SELECT vector_init('table', 'column', 'type=FLOAT32,dimension=384,distance=DOT');

-- L1 (Manhattan)
SELECT vector_init('table', 'column', 'type=FLOAT32,dimension=384,distance=L1');
```

## Troubleshooting

### Extension Not Found

If you get `ExtensionNotFoundError`, try:

```bash
# Force reinstall dependencies
npm install --force

# Or manually install the platform package
npm install @sqliteai/sqlite-vector-darwin-arm64  # Replace with your platform
```

### Platform Not Detected

The package should automatically detect your platform. If detection fails, please [open an issue](https://github.com/sqliteai/sqlite-vector/issues) with:
- Your OS and architecture
- Node.js version
- Output of `node -p "process.platform + '-' + process.arch"`

### Alpine Linux / musl

On Alpine Linux or other musl-based systems, the package automatically detects musl and installs the correct variant. If you encounter issues:

```bash
# Verify musl detection
node -e "console.log(require('@sqliteai/sqlite-vector').isMusl())"

# Should print: true
```

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/sqliteai/sqlite-vector.git
cd sqlite-vector/packages/node

# Install dependencies
npm install

# Build TypeScript
npm run build

# Run tests
npm test
```

### Project Structure

```
packages/node/
├── src/
│   ├── index.ts                     # Main entry point
│   ├── platform.ts                  # Platform detection logic
│   └── index.test.ts                # Test suite
├── dist/                            # Compiled JavaScript (generated)
├── generate-platform-packages.js   # Platform package generator
├── package.json
├── tsconfig.json
└── README.md
```

## Related Projects

- **[SQLite-AI](https://github.com/sqliteai/sqlite-ai)** - On-device AI inference and embedding generation
- **[SQLite-Sync](https://github.com/sqliteai/sqlite-sync)** - Sync on-device databases with the cloud
- **[SQLite-JS](https://github.com/sqliteai/sqlite-js)** - Define SQLite functions in JavaScript

## License

This project is licensed under the [Elastic License 2.0](../../../LICENSE.md).

For production or managed service use, please [contact SQLite Cloud, Inc](mailto:info@sqlitecloud.io) for a commercial license.

## Contributing

Contributions are welcome! Please see the [main repository](https://github.com/sqliteai/sqlite-vector) for contribution guidelines.

## Support

- 📖 [Documentation](https://github.com/sqliteai/sqlite-vector/blob/main/API.md)
- 🐛 [Report Issues](https://github.com/sqliteai/sqlite-vector/issues)
- 💬 [Discussions](https://github.com/sqliteai/sqlite-vector/discussions)
