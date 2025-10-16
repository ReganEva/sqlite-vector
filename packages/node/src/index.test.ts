import { test, describe } from 'node:test';
import assert from 'node:assert';
import {
  getCurrentPlatform,
  getPlatformPackageName,
  getBinaryName,
  isMusl,
  getExtensionPath,
  getExtensionInfo,
  ExtensionNotFoundError
} from './index.js';

describe('Platform Detection', () => {
  test('getCurrentPlatform() returns a valid platform', () => {
    const platform = getCurrentPlatform();
    const validPlatforms = [
      'darwin-arm64',
      'darwin-x86_64',
      'linux-arm64',
      'linux-arm64-musl',
      'linux-x86_64',
      'linux-x86_64-musl',
      'win32-x86_64',
    ];

    assert.ok(
      validPlatforms.includes(platform),
      `Platform ${platform} should be one of: ${validPlatforms.join(', ')}`
    );
  });

  test('getPlatformPackageName() returns correct package name format', () => {
    const packageName = getPlatformPackageName();

    assert.ok(
      packageName.startsWith('@sqliteai/sqlite-vector-'),
      'Package name should start with @sqliteai/sqlite-vector-'
    );

    assert.match(
      packageName,
      /^@sqliteai\/sqlite-vector-(darwin|linux|win32)-(arm64|x86_64)(-musl)?$/,
      'Package name should match expected format'
    );
  });

  test('getBinaryName() returns correct extension', () => {
    const binaryName = getBinaryName();

    assert.match(
      binaryName,
      /^vector\.(dylib|so|dll)$/,
      'Binary name should be vector.dylib, vector.so, or vector.dll'
    );
  });

  test('isMusl() returns a boolean', () => {
    const result = isMusl();
    assert.strictEqual(typeof result, 'boolean');
  });
});

describe('Extension Path Resolution', () => {
  test('getExtensionPath() returns a string or throws', () => {
    try {
      const path = getExtensionPath();
      assert.strictEqual(typeof path, 'string');
      assert.ok(path.length > 0, 'Path should not be empty');
    } catch (error) {
      // If it throws, it should be ExtensionNotFoundError
      assert.ok(
        error instanceof ExtensionNotFoundError,
        'Should throw ExtensionNotFoundError if extension not found'
      );
    }
  });

  test('getExtensionInfo() returns complete info object', () => {
    try {
      const info = getExtensionInfo();

      assert.ok(info.platform, 'Should have platform');
      assert.ok(info.packageName, 'Should have packageName');
      assert.ok(info.binaryName, 'Should have binaryName');
      assert.ok(info.path, 'Should have path');

      assert.strictEqual(typeof info.platform, 'string');
      assert.strictEqual(typeof info.packageName, 'string');
      assert.strictEqual(typeof info.binaryName, 'string');
      assert.strictEqual(typeof info.path, 'string');
    } catch (error) {
      // If it throws, it should be ExtensionNotFoundError
      assert.ok(
        error instanceof ExtensionNotFoundError,
        'Should throw ExtensionNotFoundError if extension not found'
      );
    }
  });
});

describe('Error Handling', () => {
  test('ExtensionNotFoundError has correct properties', () => {
    const error = new ExtensionNotFoundError('Test message');

    assert.ok(error instanceof Error);
    assert.strictEqual(error.name, 'ExtensionNotFoundError');
    assert.strictEqual(error.message, 'Test message');
  });
});
