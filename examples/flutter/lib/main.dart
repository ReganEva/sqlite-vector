// Copyright (c) 2025 SQLite Cloud, Inc.
// Licensed under the Elastic License 2.0 (see LICENSE.md).

import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_vector/sqlite_vector.dart';

void main() {
  // Load the sqlite-vector extension (call once at startup).
  sqlite3.loadSqliteVectorExtension();

  // Open an in-memory database.
  final db = sqlite3.openInMemory();

  // Verify the extension is loaded.
  final version = db.select('SELECT vector_version()');
  print('sqlite-vector version: ${version.first.values.first}');

  // Create a regular table with a BLOB column for vectors.
  db.execute('''
    CREATE TABLE items (
      id INTEGER PRIMARY KEY,
      embedding BLOB
    )
  ''');

  // Insert sample Float32 vectors (4 dimensions).
  final stmt =
      db.prepare('INSERT INTO items (embedding) VALUES (vector_as_f32(?))');
  stmt.execute(['[1.0, 2.0, 3.0, 4.0]']);
  stmt.execute(['[5.0, 6.0, 7.0, 8.0]']);
  stmt.execute(['[1.1, 2.1, 3.1, 4.1]']);
  stmt.dispose();

  // Initialize the vector index.
  db.execute(
    "SELECT vector_init('items', 'embedding', 'type=FLOAT32,dimension=4')",
  );

  // Find the 2 nearest neighbors using vector_full_scan.
  final results = db.select('''
    SELECT e.id, v.distance
    FROM items AS e
    JOIN vector_full_scan('items', 'embedding', vector_as_f32('[1.0, 2.0, 3.0, 4.0]'), 2) AS v
    ON e.id = v.rowid
  ''');

  print('Nearest neighbors:');
  for (final row in results) {
    print('  id=${row['id']}, distance=${row['distance']}');
  }

  db.dispose();
}
