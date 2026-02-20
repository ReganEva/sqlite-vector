/*
 * test_vector.c
 * Comprehensive unit tests for the SQLite Vector extension.
 *
 * Compiled with -DSQLITE_CORE so sqlite3_vector_init links statically.
 * Usage: gcc -DSQLITE_CORE ... -o test_vector && ./test_vector
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include "sqlite3.h"
#include "sqlite-vector.h"

/* ---------- Test infrastructure ---------- */

static int failures = 0;
static int passes   = 0;

#define ASSERT(cond, msg) do {                        \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else         { printf("PASS: %s\n", msg); passes++;    } \
} while (0)

/* Execute SQL that must succeed; returns SQLITE_OK or aborts the test. */
static int exec_sql(sqlite3 *db, const char *sql) {
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        printf("  SQL error (%d): %s\n  Statement: %s\n", rc, err ? err : "unknown", sql);
        sqlite3_free(err);
    }
    return rc;
}

/* ---------- Helper: create, populate, and init a vector table ---------- */

/*
 * Sets up a table named `tbl` with columns (id INTEGER PRIMARY KEY, v BLOB),
 * inserts `n` vectors of the given type converted from JSON via vector_as_<type>(),
 * and calls vector_init() with the specified type, distance, and dimension.
 *
 * `vecs` is an array of JSON strings, e.g. "[1.0, 2.0, 3.0]".
 */
static int setup_table(sqlite3 *db, const char *tbl, const char *type,
                       const char *distance, int dim,
                       const char **vecs, int n) {
    char sql[2048];

    /* Create table */
    snprintf(sql, sizeof(sql), "CREATE TABLE \"%s\" (id INTEGER PRIMARY KEY, v BLOB);", tbl);
    if (exec_sql(db, sql) != SQLITE_OK) return -1;

    /* Insert vectors */
    for (int i = 0; i < n; i++) {
        snprintf(sql, sizeof(sql),
                 "INSERT INTO \"%s\" (id, v) VALUES (%d, vector_as_%s('%s'));",
                 tbl, i + 1, type, vecs[i]);
        if (exec_sql(db, sql) != SQLITE_OK) return -1;
    }

    /* vector_init */
    snprintf(sql, sizeof(sql),
             "SELECT vector_init('%s', 'v', 'type=%s,dimension=%d,distance=%s');",
             tbl, type, dim, distance);
    if (exec_sql(db, sql) != SQLITE_OK) return -1;

    return 0;
}

/* ---------- Callback helpers for querying results ---------- */

typedef struct {
    int    count;
    int    ids[64];
    double distances[64];
} scan_result;

static int scan_cb(void *ctx, int ncols, char **vals, char **names) {
    (void)names;
    scan_result *r = (scan_result *)ctx;
    if (r->count < 64 && ncols >= 2 && vals[1]) {
        r->ids[r->count] = vals[0] ? atoi(vals[0]) : 0;
        r->distances[r->count] = atof(vals[1]);
    }
    r->count++;
    return 0;
}

/* ---------- Test: basics ---------- */

static void test_basics(sqlite3 *db) {
    printf("\n=== Basics ===\n");

    /* vector_version() */
    {
        sqlite3_stmt *stmt;
        int rc = sqlite3_prepare_v2(db, "SELECT vector_version();", -1, &stmt, NULL);
        ASSERT(rc == SQLITE_OK, "vector_version() prepares");
        if (rc == SQLITE_OK) {
            rc = sqlite3_step(stmt);
            ASSERT(rc == SQLITE_ROW, "vector_version() returns a row");
            const char *v = (const char *)sqlite3_column_text(stmt, 0);
            ASSERT(v != NULL && strlen(v) > 0, "vector_version() returns non-empty text");
        }
        sqlite3_finalize(stmt);
    }

    /* vector_backend() */
    {
        sqlite3_stmt *stmt;
        int rc = sqlite3_prepare_v2(db, "SELECT vector_backend();", -1, &stmt, NULL);
        ASSERT(rc == SQLITE_OK, "vector_backend() prepares");
        if (rc == SQLITE_OK) {
            rc = sqlite3_step(stmt);
            ASSERT(rc == SQLITE_ROW, "vector_backend() returns a row");
            const char *v = (const char *)sqlite3_column_text(stmt, 0);
            ASSERT(v != NULL && strlen(v) > 0, "vector_backend() returns non-empty text");
        }
        sqlite3_finalize(stmt);
    }
}

/* ---------- Test: vector_full_scan for a given (type, distance) pair ---------- */

static void test_full_scan(sqlite3 *db, const char *type, const char *distance,
                           int dim, const char **vecs, int nvecs,
                           const char *query_vec) {
    char tbl[64], sql[1024], msg[256];
    snprintf(tbl, sizeof(tbl), "tfs_%s_%s", type, distance);

    /* lowercase table name for uniqueness */
    for (char *p = tbl; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    if (setup_table(db, tbl, type, distance, dim, vecs, nvecs) != 0) {
        snprintf(msg, sizeof(msg), "full_scan setup %s/%s", type, distance);
        ASSERT(0, msg);
        return;
    }

    /* DOT distance returns negative dot product, so skip non-negative checks */
    int is_dot = (strcasecmp(distance, "DOT") == 0);

    /* Top-k mode (k=3) */
    {
        scan_result r = {0};
        snprintf(sql, sizeof(sql),
                 "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_%s('%s'), 3);",
                 tbl, type, query_vec);
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
        snprintf(msg, sizeof(msg), "full_scan top-k executes (%s/%s)", type, distance);
        ASSERT(rc == SQLITE_OK, msg);
        if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

        snprintf(msg, sizeof(msg), "full_scan top-k returns 3 rows (%s/%s)", type, distance);
        ASSERT(r.count == 3, msg);

        /* Distances should be non-negative (DOT returns negative dot product, so skip) */
        if (!is_dot) {
            int all_non_neg = 1;
            for (int i = 0; i < r.count; i++) {
                if (r.distances[i] < 0) all_non_neg = 0;
            }
            snprintf(msg, sizeof(msg), "full_scan top-k distances >= 0 (%s/%s)", type, distance);
            ASSERT(all_non_neg, msg);
        }

        /* Distances should be sorted ascending */
        int sorted = 1;
        for (int i = 1; i < r.count; i++) {
            if (r.distances[i] < r.distances[i - 1]) sorted = 0;
        }
        snprintf(msg, sizeof(msg), "full_scan top-k distances sorted (%s/%s)", type, distance);
        ASSERT(sorted, msg);
    }

    /* Streaming mode (no k, use LIMIT) */
    {
        scan_result r = {0};
        snprintf(sql, sizeof(sql),
                 "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_%s('%s')) LIMIT 5;",
                 tbl, type, query_vec);
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
        snprintf(msg, sizeof(msg), "full_scan stream executes (%s/%s)", type, distance);
        ASSERT(rc == SQLITE_OK, msg);
        if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

        snprintf(msg, sizeof(msg), "full_scan stream returns rows (%s/%s)", type, distance);
        ASSERT(r.count > 0, msg);

        if (!is_dot) {
            int all_non_neg = 1;
            for (int i = 0; i < r.count; i++) {
                if (r.distances[i] < 0) all_non_neg = 0;
            }
            snprintf(msg, sizeof(msg), "full_scan stream distances >= 0 (%s/%s)", type, distance);
            ASSERT(all_non_neg, msg);
        }
    }
}

/* ---------- Test: vector_quantize_scan for a given (type, qtype) pair ---------- */

static void test_quantize_scan(sqlite3 *db, const char *type, const char *qtype,
                               int dim, const char **vecs, int nvecs,
                               const char *query_vec) {
    char tbl[64], sql[1024], msg[256];
    snprintf(tbl, sizeof(tbl), "tqs_%s_%s", type, qtype);

    for (char *p = tbl; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    /* Use L2 distance (or HAMMING for BIT) */
    const char *distance = (strcasecmp(type, "BIT") == 0) ? "HAMMING" : "L2";

    if (setup_table(db, tbl, type, distance, dim, vecs, nvecs) != 0) {
        snprintf(msg, sizeof(msg), "quantize_scan setup %s/%s", type, qtype);
        ASSERT(0, msg);
        return;
    }

    /* vector_quantize */
    snprintf(sql, sizeof(sql),
             "SELECT vector_quantize('%s', 'v', 'qtype=%s');", tbl, qtype);
    if (exec_sql(db, sql) != SQLITE_OK) {
        snprintf(msg, sizeof(msg), "vector_quantize %s/%s", type, qtype);
        ASSERT(0, msg);
        return;
    }

    /* Top-k mode */
    {
        scan_result r = {0};
        snprintf(sql, sizeof(sql),
                 "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_%s('%s'), 3);",
                 tbl, type, query_vec);
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
        snprintf(msg, sizeof(msg), "quantize_scan top-k executes (%s/%s)", type, qtype);
        ASSERT(rc == SQLITE_OK, msg);
        if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

        snprintf(msg, sizeof(msg), "quantize_scan top-k returns rows (%s/%s)", type, qtype);
        ASSERT(r.count > 0, msg);
    }

    /* Streaming mode */
    {
        scan_result r = {0};
        snprintf(sql, sizeof(sql),
                 "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_%s('%s')) LIMIT 5;",
                 tbl, type, query_vec);
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
        snprintf(msg, sizeof(msg), "quantize_scan stream executes (%s/%s)", type, qtype);
        ASSERT(rc == SQLITE_OK, msg);
        if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

        snprintf(msg, sizeof(msg), "quantize_scan stream returns rows (%s/%s)", type, qtype);
        ASSERT(r.count > 0, msg);
    }
}

/* ---------- Test vectors ---------- */

/* 4-dimensional float vectors for numeric types */
static const char *float_vecs[] = {
    "[1.0, 0.0, 0.0, 0.0]",
    "[0.0, 1.0, 0.0, 0.0]",
    "[0.0, 0.0, 1.0, 0.0]",
    "[0.0, 0.0, 0.0, 1.0]",
    "[1.0, 1.0, 0.0, 0.0]",
    "[0.0, 1.0, 1.0, 0.0]",
    "[0.0, 0.0, 1.0, 1.0]",
    "[1.0, 1.0, 1.0, 0.0]",
    "[0.0, 1.0, 1.0, 1.0]",
    "[1.0, 1.0, 1.0, 1.0]",
};
static const int float_nvecs = 10;
static const char *float_query = "[0.5, 0.5, 0.5, 0.5]";

/* Integer vectors (0-255 range for U8, -128..127 for I8) */
static const char *int_vecs[] = {
    "[10, 0, 0, 0]",
    "[0, 10, 0, 0]",
    "[0, 0, 10, 0]",
    "[0, 0, 0, 10]",
    "[10, 10, 0, 0]",
    "[0, 10, 10, 0]",
    "[0, 0, 10, 10]",
    "[10, 10, 10, 0]",
    "[0, 10, 10, 10]",
    "[10, 10, 10, 10]",
};
static const int int_nvecs = 10;
static const char *int_query = "[5, 5, 5, 5]";

/* 8-dimensional BIT vectors (0 or 1 values) */
static const char *bit_vecs[] = {
    "[1, 0, 0, 0, 0, 0, 0, 0]",
    "[0, 1, 0, 0, 0, 0, 0, 0]",
    "[0, 0, 1, 0, 0, 0, 0, 0]",
    "[0, 0, 0, 1, 0, 0, 0, 0]",
    "[1, 1, 0, 0, 0, 0, 0, 0]",
    "[0, 1, 1, 0, 0, 0, 0, 0]",
    "[0, 0, 1, 1, 0, 0, 0, 0]",
    "[1, 1, 1, 0, 0, 0, 0, 0]",
    "[0, 1, 1, 1, 0, 0, 0, 0]",
    "[1, 1, 1, 1, 0, 0, 0, 0]",
};
static const int bit_nvecs = 10;
static const char *bit_query = "[1, 0, 1, 0, 1, 0, 1, 0]";

/* ---------- Test: distance function values ---------- */

typedef struct {
    const char *distance_name;
    double eps_f32;
    double eps_f16;
    double eps_bf16;
    double expected[10];
} expected_distance_case;

static const char *distance_vecs[] = {
    "[1.0, 2.0, 0.0, -1.0]",
    "[0.5, -1.5, 2.0, 1.0]",
    "[-2.0, 0.0, 1.0, 0.5]",
    "[3.0, 1.0, -1.0, 2.0]",
    "[-0.5, 2.5, 1.5, -2.0]",
    "[1.5, 1.5, 1.5, 1.5]",
    "[-1.0, -2.0, 0.5, 3.0]",
    "[2.0, -0.5, -2.5, 0.0]",
    "[0.0, 3.0, -1.0, -1.5]",
    "[-1.5, 0.5, 2.5, -0.5]"
};
static const int distance_nvecs = 10;
static const char *distance_query = "[0.75, -0.25, 1.25, -0.75]";
static const char *distance_int_vecs[] = {
    "[10, 2, 0, 7]",
    "[3, 14, 9, 1]",
    "[20, 5, 4, 12]",
    "[8, 8, 8, 8]",
    "[1, 0, 15, 6]",
    "[12, 18, 2, 4]",
    "[6, 3, 11, 19]",
    "[16, 7, 13, 5]",
    "[4, 20, 1, 10]",
    "[9, 11, 6, 14]"
};
static const int distance_int_nvecs = 10;
static const char *distance_int_query = "[7, 9, 5, 11]";

static double eps_for_type(const expected_distance_case *tc, const char *vtype) {
    if (strcasecmp(vtype, "f16") == 0) return tc->eps_f16;
    if (strcasecmp(vtype, "bf16") == 0) return tc->eps_bf16;
    return tc->eps_f32;
}

static void test_one_distance_case(sqlite3 *db, const char *vtype, const expected_distance_case *tc) {
    char tbl[64];
    char sql[1024];
    char msg[256];
    double eps = eps_for_type(tc, vtype);

    snprintf(tbl, sizeof(tbl), "tdist_%s_%s", tc->distance_name, vtype);
    for (char *p = tbl; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    if (setup_table(db, tbl, vtype, tc->distance_name, 4, distance_vecs, distance_nvecs) != 0) {
        snprintf(msg, sizeof(msg), "%s/%s distance setup", vtype, tc->distance_name);
        ASSERT(0, msg);
        return;
    }

    scan_result r = {0};
    snprintf(sql, sizeof(sql),
             "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_%s('%s')) ORDER BY id;",
             tbl, vtype, distance_query);
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
    snprintf(msg, sizeof(msg), "%s/%s distance query executes", vtype, tc->distance_name);
    ASSERT(rc == SQLITE_OK, msg);
    if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
    if (rc != SQLITE_OK) return;

    snprintf(msg, sizeof(msg), "%s/%s distance query returns all rows", vtype, tc->distance_name);
    ASSERT(r.count == distance_nvecs, msg);
    if (r.count != distance_nvecs) return;

    for (int i = 0; i < distance_nvecs; i++) {
        int id_ok = (r.ids[i] == (i + 1));
        snprintf(msg, sizeof(msg), "%s/%s row id matches expected (row %d)",
                 vtype, tc->distance_name, i + 1);
        ASSERT(id_ok, msg);

        double diff = fabs(r.distances[i] - tc->expected[i]);
        int within_eps = diff <= eps;
        snprintf(msg, sizeof(msg), "%s/%s distance within epsilon (id=%d, diff=%.8g, eps=%.3g)",
                 vtype, tc->distance_name, i + 1, diff, eps);
        ASSERT(within_eps, msg);
    }
}

static void test_distance_functions_float(sqlite3 *db) {
    const expected_distance_case cases[] = {
        {
            .distance_name = "L2",
            .eps_f32 = 1e-6,
            .eps_f16 = 1e-2,
            .eps_bf16 = 5e-2,
            .expected = {
                2.598076211353316,
                2.291287847477920,
                3.041381265149110,
                4.387482193696061,
                3.278719262151000,
                2.958039891549808,
                4.555216789572150,
                4.031128874149275,
                4.092676385936225,
                2.692582403567252
            }
        },
        {
            .distance_name = "SQUARED_L2",
            .eps_f32 = 1e-6,
            .eps_f16 = 5e-2,
            .eps_bf16 = 2e-1,
            .expected = {6.75, 5.25, 9.25, 19.25, 10.75, 8.75, 20.75, 16.25, 16.75, 7.25}
        },
        {
            .distance_name = "COSINE",
            .eps_f32 = 1e-5,
            .eps_f16 = 1e-2,
            .eps_bf16 = 5e-2,
            .expected = {
                0.753817018041334,
                0.449518117436820,
                1.164487923739942,
                1.116774841624228,
                0.598909685625288,
                0.698488655422236,
                1.299521148936577,
                1.279145263119541,
                1.150755672288882,
                0.547732983133355
            }
        },
        {
            .distance_name = "DOT",
            .eps_f32 = 1e-6,
            .eps_f16 = 1e-2,
            .eps_bf16 = 5e-2,
            .expected = {-1.0, -2.5, 0.625, 0.75, -2.375, -1.5, 1.875, 1.5, 0.875, -2.25}
        },
        {
            .distance_name = "L1",
            .eps_f32 = 1e-6,
            .eps_f16 = 1e-2,
            .eps_bf16 = 5e-2,
            .expected = {4.0, 4.0, 4.5, 8.5, 5.5, 5.0, 8.0, 6.0, 7.0, 4.5}
        }
    };
    const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    const char *types[] = {"f32", "f16", "bf16"};

    for (int t = 0; t < 3; t++) {
        for (int i = 0; i < ncases; i++) {
            test_one_distance_case(db, types[t], &cases[i]);
        }
    }
}

typedef struct {
    const char *distance_name;
    double eps_i8;
    double eps_u8;
    double expected[10];
} expected_int_distance_case;

static double eps_for_int_type(const expected_int_distance_case *tc, const char *vtype) {
    if (strcasecmp(vtype, "i8") == 0) return tc->eps_i8;
    return tc->eps_u8;
}

static void test_one_int_distance_case(sqlite3 *db, const char *vtype, const expected_int_distance_case *tc) {
    char tbl[64];
    char sql[1024];
    char msg[256];
    double eps = eps_for_int_type(tc, vtype);

    snprintf(tbl, sizeof(tbl), "tdist_%s_%s", tc->distance_name, vtype);
    for (char *p = tbl; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    if (setup_table(db, tbl, vtype, tc->distance_name, 4, distance_int_vecs, distance_int_nvecs) != 0) {
        snprintf(msg, sizeof(msg), "%s/%s int distance setup", vtype, tc->distance_name);
        ASSERT(0, msg);
        return;
    }

    scan_result r = {0};
    snprintf(sql, sizeof(sql),
             "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_%s('%s')) ORDER BY id;",
             tbl, vtype, distance_int_query);
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
    snprintf(msg, sizeof(msg), "%s/%s int distance query executes", vtype, tc->distance_name);
    ASSERT(rc == SQLITE_OK, msg);
    if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
    if (rc != SQLITE_OK) return;

    snprintf(msg, sizeof(msg), "%s/%s int distance query returns all rows", vtype, tc->distance_name);
    ASSERT(r.count == distance_int_nvecs, msg);
    if (r.count != distance_int_nvecs) return;

    for (int i = 0; i < distance_int_nvecs; i++) {
        int id_ok = (r.ids[i] == (i + 1));
        snprintf(msg, sizeof(msg), "%s/%s int row id matches expected (row %d)",
                 vtype, tc->distance_name, i + 1);
        ASSERT(id_ok, msg);

        double diff = fabs(r.distances[i] - tc->expected[i]);
        int within_eps = diff <= eps;
        snprintf(msg, sizeof(msg), "%s/%s int distance within epsilon (id=%d, diff=%.8g, eps=%.3g)",
                 vtype, tc->distance_name, i + 1, diff, eps);
        ASSERT(within_eps, msg);
    }
}

static void test_distance_functions_int(sqlite3 *db) {
    const expected_int_distance_case cases[] = {
        {
            .distance_name = "L2",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {
                9.949874371066199,
                12.529964086141668,
                13.674794331177344,
                4.472135954999580,
                15.556349186104045,
                12.806248474865697,
                11.704699910719626,
                13.601470508735444,
                12.124355652982141,
                4.242640687119285
            }
        },
        {
            .distance_name = "SQUARED_L2",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {99.0, 157.0, 187.0, 20.0, 242.0, 164.0, 137.0, 185.0, 147.0, 18.0}
        },
        {
            .distance_name = "COSINE",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {
                0.197058901598547,
                0.278725549597720,
                0.161317797973194,
                0.036913175313846,
                0.449627749704491,
                0.182558273343614,
                0.126858993881120,
                0.205091387999948,
                0.144927951966812,
                0.000283884548207
            }
        },
        {
            .distance_name = "DOT",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {-165.0, -203.0, -337.0, -256.0, -148.0, -300.0, -333.0, -295.0, -323.0, -346.0}
        },
        {
            .distance_name = "L1",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {19.0, 23.0, 19.0, 8.0, 30.0, 24.0, 21.0, 25.0, 19.0, 8.0}
        }
    };
    const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    const char *types[] = {"i8", "u8"};

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < ncases; i++) {
            test_one_int_distance_case(db, types[t], &cases[i]);
        }
    }
}

static void test_distance_functions_hamming(sqlite3 *db) {
    const char *tbl = "tdist_hamming_bit";
    const double expected[] = {3.0, 5.0, 3.0, 5.0, 4.0, 4.0, 4.0, 3.0, 5.0, 4.0};
    char sql[1024];
    char msg[256];

    if (setup_table(db, tbl, "bit", "HAMMING", 8, bit_vecs, bit_nvecs) != 0) {
        ASSERT(0, "bit/HAMMING distance setup");
        return;
    }

    scan_result r = {0};
    snprintf(sql, sizeof(sql),
             "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_bit('%s')) ORDER BY id;",
             tbl, bit_query);
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
    ASSERT(rc == SQLITE_OK, "bit/HAMMING distance query executes");
    if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
    if (rc != SQLITE_OK) return;

    ASSERT(r.count == bit_nvecs, "bit/HAMMING distance query returns all rows");
    if (r.count != bit_nvecs) return;

    for (int i = 0; i < bit_nvecs; i++) {
        int id_ok = (r.ids[i] == (i + 1));
        snprintf(msg, sizeof(msg), "bit/HAMMING row id matches expected (row %d)", i + 1);
        ASSERT(id_ok, msg);

        int exact_match = (r.distances[i] == expected[i]);
        snprintf(msg, sizeof(msg), "bit/HAMMING distance matches exactly (id=%d)", i + 1);
        ASSERT(exact_match, msg);
    }
}

/* ---------- Main ---------- */

int main(void) {
    sqlite3 *db;
    int rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        printf("FAIL: cannot open :memory: database\n");
        return 1;
    }

    /* Initialize the vector extension */
    char *errmsg = NULL;
    rc = sqlite3_vector_init(db, &errmsg, NULL);
    if (rc != SQLITE_OK) {
        printf("FAIL: sqlite3_vector_init returned %d: %s\n", rc, errmsg ? errmsg : "");
        sqlite3_close(db);
        return 1;
    }

    /* 1. Basics */
    test_basics(db);

    /* 2. vector_full_scan — float types × all distances */
    printf("\n=== vector_full_scan ===\n");
    {
        const char *float_types[] = {"f32", "f16", "bf16"};
        const char *distances[]   = {"L2", "SQUARED_L2", "COSINE", "DOT", "L1"};

        for (int t = 0; t < 3; t++) {
            for (int d = 0; d < 5; d++) {
                test_full_scan(db, float_types[t], distances[d],
                               4, float_vecs, float_nvecs, float_query);
            }
        }

        /* Integer types */
        const char *int_types[] = {"i8", "u8"};
        for (int t = 0; t < 2; t++) {
            for (int d = 0; d < 5; d++) {
                test_full_scan(db, int_types[t], distances[d],
                               4, int_vecs, int_nvecs, int_query);
            }
        }

        /* BIT — only HAMMING */
        test_full_scan(db, "bit", "HAMMING", 8, bit_vecs, bit_nvecs, bit_query);
    }

    /* 3. vector_quantize_scan — all vector types × quantization types */
    printf("\n=== vector_quantize_scan ===\n");
    {
        const char *qtypes[] = {"UINT8", "INT8", "1BIT"};

        /* Float types */
        const char *float_types[] = {"f32", "f16", "bf16"};
        for (int t = 0; t < 3; t++) {
            for (int q = 0; q < 3; q++) {
                test_quantize_scan(db, float_types[t], qtypes[q],
                                   4, float_vecs, float_nvecs, float_query);
            }
        }

        /* Integer types */
        const char *int_types[] = {"i8", "u8"};
        for (int t = 0; t < 2; t++) {
            for (int q = 0; q < 3; q++) {
                test_quantize_scan(db, int_types[t], qtypes[q],
                                   4, int_vecs, int_nvecs, int_query);
            }
        }

        /* BIT — quantize with 1BIT */
        test_quantize_scan(db, "bit", "1BIT", 8, bit_vecs, bit_nvecs, bit_query);
    }

    /* 4. Backward-compat aliases */
    printf("\n=== Backward-compat aliases ===\n");
    {
        /* Set up a table for alias tests */
        const char *tbl = "tfs_alias";
        if (setup_table(db, tbl, "f32", "L2", 4, float_vecs, float_nvecs) == 0) {
            /* vector_full_scan_stream */
            {
                scan_result r = {0};
                char sql[512];
                snprintf(sql, sizeof(sql),
                         "SELECT id, distance FROM vector_full_scan_stream('%s', 'v', vector_as_f32('%s')) LIMIT 3;",
                         tbl, float_query);
                char *err = NULL;
                rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
                ASSERT(rc == SQLITE_OK, "vector_full_scan_stream alias works");
                if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
                ASSERT(r.count > 0, "vector_full_scan_stream returns rows");
            }

            /* vector_quantize_scan_stream */
            {
                char sql[512];
                snprintf(sql, sizeof(sql),
                         "SELECT vector_quantize('%s', 'v');", tbl);
                exec_sql(db, sql);

                scan_result r = {0};
                snprintf(sql, sizeof(sql),
                         "SELECT id, distance FROM vector_quantize_scan_stream('%s', 'v', vector_as_f32('%s')) LIMIT 3;",
                         tbl, float_query);
                char *err = NULL;
                rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
                ASSERT(rc == SQLITE_OK, "vector_quantize_scan_stream alias works");
                if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
                ASSERT(r.count > 0, "vector_quantize_scan_stream returns rows");
            }
        }
    }


    /* 5. distance functions */
    printf("\n=== distance_functions ===\n");
    {
        test_distance_functions_float(db);
        test_distance_functions_int(db);
        test_distance_functions_hamming(db);
    }


    sqlite3_close(db);

    /* Summary */
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", passes, failures);
    printf("========================================\n");

    return failures > 0 ? 1 : 0;
}
