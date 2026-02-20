//
//  distance-rvv.c
//  sqlitevector
//
//  Created by Afonso Bordado on 2026/02/19.
//

#include "distance-rvv.h"
#include "distance-cpu.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern const char *distance_backend_name;

// MARK: - FLOAT32 -

float float32_distance_l2_rvv (const void *v1, const void *v2, int n) {
    printf("float32_distance_l2_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float float32_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    printf("float32_distance_l2_squared_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float float32_distance_l1_rvv (const void *v1, const void *v2, int n) {
    printf("float32_distance_l1_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float float32_distance_dot_rvv (const void *v1, const void *v2, int n) {
    printf("float32_distance_dot_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float float32_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // We accumulate the results into a vecto register
    size_t vl = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t dot_acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t magn_a_acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t magn_b_acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

  
    // Iterate by VL elements
    for (; n > 0; n -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        // In practice we use 3, and the last register gets split for the reduction operations
        vl = __riscv_vsetvl_e32m8(n);

        // Load the vectors into the registers
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);

        // Compute the dot product for the entire register, and sum the
        // results into the accumuating register
        vfloat32m8_t vdot = __riscv_vfmul_vv_f32m8(va, vb, vl);
        dot_acc = __riscv_vfredusum_vs_f32m8_f32m1(vdot, dot_acc, vl);

        // Also calculate the magnitude value for both a and b
        vfloat32m8_t magn_a = __riscv_vfmul_vv_f32m8(va, va, vl);
        magn_a_acc = __riscv_vfredusum_vs_f32m8_f32m1(magn_a, magn_a_acc, vl);
        vfloat32m8_t magn_b = __riscv_vfmul_vv_f32m8(vb, vb, vl);
        magn_b_acc = __riscv_vfredusum_vs_f32m8_f32m1(magn_b, magn_b_acc, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register, to finalize the calculations
    // TODO: With default flags this does not use the fsqrt.s/fmin.s/fmax.s instruction, we should fix that
    float dot = __riscv_vfmv_f_s_f32m1_f32(dot_acc);
    float magn_a = sqrtf(__riscv_vfmv_f_s_f32m1_f32(magn_a_acc));
    float magn_b = sqrtf(__riscv_vfmv_f_s_f32m1_f32(magn_b_acc));

    if (magn_a == 0.0f || magn_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (magn_a * magn_b);
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}


// MARK: - FLOAT16 -

float float16_distance_l2_rvv (const void *v1, const void *v2, int n) {
    printf("float16_distance_l2_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float float16_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    printf("float16_distance_l2_squared_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float float16_distance_l1_rvv (const void *v1, const void *v2, int n) {
    printf("float16_distance_l1_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float float16_distance_dot_rvv (const void *v1, const void *v2, int n) {
    printf("float16_distance_dot_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float float16_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    printf("float16_distance_cosine_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

// MARK: - BFLOAT16 -

float bfloat16_distance_l2_rvv (const void *v1, const void *v2, int n) {
    printf("bfloat16_distance_l2_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float bfloat16_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    printf("bfloat16_distance_l2_squared_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float bfloat16_distance_l1_rvv (const void *v1, const void *v2, int n) {
    printf("bfloat16_distance_l1_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float bfloat16_distance_dot_rvv (const void *v1, const void *v2, int n) {
    printf("bfloat16_distance_dot_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float bfloat16_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    printf("bfloat16_distance_cosine_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

// MARK: - UINT8 -

float uint8_distance_l2_rvv (const void *v1, const void *v2, int n) {
    printf("uint8_distance_l2_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float uint8_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    printf("uint8_distance_l2_squared_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float uint8_distance_dot_rvv (const void *v1, const void *v2, int n) {
    printf("uint8_distance_dot_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float uint8_distance_l1_rvv (const void *v1, const void *v2, int n) {
    printf("uint8_distance_l1_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float uint8_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    printf("uint8_distance_cosine_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

// MARK: - INT8 -

float int8_distance_l2_rvv (const void *v1, const void *v2, int n) {
    printf("int8_distance_l2_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float int8_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    printf("int8_distance_l2_squared_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float int8_distance_dot_rvv (const void *v1, const void *v2, int n) {
    printf("int8_distance_dot_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float int8_distance_l1_rvv (const void *v1, const void *v2, int n) {
    printf("int8_distance_l1_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

float int8_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    printf("int8_distance_cosine_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

// MARK: - BIT -

float bit1_distance_hamming_rvv (const void *v1, const void *v2, int n) {
    printf("bit1_distance_hamming_rvv: unimplemented\n");
    abort();
    return 0.0f;
}

#endif

// MARK: -

void init_distance_functions_rvv (void) {
#if defined(__riscv_vector)
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F32] = float32_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F16] = float16_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_rvv;
    
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F16] = float16_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F16] = float16_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_BF16] = bfloat16_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_rvv;
    
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F16] = float16_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_BF16] = bfloat16_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_rvv;
    
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F16] = float16_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_BF16] = bfloat16_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_rvv;
    
    // dispatch_distance_table[VECTOR_DISTANCE_HAMMING][VECTOR_TYPE_BIT] = bit1_distance_hamming_rvv;
    
    distance_backend_name = "RVV";
#endif
}
