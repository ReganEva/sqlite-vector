//
//  distance-rvv.c
//  sqlitevector
//
//  Created by Afonso Bordado on 2026/02/19.
//

#include "distance-rvv.h"
#include "distance-cpu.h"

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern const char *distance_backend_name;

// MARK: - UTILS -

// Reduces a vector by summing all of it's elements into a single scalar float
float float32_sum_vector_f32m8(vfloat32m8_t vec, size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vl = __riscv_vsetvl_e32m8(vl);
    acc = __riscv_vfredusum_vs_f32m8_f32m1(vec, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

// Reduces a vector by summing all of it's elements into a single scalar float
float float32_sum_vector_f32m4(vfloat32m4_t vec, size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vl = __riscv_vsetvl_e32m4(vl);
    acc = __riscv_vfredusum_vs_f32m4_f32m1(vec, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

// MARK: - FLOAT32 -

float float32_distance_l2_impl_rvv (const void *v1, const void *v2, int n, bool use_sqrt) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vl2 = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        vl = __riscv_vsetvl_e32m8(n);

        // Load the vectors into the registers
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);

        // L2 = (a[i] - b[i]) + acc
        vfloat32m8_t vdiff = __riscv_vfsub_vv_f32m8(va, vb, vl);
        vl2 = __riscv_vfmacc_vv_f32m8(vl2, vdiff, vdiff, vl);
        
        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l2 = float32_sum_vector_f32m8(vl2, n);
    return use_sqrt ? sqrtf(l2) : l2;
}

float float32_distance_l2_rvv (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_rvv(v1, v2, n, true);
}


float float32_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_rvv(v1, v2, n, false);
}

float float32_distance_l1_rvv (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vsad = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        vl = __riscv_vsetvl_e32m8(n);

        // Load the vectors into the registers
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);


        // SAD = abs(a[i] - b[i]) + acc
        vfloat32m8_t vdiff = __riscv_vfsub_vv_f32m8(va, vb, vl);
        vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(vdiff, vl);
        vsad = __riscv_vfadd_vv_f32m8(vsad, vabs, vl);
        
        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    return float32_sum_vector_f32m8(vsad, n);
}

float float32_distance_dot_rvv (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vdot = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        vl = __riscv_vsetvl_e32m8(i);

        // Load the vectors into the registers
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);

        // Compute the dot product for the entire register, and sum the
        // results into the accumuating register
        vdot = __riscv_vfmacc_vv_f32m8(vdot, va, vb, vl);
        
        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float dot = float32_sum_vector_f32m8(vdot, n);
    return -dot;
}

float float32_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // Use LMUL=4, we have 8 registers to work with.
    size_t vl = __riscv_vsetvl_e32m4(n);

    // Zero out the starting registers
    vfloat32m4_t vdot = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vmagn_a = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vmagn_b = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Update VL with the remaining elements
        vl = __riscv_vsetvl_e32m4(i);

        // Load the vectors into the registers
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b, vl);

        // Compute the dot product for the entire register
        vdot = __riscv_vfmacc_vv_f32m4(vdot, va, vb, vl);

        // Also calculate the magnitude value for both a and b
        vmagn_a = __riscv_vfmacc_vv_f32m4(vmagn_a, va, va, vl);
        vmagn_b = __riscv_vfmacc_vv_f32m4(vmagn_b, vb, vb, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Now do a final reduction on the registers to sum the remaining elements
    // TODO: With default flags this does not always use the fsqrt.s/fmin.s/fmax.s instruction, we should fix that
    float dot = float32_sum_vector_f32m4(vdot, n);
    float magn_a = sqrtf(float32_sum_vector_f32m4(vmagn_a, n));
    float magn_b = sqrtf(float32_sum_vector_f32m4(vmagn_b, n));

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
#if defined(__riscv_v_intrinsic)
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F32] = float32_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F16] = float16_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F16] = float16_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F16] = float16_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_BF16] = bfloat16_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F16] = float16_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_BF16] = bfloat16_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F16] = float16_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_BF16] = bfloat16_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_rvv;
    
    // dispatch_distance_table[VECTOR_DISTANCE_HAMMING][VECTOR_TYPE_BIT] = bit1_distance_hamming_rvv;
    
    distance_backend_name = "RVV";
#endif
}
