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

// Reduces a vector by summing all of it's elements into a single scalar integer
uint64_t uint64_sum_vector_u64m8(vuint64m8_t vec, size_t vl) {
    vuint64m1_t acc = __riscv_vmv_s_x_u64m1(0, 1);
    vl = __riscv_vsetvl_e64m8(vl);
    acc = __riscv_vredsum_vs_u64m8_u64m1(vec, acc, vl);
    return __riscv_vmv_x_s_u64m1_u64(acc);
}

// Reduces a vector by summing all of it's elements into a single scalar integer
uint32_t uint32_sum_vector_u32m8(vuint32m8_t vec, size_t vl) {
    vuint32m1_t acc = __riscv_vmv_s_x_u32m1(0, 1);
    vl = __riscv_vsetvl_e32m8(vl);
    acc = __riscv_vredsum_vs_u32m8_u32m1(vec, acc, vl);
    return __riscv_vmv_x_s_u32m1_u32(acc);
}

// Reduces a vector by summing all of it's elements into a single scalar integer
int32_t int32_sum_vector_i32m8(vint32m8_t vec, size_t vl) {
    vint32m1_t acc = __riscv_vmv_s_x_i32m1(0, 1);
    vl = __riscv_vsetvl_e32m8(vl);
    acc = __riscv_vredsum_vs_i32m8_i32m1(vec, acc, vl);
    return __riscv_vmv_x_s_i32m1_i32(acc);
}


// MARK: - FLOAT32 -

float float32_distance_l2_impl_rvv (const void *v1, const void *v2, int n, bool use_sqrt) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
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
    size_t vl = __riscv_vsetvlmax_e32m8();
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
    size_t vl = __riscv_vsetvlmax_e32m8();
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
    size_t vl = __riscv_vsetvlmax_e32m4();

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

float uint8_distance_l2_impl_rvv (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vint32m8_t vl2 = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vuint8m2_t va = __riscv_vle8_v_u8m2(a, vl);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(b, vl);

        // Widen these values to 16bit unsigned
        vuint16m4_t va_wide = __riscv_vwcvtu_x_x_v_u16m4(va, vl);
        vuint16m4_t vb_wide = __riscv_vwcvtu_x_x_v_u16m4(vb, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Cast these to signed values
        vint16m4_t va_wides = __riscv_vreinterpret_v_u16m4_i16m4(va_wide);
        vint16m4_t vb_wides = __riscv_vreinterpret_v_u16m4_i16m4(vb_wide);

        // L2 = (a[i] - b[i]) + acc
        // The subtract is signed, but the accumulate is unsigned
        vint32m8_t vdiff = __riscv_vwsub_vv_i32m8(va_wides, vb_wides, vl);
        vl2 = __riscv_vmacc_vv_i32m8(vl2, vdiff, vdiff, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l2 = (float) int32_sum_vector_i32m8(vl2, n);
    return use_sqrt ? sqrtf(l2) : l2;
}

float uint8_distance_l2_rvv (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_rvv(v1, v2, n, true);
}

float uint8_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_rvv(v1, v2, n, false);
}

float uint8_distance_dot_rvv (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vuint32m8_t vdot = __riscv_vmv_s_x_u32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vuint8m2_t va = __riscv_vle8_v_u8m2(a, vl);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(b, vl);

        // Widen these vectors to 16bit
        vuint16m4_t va_wide = __riscv_vwcvtu_x_x_v_u16m4(va, vl);
        vuint16m4_t vb_wide = __riscv_vwcvtu_x_x_v_u16m4(vb, vl);

        // Now we're operating on 16 bit elements
        vl = __riscv_vsetvl_e16m4(i);

        // Do a widening multiply-accumulate to 32 bits
        vdot = __riscv_vwmaccu_vv_u32m8(vdot, va_wide, vb_wide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float dot = uint32_sum_vector_u32m8(vdot, n);
    return -dot;
}

float uint8_distance_l1_rvv (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vuint32m8_t vl1 = __riscv_vmv_s_x_u32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vuint8m2_t va = __riscv_vle8_v_u8m2(a, vl);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(b, vl);

        // Compute the absolute difference by getting the min and max and subtracting them.
        vuint8m2_t vmin = __riscv_vminu_vv_u8m2(va, vb, vl);
        vuint8m2_t vmax = __riscv_vmaxu_vv_u8m2(va, vb, vl);
        vuint16m4_t vabs = __riscv_vwsubu_vv_u16m4(vmax, vmin, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Now widen it to 32bits and add to the accumulator
        vuint32m8_t vwide = __riscv_vwcvtu_x_x_v_u32m8(vabs, vl);
        vl1 = __riscv_vadd_vv_u32m8(vl1, vwide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l1 = uint32_sum_vector_u32m8(vl1, n);
    return l1;
}

float uint8_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();

    // Zero out the starting registers
    vuint32m8_t vdot = __riscv_vmv_s_x_u32m8(0, vl);
    vuint32m8_t vmagn_a = __riscv_vmv_s_x_u32m8(0, vl);
    vuint32m8_t vmagn_b = __riscv_vmv_s_x_u32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vuint8m2_t va = __riscv_vle8_v_u8m2(a, vl);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(b, vl);

        // Widen these values to 16bit unsigned
        vuint16m4_t va_wide = __riscv_vwcvtu_x_x_v_u16m4(va, vl);
        vuint16m4_t vb_wide = __riscv_vwcvtu_x_x_v_u16m4(vb, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Compute the dot product for the entire register (widening madd)
        vdot = __riscv_vwmaccu_vv_u32m8(vdot, va_wide, vb_wide, vl);

        // Also calculate the magnitude value for both a and b (widening madd)
        vmagn_a = __riscv_vwmaccu_vv_u32m8(vmagn_a, va_wide, va_wide, vl);
        vmagn_b = __riscv_vwmaccu_vv_u32m8(vmagn_b, vb_wide, vb_wide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Now do a final reduction on the registers to sum the remaining elements
    // TODO: With default flags this does not always use the fsqrt.s/fmin.s/fmax.s instruction, we should fix that
    float dot = uint32_sum_vector_u32m8(vdot, n);
    float magn_a = sqrtf(uint32_sum_vector_u32m8(vmagn_a, n));
    float magn_b = sqrtf(uint32_sum_vector_u32m8(vmagn_b, n));

    if (magn_a == 0.0f || magn_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (magn_a * magn_b);
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

// MARK: - INT8 -

float int8_distance_l2_impl_rvv (const void *v1, const void *v2, int n, bool use_sqrt) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vint32m8_t vl2 = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);

        // Widen these values to 16bit signed
        vint16m4_t va_wide = __riscv_vwcvt_x_x_v_i16m4(va, vl);
        vint16m4_t vb_wide = __riscv_vwcvt_x_x_v_i16m4(vb, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // L2 = (a[i] - b[i]) + acc
        vint32m8_t vdiff = __riscv_vwsub_vv_i32m8(va_wide, vb_wide, vl);
        vl2 = __riscv_vmacc_vv_i32m8(vl2, vdiff, vdiff, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l2 = (float) int32_sum_vector_i32m8(vl2, n);
    return use_sqrt ? sqrtf(l2) : l2;
}

float int8_distance_l2_rvv (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_rvv(v1, v2, n, true);
}

float int8_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_rvv(v1, v2, n, false);
}

float int8_distance_dot_rvv (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vint32m8_t vdot = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);

        // Widen these vectors to 16bit
        vint16m4_t va_wide = __riscv_vwcvt_x_x_v_i16m4(va, vl);
        vint16m4_t vb_wide = __riscv_vwcvt_x_x_v_i16m4(vb, vl);

        // Now we're operating on 16 bit elements
        vl = __riscv_vsetvl_e16m4(i);

        // Do a widening multiply-accumulate to 32 bits
        vdot = __riscv_vwmacc_vv_i32m8(vdot, va_wide, vb_wide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float dot = (float) int32_sum_vector_i32m8(vdot, n);
    return -dot;
}

float int8_distance_l1_rvv (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vint32m8_t vl1 = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);

        // Compute the absolute difference by getting the min and max and subtracting them.
        vint8m2_t vmin = __riscv_vmin_vv_i8m2(va, vb, vl);
        vint8m2_t vmax = __riscv_vmax_vv_i8m2(va, vb, vl);
        vint16m4_t vabs = __riscv_vwsub_vv_i16m4(vmax, vmin, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Now widen it to 32bits and add to the accumulator
        vint32m8_t vwide = __riscv_vwcvt_x_x_v_i32m8(vabs, vl);
        vl1 = __riscv_vadd_vv_i32m8(vl1, vwide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l1 = (float) int32_sum_vector_i32m8(vl1, n);
    return l1;
}

float int8_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();

    // Zero out the starting registers
    vint32m8_t vdot = __riscv_vmv_s_x_i32m8(0, vl);
    vint32m8_t vmagn_a = __riscv_vmv_s_x_i32m8(0, vl);
    vint32m8_t vmagn_b = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);

        // Widen these values to 16bit signed
        vint16m4_t va_wide = __riscv_vwcvt_x_x_v_i16m4(va, vl);
        vint16m4_t vb_wide = __riscv_vwcvt_x_x_v_i16m4(vb, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Compute the dot product for the entire register (widening madd)
        vdot = __riscv_vwmacc_vv_i32m8(vdot, va_wide, vb_wide, vl);

        // Also calculate the magnitude value for both a and b (widening madd)
        vmagn_a = __riscv_vwmacc_vv_i32m8(vmagn_a, va_wide, va_wide, vl);
        vmagn_b = __riscv_vwmacc_vv_i32m8(vmagn_b, vb_wide, vb_wide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Now do a final reduction on the registers to sum the remaining elements
    float dot = (float) int32_sum_vector_i32m8(vdot, n);
    float magn_a = sqrtf((float) int32_sum_vector_i32m8(vmagn_a, n));
    float magn_b = sqrtf((float) int32_sum_vector_i32m8(vmagn_b, n));

    if (magn_a == 0.0f || magn_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (magn_a * magn_b);
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

// MARK: - BIT -


// Counts the number of set bits on each element of a vector register
//
// TODO: RISC-V natively supports vcpop.v for population count, but only with the
// Zvbb extension, which we don't support yet. For everyone else, do a fallback implemetation.
vuint64m8_t vpopcnt_u64m8(vuint64m8_t v, size_t vl) {
    // v = v - ((v >> 1) & 0x5555555555555555ULL);
    vuint64m8_t shr1 = __riscv_vsrl_vx_u64m8(v, 1, vl);
    vuint64m8_t and1 = __riscv_vand_vx_u64m8(shr1, 0x5555555555555555ULL, vl);
    v = __riscv_vsub_vv_u64m8(v, and1, vl);
    
    // v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    vuint64m8_t shr2 = __riscv_vsrl_vx_u64m8(v, 2, vl);
    vuint64m8_t and2 = __riscv_vand_vx_u64m8(shr2, 0x3333333333333333ULL, vl);
    vuint64m8_t and3 = __riscv_vand_vx_u64m8(v, 0x3333333333333333ULL, vl);
    v = __riscv_vadd_vv_u64m8(and2, and3, vl);

    // v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    vuint64m8_t shr4 = __riscv_vsrl_vx_u64m8(v, 4, vl);
    vuint64m8_t add = __riscv_vadd_vv_u64m8(v, shr4, vl);
    v = __riscv_vand_vx_u64m8(add, 0x0f0f0f0f0f0f0f0fULL, vl);

    // v = (v * 0x0101010101010101ULL) >> 56;
    vuint64m8_t mul = __riscv_vmul_vx_u64m8(v, 0x0101010101010101ULL, vl);
    v = __riscv_vsrl_vx_u64m8(mul, 56, vl);

    return v;
}

float bit1_distance_hamming_rvv (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e64m8();
    vuint64m8_t vdistance = __riscv_vmv_s_x_u64m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        vl = __riscv_vsetvl_e64m8(n);

        // Load the vectors into the registers and cast them into a u64 inplace
        vuint64m8_t va = __riscv_vreinterpret_v_u8m8_u64m8(__riscv_vle8_v_u8m8(a, vl));
        vuint64m8_t vb = __riscv_vreinterpret_v_u8m8_u64m8(__riscv_vle8_v_u8m8(b, vl));

        vuint64m8_t xor = __riscv_vxor_vv_u64m8(va, vb, vl);
        vuint64m8_t popcnt = vpopcnt_u64m8(xor, vl);
        vdistance = __riscv_vadd_vv_u64m8(vdistance, popcnt, vl);

        // Advance the a and b pointers to the next offset. Here we multiply by 8 because
        // the vectors are defined as u8, but VL is defined in elements of 64bits.
        a = &a[vl * 8];
        b = &b[vl * 8];
    }

    // Copy the accumulator back into a scalar register
    return (float) uint64_sum_vector_u64m8(vdistance, vl);
}
#endif

// MARK: -

void init_distance_functions_rvv (void) {
#if defined(__riscv_v_intrinsic)
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F32] = float32_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F16] = float16_distance_l2_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F16] = float16_distance_l2_squared_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F16] = float16_distance_cosine_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_BF16] = bfloat16_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F16] = float16_distance_dot_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_BF16] = bfloat16_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F16] = float16_distance_l1_rvv;
    // dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_BF16] = bfloat16_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_HAMMING][VECTOR_TYPE_BIT] = bit1_distance_hamming_rvv;
    
    distance_backend_name = "RVV";
#endif
}
