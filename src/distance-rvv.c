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

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern const char *distance_backend_name;

// MARK: - FLOAT32 -

float float32_distance_l2_rvv (const void *v1, const void *v2, int n) {
    panic("float32_distance_l2_rvv: unimplemented");
    return 0.0f;
}

float float32_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    panic("float32_distance_l2_squared_rvv: unimplemented");
    return 0.0f;
}

float float32_distance_l1_rvv (const void *v1, const void *v2, int n) {
    panic("float32_distance_l1_rvv: unimplemented");
    return 0.0f;
}

float float32_distance_dot_rvv (const void *v1, const void *v2, int n) {
    panic("float32_distance_dot_rvv: unimplemented");
    return 0.0f;
}

float float32_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    panic("float32_distance_cosine_rvv: unimplemented");
    return 0.0f;
}


// MARK: - FLOAT16 -

float float16_distance_l2_rvv (const void *v1, const void *v2, int n) {
    panic("float16_distance_l2_rvv: unimplemented");
    return 0.0f;
}

float float16_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    panic("float16_distance_l2_squared_rvv: unimplemented");
    return 0.0f;
}

float float16_distance_l1_rvv (const void *v1, const void *v2, int n) {
    panic("float16_distance_l1_rvv: unimplemented");
    return 0.0f;
}

float float16_distance_dot_rvv (const void *v1, const void *v2, int n) {
    panic("float16_distance_dot_rvv: unimplemented");
    return 0.0f;
}

float float16_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    panic("float16_distance_cosine_rvv: unimplemented");
    return 0.0f;
}

// MARK: - BFLOAT16 -

float bfloat16_distance_l2_rvv (const void *v1, const void *v2, int n) {
    panic("bfloat16_distance_l2_rvv: unimplemented");
    return 0.0f;
}

float bfloat16_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    panic("bfloat16_distance_l2_squared_rvv: unimplemented");
    return 0.0f;
}

float bfloat16_distance_l1_rvv (const void *v1, const void *v2, int n) {
    panic("bfloat16_distance_l1_rvv: unimplemented");
    return 0.0f;
}

float bfloat16_distance_dot_rvv (const void *v1, const void *v2, int n) {
    panic("bfloat16_distance_dot_rvv: unimplemented");
    return 0.0f;
}

float bfloat16_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    panic("bfloat16_distance_cosine_rvv: unimplemented");
    return 0.0f;
}

// MARK: - UINT8 -

float uint8_distance_l2_rvv (const void *v1, const void *v2, int n) {
    panic("uint8_distance_l2_rvv: unimplemented");
    return 0.0f;
}

float uint8_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    panic("uint8_distance_l2_squared_rvv: unimplemented");
    return 0.0f;
}

float uint8_distance_dot_rvv (const void *v1, const void *v2, int n) {
    panic("uint8_distance_dot_rvv: unimplemented");
    return 0.0f;
}

float uint8_distance_l1_rvv (const void *v1, const void *v2, int n) {
    panic("uint8_distance_l1_rvv: unimplemented");
    return 0.0f;
}

float uint8_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    panic("uint8_distance_cosine_rvv: unimplemented");
    return 0.0f;
}

// MARK: - INT8 -

float int8_distance_l2_rvv (const void *v1, const void *v2, int n) {
    panic("int8_distance_l2_rvv: unimplemented");
    return 0.0f;
}

float int8_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    panic("int8_distance_l2_squared_rvv: unimplemented");
    return 0.0f;
}

float int8_distance_dot_rvv (const void *v1, const void *v2, int n) {
    panic("int8_distance_dot_rvv: unimplemented");
    return 0.0f;
}

float int8_distance_l1_rvv (const void *v1, const void *v2, int n) {
    panic("int8_distance_l1_rvv: unimplemented");
    return 0.0f;
}

float int8_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    panic("int8_distance_cosine_rvv: unimplemented");
    return 0.0f;
}

// MARK: - BIT -

float bit1_distance_hamming_rvv (const void *v1, const void *v2, int n) {
    panic("bit1_distance_hamming_rvv: unimplemented");
    return 0.0f;
}

#endif

// MARK: -

void init_distance_functions_rvv (void) {
#if defined(__riscv_vector)
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F32] = float32_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F16] = float16_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F16] = float16_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F16] = float16_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_BF16] = bfloat16_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F16] = float16_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_BF16] = bfloat16_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F16] = float16_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_BF16] = bfloat16_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_HAMMING][VECTOR_TYPE_BIT] = bit1_distance_hamming_rvv;
    
    distance_backend_name = "RVV";
#endif
}
