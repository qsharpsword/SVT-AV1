/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbTransformUnit_h
#define EbTransformUnit_h

#include "EbDefinitions.h"
#ifdef __cplusplus
extern "C" {
#endif
#if ENABLE_PR_1133
#define TRANSFORM_UNIT_MAX_COUNT 16
#else
#if 1//TXS_DEPTH_2
#define TRANSFORM_UNIT_MAX_COUNT 21
#else
#define TRANSFORM_UNIT_MAX_COUNT 5
#endif
#define TRANSFORM_UNIT_2Nx2N_AREA 16
#define TRANSFORM_UNIT_NxN_AREA 4
#endif
#pragma pack(push, 1)
typedef struct TransformUnit {
#if !ENABLE_PR_1133
    unsigned txb_index : 5;
    unsigned split_flag : 1;
    unsigned u_has_coeff : 1;
    unsigned v_has_coeff : 1;
    unsigned y_has_coeff : 1;
#endif
    uint16_t nz_coef_count[3];
#if !ENABLE_PR_1133
    EbBool is_only_dc[3];
#endif
    TxType transform_type[PLANE_TYPES];
} TransformUnit;
#pragma pack(pop)
#ifdef __cplusplus
}
#endif
#endif // EbTransformUnit_h
