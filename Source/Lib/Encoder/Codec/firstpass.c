/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <limits.h>
#include <math.h>
#include <stdio.h>

#include "aom_dsp_rtcd.h"
#include "EbDefinitions.h"
#include "EbRateControlProcess.h"
#include "EbSequenceControlSet.h"
#include "EbPictureControlSet.h"
#include "firstpass.h"
#include "EbLog.h"
#if FIRST_PASS_SETUP
#include "EbModeDecisionProcess.h"
#include "EbCodingLoop.h"
#include "dwt.h" // to move to firstpass.c
#include "EbPictureDecisionProcess.h"
#include "EbModeDecisionConfigurationProcess.h"
#endif
#if TWOPASS_RC
#if 0
#include "config/aom_scale_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/variance.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"
#include "aom_ports/system_state.h"
#include "aom_scale/aom_scale.h"
#include "aom_scale/yv12config.h"

#include "av1/common/entropymv.h"
#include "av1/common/quant_common.h"
#include "av1/common/reconinter.h"  // av1_setup_dst_planes()
#include "av1/common/txb_common.h"
#include "av1/encoder/aq_variance.h"
#include "av1/encoder/av1_quantize.h"
#include "av1/encoder/block.h"
#include "av1/encoder/dwt.h"
#include "av1/encoder/encodeframe.h"
#include "av1/encoder/encodemb.h"
#include "av1/encoder/encodemv.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/encode_strategy.h"
#include "av1/encoder/ethread.h"
#include "av1/encoder/extend.h"
#include "av1/encoder/firstpass.h"
#include "av1/encoder/mcomp.h"
#include "av1/encoder/rd.h"
#include "av1/encoder/reconinter_enc.h"
#endif

#define OUTPUT_FPF 0

#define FIRST_PASS_Q 10.0
#define INTRA_MODE_PENALTY 1024
#define NEW_MV_MODE_PENALTY 32
#define DARK_THRESH 64

#define NCOUNT_INTRA_THRESH 8192
#define NCOUNT_INTRA_FACTOR 3
#if !FIRST_PASS_SETUP
static INLINE int32_t frame_is_intra_only(PictureParentControlSet *pcs_ptr) {
    return pcs_ptr->frm_hdr.frame_type == KEY_FRAME ||
           pcs_ptr->frm_hdr.frame_type == INTRA_ONLY_FRAME;
}
#endif
#if 1
static AOM_INLINE void output_stats(SequenceControlSet *scs_ptr, FIRSTPASS_STATS *stats, uint64_t frame_number) {
    eb_block_on_mutex(scs_ptr->encode_context_ptr->stat_file_mutex);
    int32_t fseek_return_value = fseek(
        scs_ptr->static_config.output_stat_file, (long)frame_number * sizeof(FIRSTPASS_STATS), SEEK_SET);
    if (fseek_return_value != 0) SVT_LOG("Error in fseek  returnVal %i\n", fseek_return_value);
    fwrite(stats, sizeof(FIRSTPASS_STATS), (size_t)1, scs_ptr->static_config.output_stat_file);
// TEMP debug code
#if OUTPUT_FPF
  {
    FILE *fpfile;
    if (frame_number == 0)
        fpfile = fopen("firstpass.stt", "w");
    else
        fpfile = fopen("firstpass.stt", "a");
    fprintf(fpfile,
            "%12.0lf %12.4lf %12.0lf %12.0lf %12.0lf %12.4lf %12.4lf"
            "%12.4lf %12.4lf %12.4lf %12.4lf %12.4lf %12.4lf %12.4lf %12.4lf"
            "%12.4lf %12.4lf %12.0lf %12.4lf %12.4lf %12.4lf %12.4lf\n",
            stats->frame, stats->weight, stats->intra_error, stats->coded_error,
            stats->sr_coded_error, stats->pcnt_inter, stats->pcnt_motion,
            stats->pcnt_second_ref, stats->pcnt_neutral, stats->intra_skip_pct,
            stats->inactive_zone_rows, stats->inactive_zone_cols, stats->MVr,
            stats->mvr_abs, stats->MVc, stats->mvc_abs, stats->MVrv,
            stats->MVcv, stats->mv_in_out_count, stats->new_mv_count,
            stats->count, stats->duration);
    fclose(fpfile);
  }
#endif
    eb_release_mutex(scs_ptr->encode_context_ptr->stat_file_mutex);
}
#endif
#if TWOPASS_STAT_BUF
void av1_twopass_zero_stats(FIRSTPASS_STATS *section) {
  section->frame = 0.0;
  section->weight = 0.0;
  section->intra_error = 0.0;
  section->frame_avg_wavelet_energy = 0.0;
  section->coded_error = 0.0;
  section->sr_coded_error = 0.0;
  section->pcnt_inter = 0.0;
  section->pcnt_motion = 0.0;
  section->pcnt_second_ref = 0.0;
  section->pcnt_neutral = 0.0;
  section->intra_skip_pct = 0.0;
  section->inactive_zone_rows = 0.0;
  section->inactive_zone_cols = 0.0;
  section->MVr = 0.0;
  section->mvr_abs = 0.0;
  section->MVc = 0.0;
  section->mvc_abs = 0.0;
  section->MVrv = 0.0;
  section->MVcv = 0.0;
  section->mv_in_out_count = 0.0;
  section->new_mv_count = 0.0;
  section->count = 0.0;
  section->duration = 1.0;
}
#endif
void av1_accumulate_stats(FIRSTPASS_STATS *section,
                          const FIRSTPASS_STATS *frame) {
  section->frame += frame->frame;
  section->weight += frame->weight;
  section->intra_error += frame->intra_error;
  section->frame_avg_wavelet_energy += frame->frame_avg_wavelet_energy;
  section->coded_error += frame->coded_error;
  section->sr_coded_error += frame->sr_coded_error;
  section->pcnt_inter += frame->pcnt_inter;
  section->pcnt_motion += frame->pcnt_motion;
  section->pcnt_second_ref += frame->pcnt_second_ref;
  section->pcnt_neutral += frame->pcnt_neutral;
  section->intra_skip_pct += frame->intra_skip_pct;
  section->inactive_zone_rows += frame->inactive_zone_rows;
  section->inactive_zone_cols += frame->inactive_zone_cols;
  section->MVr += frame->MVr;
  section->mvr_abs += frame->mvr_abs;
  section->MVc += frame->MVc;
  section->mvc_abs += frame->mvc_abs;
  section->MVrv += frame->MVrv;
  section->MVcv += frame->MVcv;
  section->mv_in_out_count += frame->mv_in_out_count;
  section->new_mv_count += frame->new_mv_count;
  section->count += frame->count;
  section->duration += frame->duration;
}
void av1_end_first_pass(PictureParentControlSet *pcs_ptr) {
    SequenceControlSet *scs_ptr = pcs_ptr->scs_ptr;
    TWO_PASS *twopass = &scs_ptr->twopass;

    if (twopass->stats_buf_ctx->total_stats)
        // add the total to the end of the file
        output_stats(scs_ptr, twopass->stats_buf_ctx->total_stats, pcs_ptr->picture_number + 1);
}
#if 0
static aom_variance_fn_t get_block_variance_fn(BLOCK_SIZE bsize) {
  switch (bsize) {
    case BLOCK_8X8: return aom_mse8x8;
    case BLOCK_16X8: return aom_mse16x8;
    case BLOCK_8X16: return aom_mse8x16;
    default: return aom_mse16x16;
  }
}

static unsigned int get_prediction_error(BLOCK_SIZE bsize,
                                         const struct buf_2d *src,
                                         const struct buf_2d *ref) {
  unsigned int sse;
  const aom_variance_fn_t fn = get_block_variance_fn(bsize);
  fn(src->buf, src->stride, ref->buf, ref->stride, &sse);
  return sse;
}

#if CONFIG_AV1_HIGHBITDEPTH
static aom_variance_fn_t highbd_get_block_variance_fn(BLOCK_SIZE bsize,
                                                      int bd) {
  switch (bd) {
    default:
      switch (bsize) {
        case BLOCK_8X8: return aom_highbd_8_mse8x8;
        case BLOCK_16X8: return aom_highbd_8_mse16x8;
        case BLOCK_8X16: return aom_highbd_8_mse8x16;
        default: return aom_highbd_8_mse16x16;
      }
      break;
    case 10:
      switch (bsize) {
        case BLOCK_8X8: return aom_highbd_10_mse8x8;
        case BLOCK_16X8: return aom_highbd_10_mse16x8;
        case BLOCK_8X16: return aom_highbd_10_mse8x16;
        default: return aom_highbd_10_mse16x16;
      }
      break;
    case 12:
      switch (bsize) {
        case BLOCK_8X8: return aom_highbd_12_mse8x8;
        case BLOCK_16X8: return aom_highbd_12_mse16x8;
        case BLOCK_8X16: return aom_highbd_12_mse8x16;
        default: return aom_highbd_12_mse16x16;
      }
      break;
  }
}

static unsigned int highbd_get_prediction_error(BLOCK_SIZE bsize,
                                                const struct buf_2d *src,
                                                const struct buf_2d *ref,
                                                int bd) {
  unsigned int sse;
  const aom_variance_fn_t fn = highbd_get_block_variance_fn(bsize, bd);
  fn(src->buf, src->stride, ref->buf, ref->stride, &sse);
  return sse;
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

// Refine the motion search range according to the frame dimension
// for first pass test.
static int get_search_range(const InitialDimensions *initial_dimensions) {
  int sr = 0;
  const int dim = AOMMIN(initial_dimensions->width, initial_dimensions->height);

  while ((dim << sr) < MAX_FULL_PEL_VAL) ++sr;
  return sr;
}

static AOM_INLINE void first_pass_motion_search(AV1_COMP *cpi, MACROBLOCK *x,
                                                const MV *ref_mv,
                                                FULLPEL_MV *best_mv,
                                                int *best_motion_err) {
  MACROBLOCKD *const xd = &x->e_mbd;
  FULLPEL_MV start_mv = get_fullmv_from_mv(ref_mv);
  int tmp_err;
  const BLOCK_SIZE bsize = xd->mi[0]->sb_type;
  const int new_mv_mode_penalty = NEW_MV_MODE_PENALTY;
  const int sr = get_search_range(&cpi->initial_dimensions);
  const int step_param = 3 + sr;

  const search_site_config *first_pass_search_sites =
      &cpi->mv_search_params.search_site_cfg[SS_CFG_FPF];
  FULLPEL_MOTION_SEARCH_PARAMS ms_params;
  av1_make_default_fullpel_ms_params(&ms_params, cpi, x, bsize, ref_mv,
                                     first_pass_search_sites,
                                     /*fine_search_interval=*/0);
  ms_params.search_method = NSTEP;

  FULLPEL_MV this_best_mv;
  tmp_err = av1_full_pixel_search(start_mv, &ms_params, step_param, NULL,
                                  &this_best_mv, NULL);

  if (tmp_err < INT_MAX) {
    aom_variance_fn_ptr_t v_fn_ptr = cpi->fn_ptr[bsize];
    const MSBuffers *ms_buffers = &ms_params.ms_buffers;
    tmp_err = av1_get_mvpred_sse(&ms_params.mv_cost_params, this_best_mv,
                                 &v_fn_ptr, ms_buffers->src, ms_buffers->ref) +
              new_mv_mode_penalty;
  }

  if (tmp_err < *best_motion_err) {
    *best_motion_err = tmp_err;
    *best_mv = this_best_mv;
  }
}

static BLOCK_SIZE get_bsize(const CommonModeInfoParams *const mi_params,
                            int mb_row, int mb_col) {
  if (mi_size_wide[BLOCK_16X16] * mb_col + mi_size_wide[BLOCK_8X8] <
      mi_params->mi_cols) {
    return mi_size_wide[BLOCK_16X16] * mb_row + mi_size_wide[BLOCK_8X8] <
                   mi_params->mi_rows
               ? BLOCK_16X16
               : BLOCK_16X8;
  } else {
    return mi_size_wide[BLOCK_16X16] * mb_row + mi_size_wide[BLOCK_8X8] <
                   mi_params->mi_rows
               ? BLOCK_8X16
               : BLOCK_8X8;
  }
}

static int find_fp_qindex(aom_bit_depth_t bit_depth) {
  aom_clear_system_state();
  return av1_find_qindex(FIRST_PASS_Q, bit_depth, 0, QINDEX_RANGE - 1);
}
#endif

static double raw_motion_error_stdev(int *raw_motion_err_list,
                                     int raw_motion_err_counts) {
  int64_t sum_raw_err = 0;
  double raw_err_avg = 0;
  double raw_err_stdev = 0;
  if (raw_motion_err_counts == 0) return 0;

  int i;
  for (i = 0; i < raw_motion_err_counts; i++) {
    sum_raw_err += raw_motion_err_list[i];
  }
  raw_err_avg = (double)sum_raw_err / raw_motion_err_counts;
  for (i = 0; i < raw_motion_err_counts; i++) {
    raw_err_stdev += (raw_motion_err_list[i] - raw_err_avg) *
                     (raw_motion_err_list[i] - raw_err_avg);
  }
  // Calculate the standard deviation for the motion error of all the inter
  // blocks of the 0,0 motion using the last source
  // frame as the reference.
  raw_err_stdev = sqrt(raw_err_stdev / raw_motion_err_counts);
  return raw_err_stdev;
}
#define UL_INTRA_THRESH 50
#define INVALID_ROW -1
#if 0
// Computes and returns the intra pred error of a block.
// intra pred error: sum of squared error of the intra predicted residual.
// Inputs:
//   cpi: the encoder setting. Only a few params in it will be used.
//   this_frame: the current frame buffer.
//   tile: tile information (not used in first pass, already init to zero)
//   mb_row: row index in the unit of first pass block size.
//   mb_col: column index in the unit of first pass block size.
//   y_offset: the offset of y frame buffer, indicating the starting point of
//             the current block.
//   uv_offset: the offset of u and v frame buffer, indicating the starting
//              point of the current block.
//   fp_block_size: first pass block size.
//   qindex: quantization step size to encode the frame.
//   stats: frame encoding stats.
// Modifies:
//   stats->intra_skip_count
//   stats->image_data_start_row
//   stats->intra_factor
//   stats->brightness_factor
//   stats->intra_error
//   stats->frame_avg_wavelet_energy
// Returns:
//   this_intra_error.
static int firstpass_intra_prediction(
    AV1_COMP *cpi, ThreadData *td, YV12_BUFFER_CONFIG *const this_frame,
    const TileInfo *const tile, const int mb_row, const int mb_col,
    const int y_offset, const int uv_offset, const BLOCK_SIZE fp_block_size,
    const int qindex, FRAME_STATS *const stats) {
  const AV1_COMMON *const cm = &cpi->common;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const SequenceHeader *const seq_params = &cm->seq_params;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int mb_scale = mi_size_wide[fp_block_size];
  const int use_dc_pred = (mb_col || mb_row) && (!mb_col || !mb_row);
  const int num_planes = av1_num_planes(cm);
  const BLOCK_SIZE bsize = get_bsize(mi_params, mb_row, mb_col);

  aom_clear_system_state();
  set_mi_offsets(mi_params, xd, mb_row * mb_scale, mb_col * mb_scale);
  xd->plane[0].dst.buf = this_frame->y_buffer + y_offset;
  xd->plane[1].dst.buf = this_frame->u_buffer + uv_offset;
  xd->plane[2].dst.buf = this_frame->v_buffer + uv_offset;
  xd->left_available = (mb_col != 0);
  xd->mi[0]->sb_type = bsize;
  xd->mi[0]->ref_frame[0] = INTRA_FRAME;
  set_mi_row_col(xd, tile, mb_row * mb_scale, mi_size_high[bsize],
                 mb_col * mb_scale, mi_size_wide[bsize], mi_params->mi_rows,
                 mi_params->mi_cols);
  set_plane_n4(xd, mi_size_wide[bsize], mi_size_high[bsize], num_planes);
  xd->mi[0]->segment_id = 0;
  xd->lossless[xd->mi[0]->segment_id] = (qindex == 0);
  xd->mi[0]->mode = DC_PRED;
  xd->mi[0]->tx_size =
      use_dc_pred ? (bsize >= fp_block_size ? TX_16X16 : TX_8X8) : TX_4X4;

  av1_encode_intra_block_plane(cpi, x, bsize, 0, DRY_RUN_NORMAL, 0);
  int this_intra_error = aom_get_mb_ss(x->plane[0].src_diff);

  if (this_intra_error < UL_INTRA_THRESH) {
    ++stats->intra_skip_count;
  } else if ((mb_col > 0) && (stats->image_data_start_row == INVALID_ROW)) {
    stats->image_data_start_row = mb_row;
  }

  if (seq_params->use_highbitdepth) {
    switch (seq_params->bit_depth) {
      case AOM_BITS_8: break;
      case AOM_BITS_10: this_intra_error >>= 4; break;
      case AOM_BITS_12: this_intra_error >>= 8; break;
      default:
        assert(0 &&
               "seq_params->bit_depth should be AOM_BITS_8, "
               "AOM_BITS_10 or AOM_BITS_12");
        return -1;
    }
  }

  aom_clear_system_state();
  double log_intra = log(this_intra_error + 1.0);
  if (log_intra < 10.0) {
    stats->intra_factor += 1.0 + ((10.0 - log_intra) * 0.05);
  } else {
    stats->intra_factor += 1.0;
  }

  int level_sample;
  if (seq_params->use_highbitdepth) {
    level_sample = CONVERT_TO_SHORTPTR(x->plane[0].src.buf)[0];
  } else {
    level_sample = x->plane[0].src.buf[0];
  }
  if ((level_sample < DARK_THRESH) && (log_intra < 9.0)) {
    stats->brightness_factor += 1.0 + (0.01 * (DARK_THRESH - level_sample));
  } else {
    stats->brightness_factor += 1.0;
  }

  // Intrapenalty below deals with situations where the intra and inter
  // error scores are very low (e.g. a plain black frame).
  // We do not have special cases in first pass for 0,0 and nearest etc so
  // all inter modes carry an overhead cost estimate for the mv.
  // When the error score is very low this causes us to pick all or lots of
  // INTRA modes and throw lots of key frames.
  // This penalty adds a cost matching that of a 0,0 mv to the intra case.
  this_intra_error += INTRA_MODE_PENALTY;

  // Accumulate the intra error.
  stats->intra_error += (int64_t)this_intra_error;

  const int hbd = is_cur_buf_hbd(xd);
  const int stride = x->plane[0].src.stride;
  uint8_t *buf = x->plane[0].src.buf;
  for (int r8 = 0; r8 < 2; ++r8) {
    for (int c8 = 0; c8 < 2; ++c8) {
      stats->frame_avg_wavelet_energy += av1_haar_ac_sad_8x8_uint8_input(
          buf + c8 * 8 + r8 * 8 * stride, stride, hbd);
    }
  }

  return this_intra_error;
}

// Returns the sum of square error between source and reference blocks.
static int get_prediction_error_bitdepth(const int is_high_bitdepth,
                                         const int bitdepth,
                                         const BLOCK_SIZE block_size,
                                         const struct buf_2d *src,
                                         const struct buf_2d *ref) {
  (void)is_high_bitdepth;
  (void)bitdepth;
#if CONFIG_AV1_HIGHBITDEPTH
  if (is_high_bitdepth) {
    return highbd_get_prediction_error(block_size, src, ref, bitdepth);
  }
#endif  // CONFIG_AV1_HIGHBITDEPTH
  return get_prediction_error(block_size, src, ref);
}
#endif
// Accumulates motion vector stats.
// Modifies member variables of "stats".
/*static*/ void accumulate_mv_stats(const MV best_mv, const FULLPEL_MV mv,
                                const int mb_row, const int mb_col,
                                const int mb_rows, const int mb_cols,
                                MV *last_mv, FRAME_STATS *stats) {
  if (is_zero_mv(&best_mv)) return;

  ++stats->mv_count;
  // Non-zero vector, was it different from the last non zero vector?
  if (!is_equal_mv(&best_mv, last_mv)) ++stats->new_mv_count;
  *last_mv = best_mv;

  // Does the row vector point inwards or outwards?
  if (mb_row < mb_rows / 2) {
    if (mv.row > 0) {
      --stats->sum_in_vectors;
    } else if (mv.row < 0) {
      ++stats->sum_in_vectors;
    }
  } else if (mb_row > mb_rows / 2) {
    if (mv.row > 0) {
      ++stats->sum_in_vectors;
    } else if (mv.row < 0) {
      --stats->sum_in_vectors;
    }
  }

  // Does the col vector point inwards or outwards?
  if (mb_col < mb_cols / 2) {
    if (mv.col > 0) {
      --stats->sum_in_vectors;
    } else if (mv.col < 0) {
      ++stats->sum_in_vectors;
    }
  } else if (mb_col > mb_cols / 2) {
    if (mv.col > 0) {
      ++stats->sum_in_vectors;
    } else if (mv.col < 0) {
      --stats->sum_in_vectors;
    }
  }
}
#if 0
#define LOW_MOTION_ERROR_THRESH 25
// Computes and returns the inter prediction error from the last frame.
// Computes inter prediction errors from the golden and alt ref frams and
// Updates stats accordingly.
// Inputs:
//   cpi: the encoder setting. Only a few params in it will be used.
//   last_frame: the frame buffer of the last frame.
//   golden_frame: the frame buffer of the golden frame.
//   alt_ref_frame: the frame buffer of the alt ref frame.
//   mb_row: row index in the unit of first pass block size.
//   mb_col: column index in the unit of first pass block size.
//   recon_yoffset: the y offset of the reconstructed  frame buffer,
//                  indicating the starting point of the current block.
//   recont_uvoffset: the u/v offset of the reconstructed frame buffer,
//                    indicating the starting point of the current block.
//   src_yoffset: the y offset of the source frame buffer.
//   alt_ref_frame_offset: the y offset of the alt ref frame buffer.
//   fp_block_size: first pass block size.
//   this_intra_error: the intra prediction error of this block.
//   raw_motion_err_counts: the count of raw motion vectors.
//   raw_motion_err_list: the array that records the raw motion error.
//   best_ref_mv: best reference mv found so far.
//   last_mv: last mv.
//   stats: frame encoding stats.
//  Modifies:
//    raw_motion_err_list
//    best_ref_mv
//    last_mv
//    stats: many member params in it.
//  Returns:
//    this_inter_error
static int firstpass_inter_prediction(
    AV1_COMP *cpi, ThreadData *td, const YV12_BUFFER_CONFIG *const last_frame,
    const YV12_BUFFER_CONFIG *const golden_frame,
    const YV12_BUFFER_CONFIG *const alt_ref_frame, const int mb_row,
    const int mb_col, const int recon_yoffset, const int recon_uvoffset,
    const int src_yoffset, const int alt_ref_frame_yoffset,
    const BLOCK_SIZE fp_block_size, const int this_intra_error,
    const int raw_motion_err_counts, int *raw_motion_err_list, MV *best_ref_mv,
    MV *last_mv, FRAME_STATS *stats) {
  int this_inter_error = this_intra_error;
  AV1_COMMON *const cm = &cpi->common;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  CurrentFrame *const current_frame = &cm->current_frame;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  const int is_high_bitdepth = is_cur_buf_hbd(xd);
  const int bitdepth = xd->bd;
  const int mb_scale = mi_size_wide[fp_block_size];
  const BLOCK_SIZE bsize = get_bsize(mi_params, mb_row, mb_col);
  const int fp_block_size_height = block_size_wide[fp_block_size];
  // Assume 0,0 motion with no mv overhead.
  FULLPEL_MV mv = kZeroFullMv;
  FULLPEL_MV tmp_mv = kZeroFullMv;
  xd->plane[0].pre[0].buf = last_frame->y_buffer + recon_yoffset;
  // Set up limit values for motion vectors to prevent them extending
  // outside the UMV borders.
  av1_set_mv_col_limits(mi_params, &x->mv_limits, (mb_col << FP_MIB_SIZE_LOG2),
                        (fp_block_size_height >> MI_SIZE_LOG2),
                        cpi->oxcf.border_in_pixels);

  int motion_error =
      get_prediction_error_bitdepth(is_high_bitdepth, bitdepth, bsize,
                                    &x->plane[0].src, &xd->plane[0].pre[0]);

  // Compute the motion error of the 0,0 motion using the last source
  // frame as the reference. Skip the further motion search on
  // reconstructed frame if this error is small.
  struct buf_2d unscaled_last_source_buf_2d;
  unscaled_last_source_buf_2d.buf =
      cpi->unscaled_last_source->y_buffer + src_yoffset;
  unscaled_last_source_buf_2d.stride = cpi->unscaled_last_source->y_stride;
  const int raw_motion_error = get_prediction_error_bitdepth(
      is_high_bitdepth, bitdepth, bsize, &x->plane[0].src,
      &unscaled_last_source_buf_2d);
  raw_motion_err_list[raw_motion_err_counts] = raw_motion_error;

  // TODO(pengchong): Replace the hard-coded threshold
  if (raw_motion_error > LOW_MOTION_ERROR_THRESH) {
    // Test last reference frame using the previous best mv as the
    // starting point (best reference) for the search.
    first_pass_motion_search(cpi, x, best_ref_mv, &mv, &motion_error);

    // If the current best reference mv is not centered on 0,0 then do a
    // 0,0 based search as well.
    if (!is_zero_mv(best_ref_mv)) {
      int tmp_err = INT_MAX;
      first_pass_motion_search(cpi, x, &kZeroMv, &tmp_mv, &tmp_err);

      if (tmp_err < motion_error) {
        motion_error = tmp_err;
        mv = tmp_mv;
      }
    }

    // Motion search in 2nd reference frame.
    int gf_motion_error = motion_error;
    if ((current_frame->frame_number > 1) && golden_frame != NULL) {
      // Assume 0,0 motion with no mv overhead.
      xd->plane[0].pre[0].buf = golden_frame->y_buffer + recon_yoffset;
      xd->plane[0].pre[0].stride = golden_frame->y_stride;
      gf_motion_error =
          get_prediction_error_bitdepth(is_high_bitdepth, bitdepth, bsize,
                                        &x->plane[0].src, &xd->plane[0].pre[0]);
      first_pass_motion_search(cpi, x, &kZeroMv, &tmp_mv, &gf_motion_error);
    }
    if (gf_motion_error < motion_error && gf_motion_error < this_intra_error) {
      ++stats->second_ref_count;
    }
    // In accumulating a score for the 2nd reference frame take the
    // best of the motion predicted score and the intra coded error
    // (just as will be done for) accumulation of "coded_error" for
    // the last frame.
    if ((current_frame->frame_number > 1) && golden_frame != NULL) {
      stats->sr_coded_error += AOMMIN(gf_motion_error, this_intra_error);
    } else {
      // TODO(chengchen): I believe logically this should also be changed to
      // stats->sr_coded_error += AOMMIN(gf_motion_error, this_intra_error).
      stats->sr_coded_error += motion_error;
    }

    // Motion search in 3rd reference frame.
    int alt_motion_error = motion_error;
    if (alt_ref_frame != NULL) {
      xd->plane[0].pre[0].buf = alt_ref_frame->y_buffer + alt_ref_frame_yoffset;
      xd->plane[0].pre[0].stride = alt_ref_frame->y_stride;
      alt_motion_error =
          get_prediction_error_bitdepth(is_high_bitdepth, bitdepth, bsize,
                                        &x->plane[0].src, &xd->plane[0].pre[0]);
      first_pass_motion_search(cpi, x, &kZeroMv, &tmp_mv, &alt_motion_error);
    }
    if (alt_motion_error < motion_error && alt_motion_error < gf_motion_error &&
        alt_motion_error < this_intra_error) {
      ++stats->third_ref_count;
    }
    // In accumulating a score for the 3rd reference frame take the
    // best of the motion predicted score and the intra coded error
    // (just as will be done for) accumulation of "coded_error" for
    // the last frame.
    if (alt_ref_frame != NULL) {
      stats->tr_coded_error += AOMMIN(alt_motion_error, this_intra_error);
    } else {
      // TODO(chengchen): I believe logically this should also be changed to
      // stats->tr_coded_error += AOMMIN(alt_motion_error, this_intra_error).
      stats->tr_coded_error += motion_error;
    }

    // Reset to last frame as reference buffer.
    xd->plane[0].pre[0].buf = last_frame->y_buffer + recon_yoffset;
    xd->plane[1].pre[0].buf = last_frame->u_buffer + recon_uvoffset;
    xd->plane[2].pre[0].buf = last_frame->v_buffer + recon_uvoffset;
  } else {
    stats->sr_coded_error += motion_error;
    stats->tr_coded_error += motion_error;
  }

  // Start by assuming that intra mode is best.
  best_ref_mv->row = 0;
  best_ref_mv->col = 0;

  if (motion_error <= this_intra_error) {
    aom_clear_system_state();

    // Keep a count of cases where the inter and intra were very close
    // and very low. This helps with scene cut detection for example in
    // cropped clips with black bars at the sides or top and bottom.
    if (((this_intra_error - INTRA_MODE_PENALTY) * 9 <= motion_error * 10) &&
        (this_intra_error < (2 * INTRA_MODE_PENALTY))) {
      stats->neutral_count += 1.0;
      // Also track cases where the intra is not much worse than the inter
      // and use this in limiting the GF/arf group length.
    } else if ((this_intra_error > NCOUNT_INTRA_THRESH) &&
               (this_intra_error < (NCOUNT_INTRA_FACTOR * motion_error))) {
      stats->neutral_count +=
          (double)motion_error / DOUBLE_DIVIDE_CHECK((double)this_intra_error);
    }

    const MV best_mv = get_mv_from_fullmv(&mv);
    this_inter_error = motion_error;
    xd->mi[0]->mode = NEWMV;
    xd->mi[0]->mv[0].as_mv = best_mv;
    xd->mi[0]->tx_size = TX_4X4;
    xd->mi[0]->ref_frame[0] = LAST_FRAME;
    xd->mi[0]->ref_frame[1] = NONE_FRAME;
    av1_enc_build_inter_predictor(cm, xd, mb_row * mb_scale, mb_col * mb_scale,
                                  NULL, bsize, AOM_PLANE_Y, AOM_PLANE_Y);
    av1_encode_sby_pass1(cpi, x, bsize);
    stats->sum_mvr += best_mv.row;
    stats->sum_mvr_abs += abs(best_mv.row);
    stats->sum_mvc += best_mv.col;
    stats->sum_mvc_abs += abs(best_mv.col);
    stats->sum_mvrs += best_mv.row * best_mv.row;
    stats->sum_mvcs += best_mv.col * best_mv.col;
    ++stats->inter_count;

    *best_ref_mv = best_mv;
    accumulate_mv_stats(best_mv, mv, mb_row, mb_col, mi_params->mb_rows,
                        mi_params->mb_cols, last_mv, stats);
  }

  return this_inter_error;
}
#endif
// Updates the first pass stats of this frame.
// Input:
//   cpi: the encoder setting. Only a few params in it will be used.
//   stats: stats accumulated for this frame.
//   raw_err_stdev: the statndard deviation for the motion error of all the
//                  inter blocks of the (0,0) motion using the last source
//                  frame as the reference.
//   frame_number: current frame number.
//   ts_duration: Duration of the frame / collection of frames.
// Updates:
//   twopass->total_stats: the accumulated stats.
//   twopass->stats_buf_ctx->stats_in_end: the pointer to the current stats,
//                                         update its value and its position
//                                         in the buffer.
static void update_firstpass_stats(PictureParentControlSet *pcs_ptr, //AV1_COMP *cpi,
                                   const FRAME_STATS *const stats,
                                   const double raw_err_stdev,
                                   const int frame_number,
                                   const int64_t ts_duration) {
#if TWOPASS_STAT_BUF
    SequenceControlSet *scs_ptr = pcs_ptr->scs_ptr;
    TWO_PASS *twopass = &scs_ptr->twopass;
#else
  TWO_PASS *twopass = &pcs_ptr->twopass;
  SequenceControlSet *scs_ptr = pcs_ptr->scs_ptr;
#endif
  const uint32_t mb_cols = (scs_ptr->seq_header.max_frame_width  + 16 - 1) / 16;
  const uint32_t mb_rows = (scs_ptr->seq_header.max_frame_height + 16 - 1) / 16;
  //AV1_COMMON *const cm = &cpi->common;
  //const CommonModeInfoParams *const mi_params = &cm->mi_params;
  FIRSTPASS_STATS *this_frame_stats = twopass->stats_buf_ctx->stats_in_end;
  FIRSTPASS_STATS fps;
  // The minimum error here insures some bit allocation to frames even
  // in static regions. The allocation per MB declines for larger formats
  // where the typical "real" energy per MB also falls.
  // Initial estimate here uses sqrt(mbs) to define the min_err, where the
  // number of mbs is proportional to the image area.
  const int num_mbs = mb_rows * mb_cols;
                      //(cpi->oxcf.resize_cfg.resize_mode != RESIZE_NONE)
                      //    ? cpi->initial_mbs
                      //    : mi_params->MBs;
  const double min_err = 200 * sqrt(num_mbs);

  fps.weight = stats->intra_factor * stats->brightness_factor;
  fps.frame = frame_number;
  fps.coded_error = (double)(stats->coded_error >> 8) + min_err;
  fps.sr_coded_error = (double)(stats->sr_coded_error >> 8) + min_err;
  fps.tr_coded_error = (double)(stats->tr_coded_error >> 8) + min_err;
  fps.intra_error = (double)(stats->intra_error >> 8) + min_err;
  fps.frame_avg_wavelet_energy = (double)stats->frame_avg_wavelet_energy;
  fps.count = 1.0;
  fps.pcnt_inter = (double)stats->inter_count / num_mbs;
  fps.pcnt_second_ref = (double)stats->second_ref_count / num_mbs;
  fps.pcnt_third_ref = (double)stats->third_ref_count / num_mbs;
  fps.pcnt_neutral = (double)stats->neutral_count / num_mbs;
  fps.intra_skip_pct = (double)stats->intra_skip_count / num_mbs;
  fps.inactive_zone_rows = (double)stats->image_data_start_row;
  fps.inactive_zone_cols = (double)0;  // TODO(paulwilkins): fix
  fps.raw_error_stdev = raw_err_stdev;

  if (stats->mv_count > 0) {
    fps.MVr = (double)stats->sum_mvr / stats->mv_count;
    fps.mvr_abs = (double)stats->sum_mvr_abs / stats->mv_count;
    fps.MVc = (double)stats->sum_mvc / stats->mv_count;
    fps.mvc_abs = (double)stats->sum_mvc_abs / stats->mv_count;
    fps.MVrv = ((double)stats->sum_mvrs -
                ((double)stats->sum_mvr * stats->sum_mvr / stats->mv_count)) /
               stats->mv_count;
    fps.MVcv = ((double)stats->sum_mvcs -
                ((double)stats->sum_mvc * stats->sum_mvc / stats->mv_count)) /
               stats->mv_count;
    fps.mv_in_out_count = (double)stats->sum_in_vectors / (stats->mv_count * 2);
    fps.new_mv_count = stats->new_mv_count;
    fps.pcnt_motion = (double)stats->mv_count / num_mbs;
  } else {
    fps.MVr = 0.0;
    fps.mvr_abs = 0.0;
    fps.MVc = 0.0;
    fps.mvc_abs = 0.0;
    fps.MVrv = 0.0;
    fps.MVcv = 0.0;
    fps.mv_in_out_count = 0.0;
    fps.new_mv_count = 0.0;
    fps.pcnt_motion = 0.0;
  }

  // TODO(paulwilkins):  Handle the case when duration is set to 0, or
  // something less than the full time between subsequent values of
  // cpi->source_time_stamp.
  fps.duration = (double)ts_duration;

  // We will store the stats inside the persistent twopass struct (and NOT the
  // local variable 'fps'), and then cpi->output_pkt_list will point to it.
  *this_frame_stats = fps;
  output_stats(scs_ptr, this_frame_stats, pcs_ptr->picture_number);
  if (twopass->stats_buf_ctx->total_stats != NULL) {
    av1_accumulate_stats(twopass->stats_buf_ctx->total_stats, &fps);
  }
  /*In the case of two pass, first pass uses it as a circular buffer,
   * when LAP is enabled it is used as a linear buffer*/
  twopass->stats_buf_ctx->stats_in_end++;
  if ((scs_ptr->use_output_stat_file) && (twopass->stats_buf_ctx->stats_in_end >=
                                twopass->stats_buf_ctx->stats_in_buf_end)) {
    twopass->stats_buf_ctx->stats_in_end =
        twopass->stats_buf_ctx->stats_in_start;
  }
}

#if 0
static void print_reconstruction_frame(
    const YV12_BUFFER_CONFIG *const last_frame, int frame_number,
    int do_print) {
  if (!do_print) return;

  char filename[512];
  FILE *recon_file;
  snprintf(filename, sizeof(filename), "enc%04d.yuv", frame_number);

  if (frame_number == 0) {
    recon_file = fopen(filename, "wb");
  } else {
    recon_file = fopen(filename, "ab");
  }

  fwrite(last_frame->buffer_alloc, last_frame->frame_size, 1, recon_file);
  fclose(recon_file);
}
#endif

static FRAME_STATS accumulate_frame_stats(FRAME_STATS *mb_stats, int mb_rows,
                                          int mb_cols) {
  FRAME_STATS stats = { 0 };
  int i, j;

  stats.image_data_start_row = INVALID_ROW;
  for (j = 0; j < mb_rows; j++) {
    for (i = 0; i < mb_cols; i++) {
      FRAME_STATS mb_stat = mb_stats[j * mb_cols + i];
      stats.brightness_factor += mb_stat.brightness_factor;
      stats.coded_error += mb_stat.coded_error;
      stats.frame_avg_wavelet_energy += mb_stat.frame_avg_wavelet_energy;
      if (stats.image_data_start_row == INVALID_ROW &&
          mb_stat.image_data_start_row != INVALID_ROW) {
        stats.image_data_start_row = mb_stat.image_data_start_row;
      }
      stats.inter_count += mb_stat.inter_count;
      stats.intra_error += mb_stat.intra_error;
      stats.intra_factor += mb_stat.intra_factor;
      stats.intra_skip_count += mb_stat.intra_skip_count;
      stats.mv_count += mb_stat.mv_count;
      stats.neutral_count += mb_stat.neutral_count;
      stats.new_mv_count += mb_stat.new_mv_count;
      stats.second_ref_count += mb_stat.second_ref_count;
      stats.sr_coded_error += mb_stat.sr_coded_error;
      stats.sum_in_vectors += mb_stat.sum_in_vectors;
      stats.sum_mvc += mb_stat.sum_mvc;
      stats.sum_mvc_abs += mb_stat.sum_mvc_abs;
      stats.sum_mvcs += mb_stat.sum_mvcs;
      stats.sum_mvr += mb_stat.sum_mvr;
      stats.sum_mvr_abs += mb_stat.sum_mvr_abs;
      stats.sum_mvrs += mb_stat.sum_mvrs;
      stats.third_ref_count += mb_stat.third_ref_count;
      stats.tr_coded_error += mb_stat.tr_coded_error;
    }
  }
  return stats;
}
/**************************************************
* average_non_16x16_stats
* Handle stat for non 16x16 blocks. For non 16x16 blocks, some of the stats are increased multiple times
* First find the last block in the 16x16 area and then devide the stats by the number of small blocks
 **************************************************/
// Handle stat for non 16x16 blocks. For non 16x16 blocks, some of the stats are increased multiple times
// First find the last block in the 16x16 area and then devide the stats by the number of small blocks
void average_non_16x16_stats(FRAME_STATS *mb_stats, int blk_num) {
    mb_stats->brightness_factor /= blk_num;
    mb_stats->frame_avg_wavelet_energy /= blk_num;
    mb_stats->inter_count /= blk_num;
    mb_stats->intra_skip_count /= blk_num;
    mb_stats->mv_count /= blk_num;
    mb_stats->neutral_count /= blk_num; //anaghdin check the calculation
    mb_stats->new_mv_count /= blk_num;
    mb_stats->second_ref_count /= blk_num;
    mb_stats->sum_in_vectors /= blk_num;
    mb_stats->sum_mvc /= blk_num;
    mb_stats->sum_mvc_abs /= blk_num;
    mb_stats->sum_mvcs /= blk_num;
    mb_stats->sum_mvr /= blk_num;
    mb_stats->sum_mvr_abs /= blk_num;
    mb_stats->sum_mvrs /= blk_num;
    mb_stats->third_ref_count /= blk_num;
    mb_stats->intra_factor /= blk_num;
}
/**************************************************
 * Reset first pass stat
 **************************************************/
void setup_firstpass_data(PictureParentControlSet *pcs_ptr) {

    SequenceControlSet *scs_ptr = pcs_ptr->scs_ptr;
    FirstPassData *firstpass_data = &pcs_ptr->firstpass_data;
    const uint32_t mb_cols = (scs_ptr->seq_header.max_frame_width + 16 - 1) / 16;
    const uint32_t mb_rows = (scs_ptr->seq_header.max_frame_height + 16 - 1) / 16;
    const uint32_t num_mbs = mb_cols * mb_rows;
    //anaghdin: set the init in another place. It might not be initialized for LAD=0
 //   memset(firstpass_data->raw_motion_err_list, 0, sizeof(*firstpass_data->raw_motion_err_list) * num_mbs);
    memset(firstpass_data->mb_stats, 0, sizeof(*firstpass_data->mb_stats) * num_mbs);
    for (uint32_t i = 0; i < num_mbs; i++)
        firstpass_data->mb_stats[i].image_data_start_row = INVALID_ROW;
}
#if 0
static void free_firstpass_data(FirstPassData *firstpass_data) {
  aom_free(firstpass_data->raw_motion_err_list);
  aom_free(firstpass_data->mb_stats);
}

int av1_get_mb_rows_in_tile(TileInfo tile) {
  int mi_rows_aligned_to_mb =
      ALIGN_POWER_OF_TWO(tile.mi_row_end - tile.mi_row_start, FP_MIB_SIZE_LOG2);
  int mb_rows = mi_rows_aligned_to_mb >> FP_MIB_SIZE_LOG2;

  return mb_rows;
}

int av1_get_mb_cols_in_tile(TileInfo tile) {
  int mi_cols_aligned_to_mb =
      ALIGN_POWER_OF_TWO(tile.mi_col_end - tile.mi_col_start, FP_MIB_SIZE_LOG2);
  int mb_cols = mi_cols_aligned_to_mb >> FP_MIB_SIZE_LOG2;

  return mb_cols;
}

#define FIRST_PASS_ALT_REF_DISTANCE 16
static void first_pass_tile(AV1_COMP *cpi, ThreadData *td,
                            TileDataEnc *tile_data) {
  TileInfo *tile = &tile_data->tile_info;
  for (int mi_row = tile->mi_row_start; mi_row < tile->mi_row_end;
       mi_row += FP_MIB_SIZE) {
    av1_first_pass_row(cpi, td, tile_data, mi_row >> FP_MIB_SIZE_LOG2);
  }
}

static void first_pass_tiles(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  const int tile_cols = cm->tiles.cols;
  const int tile_rows = cm->tiles.rows;
  for (int tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_cols; ++tile_col) {
      TileDataEnc *const tile_data =
          &cpi->tile_data[tile_row * tile_cols + tile_col];
      first_pass_tile(cpi, &cpi->td, tile_data);
    }
  }
}

void av1_first_pass_row(AV1_COMP *cpi, ThreadData *td, TileDataEnc *tile_data,
                        int mb_row) {
  MACROBLOCK *const x = &td->mb;
  AV1_COMMON *const cm = &cpi->common;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  CurrentFrame *const current_frame = &cm->current_frame;
  const SequenceHeader *const seq_params = &cm->seq_params;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCKD *const xd = &x->e_mbd;
  TileInfo *tile = &tile_data->tile_info;
  const int qindex = find_fp_qindex(seq_params->bit_depth);
  // First pass coding proceeds in raster scan order with unit size of 16x16.
  const BLOCK_SIZE fp_block_size = BLOCK_16X16;
  const int fp_block_size_width = block_size_high[fp_block_size];
  const int fp_block_size_height = block_size_wide[fp_block_size];
  int raw_motion_err_counts = 0;
  int mb_row_in_tile = mb_row - (tile->mi_row_start >> FP_MIB_SIZE_LOG2);
  int mb_col_start = tile->mi_col_start >> FP_MIB_SIZE_LOG2;
  int mb_cols_in_tile = av1_get_mb_cols_in_tile(*tile);
  MultiThreadInfo *const mt_info = &cpi->mt_info;
  AV1EncRowMultiThreadInfo *const enc_row_mt = &mt_info->enc_row_mt;
  AV1EncRowMultiThreadSync *const row_mt_sync = &tile_data->row_mt_sync;

  const YV12_BUFFER_CONFIG *const last_frame =
      get_ref_frame_yv12_buf(cm, LAST_FRAME);
  const YV12_BUFFER_CONFIG *golden_frame =
      get_ref_frame_yv12_buf(cm, GOLDEN_FRAME);
  const YV12_BUFFER_CONFIG *alt_ref_frame = NULL;
  const int alt_ref_offset =
      FIRST_PASS_ALT_REF_DISTANCE -
      (current_frame->frame_number % FIRST_PASS_ALT_REF_DISTANCE);
  if (alt_ref_offset < FIRST_PASS_ALT_REF_DISTANCE) {
    const struct lookahead_entry *const alt_ref_frame_buffer =
        av1_lookahead_peek(cpi->lookahead, alt_ref_offset,
                           cpi->compressor_stage);
    if (alt_ref_frame_buffer != NULL) {
      alt_ref_frame = &alt_ref_frame_buffer->img;
    }
  }
  YV12_BUFFER_CONFIG *const this_frame = &cm->cur_frame->buf;

  PICK_MODE_CONTEXT *ctx = td->firstpass_ctx;
  FRAME_STATS *mb_stats =
      cpi->firstpass_data.mb_stats + mb_row * mi_params->mb_cols + mb_col_start;
  int *raw_motion_err_list = cpi->firstpass_data.raw_motion_err_list +
                             mb_row * mi_params->mb_cols + mb_col_start;
  MV *first_top_mv = &tile_data->firstpass_top_mv;

  for (int i = 0; i < num_planes; ++i) {
    x->plane[i].coeff = ctx->coeff[i];
    x->plane[i].qcoeff = ctx->qcoeff[i];
    x->plane[i].eobs = ctx->eobs[i];
    x->plane[i].txb_entropy_ctx = ctx->txb_entropy_ctx[i];
    x->plane[i].dqcoeff = ctx->dqcoeff[i];
  }

  const int src_y_stride = cpi->source->y_stride;
  const int recon_y_stride = this_frame->y_stride;
  const int recon_uv_stride = this_frame->uv_stride;
  const int uv_mb_height =
      fp_block_size_height >> (this_frame->y_height > this_frame->uv_height);

  MV best_ref_mv = kZeroMv;
  MV last_mv;

  // Reset above block coeffs.
  xd->up_available = (mb_row_in_tile != 0);
  int recon_yoffset = (mb_row * recon_y_stride * fp_block_size_height) +
                      (mb_col_start * fp_block_size_width);
  int src_yoffset = (mb_row * src_y_stride * fp_block_size_height) +
                    (mb_col_start * fp_block_size_width);
  int recon_uvoffset =
      (mb_row * recon_uv_stride * uv_mb_height) + (mb_col_start * uv_mb_height);
  int alt_ref_frame_yoffset =
      (alt_ref_frame != NULL)
          ? (mb_row * alt_ref_frame->y_stride * fp_block_size_height) +
                (mb_col_start * fp_block_size_width)
          : -1;

  // Set up limit values for motion vectors to prevent them extending
  // outside the UMV borders.
  av1_set_mv_row_limits(mi_params, &x->mv_limits, (mb_row << FP_MIB_SIZE_LOG2),
                        (fp_block_size_height >> MI_SIZE_LOG2),
                        cpi->oxcf.border_in_pixels);

  av1_setup_src_planes(x, cpi->source, mb_row << FP_MIB_SIZE_LOG2,
                       tile->mi_col_start, num_planes, fp_block_size);

  // Fix - zero the 16x16 block first. This ensures correct this_intra_error for
  // block sizes smaller than 16x16.
  av1_zero_array(x->plane[0].src_diff, 256);

  for (int mi_col = tile->mi_col_start; mi_col < tile->mi_col_end;
       mi_col += FP_MIB_SIZE) {
    int mb_col = mi_col >> FP_MIB_SIZE_LOG2;
    int mb_col_in_tile = mb_col - mb_col_start;

    (*(enc_row_mt->sync_read_ptr))(row_mt_sync, mb_row_in_tile, mb_col_in_tile);

    if (mb_col_in_tile == 0) {
      last_mv = *first_top_mv;
    }
    int this_intra_error = firstpass_intra_prediction(
        cpi, td, this_frame, tile, mb_row, mb_col, recon_yoffset,
        recon_uvoffset, fp_block_size, qindex, mb_stats);

    if (!frame_is_intra_only(cm)) {
      const int this_inter_error = firstpass_inter_prediction(
          cpi, td, last_frame, golden_frame, alt_ref_frame, mb_row, mb_col,
          recon_yoffset, recon_uvoffset, src_yoffset, alt_ref_frame_yoffset,
          fp_block_size, this_intra_error, raw_motion_err_counts,
          raw_motion_err_list, &best_ref_mv, &last_mv, mb_stats);
      if (mb_col_in_tile == 0) {
        *first_top_mv = last_mv;
      }
      mb_stats->coded_error += this_inter_error;
      ++raw_motion_err_counts;
    } else {
      mb_stats->sr_coded_error += this_intra_error;
      mb_stats->tr_coded_error += this_intra_error;
      mb_stats->coded_error += this_intra_error;
    }

    // Adjust to the next column of MBs.
    x->plane[0].src.buf += fp_block_size_width;
    x->plane[1].src.buf += uv_mb_height;
    x->plane[2].src.buf += uv_mb_height;

    recon_yoffset += fp_block_size_width;
    src_yoffset += fp_block_size_width;
    recon_uvoffset += uv_mb_height;
    alt_ref_frame_yoffset += fp_block_size_width;
    mb_stats++;

    (*(enc_row_mt->sync_write_ptr))(row_mt_sync, mb_row_in_tile, mb_col_in_tile,
                                    mb_cols_in_tile);
  }
}
#endif
//void av1_first_pass(AV1_COMP *cpi, const int64_t ts_duration)
void av1_first_pass(PictureParentControlSet *pcs_ptr, const int64_t ts_duration)
{
  SequenceControlSet *scs_ptr = pcs_ptr->scs_ptr;
  const uint32_t mb_cols = (scs_ptr->seq_header.max_frame_width  + 16 - 1) / 16;
  const uint32_t mb_rows = (scs_ptr->seq_header.max_frame_height + 16 - 1) / 16;
#if 0
  MACROBLOCK *const x = &cpi->td.mb;
  AV1_COMMON *const cm = &cpi->common;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  CurrentFrame *const current_frame = &cm->current_frame;
  const SequenceHeader *const seq_params = &cm->seq_params;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCKD *const xd = &x->e_mbd;
  const int qindex = find_fp_qindex(seq_params->bit_depth);
  // Detect if the key frame is screen content type.
  if (frame_is_intra_only(cm)) {
    FeatureFlags *const features = &cm->features;
    av1_set_screen_content_options(cpi, features);
    cpi->is_screen_content_type = features->allow_screen_content_tools;
  }
  // First pass coding proceeds in raster scan order with unit size of 16x16.
  const BLOCK_SIZE fp_block_size = BLOCK_16X16;

  setup_firstpass_data(cm, &cpi->firstpass_data);
#endif
  int *raw_motion_err_list = pcs_ptr->firstpass_data.raw_motion_err_list;
  FRAME_STATS *mb_stats = pcs_ptr->firstpass_data.mb_stats;

#if 0
  // multi threading info
  MultiThreadInfo *const mt_info = &cpi->mt_info;
  AV1EncRowMultiThreadInfo *const enc_row_mt = &mt_info->enc_row_mt;

  const int tile_cols = cm->tiles.cols;
  const int tile_rows = cm->tiles.rows;
  if (cpi->allocated_tiles < tile_cols * tile_rows) {
    av1_row_mt_mem_dealloc(cpi);
    av1_alloc_tile_data(cpi);
  }

  av1_init_tile_data(cpi);

  const YV12_BUFFER_CONFIG *const last_frame =
      get_ref_frame_yv12_buf(cm, LAST_FRAME);
  const YV12_BUFFER_CONFIG *golden_frame =
      get_ref_frame_yv12_buf(cm, GOLDEN_FRAME);
  YV12_BUFFER_CONFIG *const this_frame = &cm->cur_frame->buf;
  // First pass code requires valid last and new frame buffers.
  assert(this_frame != NULL);
  assert(frame_is_intra_only(cm) || (last_frame != NULL));

  av1_setup_frame_size(cpi);
  aom_clear_system_state();

  set_mi_offsets(mi_params, xd, 0, 0);
  xd->mi[0]->sb_type = fp_block_size;

  // Do not use periodic key frames.
  cpi->rc.frames_to_key = INT_MAX;

  av1_set_quantizer(cm, cpi->oxcf.q_cfg.qm_minlevel,
                    cpi->oxcf.q_cfg.qm_maxlevel, qindex,
                    cpi->oxcf.q_cfg.enable_chroma_deltaq);

  av1_setup_block_planes(xd, seq_params->subsampling_x,
                         seq_params->subsampling_y, num_planes);

  av1_setup_src_planes(x, cpi->source, 0, 0, num_planes, fp_block_size);
  av1_setup_dst_planes(xd->plane, seq_params->sb_size, this_frame, 0, 0, 0,
                       num_planes);

  if (!frame_is_intra_only(cm)) {
    av1_setup_pre_planes(xd, 0, last_frame, 0, 0, NULL, num_planes);
  }

  set_mi_offsets(mi_params, xd, 0, 0);

  // Don't store luma on the fist pass since chroma is not computed
  xd->cfl.store_y = 0;
  av1_frame_init_quantizer(cpi);

  av1_init_mv_probs(cm);
  av1_initialize_rd_consts(cpi);

  enc_row_mt->sync_read_ptr = av1_row_mt_sync_read_dummy;
  enc_row_mt->sync_write_ptr = av1_row_mt_sync_write_dummy;

  if (mt_info->num_workers > 1) {
    enc_row_mt->sync_read_ptr = av1_row_mt_sync_read;
    enc_row_mt->sync_write_ptr = av1_row_mt_sync_write;
    av1_fp_encode_tiles_row_mt(cpi);
  } else {
    first_pass_tiles(cpi);
  }
#endif

  FRAME_STATS stats =
      accumulate_frame_stats(mb_stats, mb_rows, mb_cols);
  int total_raw_motion_err_count =
      frame_is_intra_only(pcs_ptr) ? 0 : mb_rows * mb_cols;
  const double raw_err_stdev =
      raw_motion_error_stdev(raw_motion_err_list, total_raw_motion_err_count);
#if 0
  free_firstpass_data(&cpi->firstpass_data);

  // Clamp the image start to rows/2. This number of rows is discarded top
  // and bottom as dead data so rows / 2 means the frame is blank.
  if ((stats.image_data_start_row > mi_params->mb_rows / 2) ||
      (stats.image_data_start_row == INVALID_ROW)) {
    stats.image_data_start_row = mi_params->mb_rows / 2;
  }
  // Exclude any image dead zone
  if (stats.image_data_start_row > 0) {
    stats.intra_skip_count =
        AOMMAX(0, stats.intra_skip_count -
                      (stats.image_data_start_row * mi_params->mb_cols * 2));
  }
#endif
//#if TWOPASS_STAT_BUF
//  TWO_PASS *twopass = &scs_ptr->twopass;
//#else
//  TWO_PASS *twopass = &pcs_ptr->twopass;
//#endif
  const int num_mbs = mb_rows * mb_cols;
                      /*(cpi->oxcf.resize_cfg.resize_mode != RESIZE_NONE)
                          ? cpi->initial_mbs
                          : mi_params->MBs;*/
  stats.intra_factor = stats.intra_factor / (double)num_mbs;
  stats.brightness_factor = stats.brightness_factor / (double)num_mbs;
  //FIRSTPASS_STATS *this_frame_stats = twopass->stats_buf_ctx->stats_in_end;
  update_firstpass_stats(pcs_ptr, &stats, raw_err_stdev,
                         (const int)pcs_ptr->picture_number/*current_frame->frame_number*/, ts_duration);

#if 0
  // Copy the previous Last Frame back into gf buffer if the prediction is good
  // enough... but also don't allow it to lag too far.
  if ((twopass->sr_update_lag > 3) ||
      ((current_frame->frame_number > 0) &&
       (this_frame_stats->pcnt_inter > 0.20) &&
       ((this_frame_stats->intra_error /
         DOUBLE_DIVIDE_CHECK(this_frame_stats->coded_error)) > 2.0))) {
    if (golden_frame != NULL) {
      assign_frame_buffer_p(
          &cm->ref_frame_map[get_ref_frame_map_idx(cm, GOLDEN_FRAME)],
          cm->ref_frame_map[get_ref_frame_map_idx(cm, LAST_FRAME)]);
    }
    twopass->sr_update_lag = 1;
  } else {
    ++twopass->sr_update_lag;
  }

  aom_extend_frame_borders(this_frame, num_planes);

  // The frame we just compressed now becomes the last frame.
  assign_frame_buffer_p(
      &cm->ref_frame_map[get_ref_frame_map_idx(cm, LAST_FRAME)], cm->cur_frame);

  // Special case for the first frame. Copy into the GF buffer as a second
  // reference.
  if (current_frame->frame_number == 0 &&
      get_ref_frame_map_idx(cm, GOLDEN_FRAME) != INVALID_IDX) {
    assign_frame_buffer_p(
        &cm->ref_frame_map[get_ref_frame_map_idx(cm, GOLDEN_FRAME)],
        cm->ref_frame_map[get_ref_frame_map_idx(cm, LAST_FRAME)]);
  }

  print_reconstruction_frame(last_frame, current_frame->frame_number,
                             /*do_print=*/0);

  ++current_frame->frame_number;
#endif
}

#if FIRST_PASS_SETUP
void first_pass_frame_end(PictureParentControlSet *pcs_ptr, const int64_t ts_duration) {
    SequenceControlSet *scs_ptr = pcs_ptr->scs_ptr;
    const uint32_t mb_cols = (scs_ptr->seq_header.max_frame_width + 16 - 1) / 16;
    const uint32_t mb_rows = (scs_ptr->seq_header.max_frame_height + 16 - 1) / 16;

    int *raw_motion_err_list = pcs_ptr->firstpass_data.raw_motion_err_list;
    FRAME_STATS *mb_stats = pcs_ptr->firstpass_data.mb_stats;

    FRAME_STATS stats =
        accumulate_frame_stats(mb_stats, mb_rows, mb_cols);
    int total_raw_motion_err_count =
        frame_is_intra_only(pcs_ptr) ? 0 : mb_rows * mb_cols;
    const double raw_err_stdev =
        raw_motion_error_stdev(raw_motion_err_list, total_raw_motion_err_count);
   // free_firstpass_data(&cpi->firstpass_data);
    // anaghdin check this
    // Clamp the image start to rows/2. This number of rows is discarded top
    // and bottom as dead data so rows / 2 means the frame is blank.
    if ((stats.image_data_start_row > (int)mb_rows / 2) ||
        (stats.image_data_start_row == INVALID_ROW)) {
        stats.image_data_start_row = mb_rows / 2;
    }
    // Exclude any image dead zone
    if (stats.image_data_start_row > 0) {
        stats.intra_skip_count =
            AOMMAX(0, stats.intra_skip_count -
            (stats.image_data_start_row * (int)mb_cols * 2));
    }
//#if TWOPASS_STAT_BUF
//    TWO_PASS *twopass = &scs_ptr->twopass;
//#else
//    TWO_PASS *twopass = &pcs_ptr->twopass;
//#endif
    const int num_mbs = mb_rows * mb_cols;
    /*(cpi->oxcf.resize_cfg.resize_mode != RESIZE_NONE)
        ? cpi->initial_mbs
        : mi_params->MBs;*/
    stats.intra_factor = stats.intra_factor / (double)num_mbs;
    stats.brightness_factor = stats.brightness_factor / (double)num_mbs;
    //FIRSTPASS_STATS *this_frame_stats = twopass->stats_buf_ctx->stats_in_end;
    update_firstpass_stats(pcs_ptr, &stats, raw_err_stdev,
        (const int)pcs_ptr->picture_number/*current_frame->frame_number*/, ts_duration);

}
#endif
#if FIRST_PASS_SETUP
/******************************************************
* Derive Pre-Analysis settings for first pass
Input   : encoder mode and tune
Output  : Pre-Analysis signal(s)
******************************************************/
extern EbErrorType first_pass_signal_derivation_pre_analysis(SequenceControlSet *     scs_ptr,
    PictureParentControlSet *pcs_ptr) {
    EbErrorType return_error = EB_ErrorNone;
    // Derive HME Flag
    pcs_ptr->enable_hme_flag = 1;
    pcs_ptr->enable_hme_level0_flag = 1;
    pcs_ptr->enable_hme_level1_flag = 1;
    pcs_ptr->enable_hme_level2_flag = 1;

    //// Set here to allocate resources for the downsampled pictures used in HME (generated in PictureAnalysis)
    //// Will be later updated for SC/NSC in PictureDecisionProcess
    pcs_ptr->tf_enable_hme_flag = 0;
    pcs_ptr->tf_enable_hme_level0_flag = 0;
    pcs_ptr->tf_enable_hme_level1_flag = 0;
    pcs_ptr->tf_enable_hme_level2_flag = 0;
    scs_ptr->seq_header.enable_intra_edge_filter = 0;
    scs_ptr->seq_header.pic_based_rate_est = 0;
    scs_ptr->seq_header.enable_restoration = 0;
    scs_ptr->seq_header.enable_cdef = 0;
    scs_ptr->seq_header.enable_warped_motion = 0;

    return return_error;
}

#endif

#if FIRST_PASS_SETUP
extern EbErrorType av1_intra_full_cost(PictureControlSet *pcs_ptr, ModeDecisionContext *context_ptr,
                                       struct ModeDecisionCandidateBuffer *candidate_buffer_ptr,
                                       BlkStruct *blk_ptr, uint64_t *y_distortion,
                                       uint64_t *cb_distortion, uint64_t *cr_distortion,
                                       uint64_t lambda, uint64_t *y_coeff_bits,
                                       uint64_t *cb_coeff_bits, uint64_t *cr_coeff_bits,
                                       BlockSize bsize);

extern EbErrorType av1_inter_full_cost(PictureControlSet *pcs_ptr, ModeDecisionContext *context_ptr,
                                       struct ModeDecisionCandidateBuffer *candidate_buffer_ptr,
                                       BlkStruct *blk_ptr, uint64_t *y_distortion,
                                       uint64_t *cb_distortion, uint64_t *cr_distortion,
                                       uint64_t lambda, uint64_t *y_coeff_bits,
                                       uint64_t *cb_coeff_bits, uint64_t *cr_coeff_bits,
                                       BlockSize bsize);
const EbPredictionFunc product_prediction_fun_table[3] ;

const EbAv1FullCostFunc av1_product_full_cost_func_table[3] ;

void perform_tx_partitioning(ModeDecisionCandidateBuffer *candidate_buffer,
                             ModeDecisionContext *context_ptr, PictureControlSet *pcs_ptr,
                             uint64_t ref_fast_cost, uint8_t start_tx_depth, uint8_t end_tx_depth,
#if QP2QINDEX
                             uint32_t qindex, uint32_t *y_count_non_zero_coeffs, uint64_t *y_coeff_bits,
#else
                             uint32_t qp, uint32_t *y_count_non_zero_coeffs, uint64_t *y_coeff_bits,
#endif
                             uint64_t *y_full_distortion);

extern void first_pass_loop_core(PictureControlSet *pcs_ptr, /*SuperBlock *sb_ptr, */BlkStruct *blk_ptr,
    ModeDecisionContext *context_ptr, ModeDecisionCandidateBuffer *candidate_buffer,
    ModeDecisionCandidate *candidate_ptr, EbPictureBufferDesc *input_picture_ptr,
    uint32_t input_origin_index,// uint32_t input_cb_origin_in_index,
    uint32_t blk_origin_index,// uint32_t blk_chroma_origin_index,
    uint64_t ref_fast_cost) {
    uint64_t y_full_distortion[DIST_CALC_TOTAL];
    uint32_t count_non_zero_coeffs[3][MAX_NUM_OF_TU_PER_CU];

    uint64_t cb_full_distortion[DIST_CALC_TOTAL];
    uint64_t cr_full_distortion[DIST_CALC_TOTAL];

    uint64_t y_coeff_bits;
    uint64_t cb_coeff_bits = 0;
    uint64_t cr_coeff_bits = 0;

    uint32_t full_lambda = context_ptr->hbd_mode_decision ?
        context_ptr->full_lambda_md[EB_10_BIT_MD] :
        context_ptr->full_lambda_md[EB_8_BIT_MD];
#if FIX_CFL_OFF
    int32_t is_inter = (candidate_buffer->candidate_ptr->type == INTER_MODE ||
        candidate_buffer->candidate_ptr->use_intrabc)
        ? EB_TRUE
        : EB_FALSE;
#endif

    // initialize TU Split
    y_full_distortion[DIST_CALC_RESIDUAL] = 0;
    y_full_distortion[DIST_CALC_PREDICTION] = 0;
    y_coeff_bits = 0;

    candidate_ptr->full_distortion = 0;

    memset(candidate_ptr->eob[0], 0, sizeof(uint16_t));
    memset(candidate_ptr->eob[1], 0, sizeof(uint16_t));
    memset(candidate_ptr->eob[2], 0, sizeof(uint16_t));

    candidate_ptr->chroma_distortion = 0;
    candidate_ptr->chroma_distortion_inter_depth = 0;
    // Set Skip Flag
    candidate_ptr->skip_flag = EB_FALSE;

    product_prediction_fun_table[candidate_ptr->type](
        context_ptr->hbd_mode_decision, context_ptr, pcs_ptr, candidate_buffer);

    // Initialize luma CBF
    candidate_ptr->y_has_coeff = 0;
    candidate_ptr->u_has_coeff = 0;
    candidate_ptr->v_has_coeff = 0;

    // Initialize tx type
    for (int tu_index = 0; tu_index < MAX_TXB_COUNT; tu_index++)
        candidate_ptr->transform_type[tu_index] = DCT_DCT;
    uint8_t start_tx_depth = 0;
    uint8_t end_tx_depth = 0;
    if (context_ptr->md_tx_size_search_mode == 0) {
        start_tx_depth = end_tx_depth;
    }
    else if (context_ptr->md_staging_tx_size_mode == 0) {
        start_tx_depth = end_tx_depth = candidate_buffer->candidate_ptr->tx_depth;
    }
    //Y Residual: residual for INTRA is computed inside the TU loop
    if (is_inter)
        //Y Residual
        residual_kernel(input_picture_ptr->buffer_y,
            input_origin_index,
            input_picture_ptr->stride_y,
            candidate_buffer->prediction_ptr->buffer_y,
            blk_origin_index,
            candidate_buffer->prediction_ptr->stride_y,
            (int16_t *)candidate_buffer->residual_ptr->buffer_y,
            blk_origin_index,
            candidate_buffer->residual_ptr->stride_y,
            context_ptr->hbd_mode_decision,
            context_ptr->blk_geom->bwidth,
            context_ptr->blk_geom->bheight);

    perform_tx_partitioning(candidate_buffer,
        context_ptr,
        pcs_ptr,
        ref_fast_cost,
        start_tx_depth,
        end_tx_depth,
#if QP2QINDEX
        context_ptr->blk_ptr->qindex,
#else
        context_ptr->blk_ptr->qp,
#endif
        &(*count_non_zero_coeffs[0]),
        &y_coeff_bits,
        &y_full_distortion[0]);

    candidate_ptr->chroma_distortion_inter_depth = 0;
    candidate_ptr->chroma_distortion = 0;

    //CHROMA

    cb_full_distortion[DIST_CALC_RESIDUAL] = 0;
    cr_full_distortion[DIST_CALC_RESIDUAL] = 0;
    cb_full_distortion[DIST_CALC_PREDICTION] = 0;
    cr_full_distortion[DIST_CALC_PREDICTION] = 0;

    cb_coeff_bits = 0;
    cr_coeff_bits = 0;
#if 0 //first_pass_opt
    // FullLoop and TU search
#if QP2QINDEX
    uint16_t cb_qindex = context_ptr->qp_index;
    uint16_t cr_qindex = context_ptr->qp_index;
#else
    uint16_t cb_qp = context_ptr->qp;
    uint16_t cr_qp = context_ptr->qp;
#endif
    if (context_ptr->md_staging_skip_full_chroma == EB_FALSE) {
        if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
            //Cb Residual
            residual_kernel(input_picture_ptr->buffer_cb,
                input_cb_origin_in_index,
                input_picture_ptr->stride_cb,
                candidate_buffer->prediction_ptr->buffer_cb,
                blk_chroma_origin_index,
                candidate_buffer->prediction_ptr->stride_cb,
                (int16_t *)candidate_buffer->residual_ptr->buffer_cb,
                blk_chroma_origin_index,
                candidate_buffer->residual_ptr->stride_cb,
                context_ptr->hbd_mode_decision,
                context_ptr->blk_geom->bwidth_uv,
                context_ptr->blk_geom->bheight_uv);

            //Cr Residual
            residual_kernel(input_picture_ptr->buffer_cr,
                input_cb_origin_in_index,
                input_picture_ptr->stride_cr,
                candidate_buffer->prediction_ptr->buffer_cr,
                blk_chroma_origin_index,
                candidate_buffer->prediction_ptr->stride_cr,
                (int16_t *)candidate_buffer->residual_ptr->buffer_cr,
                blk_chroma_origin_index,
                candidate_buffer->residual_ptr->stride_cr,
                context_ptr->hbd_mode_decision,
                context_ptr->blk_geom->bwidth_uv,
                context_ptr->blk_geom->bheight_uv);
        }
        if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
            full_loop_r(sb_ptr,
                candidate_buffer,
                context_ptr,
                input_picture_ptr,
                pcs_ptr,
                PICTURE_BUFFER_DESC_CHROMA_MASK,
#if QP2QINDEX
                cb_qindex,
                cr_qindex,
#else
                cb_qp,
                cr_qp,
#endif
                &(*count_non_zero_coeffs[1]),
                &(*count_non_zero_coeffs[2]));

            cu_full_distortion_fast_txb_mode_r(sb_ptr,
                candidate_buffer,
                context_ptr,
                candidate_ptr,
                pcs_ptr,
                input_picture_ptr,
                cb_full_distortion,
                cr_full_distortion,
                count_non_zero_coeffs,
                COMPONENT_CHROMA,
                &cb_coeff_bits,
                &cr_coeff_bits,
                1);
        }
    }
#endif
    candidate_ptr->block_has_coeff =
        (candidate_ptr->y_has_coeff | candidate_ptr->u_has_coeff | candidate_ptr->v_has_coeff)
        ? EB_TRUE
        : EB_FALSE;

    //ALL PLANE
    av1_product_full_cost_func_table[candidate_ptr->type](pcs_ptr,
        context_ptr,
        candidate_buffer,
        blk_ptr,
        y_full_distortion,
        cb_full_distortion,
        cr_full_distortion,
        full_lambda,
        &y_coeff_bits,
        &cb_coeff_bits,
        &cr_coeff_bits,
        context_ptr->blk_geom->bsize);
#if SB_CLASSIFIER
    uint16_t txb_count = context_ptr->blk_geom->txb_count[candidate_buffer->candidate_ptr->tx_depth];
    candidate_ptr->count_non_zero_coeffs = 0;
    for (uint8_t txb_itr = 0; txb_itr < txb_count; txb_itr++)
        candidate_ptr->count_non_zero_coeffs += count_non_zero_coeffs[0][txb_itr];
#endif
}
// anaghdin to move to firstpass.c and remove
#define FIRST_PASS_Q 10.0
#define INTRA_MODE_PENALTY 1024
#define NEW_MV_MODE_PENALTY 32
#define DARK_THRESH 64
#define UL_INTRA_THRESH 50
#define INVALID_ROW -1
#define NCOUNT_INTRA_THRESH 8192
#define NCOUNT_INTRA_FACTOR 3
#define LOW_MOTION_ERROR_THRESH 25
// Computes and returns the intra pred error of a block.
// intra pred error: sum of squared error of the intra predicted residual.
// Inputs:
//   cpi: the encoder setting. Only a few params in it will be used.
//   this_frame: the current frame buffer.
//   tile: tile information (not used in first pass, already init to zero)
//   mb_row: row index in the unit of first pass block size.
//   mb_col: column index in the unit of first pass block size.
//   y_offset: the offset of y frame buffer, indicating the starting point of
//             the current block.
//   uv_offset: the offset of u and v frame buffer, indicating the starting
//              point of the current block.
//   fp_block_size: first pass block size.
//   qindex: quantization step size to encode the frame.
//   stats: frame encoding stats.
// Modifies:
//   stats->intra_skip_count
//   stats->image_data_start_row
//   stats->intra_factor
//   stats->brightness_factor
//   stats->intra_error
//   stats->frame_avg_wavelet_energy
// Returns:
//   this_intra_error.
static int firstpass_intra_prediction(PictureControlSet *pcs_ptr, BlkStruct *blk_ptr,
    ModeDecisionContext *context_ptr, ModeDecisionCandidateBuffer *candidate_buffer,
    ModeDecisionCandidate *candidate_ptr, EbPictureBufferDesc *input_picture_ptr,
    uint32_t input_origin_index,// uint32_t input_cb_origin_in_index,
    uint32_t blk_origin_index,// uint32_t blk_chroma_origin_index,
    uint64_t ref_fast_cost, FRAME_STATS *const stats){

    int32_t       mb_row = context_ptr->blk_origin_y >> 4;
    int32_t       mb_col = context_ptr->blk_origin_x >> 4;
    const int use_dc_pred = (mb_col || mb_row) && (!mb_col || !mb_row);
    const BlockSize bsize = context_ptr->blk_geom->bsize;

    // Initialize tx_depth
    candidate_buffer->candidate_ptr->tx_depth =
        use_dc_pred ? 0 :
        (bsize == BLOCK_16X16 ? 2 : bsize == BLOCK_8X8 ? 1: 0);
    candidate_buffer->candidate_ptr->fast_luma_rate = 0;
    candidate_buffer->candidate_ptr->fast_chroma_rate = 0;
    context_ptr->md_staging_skip_interpolation_search = EB_TRUE;
    context_ptr->md_staging_skip_chroma_pred = EB_FALSE;
    context_ptr->md_staging_tx_size_mode = 0;
    context_ptr->md_staging_skip_full_chroma = EB_FALSE;
    context_ptr->md_staging_skip_rdoq = EB_TRUE;
    context_ptr->md_staging_spatial_sse_full_loop = context_ptr->spatial_sse_full_loop;

    first_pass_loop_core(pcs_ptr,
        //context_ptr->sb_ptr,
        blk_ptr,
        context_ptr,
        candidate_buffer,
        candidate_ptr,
        input_picture_ptr,
        input_origin_index,
        //input_cb_origin_in_index,
        blk_origin_index,
        //blk_chroma_origin_index,
        ref_fast_cost);

    EbSpatialFullDistType spatial_full_dist_type_fun = context_ptr->hbd_mode_decision
        ? full_distortion_kernel16_bits
        : spatial_full_distortion_kernel;

    int this_intra_error = (uint32_t)(
        spatial_full_dist_type_fun(input_picture_ptr->buffer_y,
            input_origin_index,
            input_picture_ptr->stride_y,
            candidate_buffer->prediction_ptr->buffer_y,
            blk_origin_index,
            candidate_buffer->prediction_ptr->stride_y,
            context_ptr->blk_geom->bwidth,
            context_ptr->blk_geom->bheight));

    if (this_intra_error < UL_INTRA_THRESH) {
        ++stats->intra_skip_count;
    }
    else if ((mb_col > 0) && (stats->image_data_start_row == INVALID_ROW)) {
        stats->image_data_start_row = mb_row;
    }

    if (pcs_ptr->parent_pcs_ptr->av1_cm->use_highbitdepth) {
        switch (pcs_ptr->parent_pcs_ptr->av1_cm->bit_depth) {
        case AOM_BITS_8: break;
        case AOM_BITS_10: this_intra_error >>= 4; break;
        case AOM_BITS_12: this_intra_error >>= 8; break;
        default:
            assert(0 &&
                "seq_params->bit_depth should be AOM_BITS_8, "
                "AOM_BITS_10 or AOM_BITS_12");
            return -1;
        }
    }

   // aom_clear_system_state();
    double log_intra = log(this_intra_error + 1.0);
    if (log_intra < 10.0)
        stats->intra_factor += 1.0 + ((10.0 - log_intra) * 0.05);
    else
        stats->intra_factor += 1.0;

    int level_sample;
    if (pcs_ptr->parent_pcs_ptr->av1_cm->use_highbitdepth)
        level_sample = CONVERT_TO_SHORTPTR(input_picture_ptr->buffer_y)[input_origin_index];
    else
        level_sample = input_picture_ptr->buffer_y[input_origin_index];
    if ((level_sample < DARK_THRESH) && (log_intra < 9.0))
        stats->brightness_factor += 1.0 + (0.01 * (DARK_THRESH - level_sample));
    else
        stats->brightness_factor += 1.0;
    // Intrapenalty below deals with situations where the intra and inter
    // error scores are very low (e.g. a plain black frame).
    // We do not have special cases in first pass for 0,0 and nearest etc so
    // all inter modes carry an overhead cost estimate for the mv.
    // When the error score is very low this causes us to pick all or lots of
    // INTRA modes and throw lots of key frames.
    // This penalty adds a cost matching that of a 0,0 mv to the intra case.
    this_intra_error += INTRA_MODE_PENALTY;

    const int hbd = context_ptr->hbd_mode_decision;
    const int stride = input_picture_ptr->stride_y;
    uint8_t *buf = &input_picture_ptr->buffer_y[input_origin_index];
    for (int r8 = 0; r8 < 2; ++r8) {
        for (int c8 = 0; c8 < 2; ++c8) {
            stats->frame_avg_wavelet_energy += av1_haar_ac_sad_8x8_uint8_input(
                buf + c8 * 8 + r8 * 8 * stride, stride, hbd);
        }
    }
    // Accumulate the intra error.
    stats->intra_error += (int64_t)this_intra_error;
    return this_intra_error;
}
// Computes and returns the inter prediction error from the last frame.
// Computes inter prediction errors from the golden and alt ref frams and
// Updates stats accordingly.
// Inputs:
//   cpi: the encoder setting. Only a few params in it will be used.
//   last_frame: the frame buffer of the last frame.
//   golden_frame: the frame buffer of the golden frame.
//   alt_ref_frame: the frame buffer of the alt ref frame.
//   mb_row: row index in the unit of first pass block size.
//   mb_col: column index in the unit of first pass block size.
//   recon_yoffset: the y offset of the reconstructed  frame buffer,
//                  indicating the starting point of the current block.
//   recont_uvoffset: the u/v offset of the reconstructed frame buffer,
//                    indicating the starting point of the current block.
//   src_yoffset: the y offset of the source frame buffer.
//   alt_ref_frame_offset: the y offset of the alt ref frame buffer.
//   fp_block_size: first pass block size.
//   this_intra_error: the intra prediction error of this block.
//   raw_motion_err_counts: the count of raw motion vectors.
//   raw_motion_err_list: the array that records the raw motion error.
//   best_ref_mv: best reference mv found so far.
//   last_mv: last mv.
//   stats: frame encoding stats.
//  Modifies:
//    raw_motion_err_list
//    best_ref_mv
//    last_mv
//    stats: many member params in it.
//  Returns:
//    this_inter_error
static int firstpass_inter_prediction(PictureControlSet *pcs_ptr, BlkStruct *blk_ptr,
    ModeDecisionContext *context_ptr, ModeDecisionCandidateBuffer *candidate_buffer,
    ModeDecisionCandidate *candidate_ptr, EbPictureBufferDesc *input_picture_ptr,
    uint32_t input_origin_index,// uint32_t input_cb_origin_in_index,
    uint32_t blk_origin_index,// uint32_t blk_chroma_origin_index,
    uint64_t ref_fast_cost, uint32_t fast_candidate_total_count, const int this_intra_error,
    /*int *raw_motion_err_list, */MV *best_ref_mv,
    MV *last_mv, FRAME_STATS *stats) {

    int32_t       mb_row = context_ptr->blk_origin_y >> 4;
    int32_t       mb_col = context_ptr->blk_origin_x >> 4;
    const uint32_t mb_cols = (pcs_ptr->parent_pcs_ptr->scs_ptr->seq_header.max_frame_width + 16 - 1) / 16;
    const uint32_t mb_rows = (pcs_ptr->parent_pcs_ptr->scs_ptr->seq_header.max_frame_height + 16 - 1) / 16;
    int this_inter_error = this_intra_error;
    //const int is_high_bitdepth = context_ptr->hbd_mode_decision;
    //const int bitdepth = pcs_ptr->parent_pcs_ptr->av1_cm->bit_depth;
    const BlockSize bsize = context_ptr->blk_geom->bsize;
    // Assume 0,0 motion with no mv overhead.
    FULLPEL_MV mv = kZeroFullMv;
  //  FULLPEL_MV tmp_mv = kZeroFullMv;
    //xd->plane[0].pre[0].buf = last_frame->y_buffer + recon_yoffset;
    //// Set up limit values for motion vectors to prevent them extending
    //// outside the UMV borders.
    //av1_set_mv_col_limits(mi_params, &x->mv_limits, (mb_col << FP_MIB_SIZE_LOG2),
    //    (fp_block_size_height >> MI_SIZE_LOG2),
    //    cpi->oxcf.border_in_pixels);

    uint32_t full_lambda = context_ptr->full_lambda_md[EB_8_BIT_MD];
    int errorperbit = full_lambda >> RD_EPB_SHIFT;
    errorperbit += (errorperbit == 0);
    EbSpatialFullDistType spatial_full_dist_type_fun = context_ptr->hbd_mode_decision
        ? full_distortion_kernel16_bits
        : spatial_full_distortion_kernel;

    int motion_error = 0;
           // get_prediction_error_bitdepth(is_high_bitdepth, bitdepth, bsize,
       //     &x->plane[0].src, &xd->plane[0].pre[0]);
    // Compute the motion error of the 0,0 motion using the last source
    // frame as the reference. Skip the further motion search on
    // reconstructed frame if this error is small.
    //const int raw_motion_error = raw_motion_err_list[0];

    // TODO(pengchong): Replace the hard-coded threshold
    // anaghdin to check
    if (1)//(raw_motion_error > LOW_MOTION_ERROR_THRESH)
    {
        //// Test last reference frame using the previous best mv as the
        //// starting point (best reference) for the search.
        //first_pass_motion_search(cpi, x, best_ref_mv, &mv, &motion_error);

        //// If the current best reference mv is not centered on 0,0 then do a
        //// 0,0 based search as well.
        //if (!is_zero_mv(best_ref_mv)) {
        //    int tmp_err = INT_MAX;
        //    first_pass_motion_search(cpi, x, &kZeroMv, &tmp_mv, &tmp_err);

        //    if (tmp_err < motion_error) {
        //        motion_error = tmp_err;
        //        mv = tmp_mv;
        //    }
        //}

        uint32_t cand_index = 1;
        ModeDecisionCandidateBuffer **candidate_buffer_ptr_array_base =
            context_ptr->candidate_buffer_ptr_array;
        ModeDecisionCandidateBuffer **candidate_buffer_ptr_array =
            &(candidate_buffer_ptr_array_base[0]);

        candidate_buffer = candidate_buffer_ptr_array[cand_index];
        candidate_ptr = candidate_buffer->candidate_ptr =
            &context_ptr->fast_candidate_array[cand_index];
        context_ptr->best_candidate_index_array[cand_index] = cand_index;
        // Initialize tx_depth
        candidate_buffer->candidate_ptr->tx_depth =
            (bsize == BLOCK_16X16 ? 2 : bsize == BLOCK_8X8 ? 1 : 0);
        candidate_buffer->candidate_ptr->fast_luma_rate = 0;
        candidate_buffer->candidate_ptr->fast_chroma_rate = 0;
        candidate_buffer->candidate_ptr->interp_filters = 0;

        first_pass_loop_core(pcs_ptr,
            //context_ptr->sb_ptr,
            blk_ptr,
            context_ptr,
            candidate_buffer,
            candidate_ptr,
            input_picture_ptr,
            input_origin_index,
            //input_cb_origin_in_index,
            blk_origin_index,
            //blk_chroma_origin_index,
            ref_fast_cost);

        // anaghdin to check the above logic
        mv.col = candidate_buffer->candidate_ptr->motion_vector_xl0>>3;
        mv.row = candidate_buffer->candidate_ptr->motion_vector_yl0>>3;

        last_mv->col = candidate_buffer->candidate_ptr->motion_vector_pred_x[REF_LIST_0];
        last_mv->row = candidate_buffer->candidate_ptr->motion_vector_pred_y[REF_LIST_0];

        motion_error = (uint32_t)(
            spatial_full_dist_type_fun(input_picture_ptr->buffer_y,
                input_origin_index,
                input_picture_ptr->stride_y,
                candidate_buffer->prediction_ptr->buffer_y,
                blk_origin_index,
                candidate_buffer->prediction_ptr->stride_y,
                context_ptr->blk_geom->bwidth,
                context_ptr->blk_geom->bheight));

        // Assume 0,0 motion with no mv overhead.
        if (mv.col != 0 && mv.row != 0) {
            const MV temp_full_mv = get_mv_from_fullmv(&mv);
            motion_error += mv_err_cost(&temp_full_mv, last_mv, context_ptr->md_rate_estimation_ptr->nmv_vec_cost, context_ptr->md_rate_estimation_ptr->nmvcoststack, errorperbit) +
                NEW_MV_MODE_PENALTY;
        }

        // Motion search in 2nd reference frame.
        int gf_motion_error = motion_error;
//    // Assume 0,0 motion with no mv overhead.
//    gf_motion_error =
//        get_prediction_error_bitdepth(is_high_bitdepth, bitdepth, bsize,
//            &x->plane[0].src, &xd->plane[0].pre[0]);
//    first_pass_motion_search(cpi, x, &kZeroMv, &tmp_mv, &gf_motion_error);
//}
       if (fast_candidate_total_count > 2) {
            cand_index++;
            candidate_buffer = candidate_buffer_ptr_array[cand_index];
            candidate_ptr = candidate_buffer->candidate_ptr =
                &context_ptr->fast_candidate_array[cand_index];
            context_ptr->best_candidate_index_array[cand_index] = cand_index;
            // Initialize tx_depth
            candidate_buffer->candidate_ptr->tx_depth =
                (bsize == BLOCK_16X16 ? 2 : bsize == BLOCK_8X8 ? 1 : 0);
            candidate_buffer->candidate_ptr->fast_luma_rate = 0;
            candidate_buffer->candidate_ptr->fast_chroma_rate = 0;
            candidate_buffer->candidate_ptr->interp_filters = 0;
            // anaghdin: no need to do everything, we just need the prediction
            first_pass_loop_core(pcs_ptr,
                //context_ptr->sb_ptr,
                blk_ptr,
                context_ptr,
                candidate_buffer,
                candidate_ptr,
                input_picture_ptr,
                input_origin_index,
                //input_cb_origin_in_index,
                blk_origin_index,
                //blk_chroma_origin_index,
                ref_fast_cost);

            gf_motion_error = (uint32_t)(
                spatial_full_dist_type_fun(input_picture_ptr->buffer_y,
                    input_origin_index,
                    input_picture_ptr->stride_y,
                    candidate_buffer->prediction_ptr->buffer_y,
                    blk_origin_index,
                    candidate_buffer->prediction_ptr->stride_y,
                    context_ptr->blk_geom->bwidth,
                    context_ptr->blk_geom->bheight));
            FULLPEL_MV gf_mv;
            gf_mv.col = candidate_buffer->candidate_ptr->motion_vector_xl1 >> 3;
            gf_mv.row = candidate_buffer->candidate_ptr->motion_vector_yl1 >> 3;

            // Assume 0,0 motion with no mv overhead.
            if (gf_mv.col != 0 && gf_mv.row != 0) {
                const MV temp_full_mv = get_mv_from_fullmv(&gf_mv);
                gf_motion_error += mv_err_cost(&temp_full_mv, &kZeroMv, context_ptr->md_rate_estimation_ptr->nmv_vec_cost, context_ptr->md_rate_estimation_ptr->nmvcoststack, errorperbit) +
                    NEW_MV_MODE_PENALTY;
            }
        }

        if (gf_motion_error < motion_error && gf_motion_error < this_intra_error) {
            ++stats->second_ref_count;
        }
        // In accumulating a score for the 2nd reference frame take the
        // best of the motion predicted score and the intra coded error
        // (just as will be done for) accumulation of "coded_error" for
        // the last frame.
        if (fast_candidate_total_count > 2) {
        //    if ((current_frame->frame_number > 1) && golden_frame != NULL) {
            stats->sr_coded_error += AOMMIN(gf_motion_error, this_intra_error);
        }
        else {
            // TODO(chengchen): I believe logically this should also be changed to
            // stats->sr_coded_error += AOMMIN(gf_motion_error, this_intra_error).
            stats->sr_coded_error += motion_error;
        }

        // Motion search in 3rd reference frame.
        int alt_motion_error = motion_error;
        //if (alt_ref_frame != NULL) {
        //    xd->plane[0].pre[0].buf = alt_ref_frame->y_buffer + alt_ref_frame_yoffset;
        //    xd->plane[0].pre[0].stride = alt_ref_frame->y_stride;
        //    alt_motion_error =
        //        get_prediction_error_bitdepth(is_high_bitdepth, bitdepth, bsize,
        //            &x->plane[0].src, &xd->plane[0].pre[0]);
        //    first_pass_motion_search(cpi, x, &kZeroMv, &tmp_mv, &alt_motion_error);
        //}
        if (alt_motion_error < motion_error && alt_motion_error < gf_motion_error &&
            alt_motion_error < this_intra_error) {
            ++stats->third_ref_count;
        }
        // In accumulating a score for the 3rd reference frame take the
        // best of the motion predicted score and the intra coded error
        // (just as will be done for) accumulation of "coded_error" for
        // the last frame.
        if (0/*alt_ref_frame != NULL*/) {
            stats->tr_coded_error += AOMMIN(alt_motion_error, this_intra_error);
        }
        else {
            // TODO(chengchen): I believe logically this should also be changed to
            // stats->tr_coded_error += AOMMIN(alt_motion_error, this_intra_error).
            stats->tr_coded_error += motion_error;
        }
    }
    else {
        stats->sr_coded_error += motion_error;
        stats->tr_coded_error += motion_error;
    }

    // Start by assuming that intra mode is best.
    best_ref_mv->row = 0;
    best_ref_mv->col = 0;

    if (motion_error <= this_intra_error) {
        aom_clear_system_state();

        // Keep a count of cases where the inter and intra were very close
        // and very low. This helps with scene cut detection for example in
        // cropped clips with black bars at the sides or top and bottom.
        if (((this_intra_error - INTRA_MODE_PENALTY) * 9 <= motion_error * 10) &&
            (this_intra_error < (2 * INTRA_MODE_PENALTY))) {
            stats->neutral_count += 1.0;
            // Also track cases where the intra is not much worse than the inter
            // and use this in limiting the GF/arf group length.
        }
        else if ((this_intra_error > NCOUNT_INTRA_THRESH) &&
            (this_intra_error < (NCOUNT_INTRA_FACTOR * motion_error))) {
            stats->neutral_count +=
                (double)motion_error / DOUBLE_DIVIDE_CHECK((double)this_intra_error);
        }
        const MV best_mv = get_mv_from_fullmv(&mv);
        this_inter_error = motion_error;
        stats->sum_mvr += best_mv.row;
        stats->sum_mvr_abs += abs(best_mv.row);
        stats->sum_mvc += best_mv.col;
        stats->sum_mvc_abs += abs(best_mv.col);
        stats->sum_mvrs += best_mv.row * best_mv.row;
        stats->sum_mvcs += best_mv.col * best_mv.col;
        ++stats->inter_count;

        *best_ref_mv = best_mv;
        accumulate_mv_stats(best_mv, mv, mb_row, mb_col, mb_rows,
            mb_cols, last_mv, stats);
    }

    return this_inter_error;
}
#endif

#if FIRST_PASS_SETUP
void soft_cycles_reduction_mrp(ModeDecisionContext *context_ptr, uint8_t *mrp_level);
void set_inter_inter_distortion_based_reference_pruning_controls(
    ModeDecisionContext *mdctxt, uint8_t inter_inter_distortion_based_reference_pruning_mode) ;
void set_inter_comp_controls(ModeDecisionContext *mdctxt, uint8_t inter_comp_mode);
/******************************************************
* Derive md Settings(feature signals) that could be
  changed  at the block level
******************************************************/
extern EbErrorType first_pass_signal_derivation_block(
    ModeDecisionContext   *context_ptr) {

    EbErrorType return_error = EB_ErrorNone;


    // Set inter_inter_distortion_based_reference_pruning
    context_ptr->inter_inter_distortion_based_reference_pruning = 0;

    soft_cycles_reduction_mrp(context_ptr, &context_ptr->inter_inter_distortion_based_reference_pruning);
    set_inter_inter_distortion_based_reference_pruning_controls(context_ptr, context_ptr->inter_inter_distortion_based_reference_pruning);


    // set compound_types_to_try
    set_inter_comp_controls(context_ptr, 0);

    context_ptr->compound_types_to_try = MD_COMP_AVG;

    BlkStruct *similar_cu = &context_ptr->md_blk_arr_nsq[context_ptr->similar_blk_mds];
    if (context_ptr->compound_types_to_try > MD_COMP_AVG && context_ptr->similar_blk_avail) {
        int32_t is_src_compound = similar_cu->pred_mode >= NEAREST_NEARESTMV;
#if INTER_COMP_REDESIGN
        if (context_ptr->inter_comp_ctrls.similar_previous_blk == 1) {
#else
        if (context_ptr->comp_similar_mode == 1) {
#endif
            context_ptr->compound_types_to_try = !is_src_compound ? MD_COMP_AVG : context_ptr->compound_types_to_try;
        }
#if INTER_COMP_REDESIGN
        else if (context_ptr->inter_comp_ctrls.similar_previous_blk == 2) {
#else
        else if (context_ptr->comp_similar_mode == 2) {
#endif
            context_ptr->compound_types_to_try = !is_src_compound ? MD_COMP_AVG : similar_cu->interinter_comp.type;
        }
        }
#if INTER_COMP_REDESIGN
    // Do not add MD_COMP_WEDGE  beyond this point
    if (get_wedge_params_bits(context_ptr->blk_geom->bsize) == 0)
        context_ptr->compound_types_to_try = MIN(context_ptr->compound_types_to_try, MD_COMP_DIFF0);
#endif
    context_ptr->inject_inter_candidates = 1;
    if (context_ptr->pd_pass > PD_PASS_1 && context_ptr->similar_blk_avail) {
        int32_t is_src_intra = similar_cu->pred_mode <= PAETH_PRED;
        if (context_ptr->intra_similar_mode)
            context_ptr->inject_inter_candidates = is_src_intra ? 0 : context_ptr->inject_inter_candidates;
    }

    return return_error;
}

#endif
#if FIRST_PASS_SETUP
void product_coding_loop_init_fast_loop(ModeDecisionContext *context_ptr,
                                        NeighborArrayUnit *  skip_coeff_neighbor_array,
                                        NeighborArrayUnit *  inter_pred_dir_neighbor_array,
                                        NeighborArrayUnit *  ref_frame_type_neighbor_array,
                                        NeighborArrayUnit *  intra_luma_mode_neighbor_array,
                                        NeighborArrayUnit *  skip_flag_neighbor_array,
                                        NeighborArrayUnit *  mode_type_neighbor_array,
                                        NeighborArrayUnit *  leaf_depth_neighbor_array,
                                        NeighborArrayUnit *  leaf_partition_neighbor_array);
void read_refine_me_mvs(PictureControlSet *pcs_ptr, ModeDecisionContext *context_ptr,
    EbPictureBufferDesc *input_picture_ptr, uint32_t input_origin_index,
    uint32_t blk_origin_index);
void perform_md_reference_pruning(PictureControlSet *pcs_ptr, ModeDecisionContext *context_ptr,
                         EbPictureBufferDesc *input_picture_ptr, uint32_t blk_origin_index);
void    predictive_me_search(PictureControlSet *pcs_ptr, ModeDecisionContext *context_ptr,
                             EbPictureBufferDesc *input_picture_ptr, uint32_t input_origin_index,
                             uint32_t blk_origin_index);
EbErrorType generate_md_stage_0_cand(SuperBlock *sb_ptr, ModeDecisionContext *context_ptr,
                                     uint32_t *         fast_candidate_total_count,
                                     PictureControlSet *pcs_ptr);
void av1_perform_inverse_transform_recon(ModeDecisionContext *        context_ptr,
                                         ModeDecisionCandidateBuffer *candidate_buffer);
#if FIX_WARNINGS
void distortion_based_modulator(ModeDecisionContext *context_ptr,
#else
void distortion_based_modulator(PictureControlSet *pcs_ptr,ModeDecisionContext *context_ptr,

#endif
    EbPictureBufferDesc *input_picture_ptr, uint32_t input_origin_index,
    EbPictureBufferDesc *recon_ptr, uint32_t blk_origin_index);


extern void first_pass_md_encode_block(PictureControlSet *pcs_ptr,
    ModeDecisionContext *context_ptr, EbPictureBufferDesc *input_picture_ptr,
    ModeDecisionCandidateBuffer *bestcandidate_buffers[5]) {
    ModeDecisionCandidateBuffer **candidate_buffer_ptr_array_base =
        context_ptr->candidate_buffer_ptr_array;
    ModeDecisionCandidateBuffer **candidate_buffer_ptr_array;
    const BlockGeom *             blk_geom = context_ptr->blk_geom;
    ModeDecisionCandidateBuffer * candidate_buffer;
    ModeDecisionCandidate *       fast_candidate_array = context_ptr->fast_candidate_array;
    uint32_t                      candidate_index;
    uint32_t                      fast_candidate_total_count;
    uint32_t                      best_intra_mode = EB_INTRA_MODE_INVALID;
    const uint32_t                input_origin_index =
        (context_ptr->blk_origin_y + input_picture_ptr->origin_y) * input_picture_ptr->stride_y +
        (context_ptr->blk_origin_x + input_picture_ptr->origin_x);

    //const uint32_t input_cb_origin_in_index =
    //    ((context_ptr->round_origin_y >> 1) + (input_picture_ptr->origin_y >> 1)) *
    //    input_picture_ptr->stride_cb +
    //    ((context_ptr->round_origin_x >> 1) + (input_picture_ptr->origin_x >> 1));
#if SB64_MEM_OPT
    const uint32_t blk_origin_index = blk_geom->origin_x + blk_geom->origin_y * context_ptr->sb_size;
    //const uint32_t blk_chroma_origin_index =
    //    ROUND_UV(blk_geom->origin_x) / 2 + ROUND_UV(blk_geom->origin_y) / 2 * (context_ptr->sb_size >> 1);
#else
    const uint32_t blk_origin_index = blk_geom->origin_x + blk_geom->origin_y * SB_STRIDE_Y;
    //const uint32_t blk_chroma_origin_index =
    //    ROUND_UV(blk_geom->origin_x) / 2 + ROUND_UV(blk_geom->origin_y) / 2 * SB_STRIDE_UV;
#endif
    BlkStruct *blk_ptr = context_ptr->blk_ptr;
    candidate_buffer_ptr_array = &(candidate_buffer_ptr_array_base[0]);
    first_pass_signal_derivation_block(
        context_ptr);

    blk_ptr->av1xd->tile.mi_col_start = context_ptr->sb_ptr->tile_info.mi_col_start;
    blk_ptr->av1xd->tile.mi_col_end = context_ptr->sb_ptr->tile_info.mi_col_end;
    blk_ptr->av1xd->tile.mi_row_start = context_ptr->sb_ptr->tile_info.mi_row_start;
    blk_ptr->av1xd->tile.mi_row_end = context_ptr->sb_ptr->tile_info.mi_row_end;

    product_coding_loop_init_fast_loop(context_ptr,
        context_ptr->skip_coeff_neighbor_array,
        context_ptr->inter_pred_dir_neighbor_array,
        context_ptr->ref_frame_type_neighbor_array,
        context_ptr->intra_luma_mode_neighbor_array,
        context_ptr->skip_flag_neighbor_array,
        context_ptr->mode_type_neighbor_array,
        context_ptr->leaf_depth_neighbor_array,
        context_ptr->leaf_partition_neighbor_array);

    FrameHeader *frm_hdr = &pcs_ptr->parent_pcs_ptr->frm_hdr;
    // Generate MVP(s)
    if (!context_ptr->md_skip_mvp_generation) {
        if (frm_hdr->allow_intrabc) // pcs_ptr->slice_type == I_SLICE
            generate_av1_mvp_table(&context_ptr->sb_ptr->tile_info,
                context_ptr,
                context_ptr->blk_ptr,
                context_ptr->blk_geom,
                context_ptr->blk_origin_x,
                context_ptr->blk_origin_y,
                pcs_ptr->parent_pcs_ptr->ref_frame_type_arr,
                1,
                pcs_ptr);
        else if (pcs_ptr->slice_type != I_SLICE)
            generate_av1_mvp_table(&context_ptr->sb_ptr->tile_info,
                context_ptr,
                context_ptr->blk_ptr,
                context_ptr->blk_geom,
                context_ptr->blk_origin_x,
                context_ptr->blk_origin_y,
                pcs_ptr->parent_pcs_ptr->ref_frame_type_arr,
                pcs_ptr->parent_pcs_ptr->tot_ref_frame_types,
                pcs_ptr);
    }
    else {
        mvp_bypass_init(pcs_ptr, context_ptr);
    }
    // Read and (if needed) perform 1/8 Pel ME MVs refinement
#if ADD_MD_NSQ_SEARCH
    if (pcs_ptr->slice_type != I_SLICE)
#endif
        read_refine_me_mvs(
            pcs_ptr, context_ptr, input_picture_ptr, input_origin_index, blk_origin_index);
#if 0 //remove
#if PME_SORT_REF
    for (uint32_t li = 0; li < MAX_NUM_OF_REF_PIC_LIST; ++li) {
        for (uint32_t ri = 0; ri < REF_LIST_MAX_DEPTH; ++ri) {
            context_ptr->pme_res[li][ri].dist = 0xFFFFFFFF;
            context_ptr->pme_res[li][ri].list_i = li;
            context_ptr->pme_res[li][ri].ref_i = ri;
#if !INTER_COMP_REDESIGN
            context_ptr->pme_res[li][ri].do_ref = 1;
#endif
        }
    }
#endif
#endif
#if MD_REFERENCE_MASKING
    // Perform md reference pruning
    perform_md_reference_pruning(
        pcs_ptr, context_ptr, input_picture_ptr, blk_origin_index);
#endif
    // Perform ME search around the best MVP
    if (context_ptr->predictive_me_level)
        predictive_me_search(
            pcs_ptr, context_ptr, input_picture_ptr, input_origin_index, blk_origin_index);
    //for every CU, perform Luma DC/V/H/S intra prediction to be used later in inter-intra search

    context_ptr->inject_inter_candidates = 1;// anaghdin add signal_derivation_block
    // anaghdin create a new one
    generate_md_stage_0_cand(
        context_ptr->sb_ptr, context_ptr, &fast_candidate_total_count, pcs_ptr);

    uint64_t ref_fast_cost = MAX_MODE_COST;

    int32_t       mb_row = context_ptr->blk_origin_y >> 4;
    int32_t       mb_col = context_ptr->blk_origin_x >> 4;
    const uint32_t mb_cols = (pcs_ptr->parent_pcs_ptr->scs_ptr->seq_header.max_frame_width + 16 - 1) / 16;
    FRAME_STATS *mb_stats =
        pcs_ptr->parent_pcs_ptr->firstpass_data.mb_stats + mb_row * mb_cols + mb_col;
    //int *raw_motion_err_list = pcs_ptr->parent_pcs_ptr->firstpass_data.raw_motion_err_list +
    //    mb_row * mb_cols + mb_col;

    ModeDecisionCandidate *      candidate_ptr;
    uint32_t cand_index = 0;
    candidate_buffer = candidate_buffer_ptr_array[cand_index];
    candidate_ptr = candidate_buffer->candidate_ptr =
        &fast_candidate_array[cand_index];

    int this_intra_error = firstpass_intra_prediction(pcs_ptr,
        blk_ptr,
        context_ptr,
        candidate_buffer,
        candidate_ptr,
        input_picture_ptr,
        input_origin_index,
        //input_cb_origin_in_index,
        blk_origin_index,
        //blk_chroma_origin_index,
        ref_fast_cost,
        mb_stats);

    int this_inter_error = this_intra_error;
    if (pcs_ptr->slice_type != I_SLICE && fast_candidate_total_count > 1) {
        MV firstpass_top_mv = kZeroMv;
        MV *best_ref_mv = &firstpass_top_mv; // anaghdin to set we might need later if we modify me
        MV last_mv = kZeroMv;// anaghdin: for now we overright it internaly with the mv pred
        this_inter_error = firstpass_inter_prediction(pcs_ptr,
            blk_ptr,
            context_ptr,
            candidate_buffer,
            candidate_ptr,
            input_picture_ptr,
            input_origin_index,
            //input_cb_origin_in_index,
            blk_origin_index,
            //blk_chroma_origin_index,
            ref_fast_cost,
            fast_candidate_total_count,
            this_intra_error,
            //raw_motion_err_list,
            best_ref_mv,
            &last_mv,
            mb_stats);

        mb_stats->coded_error += this_inter_error;
    } else {
        mb_stats->sr_coded_error += this_intra_error;
        mb_stats->tr_coded_error += this_intra_error;
        mb_stats->coded_error += this_intra_error;
    }
    // choose between Intra and inter LAST based on inter/intra error
    if (this_inter_error < this_intra_error)
        context_ptr->best_candidate_index_array[0] = 1;
    else
        context_ptr->best_candidate_index_array[0] = 0;
    // Handle stat for non 16x16 blocks. For non 16x16 blocks, some of the stats are increased multiple times
    // First find the last block in the 16x16 area and then devide the stats by the number of small blocks
    if (context_ptr->blk_geom->bsize != BLOCK_16X16 &&
        (context_ptr->blk_origin_x + context_ptr->blk_geom->bwidth == pcs_ptr->parent_pcs_ptr->aligned_width ||
         (context_ptr->blk_geom->origin_x +  context_ptr->blk_geom->bwidth) % FORCED_BLK_SIZE == 0) &&
        (context_ptr->blk_origin_y + context_ptr->blk_geom->bheight == pcs_ptr->parent_pcs_ptr->aligned_height ||
            (context_ptr->blk_geom->origin_y + context_ptr->blk_geom->bheight)% FORCED_BLK_SIZE== 0)) {
        int blk_num =  (((context_ptr->blk_geom->origin_x % FORCED_BLK_SIZE) + context_ptr->blk_geom->bwidth) / context_ptr->blk_geom->bwidth)*
            (((context_ptr->blk_geom->origin_y % FORCED_BLK_SIZE) + context_ptr->blk_geom->bheight) / context_ptr->blk_geom->bheight);
        average_non_16x16_stats(mb_stats, blk_num);
    }

// Full Mode Decision (choose the best mode)
    candidate_index = product_full_mode_decision(
        context_ptr,
        blk_ptr,
        candidate_buffer_ptr_array,
        1,//fast_candidate_total_count,//context_ptr->md_stage_3_total_count,
#if M8_CLEAN_UP
        context_ptr->best_candidate_index_array,
#else
        (context_ptr->full_loop_escape == 2) ? context_ptr->sorted_candidate_index_array
        : context_ptr->best_candidate_index_array,
#endif
        context_ptr->prune_ref_frame_for_rec_partitions,
        &best_intra_mode);
    candidate_buffer = candidate_buffer_ptr_array[candidate_index];

    bestcandidate_buffers[0] = candidate_buffer;
    uint8_t sq_index = LOG2F(context_ptr->blk_geom->sq_size) - 2;
    if (context_ptr->blk_geom->shape == PART_N) {
        context_ptr->parent_sq_type[sq_index] = candidate_buffer->candidate_ptr->type;

        context_ptr->parent_sq_has_coeff[sq_index] =
            (candidate_buffer->candidate_ptr->y_has_coeff ||
                candidate_buffer->candidate_ptr->u_has_coeff ||
                candidate_buffer->candidate_ptr->v_has_coeff)
            ? 1
            : 0;

        context_ptr->parent_sq_pred_mode[sq_index] = candidate_buffer->candidate_ptr->pred_mode;
    }
#if REMOVE_UNUSED_CODE_PH2
    av1_perform_inverse_transform_recon(
        context_ptr, candidate_buffer);
#if !CLEAN_UP_SB_DATA_8
    blk_ptr,
#endif
#else
    av1_perform_inverse_transform_recon(
        pcs_ptr, context_ptr, candidate_buffer,
#if !CLEAN_UP_SB_DATA_8
        blk_ptr,
#endif
        context_ptr->blk_geom);
#endif
    if (!context_ptr->blk_geom->has_uv) {
        // Store the luma data for 4x* and *x4 blocks to be used for CFL
        EbPictureBufferDesc *recon_ptr = candidate_buffer->recon_ptr;
        uint32_t             rec_luma_offset = context_ptr->blk_geom->origin_x +
            context_ptr->blk_geom->origin_y * recon_ptr->stride_y;
        if (context_ptr->hbd_mode_decision) {
            for (uint32_t j = 0; j < context_ptr->blk_geom->bheight; ++j)
                memcpy(context_ptr->cfl_temp_luma_recon16bit + rec_luma_offset +
                    j * recon_ptr->stride_y,
                    ((uint16_t *)recon_ptr->buffer_y) +
                    (rec_luma_offset + j * recon_ptr->stride_y),
                    sizeof(uint16_t) * context_ptr->blk_geom->bwidth);
        }
        else {
            for (uint32_t j = 0; j < context_ptr->blk_geom->bheight; ++j)
                memcpy(&context_ptr
                    ->cfl_temp_luma_recon[rec_luma_offset + j * recon_ptr->stride_y],
                    recon_ptr->buffer_y + rec_luma_offset + j * recon_ptr->stride_y,
                    context_ptr->blk_geom->bwidth);
        }
    }
    //copy neigh recon data in blk_ptr
    {
        uint32_t             j;
        EbPictureBufferDesc *recon_ptr = candidate_buffer->recon_ptr;
        uint32_t             rec_luma_offset = context_ptr->blk_geom->origin_x +
            context_ptr->blk_geom->origin_y * recon_ptr->stride_y;

        uint32_t rec_cb_offset = ((((context_ptr->blk_geom->origin_x >> 3) << 3) +
            ((context_ptr->blk_geom->origin_y >> 3) << 3) *
            candidate_buffer->recon_ptr->stride_cb) >>
            1);
        uint32_t rec_cr_offset = ((((context_ptr->blk_geom->origin_x >> 3) << 3) +
            ((context_ptr->blk_geom->origin_y >> 3) << 3) *
            candidate_buffer->recon_ptr->stride_cr) >>
            1);
#if CLEAN_UP_SB_DATA_3
        if (!context_ptr->hbd_mode_decision) {
#if SSE_BASED_SPLITTING
#if FIX_WARNINGS
            distortion_based_modulator(context_ptr, input_picture_ptr, input_origin_index,
#else
            distortion_based_modulator(pcs_ptr, context_ptr, input_picture_ptr, input_origin_index,
#endif
                recon_ptr, blk_origin_index);
#endif
            memcpy(context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_top_recon[0],
                recon_ptr->buffer_y + rec_luma_offset +
                (context_ptr->blk_geom->bheight - 1) * recon_ptr->stride_y,
                context_ptr->blk_geom->bwidth);
            if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
                memcpy(context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_top_recon[1],
                    recon_ptr->buffer_cb + rec_cb_offset +
                    (context_ptr->blk_geom->bheight_uv - 1) * recon_ptr->stride_cb,
                    context_ptr->blk_geom->bwidth_uv);
                memcpy(context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_top_recon[2],
                    recon_ptr->buffer_cr + rec_cr_offset +
                    (context_ptr->blk_geom->bheight_uv - 1) * recon_ptr->stride_cr,
                    context_ptr->blk_geom->bwidth_uv);
            }

            for (j = 0; j < context_ptr->blk_geom->bheight; ++j)
                context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_left_recon[0][j] =
                recon_ptr->buffer_y[rec_luma_offset + context_ptr->blk_geom->bwidth - 1 +
                j * recon_ptr->stride_y];

            if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
                for (j = 0; j < context_ptr->blk_geom->bheight_uv; ++j) {
                    context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_left_recon[1][j] =
                        recon_ptr->buffer_cb[rec_cb_offset + context_ptr->blk_geom->bwidth_uv -
                        1 + j * recon_ptr->stride_cb];
                    context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_left_recon[2][j] =
                        recon_ptr->buffer_cr[rec_cr_offset + context_ptr->blk_geom->bwidth_uv -
                        1 + j * recon_ptr->stride_cr];
                }
            }
        }
        else {
            uint16_t sz = sizeof(uint16_t);
            memcpy(context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_top_recon_16bit[0],
                recon_ptr->buffer_y +
                sz * (rec_luma_offset +
                (context_ptr->blk_geom->bheight - 1) * recon_ptr->stride_y),
                sz * context_ptr->blk_geom->bwidth);
            if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
                memcpy(context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_top_recon_16bit[1],
                    recon_ptr->buffer_cb +
                    sz * (rec_cb_offset + (context_ptr->blk_geom->bheight_uv - 1) *
                        recon_ptr->stride_cb),
                    sz * context_ptr->blk_geom->bwidth_uv);
                memcpy(context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_top_recon_16bit[2],
                    recon_ptr->buffer_cr +
                    sz * (rec_cr_offset + (context_ptr->blk_geom->bheight_uv - 1) *
                        recon_ptr->stride_cr),
                    sz * context_ptr->blk_geom->bwidth_uv);
            }

            for (j = 0; j < context_ptr->blk_geom->bheight; ++j)
                context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_left_recon_16bit[0][j] =
                ((uint16_t *)
                    recon_ptr->buffer_y)[rec_luma_offset + context_ptr->blk_geom->bwidth -
                1 + j * recon_ptr->stride_y];

            if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
                for (j = 0; j < context_ptr->blk_geom->bheight_uv; ++j) {
                    context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_left_recon_16bit[1][j] =
                        ((uint16_t *)recon_ptr
                            ->buffer_cb)[rec_cb_offset + context_ptr->blk_geom->bwidth_uv - 1 +
                        j * recon_ptr->stride_cb];
                    context_ptr->md_local_blk_unit[context_ptr->blk_geom->blkidx_mds].neigh_left_recon_16bit[2][j] =
                        ((uint16_t *)recon_ptr
                            ->buffer_cr)[rec_cr_offset + context_ptr->blk_geom->bwidth_uv - 1 +
                        j * recon_ptr->stride_cr];
                }
            }
        }
#else
        if (!context_ptr->hbd_mode_decision) {
            memcpy(blk_ptr->neigh_top_recon[0],
                recon_ptr->buffer_y + rec_luma_offset +
                (context_ptr->blk_geom->bheight - 1) * recon_ptr->stride_y,
                context_ptr->blk_geom->bwidth);
            if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
                memcpy(blk_ptr->neigh_top_recon[1],
                    recon_ptr->buffer_cb + rec_cb_offset +
                    (context_ptr->blk_geom->bheight_uv - 1) * recon_ptr->stride_cb,
                    context_ptr->blk_geom->bwidth_uv);
                memcpy(blk_ptr->neigh_top_recon[2],
                    recon_ptr->buffer_cr + rec_cr_offset +
                    (context_ptr->blk_geom->bheight_uv - 1) * recon_ptr->stride_cr,
                    context_ptr->blk_geom->bwidth_uv);
            }

            for (j = 0; j < context_ptr->blk_geom->bheight; ++j)
                blk_ptr->neigh_left_recon[0][j] =
                recon_ptr->buffer_y[rec_luma_offset + context_ptr->blk_geom->bwidth - 1 +
                j * recon_ptr->stride_y];

            if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
                for (j = 0; j < context_ptr->blk_geom->bheight_uv; ++j) {
                    blk_ptr->neigh_left_recon[1][j] =
                        recon_ptr->buffer_cb[rec_cb_offset + context_ptr->blk_geom->bwidth_uv -
                        1 + j * recon_ptr->stride_cb];
                    blk_ptr->neigh_left_recon[2][j] =
                        recon_ptr->buffer_cr[rec_cr_offset + context_ptr->blk_geom->bwidth_uv -
                        1 + j * recon_ptr->stride_cr];
                }
            }
        }
        else {
            uint16_t sz = sizeof(uint16_t);
            memcpy(blk_ptr->neigh_top_recon_16bit[0],
                recon_ptr->buffer_y +
                sz * (rec_luma_offset +
                (context_ptr->blk_geom->bheight - 1) * recon_ptr->stride_y),
                sz * context_ptr->blk_geom->bwidth);
            if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
                memcpy(blk_ptr->neigh_top_recon_16bit[1],
                    recon_ptr->buffer_cb +
                    sz * (rec_cb_offset + (context_ptr->blk_geom->bheight_uv - 1) *
                        recon_ptr->stride_cb),
                    sz * context_ptr->blk_geom->bwidth_uv);
                memcpy(blk_ptr->neigh_top_recon_16bit[2],
                    recon_ptr->buffer_cr +
                    sz * (rec_cr_offset + (context_ptr->blk_geom->bheight_uv - 1) *
                        recon_ptr->stride_cr),
                    sz * context_ptr->blk_geom->bwidth_uv);
            }

            for (j = 0; j < context_ptr->blk_geom->bheight; ++j)
                blk_ptr->neigh_left_recon_16bit[0][j] =
                ((uint16_t *)
                    recon_ptr->buffer_y)[rec_luma_offset + context_ptr->blk_geom->bwidth -
                1 + j * recon_ptr->stride_y];

            if (context_ptr->blk_geom->has_uv && context_ptr->chroma_level <= CHROMA_MODE_1) {
                for (j = 0; j < context_ptr->blk_geom->bheight_uv; ++j) {
                    blk_ptr->neigh_left_recon_16bit[1][j] =
                        ((uint16_t *)recon_ptr
                            ->buffer_cb)[rec_cb_offset + context_ptr->blk_geom->bwidth_uv - 1 +
                        j * recon_ptr->stride_cb];
                    blk_ptr->neigh_left_recon_16bit[2][j] =
                        ((uint16_t *)recon_ptr
                            ->buffer_cr)[rec_cr_offset + context_ptr->blk_geom->bwidth_uv - 1 +
                        j * recon_ptr->stride_cr];
                }
            }
        }
#endif
    }

    context_ptr->md_local_blk_unit[blk_ptr->mds_idx].avail_blk_flag = EB_TRUE;

}
#endif

#if FIRST_PASS_SETUP

void set_tf_controls(PictureDecisionContext *context_ptr, uint8_t tf_level);
/******************************************************
* Derive Multi-Processes Settings for first pass
Input   : encoder mode and tune
Output  : Multi-Processes signal(s)
******************************************************/
EbErrorType first_pass_signal_derivation_multi_processes(
    SequenceControlSet *scs_ptr,
    PictureParentControlSet *pcs_ptr,
    PictureDecisionContext *context_ptr) {

    EbErrorType return_error = EB_ErrorNone;
    FrameHeader *frm_hdr = &pcs_ptr->frm_hdr;
    // If enabled here, the hme enable flags should also be enabled in ResourceCoordinationProcess
    // to ensure that resources are allocated for the downsampled pictures used in HME
    pcs_ptr->enable_hme_flag = 1;
    pcs_ptr->enable_hme_level0_flag = 1;
    pcs_ptr->enable_hme_level1_flag = 1;
    pcs_ptr->enable_hme_level2_flag = 1;


    pcs_ptr->tf_enable_hme_flag = 0;
    pcs_ptr->tf_enable_hme_level0_flag = 0;
    pcs_ptr->tf_enable_hme_level1_flag = 0;
    pcs_ptr->tf_enable_hme_level2_flag = 0;

    // Set the Multi-Pass PD level
    pcs_ptr->multi_pass_pd_level = MULTI_PASS_PD_OFF;

    // Set disallow_nsq
    pcs_ptr->disallow_nsq = EB_TRUE;

    pcs_ptr->max_number_of_pus_per_sb = SQUARE_PU_COUNT;
    pcs_ptr->disallow_all_nsq_blocks_below_8x8 = EB_TRUE;

    // Set disallow_all_nsq_blocks_below_16x16: 16x8, 8x16, 16x4, 4x16
    pcs_ptr->disallow_all_nsq_blocks_below_16x16 = EB_TRUE;

    pcs_ptr->disallow_all_nsq_blocks_below_64x64 = EB_TRUE;
    pcs_ptr->disallow_all_nsq_blocks_below_32x32 = EB_TRUE;
    pcs_ptr->disallow_all_nsq_blocks_above_64x64 = EB_TRUE;
    pcs_ptr->disallow_all_nsq_blocks_above_32x32 = EB_TRUE;
    // disallow_all_nsq_blocks_above_16x16
    pcs_ptr->disallow_all_nsq_blocks_above_16x16 = EB_TRUE;

    pcs_ptr->disallow_HVA_HVB_HV4 = EB_TRUE;
    pcs_ptr->disallow_HV4 = EB_TRUE;

    // Set disallow_all_non_hv_nsq_blocks_below_16x16
    pcs_ptr->disallow_all_non_hv_nsq_blocks_below_16x16 = EB_TRUE;

    // Set disallow_all_h4_v4_blocks_below_16x16
    pcs_ptr->disallow_all_h4_v4_blocks_below_16x16 = EB_TRUE;

    frm_hdr->allow_screen_content_tools = 0;
    frm_hdr->allow_intrabc = 0;

    // Palette Modes:
    //    0:OFF
    //    1:Slow    NIC=7/4/4
    //    2:        NIC=7/2/2
    //    3:        NIC=7/2/2 + No K means for non ref
    //    4:        NIC=4/2/1
    //    5:        NIC=4/2/1 + No K means for Inter frame
    //    6:        Fastest NIC=4/2/1 + No K means for non base + step for non base for most dominent
    pcs_ptr->palette_mode = 0;
    // Loop filter Level                            Settings
    // 0                                            OFF
    // 1                                            CU-BASED
    // 2                                            LIGHT FRAME-BASED
    // 3                                            FULL FRAME-BASED
    pcs_ptr->loop_filter_mode = 0;

    // CDEF Level                                   Settings
    // 0                                            OFF
    // 1                                            1 step refinement
    // 2                                            4 step refinement
    // 3                                            8 step refinement
    // 4                                            16 step refinement
    // 5                                            64 step refinement
    pcs_ptr->cdef_filter_mode = 0;

    // SG Level                                    Settings
    // 0                                            OFF
    // 1                                            0 step refinement
    // 2                                            1 step refinement
    // 3                                            4 step refinement
    // 4                                            16 step refinement
    Av1Common *cm = pcs_ptr->av1_cm;
    cm->sg_filter_mode = 0;

    // WN Level                                     Settings
    // 0                                            OFF
    // 1                                            3-Tap luma/ 3-Tap chroma
    // 2                                            5-Tap luma/ 5-Tap chroma
    // 3                                            7-Tap luma/ 5-Tap chroma
    cm->wn_filter_mode = 0;

    // Intra prediction modes                       Settings
    // 0                                            FULL
    // 1                                            LIGHT per block : disable_z2_prediction && disable_angle_refinement  for 64/32/4
    // 2                                            OFF per block : disable_angle_prediction for 64/32/4
    // 3                                            OFF : disable_angle_prediction
    // 4                                            OIS based Intra
    // 5                                            Light OIS based Intra
    pcs_ptr->intra_pred_mode = 3;

    // Set Tx Search     Settings
    // 0                 OFF
    // 1                 ON
    pcs_ptr->tx_size_search_mode = 1;

    // Assign whether to use TXS in inter classes (if TXS is ON)
    // 0 OFF - TXS in intra classes only
    // 1 ON - TXS in all classes
    // 2 ON - INTER TXS restricted to max 1 depth
    pcs_ptr->txs_in_inter_classes = 0;

    // inter intra pred                      Settings
    // 0                                     OFF
    // 1                                     FULL
    // 2                                     FAST 1 : Do not inject for non basic inter
    // 3                                     FAST 2 : 1 + MRP pruning/ similar based disable + NIC tuning
    pcs_ptr->enable_inter_intra = 0;

    // Set compound mode      Settings
    // 0                      OFF: No compond mode search : AVG only
    // 1                      ON: Full
    // 2                      ON: Fast : similar based disable
    // 3                      ON: Fast : MRP pruning/ similar based disable
    pcs_ptr->compound_mode = 0;

    // Set frame end cdf update mode      Settings
    // 0                                  OFF
    // 1                                  ON
    if (scs_ptr->static_config.frame_end_cdf_update == DEFAULT)
        pcs_ptr->frame_end_cdf_update_mode = 1;
    else
        pcs_ptr->frame_end_cdf_update_mode =
        scs_ptr->static_config.frame_end_cdf_update;

     pcs_ptr->frm_hdr.use_ref_frame_mvs = 0;

    // Global motion level                        Settings
    // GM_FULL                                    Exhaustive search mode.
    // GM_DOWN                                    Downsampled resolution with a
    // downsampling factor of 2 in each dimension GM_TRAN_ONLY Translation only
    // using ME MV.
    pcs_ptr->gm_level = GM_DOWN;

    // Exit TX size search when all coefficients are zero
    // 0: OFF
    // 1: ON
    pcs_ptr->tx_size_early_exit = 0;


    context_ptr->tf_level = 0;
    set_tf_controls(context_ptr, context_ptr->tf_level);
    // MRP control
    // 0: OFF (1,1)  ; override features
    // 1: FULL (4,3) ; override features
    // 2: (4,3) ; No-override features
    // 3: (3,3) ; No-override features
    // 4: (3,2) ; No-override features
    // 5: (2,3) ; No-override features
    // 6: (2,2) ; No-override features
    // 7: (2,1) ; No-override features
    // 8: (1,2) ; No-override features
    // 9: (1,1) ; No-override features
    // Level 0 , 1  : set ref_list0_count_try and ref_list1_count_try and Override MRP-related features
    // Level 2 .. 9 : Only set ref_list0_count_try and ref_list1_count_try
    pcs_ptr->mrp_level = 0;

    pcs_ptr->tpl_opt_flag = 1;
    return return_error;
}
#endif
#if FIRST_PASS_SETUP
void set_txt_cycle_reduction_controls(ModeDecisionContext *mdctxt, uint8_t txt_cycles_red_mode);
void set_nsq_cycle_redcution_controls(ModeDecisionContext *mdctxt, uint16_t nsq_cycles_red_mode);
void set_depth_cycle_redcution_controls(ModeDecisionContext *mdctxt, uint8_t depth_cycles_red_mode) ;
void adaptive_md_cycles_redcution_controls(ModeDecisionContext *mdctxt, uint8_t adaptive_md_cycles_red_mode);
void set_obmc_controls(ModeDecisionContext *mdctxt, uint8_t obmc_mode) ;
void set_txs_cycle_reduction_controls(ModeDecisionContext *mdctxt, uint8_t txs_cycles_red_mode);
void set_inter_intra_distortion_based_reference_pruning_controls(ModeDecisionContext *mdctxt, uint8_t inter_intra_distortion_based_reference_pruning_mode);
void set_block_based_depth_reduction_controls(ModeDecisionContext *mdctxt, uint8_t block_based_depth_reduction_level);
void md_nsq_motion_search_controls(ModeDecisionContext *mdctxt, uint8_t md_nsq_mv_search_level);
void md_subpel_search_controls(ModeDecisionContext *mdctxt, uint8_t md_subpel_search_level, EbEncMode enc_mode);
/******************************************************
* Derive EncDec Settings for first pass
Input   : encoder mode and pd pass
Output  : EncDec Kernel signal(s)
******************************************************/
EbErrorType first_pass_signal_derivation_enc_dec_kernel(
    PictureControlSet *pcs_ptr,
    ModeDecisionContext *context_ptr) {
    EbErrorType return_error = EB_ErrorNone;

    EbEncMode enc_mode = pcs_ptr->enc_mode;
    uint8_t pd_pass = context_ptr->pd_pass;
    // mrp level
    context_ptr->mrp_level = pcs_ptr->parent_pcs_ptr->mrp_level;

    // sb_classifier levels
    // Level                Settings
    // 0                    Off
    // 1                    TH 80%
    // 2                    TH 70%
    // 3                    TH 60%
    // 4                    TH 50%
    // 5                    TH 40%
    context_ptr->enable_area_based_cycles_allocation = 0;
    // Tx_search Level                                Settings
    // 0                                              OFF
    // 1                                              Tx search at encdec
    // 2                                              Tx search at inter-depth
    // 3                                              Tx search at full loop
    context_ptr->tx_search_level = TX_SEARCH_OFF;
    // Set MD tx_level
    // md_txt_search_level                            Settings
    // 0                                              FULL
    // 1                                              Tx_weight 1
    // 2                                              Tx_weight 2
    // 3                                              Tx_weight 1 + disabling rdoq and sssse
    // 4                                              Tx_weight 1 + disabling rdoq and sssse + reduced set
    context_ptr->md_txt_search_level = 1;// anaghdin??

    uint8_t txt_cycles_reduction_level = 0;
    set_txt_cycle_reduction_controls(context_ptr, txt_cycles_reduction_level);
    // Interpolation search Level                     Settings
    // 0                                              OFF
    // 1                                              Interpolation search at
    // inter-depth 2                                              Interpolation
    // search at full loop 3                                              Chroma
    // blind interpolation search at fast loop 4 Interpolation search at fast loop
    context_ptr->interpolation_search_level = IT_SEARCH_OFF;

    // Set Chroma Mode
    // Level                Settings
    // CHROMA_MODE_0  0     Full chroma search @ MD
    // CHROMA_MODE_1  1     Fast chroma search @ MD
    // CHROMA_MODE_2  2     Chroma blind @ MD + CFL @ EP
    // CHROMA_MODE_3  3     Chroma blind @ MD + no CFL @ EP
    context_ptr->chroma_level = CHROMA_MODE_2; //anaghdin first_pass_opt // or CHROMA_MODE_3

    // Chroma independent modes search
    // Level                Settings
    // 0                    post first md_stage
    // 1                    post last md_stage
    context_ptr->chroma_at_last_md_stage = 0;
    context_ptr->chroma_at_last_md_stage_intra_th = (uint64_t)~0;
    context_ptr->chroma_at_last_md_stage_cfl_th = (uint64_t)~0;

    // Chroma independent modes nics
    // Level                Settings
    // 0                    All supported modes.
    // 1                    All supported modes in  Intra picture and 4 in inter picture
    context_ptr->independent_chroma_nics = 0;

    // Cfl level
    // Level                Settings
    // 0                    Allow cfl
    // 1                    Disable cfl

    context_ptr->md_disable_cfl = EB_TRUE;

    // libaom_short_cuts_ths
    // 1                    faster than libaom
    // 2                    libaom - default
    context_ptr->libaom_short_cuts_ths = 2;

    // 0                    inject all supprted chroma mode
    // 1                    follow the luma injection
    context_ptr->intra_chroma_search_follows_intra_luma_injection = 1;

    // Set disallow_4x4
    context_ptr->disallow_4x4 = EB_FALSE;

    context_ptr->md_disallow_nsq = pcs_ptr->parent_pcs_ptr->disallow_nsq;

    // Set global MV injection
    // Level                Settings
    // 0                    Injection off
    // 1                    On
    context_ptr->global_mv_injection = 0;

    context_ptr->new_nearest_injection = 0;
    context_ptr->new_nearest_near_comb_injection = 0;

    // Set warped motion injection
    // Level                Settings
    // 0                    OFF
    // 1                    On
    context_ptr->warped_motion_injection = 0;

    // Set unipred3x3 injection
    // Level                Settings
    // 0                    OFF
    // 1                    ON FULL
    // 2                    Reduced set
    context_ptr->unipred3x3_injection = 0;

    // Set bipred3x3 injection
    // Level                Settings
    // 0                    OFF
    // 1                    ON FULL
    // 2                    Reduced set
    context_ptr->bipred3x3_injection = 0;

    // Level                Settings
    // 0                    Level 0: OFF
    // 1                    Level 1: sub-pel refinement off
    // 2                    Level 2: (H + V) 1/2 & 1/4 refinement only = 4 half-pel + 4 quarter-pel = 8 positions + pred_me_distortion to pa_me_distortion deviation on
    // 3                    Level 3: (H + V + D only ~ the best) 1/2 & 1/4 refinement = up to 6 half-pel + up to 6  quarter-pel = up to 12 positions + pred_me_distortion to pa_me_distortion deviation on
    // 4                    Level 4: (H + V + D) 1/2 & 1/4 refinement = 8 half-pel + 8 quarter-pel = 16 positions + pred_me_distortion to pa_me_distortion deviation on
    // 5                    Level 5: (H + V + D) 1/2 & 1/4 refinement = 8 half-pel + 8 quarter-pel = 16 positions + pred_me_distortion to pa_me_distortion deviation off
    // 6                    Level 6: (H + V + D) 1/2 & 1/4 refinement = 8 half-pel + 8 quarter-pel = 16 positions + pred_me_distortion to pa_me_distortion deviation off
    context_ptr->predictive_me_level = 0;

    // Level                    Settings
    // FALSE                    Use SSD at PME
    // TRUE                     Use SAD at PME
    context_ptr->use_sad_at_pme = EB_FALSE;

    // Derive md_staging_mode
    //
    // MD_STAGING_MODE_1
    //  ____________________________________________________________________________________________________________________________________________________________
    // |        | md_stage_0                  | md_stage_1                     | md_stage_2                              | md_stage_3                              |
    // |________|_____________________________|________________________________|_________________________________________|_________________________________________|
    // |CLASS_0 |Prediction for Luma & Chroma |Res, T, Q, Q-1 for Luma Only    |Bypassed                                 |Res, T, Q, Q-1, T-1 or Luma & Chroma     |
    // |CLASS_6 |SAD                          |No RDOQ                         |                                         |RDOQ (f(RDOQ Level))                     |
    // |CLASS_7 |                             |No Tx Type Search               |                                         |Tx Type Search (f(Tx Type Search Level)) |
    // |        |                             |No Tx Size Search               |                                         |Tx Size Search (f(Tx Size Search Level))|
    // |        |                             |SSD @ Frequency Domain          |                                         |CFL vs. Independent                      |
    // |        |                             |                                |                                         |SSD @ Spatial Domain                     |
    // |________|_____________________________|________________________________|_________________________________________|_________________________________________|
    // |CLASS_1 |Prediction for Luma Only     |IFS (f(IFS))                    |Bypassed                                 |Prediction for Luma & Chroma  (Best IF)  |
    // |CLASS_2 |Bilinear Only (IFS OFF)      |Res, T, Q, Q-1 for Luma Only    |                                         |Res, T, Q, Q-1, T-1 or Luma & Chroma     |
    // |CLASS_3 |SAD                          |No RDOQ                         |                                         |RDOQ (f(RDOQ Level))                     |
    // |CLASS_4 |                             |No Tx Type Search               |                                         |Tx Type Search (f(Tx Type Search Level)) |
    // |CLASS_5 |                             |No Tx Size Search               |                                         |Tx Size Search  (f(Tx Size Search Level))|
    // |CLASS_8 |                             |SSD @ Frequency Domain          |                                         |SSD @ Spatial Domain                     |
    // |________|_____________________________|________________________________|_________________________________________|_________________________________________|
    //
    // MD_STAGING_MODE_2
    //  ____________________________________________________________________________________________________________________________________________________________
    // |        | md_stage_0                  | md_stage_1                     | md_stage_2                              | md_stage_3                              |
    // |________|_____________________________|________________________________|_________________________________________|_________________________________________|
    // |CLASS_0 |Prediction for Luma & Chroma |Res, T, Q, Q-1 for Luma Only    |Res, T, Q, Q-1 for Luma Only             |Res, T, Q, Q-1, T-1 or Luma & Chroma     |
    // |CLASS_6 |SAD                          |No RDOQ                         |RDOQ (f(RDOQ Level))                     |RDOQ (f(RDOQ Level))                     |
    // |CLASS_7 |                             |No Tx Type Search               |Tx Type Search (f(Tx Type Search Level)) |Tx Type Search (f(Tx Type Search Level)) |
    // |        |                             |No Tx Size Search               |No Tx Size Search                        |Tx Size Search (f(Tx Size Search Level))|
    // |        |                             |SSD @ Frequency Domain          |SSD @ Frequency Domain                   |CFL vs. Independent                      |
    // |        |                             |                                |                                         |SSD @ Spatial Domain                     |
    // |________|_____________________________|________________________________|_________________________________________|_________________________________________|
    // |CLASS_1 |Prediction for Luma Only     |IFS (f(IFS))                    |Res, T, Q, Q-1  for Luma Only            |Prediction for Luma & Chroma  (Best IF)  |
    // |CLASS_2 |Bilinear Only (IFS OFF)      |Res, T, Q, Q-1 for Luma Only    |RDOQ (f(RDOQ Level))                     |Res, T, Q, Q-1, T-1 or Luma & Chroma     |
    // |CLASS_3 |SAD                          |No RDOQ                         |Tx Type Search (f(Tx Type Search Level)) |RDOQ (f(RDOQ Level))                     |
    // |CLASS_4 |                             |No Tx Type Search               |No Tx Size Search                        |Tx Type Search (f(Tx Type Search Level)) |
    // |CLASS_5 |                             |No Tx Size Search               |SSD @ Frequency Domain                   |Tx Size Search  (f(Tx Size Search Level))|
    // |CLASS_8 |                             |SSD @ Frequency Domain          |                                         |SSD @ Spatial Domain                     |
    // |________|_____________________________|________________________________|_________________________________________|_________________________________________|

    if (pd_pass == PD_PASS_0) {
        context_ptr->md_staging_mode = MD_STAGING_MODE_0;
    }
    else if (pd_pass == PD_PASS_1) {
        context_ptr->md_staging_mode = MD_STAGING_MODE_1;
    }
    else
        context_ptr->md_staging_mode = MD_STAGING_MODE_1;


    // Set md staging count level
    // Level 0              minimum count = 1
    // Level 1              set towards the best possible partitioning (to further optimize)
    // Level 2              HG: breack down or look up-table(s) are required !
    if (pd_pass == PD_PASS_0) {
        context_ptr->md_staging_count_level = 0;
    }
    else if (pd_pass == PD_PASS_1) {
        context_ptr->md_staging_count_level = 1;
    }
    else {
        context_ptr->md_staging_count_level = 2;
    }

    // Set interpolation filter search blk size
    // Level                Settings
    // 0                    ON for 8x8 and above
    // 1                    ON for 16x16 and above
    // 2                    ON for 32x32 and above
    context_ptr->interpolation_filter_search_blk_size = 0;

    // Derive Spatial SSE Flag
    context_ptr->spatial_sse_full_loop = EB_TRUE;

    context_ptr->blk_skip_decision = EB_FALSE;

    // Derive Trellis Quant Coeff Optimization Flag
    context_ptr->enable_rdoq = EB_FALSE;

    // Derive redundant block
    context_ptr->redundant_blk = EB_FALSE;

    // Set edge_skp_angle_intra
    context_ptr->edge_based_skip_angle_intra = 0;

    // Set prune_ref_frame_for_rec_partitions
    context_ptr->prune_ref_frame_for_rec_partitions = override_feature_level(context_ptr->mrp_level, 0, 0, 0);



    // md_stage_1_cand_prune_th (for single candidate removal per class)
    // Remove candidate if deviation to the best is higher than md_stage_1_cand_prune_th
        context_ptr->md_stage_1_cand_prune_th = (uint64_t)~0;

    // md_stage_2_3_cand_prune_th (for single candidate removal per class)
    // Remove candidate if deviation to the best is higher than
    // md_stage_2_3_cand_prune_th
        context_ptr->md_stage_2_3_cand_prune_th = (uint64_t)~0;

    // md_stage_2_3_class_prune_th (for class removal)
    // Remove class if deviation to the best is higher than
    // md_stage_2_3_class_prune_th

        context_ptr->md_stage_2_3_class_prune_th = (uint64_t)~0;

    context_ptr->coeff_area_based_bypass_nsq_th = 0;

    // NSQ cycles reduction level: TBD
    uint8_t nsq_cycles_red_mode = 0;
    set_nsq_cycle_redcution_controls(context_ptr, nsq_cycles_red_mode);

    // NsqCycleRControls*nsq_cycle_red_ctrls = &context_ptr->nsq_cycles_red_ctrls;
    // Overwrite allcation action when nsq_cycles_reduction th is higher.
        context_ptr->nsq_cycles_reduction_th = 0;

    // Depth cycles reduction level: TBD
    uint8_t depth_cycles_red_mode = 0;
    set_depth_cycle_redcution_controls(context_ptr, depth_cycles_red_mode);

    uint8_t adaptive_md_cycles_level = 0;
    adaptive_md_cycles_redcution_controls(context_ptr, adaptive_md_cycles_level);
    // Weighting (expressed as a percentage) applied to
    // square shape costs for determining if a and b
    // shapes should be skipped. Namely:
    // skip HA, HB, and H4 if h_cost > (weighted sq_cost)
    // skip VA, VB, and V4 if v_cost > (weighted sq_cost)
    context_ptr->sq_weight = (uint32_t)~0;
    // nsq_hv_level  needs sq_weight to be ON
    // 0: OFF
    // 1: ON 10% + skip HA/HB/H4  or skip VA/VB/V4
    // 2: ON 10% + skip HA/HB  or skip VA/VB   ,  5% + skip H4  or skip V4
    context_ptr->nsq_hv_level = 0;

    // Set pred ME full search area
    context_ptr->pred_me_full_pel_search_width = PRED_ME_FULL_PEL_REF_WINDOW_WIDTH_15;
    context_ptr->pred_me_full_pel_search_height = PRED_ME_FULL_PEL_REF_WINDOW_HEIGHT_15;

    // Set coeff_based_nsq_cand_reduction
    context_ptr->coeff_based_nsq_cand_reduction = EB_FALSE;

    // Set pic_obmc_level @ MD
    context_ptr->md_pic_obmc_level = 0;
    set_obmc_controls(context_ptr, context_ptr->md_pic_obmc_level);

    // Set enable_inter_intra @ MD
    context_ptr->md_enable_inter_intra = 0;

    // Set enable_paeth @ MD
    context_ptr->md_enable_paeth = 0;

    // Set enable_smooth @ MD
    context_ptr->md_enable_smooth = 0;

    // Set md_tx_size_search_mode @ MD
    context_ptr->md_tx_size_search_mode = pcs_ptr->parent_pcs_ptr->tx_size_search_mode;

    uint8_t txs_cycles_reduction_level = 0;
    set_txs_cycle_reduction_controls(context_ptr, txs_cycles_reduction_level);

    // Set md_filter_intra_mode @ MD
    // md_filter_intra_level specifies whether filter intra would be active
    // for a given prediction candidate in mode decision.
    // md_filter_intra_level | Settings
    // 0                      | OFF
    // 1                      | ON
    context_ptr->md_filter_intra_level = 0;

    // Set md_allow_intrabc @ MD
    context_ptr->md_allow_intrabc = 0;

    // intra_similar_mode
    // 0: OFF
    // 1: If previous similar block is intra, do not inject any inter
    context_ptr->intra_similar_mode = 0;

    // Set inter_intra_distortion_based_reference_pruning
    context_ptr->inter_intra_distortion_based_reference_pruning = 0;
    set_inter_intra_distortion_based_reference_pruning_controls(context_ptr, context_ptr->inter_intra_distortion_based_reference_pruning);

    context_ptr->block_based_depth_reduction_level = 0;
    set_block_based_depth_reduction_controls(context_ptr, context_ptr->block_based_depth_reduction_level);

    context_ptr->md_nsq_mv_search_level = 0;
    md_nsq_motion_search_controls(context_ptr, context_ptr->md_nsq_mv_search_level);

    context_ptr->md_subpel_search_level = 0;
    md_subpel_search_controls(context_ptr, context_ptr->md_subpel_search_level, enc_mode);

    // Set max_ref_count @ MD
    context_ptr->md_max_ref_count = override_feature_level(context_ptr->mrp_level, 4, 4, 1);

    // Set md_skip_mvp_generation (and use (0,0) as MVP instead)
    context_ptr->md_skip_mvp_generation = EB_FALSE;

    // Set dc_cand_only_flag
    context_ptr->dc_cand_only_flag = EB_TRUE;

    // Set intra_angle_delta @ MD
    context_ptr->md_intra_angle_delta = 0;

    // Set disable_angle_z2_prediction_flag
    context_ptr->disable_angle_z2_intra_flag = EB_TRUE;

    // Set full_cost_derivation_fast_rate_blind_flag
    context_ptr->full_cost_shut_fast_rate_flag = EB_FALSE;

    context_ptr->skip_intra = 0;

    return return_error;
}
#endif
#if FIRST_PASS_SETUP

/******************************************************
* Derive Mode Decision Config Settings for first pass
Input   : encoder mode and tune
Output  : EncDec Kernel signal(s)
******************************************************/
EbErrorType first_pass_signal_derivation_mode_decision_config_kernel(
    /*SequenceControlSet *scs_ptr, */PictureControlSet *pcs_ptr,
    ModeDecisionConfigurationContext *context_ptr) {

    EbErrorType return_error = EB_ErrorNone;

    // ADP
    context_ptr->adp_level = pcs_ptr->parent_pcs_ptr->enc_mode;

    // CDF
    pcs_ptr->update_cdf = 0;

    // Filter INTRA
    // pic_filter_intra_level specifies whether filter intra would be active
    // for a given picture.
    // pic_filter_intra_level | Settings
    // 0                      | OFF
    // 1                      | ON
    pcs_ptr->pic_filter_intra_level = 0;

    // High Precision
    FrameHeader *frm_hdr = &pcs_ptr->parent_pcs_ptr->frm_hdr;
    frm_hdr->allow_high_precision_mv = 0;

    // Warped
    frm_hdr->allow_warped_motion = 0;
    frm_hdr->is_motion_mode_switchable = frm_hdr->allow_warped_motion;

    // pic_obmc_level - pic_obmc_level is used to define md_pic_obmc_level.
    // The latter determines the OBMC settings in the function set_obmc_controls.
    // Please check the definitions of the flags/variables in the function
    // set_obmc_controls corresponding to the pic_obmc_level settings.
    //  pic_obmc_level  |              Default Encoder Settings             |     Command Line Settings
    //         0        | OFF subject to possible constraints               | OFF everywhere in encoder
    //         1        | ON subject to possible constraints                | Fully ON in PD_PASS_2
    //         2        | Faster level subject to possible constraints      | Level 2 everywhere in PD_PASS_2
    //         3        | Even faster level subject to possible constraints | Level 3 everywhere in PD_PASS_3
    pcs_ptr->parent_pcs_ptr->pic_obmc_level = 0;

    // Switchable Motion Mode
    frm_hdr->is_motion_mode_switchable = frm_hdr->is_motion_mode_switchable ||
        pcs_ptr->parent_pcs_ptr->pic_obmc_level;

    // HBD Mode
    pcs_ptr->hbd_mode_decision = EB_8_BIT_MD; //anaghdin to check for 10 bit

    return return_error;
}
#endif
#if FIRST_PASS_SETUP
void* set_me_hme_params_oq(
    MeContext                     *me_context_ptr,
    PictureParentControlSet       *pcs_ptr,
    SequenceControlSet            *scs_ptr,
    EbInputResolution                 input_resolution);
void *set_me_hme_params_from_config(SequenceControlSet *scs_ptr, MeContext *me_context_ptr) ;
void set_me_hme_ref_prune_ctrls(MeContext* context_ptr, uint8_t prune_level) ;
void set_me_sr_adjustment_ctrls(MeContext* context_ptr, uint8_t sr_adjustment_level);
/******************************************************
* Derive ME Settings for first pass
  Input   : encoder mode and tune
  Output  : ME Kernel signal(s)
******************************************************/
EbErrorType first_pass_signal_derivation_me_kernel(
    SequenceControlSet        *scs_ptr,
    PictureParentControlSet   *pcs_ptr,
    MotionEstimationContext_t   *context_ptr) {
    EbErrorType return_error = EB_ErrorNone;

    context_ptr->me_context_ptr->mrp_level = pcs_ptr->mrp_level;
    // Set ME/HME search regions

    if (scs_ptr->static_config.use_default_me_hme)
        set_me_hme_params_oq(
            context_ptr->me_context_ptr,
            pcs_ptr,
            scs_ptr,
            scs_ptr->input_resolution);
    else
        set_me_hme_params_from_config(
            scs_ptr,
            context_ptr->me_context_ptr);


    // Set HME flags
    context_ptr->me_context_ptr->enable_hme_flag = pcs_ptr->enable_hme_flag;
    context_ptr->me_context_ptr->enable_hme_level0_flag = pcs_ptr->enable_hme_level0_flag;
    context_ptr->me_context_ptr->enable_hme_level1_flag = pcs_ptr->enable_hme_level1_flag;
    context_ptr->me_context_ptr->enable_hme_level2_flag = pcs_ptr->enable_hme_level2_flag;

    // HME Search Method
    context_ptr->me_context_ptr->hme_search_method = SUB_SAD_SEARCH; //anaghdin first_pass_opt

    // ME Search Method
    context_ptr->me_context_ptr->me_search_method = FULL_SAD_SEARCH;

    context_ptr->me_context_ptr->compute_global_motion = EB_FALSE;

    // Set hme/me based reference pruning level (0-4)
    set_me_hme_ref_prune_ctrls(context_ptr->me_context_ptr, 0);

    // Set hme-based me sr adjustment level
    set_me_sr_adjustment_ctrls(context_ptr->me_context_ptr, 0);

    return return_error;
};
#endif
#endif  // TWOPASS_RC
