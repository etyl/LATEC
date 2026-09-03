"""Closed-loop LATEC codec: progressive prediction and quantization.

The encoder simulates the decoder: reconstructions fed back into the predictor
are built exclusively from dequantize() outputs, never from the original data,
so decoder-side predictions match encoder-side predictions bitwise (same
platform) and the error bound holds end to end.
"""

from __future__ import annotations

import time

import numpy as np

from .bitstream import (
    DTYPE_CODES,
    DTYPE_IDS,
    END_MODE_SHIFT,
    FLAG_CUBIC,
    FLAG_INTERP,
    FLAG_RANS,
    Header,
    pack_stage,
    read_stream,
    unpack_stage,
    write_stream,
)
from .levels import mask_tiles, stage_ebs, stage_masks
from .predictor import (
    END_MODE_MAX,
    InterpPredictor,
    default_interp_center,
    default_interp_end_mode,
)
from .quantizer import dequantize, quantize
from .rans import (
    build_laplace_tables,
    choose_kexp,
    choose_shape,
    model_bits,
    scale_to_level,
)


def compress(
    img: np.ndarray,
    eb: float,
    predictor,
    levels: int = 4,
    anchor_stride: int = 16,
    anchor_block: int = 1,
    radius: int = 1 << 15,
    zstd_level: int = 9,
    eb_ratio: float | None = None,
    tune: str = "fast",
    tune_size_slack: float = 1.05,
    verbose: bool = False,
) -> tuple[bytes, dict]:
    """Compress an image or an arbitrary-rank scalar field. Returns (stream, stats).

    An (H, W) or (H, W, C∈{1,3}) array is treated as an image (channel axis last,
    classic SZ semantics). Any other shape is a single-channel field of arbitrary
    rank (scientific data). Both interpolation and GNN predictors process the
    complete spatial field without padding or prediction seams.

    ``eb_ratio`` scales the per-level error bound (coarse levels tighter, see
    ``levels.stage_ebs``): pass a value in (0, 1] to fix it. With ``eb_ratio``
    omitted, ``tune`` controls the search: ``fast`` runs one candidate, ``size``
    sweeps candidates and keeps the smallest stream, and ``rd`` keeps the
    lowest reconstruction SSE within ``tune_size_slack`` of the smallest stream.
    1.0 reproduces flat-eb classic SZ."""
    # Two input shapes. An "image" carries a trailing channel axis (2-D, or 3-D
    # with 1 or 3 channels) and keeps SZ's classic (H, W, C) semantics. Anything
    # else is a single-channel scalar field of arbitrary rank (scientific data):
    # spatial = the whole shape, C = 1. Everything downstream (levels, interp/GNN
    # predictors, quantizer) is rank-generic.
    is_image = img.ndim <= 2 or (img.ndim == 3 and img.shape[-1] in (1, 3))
    # asarray, not astype: a float32 field (the scientific case, and the largest)
    # is read-only here, so converting it would only buy a second copy of the
    # input.
    if is_image:
        if img.ndim == 2:
            img = img[..., None]
        spatial = img.shape[:-1]
        c = img.shape[-1]
        fimg = np.asarray(img, np.float32)
    else:
        spatial = img.shape
        c = 1
        fimg = np.asarray(img, np.float32)[..., None]  # append a size-1 channel axis
    src_dtype = DTYPE_IDS[np.dtype(img.dtype)]

    vmin = float(fimg.min())
    vmax = float(fimg.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    # A predictor with its own schedule (interp) must agree with the header
    # params, or encoder masks and the decoder's rebuilt masks silently diverge.
    for name, val in (
        ("levels", levels),
        ("anchor_stride", anchor_stride),
        ("anchor_block", anchor_block),
    ):
        got = getattr(predictor, name, val)
        if got != val:
            raise ValueError(f"predictor {name}={got} != compress {name}={val}")

    # channel axis to front: (C, *spatial). moveaxis generalizes transpose(2,0,1).
    field = np.moveaxis(fimg, -1, 0)
    region_shape = field.shape[1:]
    # interp supplies its own sub-pass split; others use the plain dyadic schedule
    make_masks = getattr(predictor, "stage_masks", stage_masks)
    masks = make_masks(region_shape, levels, anchor_stride, anchor_block)
    flags = getattr(predictor, "stream_flag", 0)
    use_rans = getattr(predictor, "provides_scale", False)
    flags |= FLAG_RANS if use_rans else 0
    round_output = np.issubdtype(np.dtype(img.dtype), np.integer)

    def new_stats():
        return {
            "predict_s": 0.0,
            "quantize_s": 0.0,
            "entropy_s": 0.0,
            "outliers": 0,
            "stage_codes": [0] * len(masks),
            "stage_outliers": [0] * len(masks),
            "stage_payload_bytes": [0] * len(masks),
            "stage_model_bits": [0.0] * len(masks),
            "stage_pred_sae": [0.0] * len(masks),
            "stage_pred_sse": [0.0] * len(masks),
            "stage_recon_sae": [0.0] * len(masks),
            "stage_recon_sse": [0.0] * len(masks),
            "stage_recon_max": [0.0] * len(masks),
        }

    def encode(ebs):
        st = new_stats()
        payload, recon = _compress_region(
            field, masks, ebs, predictor, radius, round_output, st, zstd_level
        )
        return payload, recon, len(payload), st

    def ebs_for(ratio):
        return stage_ebs(region_shape, levels, anchor_stride, anchor_block, eb, ratio)

    def make_header(center, ratio, end_mode):
        return Header(
            channels=c,
            src_dtype=src_dtype,
            spatial=tuple(spatial),
            eb=float(eb),
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=anchor_block,
            radius=radius,
            max_radius=getattr(predictor, "max_radius", 0),
            agg_level=(
                -1
                if getattr(predictor, "agg_level", None) is None
                else int(predictor.agg_level)
            ),
            vmin=vmin,
            vmax=vmax,
            ckpt_hash=getattr(predictor, "checkpoint_hash", b"\0" * 16),
            flags=flags | (int(end_mode) << END_MODE_SHIFT),
            interp_center=center,
            eb_ratio=ratio,
        )

    if tune not in ("fast", "size", "rd"):
        raise ValueError("tune must be 'fast', 'size', or 'rd'")

    # Encoder-side tuning: optionally sweep per-level eb ratio (coarse-error
    # damping) and, for interp, centre-combine mode. The finest level always
    # carries the full eb, so every candidate satisfies the bound.
    tunable = getattr(predictor, "tunable", False)
    has_center = hasattr(predictor, "center")
    has_end_mode = hasattr(predictor, "end_mode")
    # Resolve the rank-aware ``center`` / ``end_mode`` (None = auto) now that the
    # region rank is known, so fast mode and the tune sweep both start from the
    # right base.
    if has_center and getattr(predictor, "center") is None:
        predictor.center = default_interp_center(len(region_shape))
    if has_end_mode and getattr(predictor, "end_mode") is None:
        predictor.end_mode = default_interp_end_mode(len(region_shape))
    if eb_ratio is not None:
        ratio_cands = [eb_ratio]
    elif tunable and tune != "fast":
        ratio_cands = [1.0, 0.9, 0.8, 0.7]
    else:  # single encode: predictor's best fixed default (see fast_eb_ratio)
        ratio_cands = [getattr(predictor, "fast_eb_ratio", 1.0)]
    sweep = tunable and tune != "fast" and eb_ratio is None
    base_center = getattr(predictor, "center", 0)
    center_cands = [base_center]
    if sweep and has_center:
        center_cands += [c for c in (0, 1, 2) if c != base_center]
    base_end = getattr(predictor, "end_mode", 0)

    candidates = []

    def run(ratio, center, end_mode):
        """Encode one candidate, record it, and return it."""
        if has_center:
            predictor.center = center
        if has_end_mode:
            predictor.end_mode = end_mode
        payload, rc, raw_bytes, st = encode(ebs_for(ratio))
        header = make_header(center, ratio, end_mode)
        t0 = time.time()
        stream = write_stream(header, payload, zstd_level)
        st["entropy_s"] += time.time() - t0
        st["raw_payload_bytes"] = raw_bytes
        st["compressed_bytes"] = len(stream)
        st["recon_sse"] = float(sum(st["stage_recon_sse"]))
        cand = (len(stream), st["recon_sse"], ratio, center, end_mode, stream, rc, st)
        candidates.append(cand)
        return cand

    # Knobs are swept in sequence, the way SZ3 tunes its own interpolation
    # settings, rather than as a full product: 4 end modes x 4 ratios x 3 centres
    # is 48 encodes, which is unusable on a real field.
    #
    # Line-end handling goes FIRST, and the order matters: it has by far the
    # largest single effect (-37% on s3d, where the others are worth a few
    # percent) and it interacts strongly with them, so conditioning ratio/centre
    # on it beats the reverse. Measured on the s3d 160^3 crop at rel eb 1e-3 --
    # ends-last picks (0.8, centre 2, mode 3) for 25223 B, ends-first reaches
    # (0.9, centre 0, mode 3) for 21096 B.
    seen = set()

    def run_once(ratio, center, end_mode):
        if (ratio, center, end_mode) not in seen:
            seen.add((ratio, center, end_mode))
            run(ratio, center, end_mode)

    run_once(ratio_cands[0], base_center, base_end)
    chosen_end = base_end
    if sweep and has_end_mode:
        for end_mode in range(END_MODE_MAX + 1):
            run_once(ratio_cands[0], base_center, end_mode)
        chosen_end = min(candidates, key=lambda c: (c[0], c[1]))[4]

    for ratio in ratio_cands:
        for center in center_cands:
            run_once(ratio, center, chosen_end)

    if tune == "rd" and eb_ratio is None and tunable and len(candidates) > 1:
        min_size = min(c[0] for c in candidates)
        size_limit = min_size * tune_size_slack
        feasible = [c for c in candidates if c[0] <= size_limit]
        best = min(feasible, key=lambda c: (c[1], c[0], c[2], c[3], c[4]))
    else:
        best = min(candidates, key=lambda c: (c[0], c[1], c[2], c[3], c[4]))

    _, _, chosen_ratio, chosen_center, chosen_end, stream, recon_canvas, stats = best
    stats["tune_candidates"] = len(candidates)
    stats["search_predict_s"] = sum(c[7]["predict_s"] for c in candidates)
    stats["search_quantize_s"] = sum(c[7]["quantize_s"] for c in candidates)
    stats["search_entropy_s"] = sum(c[7]["entropy_s"] for c in candidates)
    if has_center:
        predictor.center = chosen_center
    if has_end_mode:
        predictor.end_mode = chosen_end
    stats["eb_ratio"] = chosen_ratio
    stats["interp_center"] = chosen_center
    stats["interp_end_mode"] = chosen_end
    stats["tune"] = tune
    stats["tune_size_slack"] = (
        tune_size_slack if tune == "rd" and eb_ratio is None and tunable else 1.0
    )
    if verbose:
        print(
            f"tuned: eb_ratio={chosen_ratio} center={chosen_center} "
            f"end_mode={chosen_end} "
            f"compressed={len(stream)} bytes raw={stats['raw_payload_bytes']} bytes "
            f"sse={stats['recon_sse']:.6g}"
        )

    stats["recon"] = _finalize(
        recon_canvas, make_header(chosen_center, chosen_ratio, chosen_end)
    )
    stats["original_bytes"] = img.nbytes
    stats["ratio"] = img.nbytes / len(stream)
    return stream, stats


def decompress(stream: bytes, predictor_factory=None) -> np.ndarray:
    """Decompress a LATEC stream. ``predictor_factory(header) -> predictor``;
    defaults to InterpPredictor for interpolation streams."""
    header, payload = read_stream(stream)
    if predictor_factory is None:
        if header.flags & FLAG_INTERP:
            predictor_factory = lambda hdr: InterpPredictor(
                "cubic" if hdr.flags & FLAG_CUBIC else "linear",
                hdr.levels,
                hdr.anchor_stride,
                hdr.anchor_block,
            )
        else:
            raise ValueError("stream needs a predictor_factory")
    predictor = predictor_factory(header)
    if hasattr(predictor, "center"):
        predictor.center = header.interp_center
    if hasattr(predictor, "end_mode"):
        predictor.end_mode = (header.flags >> END_MODE_SHIFT) & END_MODE_MAX

    make_masks = getattr(predictor, "stage_masks", stage_masks)
    spatial = tuple(header.spatial)
    masks = make_masks(
        spatial, header.levels, header.anchor_stride, header.anchor_block
    )
    ebs = stage_ebs(
        spatial,
        header.levels,
        header.anchor_stride,
        header.anchor_block,
        header.eb,
        header.eb_ratio,
    )
    canvas = _decompress_region(payload, masks, ebs, header, predictor)
    return _finalize(canvas, header)


def _gather_stage(field, pos, n):
    """The stage's values as a compact (C, n) array, gathered by flat index.

    ``field[:, pos]`` would do the same thing, but NumPy expands an n-D boolean
    index into one int64 array per axis first (see ``levels.mask_tiles``), which
    at rank 5 is 40 bytes of temporary per point of the stage."""
    out = np.empty((field.shape[0], n), field.dtype)
    flat_field = field.reshape(field.shape[0], -1)
    at = 0
    for flat in mask_tiles(pos, field.shape[1:]):
        out[:, at : at + len(flat)] = flat_field[:, flat]
        at += len(flat)
    return out


def _scatter_stage(recon, pos, values):
    """Write a compact (C, n) stage back into the reconstruction. Inverse of
    ``_gather_stage``; same reason for not using ``recon[:, pos] = values``."""
    flat_recon = recon.reshape(recon.shape[0], -1)
    at = 0
    for flat in mask_tiles(pos, recon.shape[1:]):
        flat_recon[:, flat] = values[:, at : at + len(flat)]
        at += len(flat)


_STATS_TILE = 1 << 20


def _stage_error_stats(stats, stage_idx, values, pred, recon, tile=_STATS_TILE):
    """Accumulate one stage's prediction/reconstruction error moments.

    Diagnostics only, but the natural form -- three float64 copies of the whole
    stage plus an abs/square temporary each -- was both the largest host
    allocation in an encode and its most expensive host operation. The moments
    are identical to a few ULP when the differences are taken in float32 a tile
    at a time and accumulated in float64, and the working set then stays in
    cache instead of streaming several times the field through memory.
    """
    p_sae = p_sse = r_sae = r_sse = 0.0
    r_max = 0.0
    for a in range(0, values.shape[1], tile):
        v = values[:, a : a + tile]
        pred_err = v - pred[:, a : a + tile]
        recon_err = v - recon[:, a : a + tile]
        p_sae += float(np.abs(pred_err).sum(dtype=np.float64))
        p_sse += float(np.square(pred_err, dtype=np.float64).sum())
        np.abs(recon_err, out=recon_err)
        r_sae += float(recon_err.sum(dtype=np.float64))
        r_sse += float(np.square(recon_err, dtype=np.float64).sum())
        r_max = max(r_max, float(recon_err.max(initial=0.0)))
    stats["stage_pred_sae"][stage_idx] += p_sae
    stats["stage_pred_sse"][stage_idx] += p_sse
    stats["stage_recon_sae"][stage_idx] += r_sae
    stats["stage_recon_sse"][stage_idx] += r_sse
    stats["stage_recon_max"][stage_idx] = max(
        stats["stage_recon_max"][stage_idx], r_max
    )


def _compress_region(
    field, masks, ebs, predictor, radius, round_output, stats, zstd_level=9
):
    c = field.shape[0]
    recon = np.zeros_like(field)
    known = np.zeros(field.shape[1:], bool)
    parts = []
    use_rans = getattr(predictor, "provides_scale", False)
    for stage_idx, pos in enumerate(masks):
        n = int(np.count_nonzero(pos))
        if n == 0:
            if use_rans:
                tables = build_laplace_tables(ebs[stage_idx], radius)
                parts.append(
                    pack_stage(
                        np.zeros(0, np.uint32),
                        np.zeros(0, np.float32),
                        rans_levels=np.zeros(0, np.uint8),
                        rans_tables=tables,
                    )
                )
            else:
                parts.append(
                    pack_stage(
                        np.zeros(0, np.uint32),
                        np.zeros(0, np.float32),
                        radius=radius,
                        zstd_level=zstd_level,
                    )
                )
            continue
        eb = ebs[stage_idx]
        scale = None
        if stage_idx == 0:
            pred = np.zeros((c, n), np.float32)
            if use_rans:
                scale = np.full((c, n), eb, np.float32)
        else:
            t0 = time.time()
            if getattr(predictor, "provides_scale", False):
                pred, scale = predictor.predict(recon, known, pos, eb=eb)
            else:
                pred = predictor.predict(recon, known, pos)
            stats["predict_s"] += time.time() - t0
        t0 = time.time()
        values = _gather_stage(field, pos, n)
        codes, outliers = quantize(values, pred, eb, radius, round_output=round_output)
        stage_recon = dequantize(pred, codes, outliers, eb, radius).reshape(c, n)
        _scatter_stage(recon, pos, stage_recon)
        _stage_error_stats(stats, stage_idx, values, pred, stage_recon)
        stats["quantize_s"] += time.time() - t0
        known |= pos
        stats["outliers"] += len(outliers)
        stats["stage_outliers"][stage_idx] += len(outliers)
        stats["stage_codes"][stage_idx] += n * c
        t0 = time.time()
        if use_rans:
            tables = build_laplace_tables(eb, radius)
            levels64 = scale_to_level(scale, eb).reshape(-1)
            kexp = choose_kexp(codes, radius)
            stats["stage_model_bits"][stage_idx] += model_bits(
                codes,
                levels64,
                tables,
                kexp=kexp,
                shape=choose_shape(codes, levels64, tables, kexp),
            )
            part = pack_stage(codes, outliers, rans_levels=levels64, rans_tables=tables)
        else:
            part = pack_stage(
                codes, outliers, radius=radius, zstd_level=zstd_level
            )
        stats["stage_payload_bytes"][stage_idx] += len(part)
        parts.append(part)
        stats["entropy_s"] += time.time() - t0
    return b"".join(parts), recon


def _decompress_region(payload, masks, ebs, header, predictor):
    c = header.channels
    region = tuple(header.spatial)
    recon = np.zeros((c, *region), np.float32)
    known = np.zeros(region, bool)
    off = 0
    use_rans = bool(header.flags & FLAG_RANS)
    for stage_idx, pos in enumerate(masks):
        n = int(np.count_nonzero(pos))
        if n == 0:
            if use_rans:
                tables = build_laplace_tables(ebs[stage_idx], header.radius)
                codes, outliers, off = unpack_stage(
                    payload, off, rans_levels=np.zeros(0, np.uint8), rans_tables=tables
                )
            else:
                codes, outliers, off = unpack_stage(
                    payload, off, radius=header.radius
                )
            continue
        if stage_idx == 0:
            pred = np.zeros((c, n), np.float32)
            scale = np.full((c, n), ebs[stage_idx], np.float32)
        else:
            if use_rans:
                pred, scale = predictor.predict(recon, known, pos, eb=ebs[stage_idx])
            elif getattr(predictor, "provides_scale", False):
                pred, _scale = predictor.predict(recon, known, pos, eb=ebs[stage_idx])
                scale = None
            else:
                pred = predictor.predict(recon, known, pos)
                scale = None
        if use_rans:
            tables = build_laplace_tables(ebs[stage_idx], header.radius)
            levels64 = scale_to_level(scale, ebs[stage_idx]).reshape(-1)
            codes, outliers, off = unpack_stage(
                payload, off, rans_levels=levels64, rans_tables=tables
            )
        else:
            codes, outliers, off = unpack_stage(payload, off, radius=header.radius)
        _scatter_stage(
            recon,
            pos,
            dequantize(pred, codes, outliers, ebs[stage_idx], header.radius).reshape(
                c, n
            ),
        )
        known |= pos
    return recon


def _finalize(canvas: np.ndarray, header: Header) -> np.ndarray:
    out = np.moveaxis(canvas, 0, -1)  # (C, *spatial) -> (*spatial, C)
    dtype = DTYPE_CODES[header.src_dtype]
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    # copy=False: for the float32 scientific case this is the reconstruction
    # itself rather than a second full-size array.
    out = out.astype(dtype, copy=False)
    if header.channels == 1:
        out = out[..., 0]
    return out
