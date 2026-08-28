"""
sd-webui-ChebyCast
==================

Solver-step-aware Chebyshev feature forecasting for Forge-derived
Stable Diffusion WebUIs.

ChebyCast is inspired by Spectrum (Han et al., CVPR 2026), but it is not a
faithful port. It keeps the Chebyshev ridge-regression forecasting idea while
changing the execution model for Forge wrappers and fixed-step multi-stage
samplers.

Hook:
    model_function_wrapper

Design targets:
    - reForge / Forge Classic / Forge / Forge Neo
    - SDXL-family models
    - Rank-agnostic wrapper outputs, including Anima / NextDiT candidates
    - Fixed-grid single-stage and fixed-step multi-stage samplers

Not claimed:
    - A1111 support
    - Adaptive ODE solver support
    - Pixel-identical behavior with the official Spectrum implementation

Ordering note:
    This is not a CFG hook. sorting_priority controls UI placement only.
"""

import logging
import math
import os
import sys

import gradio as gr

from modules import scripts, shared

_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

from chebycast import (
    ChebyCastRuntime,
    apply_chebycast,
    get_model_sampling,
    remove_chebycast_patches,
)
from chebycast.core import _log, _warn

try:
    from modules.infotext_utils import PasteField
except Exception:
    PasteField = None

logger = logging.getLogger(__name__)

STAGE_GROUPING_CHOICES = ["auto", "1", "2", "3", "4", "6"]
FIT_POINTS_CHOICES = ["all stages", "step head only"]
TIME_COORD_CHOICES = ["auto", "schedule", "step", "timestep", "sigma"]

# enable, w, m, lam, window, flex, warmup, stop, group, fit, history, hires, coord
N_COMPONENTS = 13


def _has_forge_backend(p) -> bool:
    return hasattr(p, "sd_model") and hasattr(p.sd_model, "forge_objects")


def _is_secondary_pass(p) -> bool:
    """Prevent patching ADetailer and postprocessing sub-runs."""
    p_type_name = str(type(p))
    return bool(
        getattr(p, "_in_adetailer", False)
        or "Postprocessed" in p_type_name
    )


def _is_hires_pass(p) -> bool:
    return bool(getattr(p, "is_hr_pass", False))


def _effective_total_steps(p) -> int:
    """Return the solver-step count for the current sampling pass."""
    steps = int(getattr(p, "steps", 0) or 0)
    if steps <= 0:
        steps = 1

    if _is_hires_pass(p):
        hr_steps = int(getattr(p, "hr_second_pass_steps", 0) or 0)
        return hr_steps if hr_steps > 0 else steps

    if getattr(p, "init_images", None):
        denoise = float(getattr(p, "denoising_strength", 1.0) or 1.0)
        return max(1, int(min(denoise, 0.999) * steps))

    return steps


class ChebyCastScript(scripts.Script):
    sorting_priority = 19.0

    def title(self):
        return "ChebyCast"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("ChebyCast", open=False):
            enabled = gr.Checkbox(
                label="Enable ChebyCast",
                value=False,
            )

            with gr.Row():
                w = gr.Slider(
                    label="Blend weight (w)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.40,
                    info=(
                        "0 = local Newton extrapolation, "
                        "1 = pure Chebyshev fit"
                    ),
                )
                basis_count = gr.Slider(
                    label="Chebyshev bases (m)",
                    minimum=2,
                    maximum=9,
                    step=1,
                    value=4,
                    info=(
                        "Number of Chebyshev bases T0..T(m-1). "
                        "m=4 corresponds to degree 3."
                    ),
                )
                lam = gr.Slider(
                    label="Ridge regularization (lam)",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                )

            with gr.Row():
                window_size = gr.Slider(
                    label="Window size (solver steps)",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=2,
                    info="Roughly one actual solver step per N eligible steps",
                )
                flex_window = gr.Slider(
                    label="Window growth (flex)",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=0.0,
                    info="Increase the effective window after actual steps",
                )
                history = gr.Slider(
                    label="History points (K)",
                    minimum=6,
                    maximum=64,
                    step=1,
                    value=16,
                    info="Maximum sliding-window sample count for the fit",
                )

            with gr.Row():
                warmup_steps = gr.Slider(
                    label="Warmup steps",
                    minimum=0,
                    maximum=20,
                    step=1,
                    value=4,
                    info=(
                        "Leading solver steps that always run the real model. "
                        "Auto grouping enforces at least 2."
                    ),
                )
                stop_offset = gr.Slider(
                    label="Stop forecasting offset (from end)",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=3,
                    info="Trailing solver steps that always run the real model",
                )

            with gr.Row():
                stage_grouping = gr.Dropdown(
                    label="Stage grouping",
                    choices=STAGE_GROUPING_CHOICES,
                    value="auto",
                    info=(
                        "auto = follow the WebUI sampling-step counter; "
                        "N = group every N model calls into one solver step"
                    ),
                )
                fit_points = gr.Dropdown(
                    label="Fit points",
                    choices=FIT_POINTS_CHOICES,
                    value="all stages",
                    info="Choose whether intermediate stage outputs update the fit",
                )
                apply_hires = gr.Checkbox(
                    label="Apply to hires pass",
                    value=False,
                )

            time_coord = gr.Dropdown(
                label="Time coordinate",
                choices=TIME_COORD_CHOICES,
                value="auto",
                info=(
                    "auto prefers schedule, then solver-step coordinate, "
                    "then model timestep, then raw sigma"
                ),
            )

        components = [
            enabled,
            w,
            basis_count,
            lam,
            window_size,
            flex_window,
            warmup_steps,
            stop_offset,
            stage_grouping,
            fit_points,
            history,
            apply_hires,
            time_coord,
        ]

        # ui-config.json can silently override component bounds by label.
        for component in components:
            component.do_not_save_to_config = True

        if PasteField is not None:
            self.infotext_fields = [
                PasteField(enabled, "ChebyCast enabled"),
                PasteField(w, "ChebyCast w"),
                PasteField(basis_count, "ChebyCast m"),
                PasteField(lam, "ChebyCast lam"),
                PasteField(window_size, "ChebyCast window"),
                PasteField(flex_window, "ChebyCast flex window"),
                PasteField(warmup_steps, "ChebyCast warmup"),
                PasteField(stop_offset, "ChebyCast stop offset"),
                PasteField(stage_grouping, "ChebyCast stage grouping"),
                PasteField(fit_points, "ChebyCast fit points"),
                PasteField(history, "ChebyCast history"),
                PasteField(apply_hires, "ChebyCast hires"),
                PasteField(time_coord, "ChebyCast coord"),
            ]

        return components

    @staticmethod
    def _resolve_args(args):
        """Keep the trailing ChebyCast component block on compatible forks."""
        if len(args) > N_COMPONENTS:
            args = args[-N_COMPONENTS:]
        if len(args) < N_COMPONENTS:
            return None
        return args

    def process_before_every_sampling(self, p, *args, **kwargs):
        resolved = self._resolve_args(args)
        if resolved is None:
            return

        (
            enabled,
            w,
            basis_count,
            lam,
            window_size,
            flex_window,
            warmup_steps,
            stop_offset,
            stage_grouping,
            fit_points,
            history,
            apply_hires,
            time_coord,
        ) = resolved

        if not enabled:
            return
        if _is_secondary_pass(p):
            return
        if _is_hires_pass(p) and not apply_hires:
            return
        if not _has_forge_backend(p):
            _warn(
                "Requires a Forge-derived backend. "
                "A1111 is not supported."
            )
            return

        manual_group = 0
        if str(stage_grouping) != "auto":
            try:
                manual_group = int(stage_grouping)
            except Exception:
                manual_group = 0

        total_steps = _effective_total_steps(p)
        fit_all_stages = str(fit_points) == FIT_POINTS_CHOICES[0]

        # Public m is the number of bases. Internal degree is m - 1.
        degree = max(1, int(basis_count) - 1)

        unet = p.sd_model.forge_objects.unet.clone()
        remove_chebycast_patches(unet)

        runtime = ChebyCastRuntime(
            w=float(w),
            degree=degree,
            lam=float(lam),
            window_size=float(window_size),
            flex_window=float(flex_window),
            warmup_steps=int(warmup_steps),
            stop_caching_offset=int(stop_offset),
            total_steps=total_steps,
            fit_all_stages=fit_all_stages,
            max_points=int(history),
            manual_group=manual_group,
            step_provider=lambda: shared.state.sampling_step,
            model_sampling=get_model_sampling(unet),
            coord_mode=str(time_coord),
        )

        apply_chebycast(unet, runtime)
        p.sd_model.forge_objects.unet = unet

        runs = getattr(p, "_chebycast_runtimes", None)
        if runs is None:
            runs = []
            p._chebycast_runtimes = runs
        runs.append(runtime)

        if not _is_hires_pass(p):
            gp = p.extra_generation_params
            gp["ChebyCast enabled"] = True
            gp["ChebyCast w"] = float(w)
            gp["ChebyCast m"] = int(basis_count)
            gp["ChebyCast lam"] = float(lam)
            gp["ChebyCast window"] = int(window_size)
            gp["ChebyCast flex window"] = float(flex_window)
            gp["ChebyCast warmup"] = int(warmup_steps)
            gp["ChebyCast stop offset"] = int(stop_offset)
            gp["ChebyCast stage grouping"] = str(stage_grouping)
            gp["ChebyCast fit points"] = str(fit_points)
            gp["ChebyCast history"] = int(history)
            gp["ChebyCast hires"] = bool(apply_hires)
            gp["ChebyCast coord"] = str(time_coord)

        window_int = max(1, int(math.floor(float(window_size))))
        _log(
            1,
            (
                "applied: pass=%s total_steps=%d w=%.2f m=%d degree=%d "
                "lam=%.2f window=%d flex=%.2f warmup=%d "
                "stop_offset=%d grouping=%s fit_points=%s "
                "history=%d coord=%s"
            )
            % (
                "hires" if _is_hires_pass(p) else "main",
                total_steps,
                float(w),
                int(basis_count),
                degree,
                float(lam),
                window_int,
                float(flex_window),
                runtime.warmup_steps,
                int(stop_offset),
                str(stage_grouping),
                str(fit_points),
                int(history),
                str(time_coord),
            ),
        )

    def postprocess(self, p, processed, *args):
        runs = getattr(p, "_chebycast_runtimes", None)
        if not runs:
            return

        for index, runtime in enumerate(runs):
            _log(1, "run %d summary: %s" % (index, runtime.summary()))

        try:
            del p._chebycast_runtimes
        except Exception:
            pass
