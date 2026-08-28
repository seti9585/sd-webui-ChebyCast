"""
ChebyCast core
==============

Chebyshev spectral feature forecasting for diffusion sampling on Forge-derived
Stable Diffusion WebUIs.

ChebyCast is inspired by:
    Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration
    Jiaqi Han et al., CVPR 2026
    https://github.com/hanjq17/Spectrum

ChebyCast is not a faithful port of Spectrum. The implementation keeps the
Chebyshev basis fitting and ridge-regression idea while changing execution
semantics for Forge wrappers and fixed-step multi-stage samplers.

Key differences:
    1. Forecast decisions are cached per solver step rather than per raw model
       call.
    2. The time coordinate prefers the sampler schedule or solver-step position
       instead of a raw model-call counter.
    3. The local non-spectral branch uses Newton divided differences so it can
       extrapolate across non-uniform stage spacing.
    4. Forecast buffers are flattened float32 tensors and restored to the
       original output shape, avoiding a fixed latent-rank assumption.
"""

from __future__ import annotations

import logging
import math
import os
import sys

import torch

logger = logging.getLogger(__name__)

EXTENSION_NAME = "sd-webui-ChebyCast"

MARKER = "sd_webui_chebycast_v1"
MARKER_ATTR = "_sd_webui_chebycast_marker"
PREV_ATTR = "_sd_webui_chebycast_prev"

DEBUG_ENV_VAR = "SD_WEBUI_SETI_DEBUG"

# This is an independent ChebyCast safety guard, not a Spectrum parameter.
CLAMP_ABS = 30.0

# Retry Cholesky after adding a small diagonal jitter.
CHOLESKY_JITTER_SCALE = 0.000001


def _debug_level() -> int:
    try:
        return int(os.environ.get(DEBUG_ENV_VAR, "0"))
    except Exception:
        return 0


def _log(level: int, message: str) -> None:
    """Emit debug output in a way that remains visible across Forge forks."""
    if _debug_level() < level:
        return

    text = "[%s] %s" % (EXTENSION_NAME, message)
    logger.warning(text)
    print(text, file=sys.stderr)


def _warn(message: str) -> None:
    """Emit an unconditional warning."""
    text = "[%s] %s" % (EXTENSION_NAME, message)
    logger.warning(text)
    print(text, file=sys.stderr)


class ChebyshevForecaster:
    """Ridge-regressed Chebyshev fit over time-coordinate feature samples."""

    def __init__(self, degree: int, lam: float, max_points: int):
        self.degree = max(1, int(degree))
        self.lam = float(lam)
        self.max_points = max(self.degree + 2, int(max_points))
        self.taus: list[float] = []
        self.feats: list[torch.Tensor] = []
        self.shape = None
        self.dtype = None
        self._coef = None

    def reset(self) -> None:
        self.taus.clear()
        self.feats.clear()
        self.shape = None
        self.dtype = None
        self._coef = None

    def update(self, tau: float, h: torch.Tensor) -> None:
        if self.shape is not None and tuple(h.shape) != tuple(self.shape):
            self.reset()

        self.shape = tuple(h.shape)
        self.dtype = h.dtype
        self.feats.append(h.detach().reshape(-1).to(torch.float32))
        self.taus.append(float(tau))

        if len(self.taus) > self.max_points:
            self.taus.pop(0)
            self.feats.pop(0)

        self._coef = None

    def ready(self) -> bool:
        return len(self.taus) >= self.degree + 2

    def _design(self, taus: torch.Tensor) -> torch.Tensor:
        """Build T0..Tdegree with the Chebyshev three-term recurrence."""
        taus = taus.reshape(-1, 1)
        columns = [torch.ones_like(taus)]

        if self.degree >= 1:
            columns.append(taus)
            for _ in range(2, self.degree + 1):
                columns.append(
                    2.0 * taus * columns[-1] - columns[-2]
                )

        return torch.cat(columns[: self.degree + 1], dim=1)

    def _fit(self) -> bool:
        if self._coef is not None:
            return True

        try:
            features = torch.stack(self.feats, dim=0)
            device = features.device
            taus = torch.tensor(
                self.taus,
                dtype=torch.float32,
                device=device,
            )
            design = self._design(taus)
            basis_count = self.degree + 1
            eye = torch.eye(
                basis_count,
                device=device,
                dtype=torch.float32,
            )
            gram = design.T @ design + self.lam * eye

            try:
                factor = torch.linalg.cholesky(gram)
            except RuntimeError:
                jitter = CHOLESKY_JITTER_SCALE * gram.diagonal().mean()
                factor = torch.linalg.cholesky(gram + jitter * eye)

            self._coef = torch.cholesky_solve(
                design.T @ features,
                factor,
            )
            return True
        except Exception:
            self._coef = None
            return False

    def _local_extrapolation(self, tau: float) -> torch.Tensor:
        """Use Newton divided differences for non-uniform local spacing."""
        h_i = self.feats[-1]

        if len(self.feats) < 2:
            return h_i

        t_i = self.taus[-1]
        t_im1 = self.taus[-2]
        delta_1 = t_i - t_im1

        if abs(delta_1) < 0.000000000001:
            return h_i

        h_im1 = self.feats[-2]
        first_difference = (h_i - h_im1) / delta_1
        result = h_i + first_difference * (tau - t_i)

        if len(self.feats) >= 3:
            t_im2 = self.taus[-3]
            delta_0 = t_im1 - t_im2
            delta_02 = t_i - t_im2

            if (
                abs(delta_0) > 0.000000000001
                and abs(delta_02) > 0.000000000001
            ):
                h_im2 = self.feats[-3]
                previous_difference = (h_im1 - h_im2) / delta_0
                second_difference = (
                    first_difference - previous_difference
                ) / delta_02

                result = result + (
                    second_difference
                    * (tau - t_i)
                    * (tau - t_im1)
                )

        return result

    def predict(self, tau: float, w: float) -> torch.Tensor:
        """Blend local Newton extrapolation with the Chebyshev prediction."""
        local = self._local_extrapolation(tau)
        mixed = local

        if self._fit():
            device = self._coef.device
            tau_star = torch.tensor(
                [float(tau)],
                dtype=torch.float32,
                device=device,
            )
            design_star = self._design(tau_star)
            spectral = (design_star @ self._coef).squeeze(0)
            mixed = (
                (1.0 - float(w)) * local
                + float(w) * spectral
            )

        if not torch.isfinite(mixed).all():
            mixed = local
            if not torch.isfinite(mixed).all():
                mixed = self.feats[-1]

        output = torch.clamp(
            mixed,
            -CLAMP_ABS,
            CLAMP_ABS,
        )
        return output.to(self.dtype).view(self.shape)


class _TimeCoord:
    """Map per-call sigma values onto a monotone progress coordinate."""

    MODES = ("auto", "schedule", "step", "timestep", "sigma")

    def __init__(self, model_sampling, total_steps, mode="auto"):
        self.ms = model_sampling
        self.total_steps = max(2, int(total_steps))
        self.mode = mode if mode in self.MODES else "auto"
        self.sched = None
        self.t_start = None
        self.sigma_start = None
        self.source = None

        self._cur_step = None
        self._head_t = None
        self._prev_span = None

    def discover_schedule(self, conditioning: dict) -> None:
        if self.sched is not None:
            return

        if self.mode not in ("auto", "schedule"):
            self.sched = []
            return

        try:
            transformer_options = conditioning.get(
                "transformer_options",
                None,
            )
            if isinstance(transformer_options, dict):
                for key in ("sample_sigmas", "sigmas"):
                    sigmas = transformer_options.get(key, None)
                    if torch.is_tensor(sigmas) and sigmas.numel() > 1:
                        values = [
                            float(value)
                            for value in sigmas.detach().flatten().tolist()
                        ]
                        if values[0] > values[-1]:
                            self.sched = values
                            self._latch(
                                "schedule",
                                "schedule grid (%d points, %s)"
                                % (len(values), key),
                            )
                            return
        except Exception:
            pass

        self.sched = []

    def _latch(self, source: str, detail: str) -> None:
        if self.source is None:
            self.source = source
            _log(1, "time coordinate source: %s" % detail)

    def _timestep_of(self, sigma: float):
        if self.ms is None:
            return None

        try:
            device = None
            log_sigmas = getattr(self.ms, "log_sigmas", None)
            if torch.is_tensor(log_sigmas):
                device = log_sigmas.device

            sigma_tensor = torch.tensor(
                [float(sigma)],
                dtype=torch.float32,
                device=device,
            )
            timestep = self.ms.timestep(sigma_tensor)
            return float(timestep.flatten()[0].item())
        except Exception:
            return None

    def _progress_schedule(self, sigma: float):
        grid = self.sched
        if not grid:
            return None

        count = len(grid)
        sigma = min(max(float(sigma), grid[-1]), grid[0])

        for index in range(count - 1):
            if grid[index] >= sigma >= grid[index + 1]:
                span = grid[index] - grid[index + 1]
                fraction = (
                    0.0
                    if span <= 0.0
                    else (grid[index] - sigma) / span
                )
                return (
                    index + fraction
                ) / max(1, count - 1)

        return 1.0

    def _progress_step(self, sigma: float, step_id):
        if step_id is None:
            return None

        step_id = int(step_id)
        scalar = self._timestep_of(sigma)

        if scalar is None:
            scalar = -float(sigma)
        else:
            scalar = -scalar

        if self._cur_step != step_id:
            if self._head_t is not None and self._cur_step is not None:
                span = scalar - self._head_t
                if span > 0.0:
                    self._prev_span = span

            self._cur_step = step_id
            self._head_t = scalar

        fraction = 0.0
        if self._prev_span and self._prev_span > 0.0:
            fraction = (
                scalar - self._head_t
            ) / self._prev_span
            fraction = min(max(fraction, 0.0), 0.999)

        self._latch(
            "step",
            "solver step index (uniform, %d steps)"
            % self.total_steps,
        )

        return min(
            max(
                (step_id + fraction)
                / (self.total_steps - 1),
                0.0,
            ),
            1.0,
        )

    def _progress_timestep(self, sigma: float):
        timestep = self._timestep_of(sigma)
        if timestep is None:
            return None

        if self.t_start is None:
            self.t_start = max(timestep, 0.000001)

        self._latch(
            "timestep",
            (
                "model_sampling.timestep "
                "(t_start=%.3f) - non-uniform axis"
            )
            % self.t_start,
        )

        return min(
            max(1.0 - timestep / self.t_start, 0.0),
            1.0,
        )

    def _progress_sigma(self, sigma: float):
        if self.sigma_start is None:
            self.sigma_start = max(float(sigma), 0.000001)

        self._latch(
            "sigma",
            "raw sigma fallback (sigma_start=%.4f) - non-uniform axis"
            % self.sigma_start,
        )

        return min(
            max(
                1.0 - float(sigma) / self.sigma_start,
                0.0,
            ),
            1.0,
        )

    def progress(self, sigma: float, step_id=None) -> float:
        order = (
            ("schedule", "step", "timestep", "sigma")
            if self.mode == "auto"
            else (self.mode,)
        )

        for name in order:
            if name == "schedule":
                progress = self._progress_schedule(sigma)
            elif name == "step":
                progress = self._progress_step(sigma, step_id)
            elif name == "timestep":
                progress = self._progress_timestep(sigma)
            else:
                progress = self._progress_sigma(sigma)

            if progress is not None:
                return progress

        return self._progress_sigma(sigma)

    def tau(self, sigma: float, step_id=None) -> float:
        return 2.0 * self.progress(sigma, step_id) - 1.0


class ChebyCastRuntime:
    """Hold all ChebyCast state for one sampling pass."""

    def __init__(
        self,
        *,
        w: float,
        degree: int,
        lam: float,
        window_size: float,
        flex_window: float,
        warmup_steps: int,
        stop_caching_offset: int,
        total_steps: int,
        fit_all_stages: bool,
        max_points: int,
        manual_group: int,
        step_provider,
        model_sampling,
        coord_mode: str = "auto",
    ):
        self.w = float(w)
        self.degree = int(degree)
        self.lam = float(lam)
        self.window_size = float(window_size)
        self.flex_window = float(flex_window)
        self.warmup_steps = int(warmup_steps)
        self.stop_caching_offset = int(stop_caching_offset)
        self.total_steps = max(1, int(total_steps))
        self.fit_all_stages = bool(fit_all_stages)
        self.max_points = int(max_points)
        self.manual_group = int(manual_group)
        self.step_provider = step_provider
        self.coord = _TimeCoord(
            model_sampling,
            self.total_steps,
            coord_mode,
        )

        if self.manual_group <= 0:
            self.warmup_steps = max(2, self.warmup_steps)

        self._decisions: dict[int, bool] = {}
        self._curr_ws = max(1.0, self.window_size)
        self._num_forecast_streak = 0
        self._call_count = 0
        self._forecasters: dict[tuple, ChebyshevForecaster] = {}
        self._key_last_update_step: dict[tuple, int] = {}

        self.n_actual_steps = 0
        self.n_forecast_steps = 0
        self.n_ready_fallbacks = 0
        self.n_calls_actual = 0
        self.n_calls_forecast = 0
        self._calls_per_step: dict[int, int] = {}

    @staticmethod
    def _sigma_scalar(timestep) -> float:
        if torch.is_tensor(timestep):
            return float(timestep.flatten()[0].item())
        return float(timestep)

    def _step_id(self) -> int:
        if self.manual_group > 0:
            return self._call_count // self.manual_group

        try:
            return int(self.step_provider())
        except Exception:
            return 0

    def _decide(self, step_id: int, tau: float) -> bool:
        cached = self._decisions.get(step_id)
        if cached is not None:
            return cached

        tail_start = self.total_steps - self.stop_caching_offset

        if (
            step_id < self.warmup_steps
            or step_id >= tail_start
        ):
            do_actual = True
        else:
            window = max(
                1,
                int(math.floor(self._curr_ws)),
            )
            do_actual = (
                (self._num_forecast_streak + 1) % window
            ) == 0

        if do_actual:
            self._num_forecast_streak = 0
            if step_id >= self.warmup_steps:
                self._curr_ws += self.flex_window
            self.n_actual_steps += 1
        else:
            self._num_forecast_streak += 1
            self.n_forecast_steps += 1

        self._decisions[step_id] = do_actual

        _log(
            2,
            "step %d tau=%+.4f -> %s (window=%.2f)"
            % (
                step_id,
                tau,
                "ACTUAL" if do_actual else "FORECAST",
                self._curr_ws,
            ),
        )

        return do_actual

    def _forecaster_for(self, key: tuple) -> ChebyshevForecaster:
        forecaster = self._forecasters.get(key)

        if forecaster is None:
            forecaster = ChebyshevForecaster(
                self.degree,
                self.lam,
                self.max_points,
            )
            self._forecasters[key] = forecaster

        return forecaster

    def run_call(self, kwargs_dict: dict, actual_fn):
        x = kwargs_dict.get("input")
        timestep = kwargs_dict.get("timestep")
        conditioning = kwargs_dict.get("c", {}) or {}

        if x is None or timestep is None:
            return actual_fn()

        sigma = self._sigma_scalar(timestep)
        self.coord.discover_schedule(conditioning)

        step_id = self._step_id()
        self._call_count += 1
        self._calls_per_step[step_id] = (
            self._calls_per_step.get(step_id, 0) + 1
        )

        tau = self.coord.tau(sigma, step_id)

        cond_ids = kwargs_dict.get(
            "cond_or_uncond",
            None,
        )
        try:
            key = (
                tuple(cond_ids)
                if cond_ids is not None
                else (),
                tuple(x.shape),
            )
        except Exception:
            key = ((), tuple(x.shape))

        do_actual = self._decide(step_id, tau)
        forecaster = self._forecaster_for(key)

        if not do_actual and not forecaster.ready():
            do_actual = True
            self.n_ready_fallbacks += 1
            _log(
                2,
                "step %d key=%s not ready -> actual fallback"
                % (step_id, key),
            )

        if do_actual:
            output = actual_fn()

            if (
                self.fit_all_stages
                or self._key_last_update_step.get(key) != step_id
            ):
                if torch.is_tensor(output):
                    forecaster.update(tau, output)
                    self._key_last_update_step[key] = step_id

            self.n_calls_actual += 1
            return output

        prediction = forecaster.predict(
            tau,
            self.w,
        )
        self.n_calls_forecast += 1
        return prediction.to(device=x.device)

    def summary(self) -> str:
        base = (
            "steps actual=%d forecast=%d | "
            "calls actual=%d forecast=%d | "
            "ready-fallbacks=%d | coord=%s"
            % (
                self.n_actual_steps,
                self.n_forecast_steps,
                self.n_calls_actual,
                self.n_calls_forecast,
                self.n_ready_fallbacks,
                self.coord.source,
            )
        )

        histogram: dict[int, int] = {}
        for count in self._calls_per_step.values():
            histogram[count] = histogram.get(count, 0) + 1

        if not histogram:
            return base + " | per-step: none"

        parts = ", ".join(
            "%d calls x%d steps"
            % (calls, histogram[calls])
            for calls in sorted(histogram)
        )
        base += " | per-step: %s" % parts

        expected_calls = max(
            histogram,
            key=histogram.get,
        )
        odd_steps = sorted(
            step_id
            for step_id, count in self._calls_per_step.items()
            if count != expected_calls
        )

        if odd_steps:
            base += " | odd steps: %s" % odd_steps

        return base


def remove_chebycast_patches(unet) -> None:
    """Remove the ChebyCast wrapper and restore a chained prior wrapper."""
    try:
        options = getattr(unet, "model_options", None)
        if not isinstance(options, dict):
            return

        wrapper = options.get(
            "model_function_wrapper",
            None,
        )
        if wrapper is None:
            return

        if getattr(wrapper, MARKER_ATTR, None) == MARKER:
            previous = getattr(
                wrapper,
                PREV_ATTR,
                None,
            )

            if previous is not None:
                options["model_function_wrapper"] = previous
            else:
                del options["model_function_wrapper"]
    except Exception:
        logger.exception(
            "[%s] failed to remove wrapper",
            EXTENSION_NAME,
        )


def apply_chebycast(
    unet,
    runtime: ChebyCastRuntime,
) -> None:
    """Install ChebyCast on a cloned Forge UNet patcher.

    If a model_function_wrapper already exists, ChebyCast preserves it on
    actual model calls. Forecasted calls intentionally bypass the inner
    wrapper because the model call itself is being replaced by a forecast.
    """
    remove_chebycast_patches(unet)

    previous = None
    try:
        previous = unet.model_options.get(
            "model_function_wrapper",
            None,
        )
    except Exception:
        previous = None

    def chebycast_wrapper(model_function, kwargs_dict):
        def _actual():
            if previous is not None:
                return previous(
                    model_function,
                    kwargs_dict,
                )

            conditioning = kwargs_dict.get(
                "c",
                {},
            ) or {}

            return model_function(
                kwargs_dict["input"],
                kwargs_dict["timestep"],
                **conditioning,
            )

        return runtime.run_call(
            kwargs_dict,
            _actual,
        )

    setattr(
        chebycast_wrapper,
        MARKER_ATTR,
        MARKER,
    )
    setattr(
        chebycast_wrapper,
        PREV_ATTR,
        previous,
    )

    try:
        unet.set_model_unet_function_wrapper(
            chebycast_wrapper
        )
    except Exception:
        unet.model_options[
            "model_function_wrapper"
        ] = chebycast_wrapper

    if previous is not None:
        _log(
            1,
            "chained around an existing model_function_wrapper",
        )


def get_model_sampling(unet):
    """Return model_sampling when exposed by the Forge UNet patcher."""
    try:
        return unet.model.model_sampling
    except Exception:
        return None
