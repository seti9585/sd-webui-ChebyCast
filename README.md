# sd-webui-ChebyCast

**EN** | [日本語](#日本語)

Chebyshev-based feature forecasting for faster diffusion sampling on Forge-derived Stable Diffusion WebUIs, designed with solver-step-aware execution and fixed-step multi-stage samplers in mind.

ChebyCast is inspired by **Spectrum** — *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration* (Han et al., CVPR 2026) — but it is **not a faithful port** of the official implementation.

The project keeps the central idea of fitting denoiser features with Chebyshev bases and ridge regression, while changing the execution model for Forge wrappers, solver-step grouping, non-uniform intermediate stages, and rank-agnostic model outputs.

---

## Status

ChebyCast is currently an early implementation intended for practical validation on Forge-derived WebUIs.

The architecture targets:

- reForge
- Stable Diffusion WebUI Forge / Forge Classic
- Forge Neo
- SDXL-family models
- fixed-grid single-stage samplers
- fixed-step multi-stage TDE / RK samplers

Anima / NextDiT and flow-matching models are structurally compatible with the rank-agnostic forecasting path, but should be treated as **validation pending** until practical Forge / Forge Neo tests are completed.

Adaptive ODE methods are currently **unsupported / untested**.

A1111 is not supported because ChebyCast requires the Forge `forge_objects` backend.

---

## Installation

**Extensions -> Install from URL:**

```text
https://github.com/seti9585/sd-webui-ChebyCast
```

Restart the WebUI after installation.

---

## Why ChebyCast exists

Many diffusion acceleration wrappers treat every denoiser invocation as if it were a complete sampling step.

That assumption becomes problematic with multi-stage solvers. A classical four-stage RK step can evaluate the denoiser several times inside one solver step:

```text
solver step N
  k1
  k2
  k3
  k4
```

A raw model-call counter can incorrectly advance the forecasting state four times even though the sampler is still inside the same solver step.

ChebyCast instead separates:

```text
solver-step forecast decision
```

from:

```text
stage time / sigma coordinate
```

This lets all stage evaluations belonging to one fixed-step RK/TDE solver step share one actual-or-forecast decision while still receiving their own time coordinate.

---

## Main differences from Spectrum

ChebyCast uses Spectrum as its main methodological inspiration, but intentionally differs in several areas.

### Solver-step-aware execution

Actual / forecast decisions are cached per solver step rather than being decided independently for each raw model call.

This is intended for fixed-step multi-stage methods where one visible sampling step can contain multiple denoiser evaluations.

### Time coordinate

`auto` mode tries the following coordinate sources in order:

```text
schedule -> step -> timestep -> sigma
```

The goal is to avoid using the raw model-call count as the forecasting time axis.

### Local extrapolation

The non-spectral branch uses **Newton divided differences** instead of assuming uniform spacing.

This is intended to remain meaningful when intermediate solver stages occur at non-uniform coordinates.

### Rank-agnostic buffers

Wrapper outputs are flattened into float32 forecasting buffers and restored to their original shape after prediction.

ChebyCast therefore does not hard-code an SDXL-style 4-D latent rank.

### Independent safety clamp

Forecast outputs include an independent finite-value fallback and safety clamp.

This is a ChebyCast implementation detail, not a parameter from the Spectrum paper.

---

## Parameters

| Parameter | Description |
| --- | --- |
| **Blend weight (w)** | Mixes local Newton extrapolation and the Chebyshev prediction. `0` = local only, `1` = Chebyshev only. |
| **Chebyshev bases (m)** | Number of bases `T0 ... T(m-1)`. The default `m=4` means polynomial degree 3. |
| **Ridge regularization (lam)** | Ridge strength used when fitting Chebyshev coefficients. |
| **Window size** | Initial spacing of actual solver steps versus forecast steps. |
| **Window growth (flex)** | Increases the effective window after actual steps. |
| **History points (K)** | Maximum number of stored fit samples. |
| **Warmup steps** | Leading solver steps that always run the real model. |
| **Stop forecasting offset** | Trailing solver steps that always run the real model. |
| **Stage grouping** | `auto` uses the WebUI sampling-step counter; a number groups that many model calls into one solver step. |
| **Fit points** | Choose whether all actual intermediate stages or only the first actual point of each solver step update the fit. |
| **Apply to hires pass** | Enables a fresh ChebyCast runtime for the hires pass. |
| **Time coordinate** | Selects the fitting axis. `auto` prefers schedule, then step, timestep, and sigma. |

---

## About `m`

The official Spectrum README defines `algo.m` as the **number of Chebyshev bases**, with a default of `4`.

ChebyCast follows that public meaning:

```text
m = 4
```

uses:

```text
T0, T1, T2, T3
```

which is polynomial degree 3.

Internally, ChebyCast stores the polynomial degree as `m - 1`.

---

## Defaults and Spectrum defaults

ChebyCast does **not** claim to reproduce Spectrum's default parameter set.

Current ChebyCast defaults are:

| Parameter | ChebyCast |
| --- | ---: |
| `w` | 0.40 |
| `m` | 4 bases |
| `lam` | 1.00 |
| initial window | 2 |
| flex | 0.00 |
| history | 16 |
| warmup | 4 |
| stop offset | 3 |

The current official Spectrum README documents `w=1.0`, `lam=0.1`, and `m=4` as defaults, and discusses a post-publication mixture with linear interpolation.

ChebyCast's `w` instead mixes the Chebyshev predictor with its own Newton-divided-difference local extrapolator. The two `w` parameters therefore should **not** be assumed to be numerically interchangeable.

---

## Multi-stage samplers

Manual stage grouping is provided for environments where the WebUI sampling-step counter does not expose the desired solver-step boundary.

For example, a fixed four-stage method can be tested with:

```text
Stage grouping = 4
```

This option must not be treated as a general solution for adaptive ODE methods.

Adaptive methods can perform variable stage counts, rejected steps, embedded error evaluations, and repeated model calls. A fixed grouping number cannot reconstruct those boundaries reliably.

---

## Fit points

### all stages

Every actual intermediate stage can update the fit.

This can provide more samples, but those samples may come from non-uniform stage coordinates.

### step head only

Only the first actual update for each solver step is added for a given forecasting key.

This is a more conservative fit history.

---

## Wrapper compatibility note

ChebyCast can preserve an existing `model_function_wrapper` on **actual model calls** by chaining it inside the ChebyCast wrapper.

On a **forecasted call**, the model evaluation is intentionally replaced by a prediction, so the inner wrapper is not executed.

Therefore compatibility must be evaluated carefully for extensions that require their own model-function wrapper to run on every denoiser invocation.

---

## Debug output

Set the shared SETI debug environment variable before starting the WebUI:

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

ChebyCast will report the selected time-coordinate source and an end-of-run summary.

A higher debug level can be used for per-step decisions:

```powershell
$env:SD_WEBUI_SETI_DEBUG = "2"
```

---

## Validation guidance

For ON / OFF comparisons, keep these fixed:

- model
- prompt / negative prompt
- seed
- sampler
- scheduler
- steps
- CFG
- resolution

Compare:

- successful completion
- NaN / Inf
- severe color corruption
- latent instability
- structure failure
- actual denoiser-call reduction
- generation time

ChebyCast is an approximation method, so pixel-identical output is not expected.

---

## Compatibility status

| Target | Current status |
| --- | --- |
| reForge | Implementation target; validation required |
| Forge Classic / Forge | Implementation target; validation required |
| Forge Neo | Implementation target; validation required |
| SDXL | Implementation target |
| Anima / NextDiT | Architecture-compatible; validation pending |
| Standard fixed-grid samplers | Intended |
| Fixed-step multi-stage TDE / RK | Intended |
| Adaptive ODE solvers | Unsupported / untested |
| A1111 | Unsupported |

---

# 日本語

**[English](#sd-webui-chebycast)** | 日本語

Forge 派生 Stable Diffusion WebUI 向けの Chebyshev ベース feature forecasting 拡張機能です。solver-step-aware な実行方式と、固定ステップの multi-stage sampler を意識して設計しています。

ChebyCast は **Spectrum** — *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration* (Han et al., CVPR 2026) — から着想を得ていますが、**faithful port ではありません**。

denoiser feature を Chebyshev basis と ridge regression で近似する中心的な考え方を引き継ぎつつ、Forge wrapper、solver-step grouping、非等間隔の中間 stage、rank に依存しない model output を扱うために実行方式を変更しています。

---

## 現在の状態

ChebyCast は現時点では Forge 派生 WebUI 上での実機検証を前提とした初期実装です。

設計上の対象:

- reForge
- Stable Diffusion WebUI Forge / Forge Classic
- Forge Neo
- SDXL 系モデル
- 固定 grid の single-stage sampler
- 固定ステップの multi-stage TDE / RK sampler

Anima / NextDiT および flow-matching モデルについては、forecast buffer 自体は rank-agnostic ですが、Forge / Forge Neo での実機検証が完了するまでは **validation pending** とします。

Adaptive ODE method は現時点では **unsupported / untested** です。

A1111 は Forge の `forge_objects` backend を必要とするため非対応です。

---

## インストール

**Extensions -> Install from URL:**

```text
https://github.com/seti9585/sd-webui-ChebyCast
```

インストール後に WebUI を再起動してください。

---

## ChebyCast を作った理由

多くの diffusion acceleration wrapper は、denoiser の呼び出し 1 回を sampling step 1 回として扱うことがあります。

しかし multi-stage solver では、1 solver step の内部で複数回 denoiser を評価します。

```text
solver step N
  k1
  k2
  k3
  k4
```

raw model-call counter をそのまま forecasting state に使うと、sampler はまだ同じ solver step にいるのに forecasting 側だけ 4 step 進んだように扱う可能性があります。

ChebyCast では、

```text
solver-step 単位の forecast 判断
```

と、

```text
stage ごとの time / sigma coordinate
```

を分離します。

これにより、同じ fixed-step RK / TDE solver step に属する stage は同じ actual / forecast 判断を共有しつつ、それぞれの stage 座標を forecasting に使用できます。

---

## Spectrum との主な違い

ChebyCast は Spectrum を主要な着想元としていますが、意図的な変更があります。

### Solver-step-aware execution

actual / forecast の判断を raw model call ごとではなく solver step ごとに保存します。

1 sampling step の内部で複数回 denoiser を評価する fixed-step multi-stage method を想定した変更です。

### Time coordinate

`auto` では次の順番で coordinate を探します。

```text
schedule -> step -> timestep -> sigma
```

raw model-call count を forecasting の時間軸として使わないことを重視しています。

### Local extrapolation

非 spectral 側には **Newton divided differences** を使用します。

中間 stage の座標が非等間隔でも局所 extrapolation が意味を持つことを狙った変更です。

### Rank-agnostic buffer

wrapper output は float32 に flatten して forecasting buffer に保存し、prediction 後に元の shape へ戻します。

SDXL 型の 4-D latent rank を固定前提にはしていません。

### 独自 safety clamp

forecast output には finite-value fallback と safety clamp を設けています。

これは Spectrum 論文のパラメータではなく ChebyCast 独自の安全策です。

---

## パラメータ

| パラメータ | 説明 |
| --- | --- |
| **Blend weight (w)** | Newton local extrapolation と Chebyshev prediction の混合比。`0` = local のみ、`1` = Chebyshev のみ。 |
| **Chebyshev bases (m)** | `T0 ... T(m-1)` の basis 数。既定値 `m=4` は degree 3。 |
| **Ridge regularization (lam)** | Chebyshev coefficient fit の ridge 強度。 |
| **Window size** | actual solver step と forecast step の初期間隔。 |
| **Window growth (flex)** | actual step の後に effective window を増加させます。 |
| **History points (K)** | fit に保持する最大 sample 数。 |
| **Warmup steps** | 冒頭で必ず実 model を実行する solver step 数。 |
| **Stop forecasting offset** | 終端側で必ず実 model を実行する solver step 数。 |
| **Stage grouping** | `auto` は WebUI の sampling-step counter を使用。数値指定では N model call を 1 solver step にまとめます。 |
| **Fit points** | actual な中間 stage を全て fit へ入れるか、solver step ごとの先頭のみ入れるかを選択。 |
| **Apply to hires pass** | hires pass に新しい ChebyCast runtime を適用。 |
| **Time coordinate** | fitting axis。`auto` は schedule、step、timestep、sigma の順に選択。 |

---

## `m` の意味

公式 Spectrum README の `algo.m` は **Chebyshev basis の数**として定義され、既定値は `4` です。

ChebyCast も公開 UI では同じ意味に統一します。

```text
m = 4
```

なら、

```text
T0, T1, T2, T3
```

を使用し、polynomial degree は 3 です。

内部では `degree = m - 1` として保持します。

---

## 既定値と Spectrum の既定値

ChebyCast は Spectrum の既定パラメータを再現することを目的としていません。

現在の ChebyCast 既定値:

| パラメータ | ChebyCast |
| --- | ---: |
| `w` | 0.40 |
| `m` | 4 bases |
| `lam` | 1.00 |
| initial window | 2 |
| flex | 0.00 |
| history | 16 |
| warmup | 4 |
| stop offset | 3 |

現在の公式 Spectrum README では `w=1.0`、`lam=0.1`、`m=4` が既定値として記載され、公開後の補足として linear interpolation との convex mixture が説明されています。

一方 ChebyCast の `w` は Chebyshev predictor と独自の Newton-divided-difference local extrapolator を混合します。そのため、両者の `w` を同じ意味の数値として扱わないでください。

---

## Multi-stage sampler

WebUI の sampling-step counter だけでは solver-step boundary を適切に扱えない場合に備えて manual stage grouping を用意しています。

たとえば固定 4-stage method を検証する場合:

```text
Stage grouping = 4
```

ただし Adaptive ODE method への一般的な解決策ではありません。

adaptive method では stage 数が可変で、rejected step、embedded error evaluation、repeated call などが発生し得ます。固定の grouping 数では solver-step boundary を確実に再構成できません。

---

## Fit points

### all stages

actual な intermediate stage を fit history に追加します。

sample 数を増やせますが、中間 stage の座標は非等間隔になる場合があります。

### step head only

同じ forecasting key について solver step ごとの最初の actual update だけを history に追加します。

より保守的な fit history です。

---

## 他の wrapper との互換性

ChebyCast は既存の `model_function_wrapper` がある場合、**actual model call** ではその wrapper を内側へ chain できます。

一方 **forecast call** は model evaluation 自体を prediction で置き換えるため、内側の wrapper は実行されません。

したがって、全 denoiser invocation で自身の wrapper 実行を必要とする拡張機能との互換性は個別に検証する必要があります。

---

## Debug output

WebUI 起動前に PowerShell で共有 debug flag を設定します。

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

選択された time-coordinate source と run 終了時の summary が表示されます。

step ごとの判断まで表示する場合:

```powershell
$env:SD_WEBUI_SETI_DEBUG = "2"
```

---

## 検証方法

ON / OFF 比較では以下を固定してください。

- model
- prompt / negative prompt
- seed
- sampler
- scheduler
- steps
- CFG
- resolution

確認項目:

- 正常に生成完了すること
- NaN / Inf が発生しないこと
- 深刻な色崩れがないこと
- latent が発散しないこと
- 構図が大きく破綻しないこと
- denoiser call が実際に減ること
- generation time が改善すること

ChebyCast は近似法なので pixel-identical な出力は前提としません。

---

## Compatibility status

| 対象 | 現在の状態 |
| --- | --- |
| reForge | 実装対象・実機検証が必要 |
| Forge Classic / Forge | 実装対象・実機検証が必要 |
| Forge Neo | 実装対象・実機検証が必要 |
| SDXL | 実装対象 |
| Anima / NextDiT | architecture 上は対応可能・実機検証待ち |
| Standard fixed-grid sampler | Intended |
| Fixed-step multi-stage TDE / RK | Intended |
| Adaptive ODE solver | Unsupported / untested |
| A1111 | Unsupported |

---

# License / Acknowledgements / References

## License

ChebyCast is released under the MIT License. See [`LICENSE`](LICENSE).

Third-party attribution is recorded in [`NOTICE`](NOTICE).

## Acknowledgements

ChebyCast is primarily inspired by the Spectrum paper and official implementation:

- Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon
- *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*
- CVPR 2026 / arXiv:2603.01623
- Official implementation: https://github.com/hanjq17/Spectrum

The official Spectrum repository is MIT licensed and credits TaylorSeer as an inspiration for part of its codebase.

ChebyCast was developed as a separate Forge-oriented implementation. It should not be described as an official Spectrum port or a faithful reproduction of the official execution behavior.

Existing community implementations were useful references for understanding practical WebUI integration, including:

- https://github.com/hirorohi03/sd-webui-forge-spectrum
- https://github.com/hirorohi03/sd-forge-spectrum-faithful
- https://github.com/ruwwww/comfyui-spectrum-sdxl
- https://github.com/judian17/ComfyUI-Spectrum

## References

- Spectrum paper: https://arxiv.org/abs/2603.01623
- Spectrum official repository: https://github.com/hanjq17/Spectrum
- TaylorSeer: https://github.com/Shenyi-Z/TaylorSeer
