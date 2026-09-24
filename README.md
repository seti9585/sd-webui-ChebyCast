# sd-webui-ChebyCast

**EN** | [日本語](#日本語)

Forecast-based sampling acceleration for Forge-derived Stable Diffusion WebUIs.

ChebyCast reduces generation time by replacing some repeated **denoising model evaluations (U-Net calls)** with predictions based on recent real outputs.

```text
Normal:
U-Net -> U-Net -> U-Net -> U-Net -> U-Net

ChebyCast:
U-Net -> U-Net -> Forecast -> U-Net -> Forecast
```

Fewer U-Net calls can make generation faster.

Because predicted values slightly change the sampling trajectory, **ChebyCast ON is not expected to produce a pixel-identical image to ChebyCast OFF, even with the same seed.**

ChebyCast is inspired by **Spectrum** (*Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*, Han et al., CVPR 2026), but it is not an official port or a faithful reproduction of the official implementation.

ChebyCast separates solver steps from individual U-Net calls, allowing fixed-step multi-stage solvers to be handled correctly.

## Measured result

Common conditions: reForge / SDXL (Illustrious-based checkpoint) / Align Your Steps / 35 steps / CFG 7 / 896x1152 / RTX 4080 SUPER / no launch arguments / ChebyCast default settings. Sampling time from the WebUI progress bar, median of 5 runs after one warm-up run.

```text
DPM++ 2M SDE (Stage grouping 1)
U-Net calls  35 -> 22          (-37%)
Sampling     7.07s -> 4.82s    (-32%)

TDE Sampler / kutta4 (Stage grouping auto)
U-Net calls  139 -> 83         (-40%)
Sampling     26.1s -> 16.1s    (-38%)
```

The time saved grows with the time spent per step, so slower samplers and heavier settings benefit more in absolute seconds.

---

## Installation

**Extensions -> Install from URL**

```text
https://github.com/seti9585/sd-webui-ChebyCast
```

Restart the WebUI after installation.

When updating from an older version, **restart the WebUI** instead of using Reload UI. A new control was added, and Reload UI does not rebuild the panel.

---

## Quick start

1. Open the **ChebyCast** panel.
2. Enable **Enable ChebyCast**.
3. Leave the other settings at their defaults.
4. Generate normally.

Keep **Time coordinate** on `auto`.

Set **Stage grouping** according to the sampler:

| Sampler type | Examples | Stage grouping |
| --- | --- | --- |
| One U-Net call per step | Euler, Euler a, DPM++ 2M, DPM++ 2M SDE | `1` |
| TDE Sampler / RK Sampler fixed-step methods | kutta4 | `auto` |

With `auto`, standard WebUI samplers report the step number one call late, so every step boundary (Warmup, Stop offset, Skip negative) shifts by one. Setting `1` counts steps from the calls themselves and avoids this. It assumes positive and negative are evaluated together in one call (the normal case).

---

## Tuning

**Want more speed -> increase Window size (faster / larger difference from OFF)**  
**Want to protect image quality -> increase Warmup / Stop offset (more conservative / less speedup)**  
**Want a little more speed in the final steps -> Skip negative in last N steps (see below)**  
**Want to experiment with the prediction method -> w / m / lam**

With very low step counts, the forecastable middle section becomes small, so the speedup may be limited.

---

## Parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| **Window size (solver steps)** | 2 | Larger = more U-Net calls replaced by forecasts. |
| **Warmup steps** | 4 | Keeps the first solver steps on real U-Net calls. |
| **Stop forecasting offset** | 3 | Keeps the final solver steps on real U-Net calls. |
| **Blend weight (w)** | 0.40 | `0` = local extrapolation, `1` = Chebyshev prediction. |
| **Chebyshev bases (m)** | 4 | Number of Chebyshev basis functions used for fitting; larger = more flexible fit. |
| **Ridge regularization (lam)** | 1.00 | Larger = stronger regularization. |
| **Window growth (flex)** | 0.00 | Larger values make forecasting more aggressive as sampling progresses. |
| **History points (K)** | 16 | Larger values keep more real samples available for fitting. |
| **Stage grouping** | `auto` | Groups model calls into solver steps. `1` for one-call-per-step samplers, `auto` for TDE / RK Sampler (see Quick start). |
| **Fit points** | `all stages` | Chooses which real stage outputs update the fit. |
| **Apply to hires pass** | Off | Applies ChebyCast to the Hires.fix pass as well. |
| **Time coordinate** | `auto` | Chooses the sampling-progress axis used by the predictor. |
| **Skip negative in last N steps** | 0 | `0` = off. In the last N solver steps, only the positive prompt is evaluated. Limited to the Stop forecasting offset. |

---

## Skip negative in last N steps

An optional setting that is **off by default** (`0`). With `0`, ChebyCast behaves exactly as before.

### What it does

With normal CFG, each U-Net call evaluates the positive prompt and the negative prompt together.

When this setting is `1` or more, ChebyCast evaluates **only the positive prompt** in the last N solver steps and uses that result in place of the negative result as well.

```text
Normal final steps:       positive + negative -> CFG
With this setting:        positive only       -> same as CFG 1
```

Those final steps therefore run as if CFG were 1. Details are mostly settled by then, but the image is **not identical** to the result with `0`.

The speed gain is expected to be modest, because it applies only to the last few steps.

### How to use it

Set it to the same value as **Stop forecasting offset**. Those steps are always real U-Net calls, so this setting does not interfere with forecasting.

```text
Stop forecasting offset = 4
Skip negative in last N steps = 4
```

A value larger than the Stop forecasting offset is reduced to the Stop forecasting offset, and a warning is printed to the console.

For multi-stage samplers, the decision is shared by all stages of the same solver step, so one step never mixes stages with and without the negative prompt.

### Do not enable the WebUI's own setting at the same time

reForge and Forge Neo have a similar built-in option in **Settings**. When you use this ChebyCast setting, **keep the WebUI option off.**

| WebUI | Setting | Value to use |
| --- | --- | --- |
| reForge / Forge | Negative Guidance minimum sigma | `0` |
| reForge / Forge | Ignore negative prompt during early sampling | `0` |
| Forge Neo | Skip Negative Prompt during Later Steps | `0` |
| Forge Neo | Ignore Negative Prompt during Early Steps | `0` |

With the sigma value at `0`, the "all steps" checkbox has no effect, so it can be left as it is.

Reasons:

- **The two overlap.** Calls that the WebUI has already reduced to positive-only are passed through unchanged by ChebyCast, so there is no extra speedup.
- **The WebUI option decides by sigma, not by step.** In multi-stage samplers, the last stage of one step and the first stage of the next step have the same sigma. Whatever threshold you choose, one boundary step ends up with some stages with the negative prompt and some without.
- **The WebUI option applies to every generation**, including generations without ChebyCast and the Hires.fix pass. The ChebyCast setting applies only when ChebyCast is enabled, and it is saved in the PNG infotext.

ChebyCast prints a console warning when both are active.

### Reference: WebUI sigma values that roughly match N

If you previously used **Negative Guidance minimum sigma**, the table below shows the approximate value that covered the same final steps. **For comparison only.** With ChebyCast, set the WebUI option to `0` and use N instead.

Conditions: SDXL, 35 steps, reForge scheduler code with default settings (Karras rho 7.0, Beta alpha 0.6 / beta 0.6). The values are calculated from the scheduler code and were not checked against console output. Different step counts or scheduler settings give different values. These values do not apply to Anima (its sigma range is different).

| N | Align Your Steps | Karras | Beta | SGM Uniform |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.03 | 0.03 | 0.08 | 0.17 |
| 2 | 0.05 | 0.04 | 0.13 | 0.25 |
| 3 | 0.07 | 0.06 | 0.18 | 0.32 |
| 4 | 0.10 | 0.07 | 0.23 | 0.38 |
| 5 | 0.13 | 0.09 | 0.28 | 0.45 |
| 6 | 0.16 | 0.12 | 0.34 | 0.51 |

Karras and Align Your Steps pack many steps into very small sigma values near the end, so the numbers are small. Beta and SGM Uniform keep larger sigma values near the end, so the same N needs a larger number.

### When it is not applied

The step is run normally (positive and negative) when:

- positive and negative are sent to the U-Net as separate calls (for example, split because of low VRAM),
- ControlNet is active,
- the prompt layout is not one positive + one negative (for example, `AND` prompts),
- the positive-only call fails.

### Guidance extensions

In the skipped steps, guidance extensions receive a negative result that is equal to the positive result. The guidance difference becomes zero, so most extensions simply have nothing to add.

With [sd-webui-TCFG](https://github.com/seti9585/sd-webui-TCFG), [sd-webui-SkimmedCFG](https://github.com/seti9585/sd-webui-SkimmedCFG), and [sd-webui-DifferenceCFG](https://github.com/seti9585/sd-webui-DifferenceCFG), the result is the positive prediction unchanged. With [sd-webui-APG](https://github.com/seti9585/sd-webui-APG), the same is true when Momentum is `0`. When Momentum is not `0`, a small amount of guidance from earlier steps remains for the first few skipped calls and fades out.

Other guidance extensions should be tested individually.

---

## Hires.fix

ChebyCast is disabled for the Hires.fix pass by default.

Enable **Apply to hires pass** if you want to use it there. The Hires.fix pass starts with its own prediction state. **Skip negative in last N steps** also applies to the Hires.fix pass when this option is on.

---

## Why the image changes

ChebyCast does not cache and replay an old U-Net result.

It predicts what the U-Net would have returned at a new point in the sampling process. Once that prediction is used, the following sampling trajectory becomes slightly different.

So:

```text
Same seed + ChebyCast OFF
!=
Same seed + ChebyCast ON
```

This is expected behaviour.

In validation, repeated ChebyCast ON runs with the same environment, settings, and seed produced pixel-identical raw output.

---

## Multi-stage samplers

Some samplers call the U-Net several times inside one solver step.

```text
solver step N
  stage 1 -> U-Net
  stage 2 -> U-Net
  stage 3 -> U-Net
  stage 4 -> U-Net
```

In that case:

```text
1 U-Net call != 1 solver step
```

ChebyCast keeps the **real-or-forecast decision at solver-step level**, while each internal stage still has its own position on the prediction axis.

This is useful for fixed-step multi-stage methods such as classical Runge-Kutta methods.

My **TDE Sampler** and **RK Sampler** are examples of extensions that can use this type of multi-stage integration. They are not required to use ChebyCast.

Adaptive ODE solvers are not supported.

---

## Compatibility

ChebyCast requires the Forge `forge_objects` backend.

| Target | Status |
| --- | --- |
| reForge | Supported design target |
| Stable Diffusion WebUI Forge / Forge Classic | Supported design target |
| Forge Neo | Supported design target |
| SDXL-family models | Primary target |
| Anima / NextDiT | Architecture-compatible / validation pending |
| Fixed-grid samplers | Supported design target |
| Fixed-step multi-stage samplers | Supported design target |
| Adaptive ODE solvers | Not supported |
| A1111 | Not supported |

ADetailer and postprocessing sub-runs are intentionally skipped.

---

## Compatibility with other extensions

ChebyCast uses Forge's `model_function_wrapper`.

Existing wrappers are preserved on **real U-Net calls**.

On a **forecasted call**, the U-Net itself is not executed, so an inner wrapper does not run either.

In steps where **Skip negative in last N steps** is active, an inner wrapper runs with the positive part only.

Extensions that require their wrapper to execute on every denoiser call should therefore be tested individually.

---

## Debug output

Set the environment variable before launching the WebUI.

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

Level 1 reports the selected time coordinate and a run summary. When **Skip negative in last N steps** is `1` or more, the summary also shows how many calls were evaluated with the positive prompt only.

```text
neg-skip last=4 calls=15 fallbacks=0
```

`fallbacks` counts calls that were run normally for one of the reasons listed in "When it is not applied".

```powershell
$env:SD_WEBUI_SETI_DEBUG = "2"
```

Level 2 also reports real-or-forecast decisions for individual solver steps.

---

## Relationship to Spectrum

ChebyCast uses the forecasting idea described in:

**Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration**  
Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon  
CVPR 2026 / arXiv:2603.01623

ChebyCast is a separate Forge-oriented implementation.

Main differences include:

- forecast decisions are tracked per solver step rather than by raw model-call count,
- the time coordinate prefers schedule / solver step / timestep / sigma,
- the local prediction branch uses Newton divided differences,
- model outputs are stored as flattened float32 history and restored to their original shape,
- ChebyCast includes its own non-finite fallback and prediction clamp.

**Skip negative in last N steps** is not part of Spectrum. It is an optional ChebyCast feature.

---

# 日本語

**[English](#sd-webui-chebycast)** | 日本語

Forge 系 Stable Diffusion WebUI 向けの、**予測による生成高速化拡張機能**です。

ChebyCast は、画像生成中に何度も繰り返される**ノイズ除去のためのモデル計算（U-Net 呼び出し）**の一部を、直前までの実際の計算結果から予測した値で置き換えます。

```text
通常:
U-Net -> U-Net -> U-Net -> U-Net -> U-Net

ChebyCast:
U-Net -> U-Net -> 予測 -> U-Net -> 予測
```

U-Net の呼び出し回数を減らすことで、生成時間の短縮を狙います。

予測値を使うと sampling trajectory が少し変わるため、**同じ seed でも ChebyCast OFF と ON の画像はピクセル単位では一致しません。**

ChebyCast は **Spectrum**（*Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*、Han ほか、CVPR 2026）から着想を得ていますが、公式実装の移植でも、公式実装の動作を忠実に再現したものでもありません。

ChebyCast は solver step と個々の U-Net 呼び出しを分けて扱うため、固定ステップの多段 solver に対応できます。

## 実測結果

共通条件: reForge / SDXL（Illustrious 系モデル）/ Align Your Steps / 35 steps / CFG 7 / 896x1152 / RTX 4080 SUPER / 起動引数なし / ChebyCast 既定値。時間は WebUI のプログレスバーのサンプリング時間で、慣らしの 1 回を除いた 5 回の中央値です。

```text
DPM++ 2M SDE（Stage grouping 1）
U-Net calls  35 -> 22          (-37%)
Sampling     7.07s -> 4.82s    (-32%)

TDE Sampler / kutta4（Stage grouping auto）
U-Net calls  139 -> 83         (-40%)
Sampling     26.1s -> 16.1s    (-38%)
```

1 ステップにかかる時間が長いほど、短縮できる秒数も大きくなります。

---

## インストール

**Extensions -> Install from URL**

```text
https://github.com/seti9585/sd-webui-ChebyCast
```

インストール後、WebUI を再起動してください。

旧版から更新した場合も、Reload UI ではなく **WebUI を再起動**してください。設定項目が増えているため、Reload UI ではパネルが作り直されません。

---

## まず使う

1. **ChebyCast** パネルを開く
2. **Enable ChebyCast** を ON
3. 他は既定値のまま
4. そのまま生成

**Time coordinate** は `auto` のまま使ってください。

**Stage grouping** はサンプラーに合わせて設定します。

| サンプラーの種類 | 例 | Stage grouping |
| --- | --- | --- |
| 1 ステップで U-Net を 1 回呼ぶもの | Euler、Euler a、DPM++ 2M、DPM++ 2M SDE | `1` |
| TDE Sampler / RK Sampler の固定ステップ法 | kutta4 | `auto` |

`auto` のままだと、WebUI 標準のサンプラーではステップ番号が 1 呼び出し遅れて伝わるため、Warmup・Stop offset・Skip negative の境目がすべて 1 つずれます。`1` にすると、呼び出し回数からステップを数えるので、このずれが起きません。ポジティブとネガティブを 1 回の呼び出しでまとめて計算していること（通常の状態）が前提です。

---

## 調整の目安

**速度を上げたい -> Window size を大きくする（高速化↑ / OFFとの差も増えやすい）**  
**画質を守りたい -> Warmup / Stop offset を大きくする（保守的 / 高速化↓）**  
**最後の数ステップをもう少し速くしたい -> Skip negative in last N steps（後述）**  
**予測方式そのものを実験したい -> w / m / lam**

step 数が少ない設定では forecast できる中間区間が短くなるため、高速化の効果が出にくくなります。

---

## パラメータ

| 項目 | 既定値 | 内容 |
| --- | ---: | --- |
| **Window size (solver steps)** | 2 | 大きいほど多くの U-Net 呼び出しを予測へ置き換えます。 |
| **Warmup steps** | 4 | 冒頭を実際の U-Net 呼び出しのまま残します。 |
| **Stop forecasting offset** | 3 | 終盤を実際の U-Net 呼び出しのまま残します。 |
| **Blend weight (w)** | 0.40 | `0` = 局所外挿、`1` = Chebyshev 予測です。 |
| **Chebyshev bases (m)** | 4 | fit に使う Chebyshev 基底の数です。大きいほど fit の自由度が上がります。 |
| **Ridge regularization (lam)** | 1.00 | 大きいほど正則化を強くします。 |
| **Window growth (flex)** | 0.00 | 大きいほど、生成が進むにつれて forecast を積極的にします。 |
| **History points (K)** | 16 | 大きいほど、fit に保持する実測点を増やします。 |
| **Stage grouping** | `auto` | モデル呼び出しを solver step にまとめます。1 ステップ 1 回のサンプラーは `1`、TDE / RK Sampler は `auto`（「まず使う」参照）。 |
| **Fit points** | `all stages` | どの実測 stage を fit に使うかを選びます。 |
| **Apply to hires pass** | OFF | Hires.fix 側にも ChebyCast を適用します。 |
| **Time coordinate** | `auto` | 予測に使う sampling progress の軸を選びます。 |
| **Skip negative in last N steps** | 0 | `0` = OFF。最後の N solver step でポジティブプロンプトだけを計算します。上限は Stop forecasting offset です。 |

---

## Skip negative in last N steps

**既定では OFF**（`0`）の追加機能です。`0` のときの動作は従来とまったく同じです。

### 何をするか

通常の CFG では、U-Net を呼ぶたびにポジティブプロンプトとネガティブプロンプトの両方を計算します。

この項目を `1` 以上にすると、最後の N solver step では**ポジティブプロンプトだけ**を計算し、その結果をネガティブ側の結果としても使います。

```text
通常の終盤:       ポジティブ + ネガティブ -> CFG
この項目を使う:   ポジティブのみ         -> CFG 1 と同じ
```

そのため、その区間は CFG 1 相当で生成されます。終盤は細部がほぼ決まった段階ですが、`0` のときと**同じ画像にはなりません**。

最後の数ステップだけに効く機能なので、速度の向上は控えめです。

### 使い方

**Stop forecasting offset と同じ値**にしてください。この区間は常に実際の U-Net 呼び出しなので、予測の邪魔をしません。

```text
Stop forecasting offset = 4
Skip negative in last N steps = 4
```

Stop forecasting offset より大きい値を入れた場合は、Stop forecasting offset の値に切り詰め、コンソールに注意を表示します。

多段サンプラーでは、同じ solver step の全 stage に同じ判断を使います。1 つのステップの中で、ネガティブありの stage となしの stage が混ざることはありません。

### WebUI 側の設定は同時に有効にしない

reForge と Forge Neo には、**Settings** 側に似た機能があります。ChebyCast のこの項目を使うときは、**WebUI 側の設定は無効のまま**にしてください。

| WebUI | 設定 | 使う値 |
| --- | --- | --- |
| reForge / Forge | Negative Guidance minimum sigma | `0` |
| reForge / Forge | Ignore negative prompt during early sampling | `0` |
| Forge Neo | Skip Negative Prompt during Later Steps | `0` |
| Forge Neo | Ignore Negative Prompt during Early Steps | `0` |

sigma の値が `0` なら、「all steps」のチェックボックスは効かないので、そのままで構いません。

理由:

- **同じ処理が重なります。** WebUI 側がすでにポジティブだけにした呼び出しは、ChebyCast がそのまま通すので、それ以上速くなりません。
- **WebUI 側はステップではなく sigma で判定します。** 多段サンプラーでは、あるステップの最後の stage と次のステップの最初の stage が同じ sigma になります。どのしきい値を選んでも、境目のステップでネガティブありの stage となしの stage が混ざります。
- **WebUI 側はすべての生成にかかります。** ChebyCast を使わない生成や Hires.fix 側にも効きます。ChebyCast の項目は ChebyCast が有効なときだけ効き、PNG の infotext にも記録されます。

両方が有効になっていると、ChebyCast がコンソールに注意を表示します。

### 参考: N に相当する WebUI 側の sigma 値

以前 **Negative Guidance minimum sigma** を使っていた場合、同じ終盤のステップを対象にしていた値の目安は次のとおりです。**比較用の参考値です。** ChebyCast を使うときは WebUI 側を `0` にして、N で指定してください。

条件: SDXL、35 steps、reForge のスケジューラ実装と既定設定（Karras rho 7.0、Beta alpha 0.6 / beta 0.6）。値はスケジューラのコードから計算したもので、実機のコンソール表示とは照合していません。ステップ数やスケジューラの設定が変わると値も変わります。Anima には当てはまりません（sigma の範囲が異なるため）。

| N | Align Your Steps | Karras | Beta | SGM Uniform |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.03 | 0.03 | 0.08 | 0.17 |
| 2 | 0.05 | 0.04 | 0.13 | 0.25 |
| 3 | 0.07 | 0.06 | 0.18 | 0.32 |
| 4 | 0.10 | 0.07 | 0.23 | 0.38 |
| 5 | 0.13 | 0.09 | 0.28 | 0.45 |
| 6 | 0.16 | 0.12 | 0.34 | 0.51 |

Karras と Align Your Steps は終盤のステップがとても小さな sigma に詰まっているため、値が小さくなります。Beta と SGM Uniform は終盤でも sigma が比較的大きく残るため、同じ N でも値が大きくなります。

### 適用されない場合

次の場合、そのステップは通常どおり（ポジティブとネガティブの両方）計算します。

- VRAM 不足などで、ポジティブとネガティブが別々の U-Net 呼び出しに分かれている
- ControlNet が有効
- ポジティブ 1 つ + ネガティブ 1 つの形になっていない（`AND` 構文など）
- ポジティブだけの呼び出しが失敗した

### ガイダンス系拡張機能との併用

省略したステップでは、ガイダンス系拡張機能にはポジティブと同じ値がネガティブの結果として渡ります。ガイダンスの差が 0 になるので、多くの拡張機能は何もしないのと同じになります。

[sd-webui-TCFG](https://github.com/seti9585/sd-webui-TCFG)、[sd-webui-SkimmedCFG](https://github.com/seti9585/sd-webui-SkimmedCFG)、[sd-webui-DifferenceCFG](https://github.com/seti9585/sd-webui-DifferenceCFG) では、ポジティブの予測がそのまま結果になります。[sd-webui-APG](https://github.com/seti9585/sd-webui-APG) も Momentum が `0` なら同じです。Momentum が `0` 以外の場合は、それまでのステップのガイダンスが、省略区間の最初の数回だけ少し残り、すぐに弱まります。

その他のガイダンス系拡張機能は、個別に併用確認をしてください。

---

## Hires.fix

Hires.fix 側では ChebyCast は既定で無効です。

使用する場合は **Apply to hires pass** を ON にしてください。Hires.fix 側では独立した予測状態を新しく開始します。この項目が ON のときは、**Skip negative in last N steps** も Hires.fix 側に適用されます。

---

## なぜ画像が変わるのか

ChebyCast は過去の U-Net 出力をそのまま再利用するキャッシュではありません。

まだ実際には計算していない位置について、U-Net が返すはずの値を予測します。その予測値を使った時点から、その後の sampling trajectory も少し変わります。

そのため、

```text
同じ seed + ChebyCast OFF
!=
同じ seed + ChebyCast ON
```

となります。

これは手法の性質です。

実機検証では、同一環境・同一設定・同一 seed の ChebyCast ON 同士で、生ピクセルまで完全な再現性を確認しています。

---

## 多段サンプラー

サンプラーによっては、1 solver step の中で U-Net を複数回呼び出します。

```text
solver step N
  stage 1 -> U-Net
  stage 2 -> U-Net
  stage 3 -> U-Net
  stage 4 -> U-Net
```

この場合、

```text
U-Net 呼び出し 1 回 != solver step 1 回
```

です。

ChebyCast は、**実測にするか予測にするかを solver step 単位で共有**しながら、各 stage には予測軸上の個別の位置を持たせます。

これは classical Runge-Kutta のような固定ステップの多段法で利用できます。

拙作の **TDE Sampler** と **RK Sampler** は、このような多段積分を利用できる拡張機能の例です。ChebyCast の利用に必須ではありません。

adaptive ODE ソルバーには対応していません。

---

## 対応環境

ChebyCast は Forge の `forge_objects` バックエンドを必要とします。

| 対象 | 状態 |
| --- | --- |
| reForge | 対象として設計 |
| Stable Diffusion WebUI Forge / Forge Classic | 対象として設計 |
| Forge Neo | 対象として設計 |
| SDXL 系モデル | 主な対象 |
| Anima / NextDiT | アーキテクチャ上は互換・検証待ち |
| 固定ステップのサンプラー | 対象として設計 |
| 固定ステップの多段サンプラー | 対象として設計 |
| adaptive ODE ソルバー | 非対応 |
| A1111 | 非対応 |

ADetailer と postprocessing の追加 run は意図的に適用対象から除外しています。

---

## 他の拡張機能との併用

ChebyCast は Forge の `model_function_wrapper` を使います。

既存の wrapper は、**実際に U-Net を呼ぶ場合**には維持されます。

一方、**予測へ置き換えた場合**は U-Net 自体を呼ばないため、内側の wrapper も実行されません。

**Skip negative in last N steps** が働いているステップでは、内側の wrapper はポジティブ側だけで実行されます。

すべての denoiser call で wrapper が実行されることを必要とする拡張機能は、個別に併用確認が必要です。

---

## デバッグ出力

WebUI 起動前に環境変数を設定します。

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

Level 1 では、選択された time coordinate と run summary を表示します。**Skip negative in last N steps** が `1` 以上のときは、ポジティブだけで計算した呼び出し回数も表示します。

```text
neg-skip last=4 calls=15 fallbacks=0
```

`fallbacks` は、「適用されない場合」の理由で通常どおり計算した回数です。

```powershell
$env:SD_WEBUI_SETI_DEBUG = "2"
```

Level 2 では、各 solver step の実測 / 予測判断も表示します。

---

## Spectrum との関係

ChebyCast は次の研究で示された forecasting の考え方をもとにしています。

**Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration**  
Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon  
CVPR 2026 / arXiv:2603.01623

ChebyCast は Forge 系 WebUI 向けに独自に書き起こした実装です。

主な違いは次のとおりです。

- forecast 判断を raw model-call count ではなく solver step 単位で保持
- time coordinate は schedule / solver step / timestep / sigma を優先
- 局所予測に Newton divided differences を使用
- model output を float32 に平坦化して履歴保持し、元の shape に戻す
- non-finite fallback と独立した prediction clamp を実装

**Skip negative in last N steps** は Spectrum の一部ではなく、ChebyCast 独自の追加機能です。

---

# License / Acknowledgements / References

## License / ライセンス

ChebyCast is released under the MIT License. See [`LICENSE`](LICENSE).

本拡張機能は MIT License で公開しています。全文は [`LICENSE`](LICENSE) を参照してください。

---

## Acknowledgements / 謝辞

Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon  
*Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*  
CVPR 2026 / arXiv:2603.01623

Official implementation / 公式実装:

- [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)

ChebyCast is a separate Forge-oriented implementation and is not an official Spectrum port.

ChebyCast は Forge 系 WebUI 向けに独自に書き起こした実装であり、Spectrum の公式移植ではありません。

### Other implementations of the same idea / 同じ手法の他の実装

These implementations were consulted only for practical WebUI integration. No source code was copied into ChebyCast.

これらは WebUI への統合方法を検討する際の参考にしたもので、ChebyCast へコードを流用していません。

- [hirorohi03/sd-webui-forge-spectrum](https://github.com/hirorohi03/sd-webui-forge-spectrum)
- [hirorohi03/sd-forge-spectrum-faithful](https://github.com/hirorohi03/sd-forge-spectrum-faithful)
- [ruwwww/comfyui-spectrum-sdxl](https://github.com/ruwwww/comfyui-spectrum-sdxl)
- [judian17/ComfyUI-Spectrum](https://github.com/judian17/ComfyUI-Spectrum)

---

## References / 典拠

- Spectrum paper / 論文: [arXiv:2603.01623](https://arxiv.org/abs/2603.01623)
- Spectrum official repository / 公式リポジトリ: [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)
- Negative Guidance minimum sigma (original A1111 option) / WebUI 側の同種設定の元: [AUTOMATIC1111/stable-diffusion-webui PR #9177](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177)
