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

Measured (TDE Sampler / kutta4 / Align Your Steps / 35 steps / 896x1152 / RTX 4080 SUPER / no launch arguments):

```text
U-Net calls  139 -> 83       (-40%)
Time         26.1s -> 15.75s (-40%)
```

---

## Installation

**Extensions -> Install from URL**

```text
https://github.com/seti9585/sd-webui-ChebyCast
```

Restart the WebUI after installation.

---

## Quick start

1. Open the **ChebyCast** panel.
2. Enable **Enable ChebyCast**.
3. Leave the other settings at their defaults.
4. Generate normally.

For normal use, keep **Stage grouping** and **Time coordinate** on `auto`.

---

## Tuning

**Want more speed -> increase Window size (faster / larger difference from OFF)**  
**Want to protect image quality -> increase Warmup / Stop offset (more conservative / less speedup)**  
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
| **Stage grouping** | `auto` | Groups model calls into solver steps. Usually leave on `auto`. |
| **Fit points** | `all stages` | Chooses which real stage outputs update the fit. |
| **Time coordinate** | `auto` | Chooses the sampling-progress axis used by the predictor. |
| **Apply to hires pass** | Off | Applies ChebyCast to the Hires.fix pass as well. |

---

## Hires.fix

ChebyCast is disabled for the Hires.fix pass by default.

Enable **Apply to hires pass** if you want to use it there. The Hires.fix pass starts with its own prediction state.

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

Extensions that require their wrapper to execute on every denoiser call should therefore be tested individually.

---

## Debug output

Set the environment variable before launching the WebUI.

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

Level 1 reports the selected time coordinate and a run summary.

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

実測条件: TDE Sampler / kutta4 / Align Your Steps / 35 steps / 896x1152 / RTX 4080 SUPER / 起動引数なし

```text
U-Net calls  139 -> 83       (-40%)
Time         26.1s -> 15.75s (-40%)
```

---

## インストール

**Extensions -> Install from URL**

```text
https://github.com/seti9585/sd-webui-ChebyCast
```

インストール後、WebUI を再起動してください。

---

## まず使う

1. **ChebyCast** パネルを開く
2. **Enable ChebyCast** を ON
3. 他は既定値のまま
4. そのまま生成

通常は **Stage grouping** と **Time coordinate** を `auto` のまま使ってください。

---

## 調整の目安

**速度を上げたい -> Window size を大きくする（高速化↑ / OFFとの差も増えやすい）**  
**画質を守りたい -> Warmup / Stop offset を大きくする（保守的 / 高速化↓）**  
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
| **Stage grouping** | `auto` | モデル呼び出しを solver step にまとめます。通常は `auto` のままです。 |
| **Fit points** | `all stages` | どの実測 stage を fit に使うかを選びます。 |
| **Time coordinate** | `auto` | 予測に使う sampling progress の軸を選びます。 |
| **Apply to hires pass** | OFF | Hires.fix 側にも ChebyCast を適用します。 |

---

## Hires.fix

Hires.fix 側では ChebyCast は既定で無効です。

使用する場合は **Apply to hires pass** を ON にしてください。Hires.fix 側では独立した予測状態を新しく開始します。

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

すべての denoiser call で wrapper が実行されることを必要とする拡張機能は、個別に併用確認が必要です。

---

## デバッグ出力

WebUI 起動前に環境変数を設定します。

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

Level 1 では、選択された time coordinate と run summary を表示します。

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
