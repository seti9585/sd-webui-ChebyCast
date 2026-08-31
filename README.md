# sd-webui-ChebyCast

**EN** | [日本語](#日本語)

ChebyCast makes image generation faster by skipping some of the model evaluations a sampler would normally perform, and filling in the gaps with a prediction instead.

During sampling, the denoising model (the UNet) is called over and over, and its output changes smoothly from one step to the next. ChebyCast watches that change, fits a curve through the recent outputs, and uses the curve to predict what the model would have returned. On predicted steps the model is not run at all, which is where the time saving comes from.

The part that sets ChebyCast apart is that it understands samplers which call the model **several times inside a single sampling step**, such as the TDE and RK samplers. Most extensions of this kind assume "one model call equals one sampling step", which is not true for those samplers.

ChebyCast is inspired by **Spectrum** (*Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*, Han et al., CVPR 2026), but it is **not a port** of the official implementation. It keeps the central idea and changes how that idea is executed inside Forge.

---

## Requirements

ChebyCast needs the Forge `forge_objects` backend, so it works on:

- reForge
- Stable Diffusion WebUI Forge / Forge Classic
- Forge Neo

A1111 is **not supported**.

It is written for SDXL-family models and for samplers that use a fixed step grid. Adaptive ODE solvers are **not supported and untested** (see [Compatibility status](#compatibility-status)).

---

## Installation

**Extensions -> Install from URL:**

```text
https://github.com/seti9585/sd-webui-ChebyCast
```

Restart the WebUI after installing. A restart is required, not just a UI reload.

---

## Getting started

Enable ChebyCast and generate with the default values first. The defaults are chosen to be cautious: the real model always runs for the first few steps and the last few steps, and only the middle of the sampling run is allowed to use predictions.

If you use a multi-stage sampler (TDE / RK with a method such as `kutta4`), leave **Stage grouping** on `auto` for the first test. Only set it to a number if the debug output shows that the step boundaries are not being detected correctly.

Two things to know before you compare results:

- **The output will not be pixel-identical to ChebyCast OFF, even with the same seed.** A predicted step nudges the sampling path onto a slightly different course, and that difference carries through to the end. Composition, colour and overall structure are preserved, but fine detail resolves differently. This is a property of the method, not a bug.
- If the picture breaks down, increase **Warmup steps** and **Stop forecasting offset** first, and lower **Window size**. Those three control how much of the run is allowed to be predicted.

---

## Parameters

### Commonly adjusted

| Parameter | What it does |
| --- | --- |
| **Window size** | How far apart the real model calls are. A larger value skips more and runs faster, but predicts further ahead. |
| **Warmup steps** | How many steps at the beginning always run the real model. Predictions need real samples to fit against, so this cannot be zero. |
| **Stop forecasting offset** | How many steps at the end always run the real model. The final steps decide fine detail, so keeping them real protects image quality. |
| **Blend weight (w)** | How much of the prediction comes from the Chebyshev fit versus the simpler local extrapolation. `0` uses only the local method, `1` uses only the Chebyshev fit. |
| **Apply to hires pass** | Whether the hires pass also gets ChebyCast. A fresh prediction state is started for that pass. |

### Advanced

| Parameter | What it does |
| --- | --- |
| **Chebyshev bases (m)** | How many curve components are used for the fit. Higher values can follow a more complicated shape but need more real samples to stay stable. |
| **Ridge regularization (lam)** | How strongly the fit is held back from swinging around. Higher values give a smoother, more conservative curve. |
| **Window growth (flex)** | Widens the gap between real model calls as the run proceeds. `0` keeps the gap constant. |
| **History points (K)** | How many past real samples are kept for fitting. |
| **Stage grouping** | `auto` uses the WebUI sampling-step counter to find step boundaries. A number instead groups that many model calls into one step. |
| **Fit points** | Whether every real intermediate stage feeds the fit, or only the first real point of each step. |
| **Time coordinate** | Which axis the curve is fitted against. `auto` tries schedule, then step, then timestep, then sigma. |

### A note on `m`

The official Spectrum README defines `algo.m` as the **number of Chebyshev bases**, with a default of `4`. ChebyCast uses the same meaning, so `m = 4` uses the four components `T0`, `T1`, `T2` and `T3`. Internally the value is stored as `m - 1`, but nothing in the UI asks you to think in those terms.

---

## How it works

### The basic idea

The model's output over the course of a sampling run behaves like a smooth curve. If you have several points on that curve, you can fit a formula to them and read off a value at a position you have not actually computed.

ChebyCast fits that curve using **Chebyshev polynomials**. Their useful property is that the approximation error stays evenly spread across the whole interval instead of piling up at one end, which is what makes them suitable for predicting some distance ahead rather than just one step. The fit itself is a least-squares fit with a regularization term (ridge regression); the **Ridge regularization (lam)** slider is the strength of that term.

### Multi-stage samplers, and why ChebyCast exists

Many acceleration extensions count model calls and treat each one as a sampling step. That assumption breaks with multi-stage solvers, where a single solver step evaluates the model several times:

```text
solver step N
  k1
  k2
  k3
  k4
```

If the raw model-call count is used as the time axis, the sampler is still partway through step N while the prediction state believes four steps have already gone by. The prediction is then asked about a future that does not exist yet.

ChebyCast separates two things that other implementations conflate:

- **the decision** of whether a solver step is real or predicted, which is made once per solver step
- **the position on the time axis**, which each individual stage gets for itself

So every stage inside one fixed-step RK or TDE solver step shares a single real-or-predicted decision, while still being fitted at its own place on the curve.

### Time coordinate

`auto` looks for a usable axis in this order:

```text
schedule -> step -> timestep -> sigma
```

The point of the ordering is to avoid ever falling back on the raw model-call count.

### Fit points

**all stages** lets every real intermediate stage update the fit. This gives more samples to work with, but those samples sit at uneven positions along the axis.

**step head only** adds just the first real update of each solver step. Fewer samples, more conservative.

---

## Main differences from Spectrum

ChebyCast takes Spectrum as its starting point and deliberately departs from it in several places.

| Area | ChebyCast |
| --- | --- |
| Execution unit | Real-or-predicted decisions are cached per solver step, not per model call |
| Time axis | Never the raw model-call count; `auto` prefers schedule, step, timestep, sigma |
| Local extrapolation | Newton divided differences, which stay meaningful when the sample positions are unevenly spaced |
| Output buffers | Model outputs are flattened and restored afterwards, so no fixed latent shape is assumed |
| Safety | An independent check for non-finite values and a clamp on predicted output |

The safety check is a ChebyCast implementation detail. It is not a parameter from the Spectrum paper.

---

## Defaults

ChebyCast does **not** aim to reproduce Spectrum's default settings.

| Parameter | ChebyCast default |
| --- | ---: |
| Blend weight (w) | 0.40 |
| Chebyshev bases (m) | 4 |
| Ridge regularization (lam) | 1.00 |
| Window size | 2 |
| Window growth (flex) | 0.00 |
| History points (K) | 16 |
| Warmup steps | 4 |
| Stop forecasting offset | 3 |

The official Spectrum README currently documents `w = 1.0`, `lam = 0.1` and `m = 4`, and describes a post-publication mixture with linear interpolation.

The two `w` values are **not interchangeable**. Spectrum's `w` mixes the Chebyshev prediction with a linear interpolation; ChebyCast's `w` mixes it with the Newton divided-difference extrapolator described above. The same number will not mean the same thing in both.

---

## Manual stage grouping

If the WebUI sampling-step counter does not expose the step boundary you need, you can state it yourself. For a fixed four-stage method:

```text
Stage grouping = 4
```

This is a workaround for fixed-step methods only. It is **not** a way to support adaptive ODE solvers. Adaptive methods vary their stage count, reject and retry steps, and make extra evaluations for error estimation. No single grouping number can reconstruct those boundaries.

---

## Compatibility with other extensions

ChebyCast can keep an existing `model_function_wrapper` alive on **real model calls** by chaining it inside its own wrapper.

On a **predicted call** the model evaluation is replaced by a prediction, so the inner wrapper does not run at all.

Any extension that needs its own wrapper to run on every single denoiser call therefore has to be checked individually before using it together with ChebyCast.

---

## Debug output

Set the shared debug variable in PowerShell before launching the WebUI:

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

Level 1 reports which time-coordinate source was selected and prints a summary at the end of the run.

```powershell
$env:SD_WEBUI_SETI_DEBUG = "2"
```

Level 2 additionally reports the real-or-predicted decision for each step.

Output goes to both the module logger and stderr, because some backends suppress module-level logger output.

---

## Validation guidance

For an ON / OFF comparison, hold all of these fixed: model, prompt and negative prompt, seed, sampler, scheduler, steps, CFG and resolution.

Then check:

- generation completes without an error
- no NaN or Inf appears in the log
- no severe colour corruption
- the latent does not diverge
- the composition does not collapse
- the number of real model calls actually went down
- generation time actually improved

As noted in [Getting started](#getting-started), pixel-identical output is not the goal and should not be used as the pass criterion.

---

## Compatibility status

| Target | Status |
| --- | --- |
| reForge | Implementation target; real-machine validation required |
| Forge Classic / Forge | Implementation target; real-machine validation required |
| Forge Neo | Implementation target; real-machine validation required |
| SDXL | Implementation target |
| Anima / NextDiT | Architecture-compatible; validation pending |
| Fixed-grid single-stage samplers | Intended |
| Fixed-step multi-stage TDE / RK samplers | Intended |
| Adaptive ODE solvers | Not supported, untested |
| A1111 | Not supported |

ChebyCast is an early implementation. The status above will be updated as real-machine testing is completed.

---

# 日本語

**[English](#sd-webui-chebycast)** | 日本語

ChebyCast は、画像生成を高速化する拡張機能です。

生成中はノイズ除去モデル（UNet）が何度も呼び出されますが、その出力はステップごとに滑らかに変化していきます。ChebyCast はその変化を記録し、直近の出力から曲線をあてはめて「次に出てくるはずの値」を計算で予測します。予測したステップではモデルを実行しないため、その分だけ生成が速くなります。

他の同種の拡張機能との最大の違いは、**1 ステップの内部でモデルを複数回呼び出すサンプラー**（TDE Sampler や RK Sampler など）に正しく対応している点です。多くの実装は「モデル呼び出し 1 回 = サンプリング 1 ステップ」を前提としていますが、これらのサンプラーではその前提が成り立ちません。

ChebyCast は **Spectrum**（*Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*、Han ほか、CVPR 2026）から着想を得ていますが、公式実装の移植ではありません。中心となる考え方を引き継いだうえで、Forge 上でどう動かすかを作り直しています。

---

## 動作条件

ChebyCast は Forge の `forge_objects` バックエンドを必要とします。対象は次のとおりです。

- reForge
- Stable Diffusion WebUI Forge / Forge Classic
- Forge Neo

A1111 は**非対応**です。

対象モデルは SDXL 系、対象サンプラーはステップ幅が固定されたものです。ステップ幅を自動調整するサンプラー（adaptive ODE ソルバー）は**非対応・未検証**です。詳細は[対応状況](#対応状況)を参照してください。

---

## インストール

**Extensions -> Install from URL:**

```text
https://github.com/seti9585/sd-webui-ChebyCast
```

インストール後、WebUI を**再起動**してください。UI のリロードだけでは反映されません。

---

## まず試す設定

最初は既定値のまま有効にして生成してみてください。既定値は安全側に寄せてあり、生成の最初の数ステップと最後の数ステップでは必ず実際のモデルを実行し、途中の区間だけ予測に置き換えるようになっています。

TDE / RK Sampler で `kutta4` のような多段の解法を使っている場合も、最初は **Stage grouping** を `auto` のままにしてください。デバッグ出力を見てステップの区切りが正しく検出されていないと分かったときだけ、数値を指定します。

比較の前に、次の 2 点を把握しておいてください。

- **同じシードでも、ChebyCast を切ったときとまったく同じ画像にはなりません。** 予測したステップが生成の進み方をわずかにずらし、そのずれが最後まで残るためです。構図・配色・全体の造形は保たれますが、細部の描かれ方は変わります。これは不具合ではなく、この手法の性質です。
- 絵が破綻する場合は、まず **Warmup steps** と **Stop forecasting offset** を増やし、**Window size** を下げてください。この 3 つが「生成のどれだけを予測に任せるか」を決めています。

---

## パラメータ

UI 上の項目名は英語のままです。以下は各項目が何をするかの説明です。

### よく調整するもの

| 項目 | 内容 |
| --- | --- |
| **Window size** | 実際にモデルを実行する間隔です。大きくするほど省略が増えて速くなりますが、その分だけ遠い先を予測することになります。 |
| **Warmup steps** | 冒頭で必ず実際のモデルを実行するステップ数です。予測の土台となる実測値が必要なため、ゼロにはできません。 |
| **Stop forecasting offset** | 終盤で必ず実際のモデルを実行するステップ数です。細部は終盤のステップで決まるため、ここを実測のまま残すことが画質の保護になります。 |
| **Blend weight (w)** | 予測値のうち、チェビシェフによるあてはめと、単純な近傍からの外挿を、どの比率で混ぜるかです。`0` で外挿のみ、`1` であてはめのみになります。 |
| **Apply to hires pass** | hires 側にも ChebyCast を適用するかどうかです。適用する場合、hires 側は予測の状態をゼロから始めます。 |

### 通常は触らないもの

| 項目 | 内容 |
| --- | --- |
| **Chebyshev bases (m)** | あてはめに使う曲線の成分の数です。多いほど複雑な形に追従できますが、安定させるにはより多くの実測値が必要になります。 |
| **Ridge regularization (lam)** | あてはめた曲線が大きく振れないように抑える強さです。大きいほど滑らかで保守的な曲線になります。 |
| **Window growth (flex)** | 生成が進むにつれて、実際にモデルを実行する間隔を広げていきます。`0` なら間隔は一定のままです。 |
| **History points (K)** | あてはめに使う過去の実測値を、何点まで保持するかです。 |
| **Stage grouping** | `auto` は WebUI のステップカウンタからステップの区切りを判定します。数値を指定した場合は、その回数のモデル呼び出しを 1 ステップとしてまとめます。 |
| **Fit points** | 実際に計算した中間段階をすべてあてはめに使うか、各ステップの最初の 1 点だけを使うかを選びます。 |
| **Time coordinate** | あてはめの横軸に何を使うかです。`auto` は schedule、step、timestep、sigma の順に使えるものを探します。 |

### `m` について

公式 Spectrum の README では `algo.m` は**チェビシェフ基底の個数**と定義され、既定値は `4` です。ChebyCast も同じ意味で使っています。`m = 4` なら `T0`、`T1`、`T2`、`T3` の 4 成分を使います。内部では `m - 1` の形で保持していますが、UI の操作上それを意識する必要はありません。

---

## 仕組み

### 基本的な考え方

生成中のモデル出力は、全体として滑らかな曲線のようにふるまいます。曲線上の点がいくつか分かっていれば、そこに数式をあてはめて、まだ実際には計算していない位置の値を読み取ることができます。

ChebyCast はこのあてはめに**チェビシェフ多項式**を使います。チェビシェフ多項式には、近似の誤差が区間の一方の端に集中せず全体に均等に散らばるという性質があります。1 ステップ先だけでなくある程度先まで予測したい場合に、この性質が効いてきます。

あてはめの計算自体は、値が大きく振れないように抑える項を加えた最小二乗法（リッジ回帰）です。**Ridge regularization (lam)** はこの抑制項の強さにあたります。

### 多段サンプラーへの対応 — この拡張機能を作った理由

多くの高速化拡張機能は、モデルの呼び出し回数をそのまま数えて、1 回を 1 ステップとして扱います。しかし多段の解法では、1 ステップの内部でモデルを複数回評価します。

```text
solver step N
  k1
  k2
  k3
  k4
```

呼び出し回数をそのまま予測の時間軸に使うと、サンプラーはまだ N ステップ目の途中にいるのに、予測側は 4 ステップ進んだと認識してしまいます。結果として、まだ存在しない未来について予測を求めることになります。

ChebyCast は、他の実装がひとまとめにしている次の 2 つを分離しています。

- **実測にするか予測にするかの判断** — ソルバーのステップごとに 1 回だけ行う
- **時間軸上の位置** — 各段階がそれぞれ自分の位置を持つ

これにより、同じステップに属する各段階は同一の判断を共有しつつ、あてはめには自分の正しい位置を使えます。

### Time coordinate（横軸の選択）

`auto` は次の順で使える軸を探します。

```text
schedule -> step -> timestep -> sigma
```

この順序の目的は、モデル呼び出し回数を軸として使う事態を避けることにあります。

### Fit points（あてはめに使う点）

**all stages** は、実際に計算した中間段階をすべてあてはめに使います。点数は増えますが、それらの点は軸の上に不均等に並びます。

**step head only** は、各ステップで最初に実測した 1 点だけを使います。点数は減りますが、より保守的です。

---

## Spectrum との主な違い

ChebyCast は Spectrum を出発点としつつ、いくつかの点を意図的に変更しています。

| 箇所 | ChebyCast |
| --- | --- |
| 判断の単位 | 実測か予測かの判断を、モデル呼び出しごとではなくソルバーのステップごとに保持 |
| 時間軸 | モデル呼び出し回数は使わない。`auto` は schedule、step、timestep、sigma の順 |
| 近傍からの外挿 | ニュートンの差分商を使用。点の間隔が不均等でも意味を保つ |
| 出力の保持 | モデル出力を平坦化して保持し、予測後に元の形に戻す。特定の次元数を前提にしない |
| 安全策 | 予測値に対する有限値チェックと上下限の制限 |

安全策は ChebyCast 独自の実装であり、Spectrum 論文のパラメータではありません。

---

## 既定値

ChebyCast は Spectrum の既定値を再現することを目的としていません。

| 項目 | ChebyCast の既定値 |
| --- | ---: |
| Blend weight (w) | 0.40 |
| Chebyshev bases (m) | 4 |
| Ridge regularization (lam) | 1.00 |
| Window size | 2 |
| Window growth (flex) | 0.00 |
| History points (K) | 16 |
| Warmup steps | 4 |
| Stop forecasting offset | 3 |

公式 Spectrum の README では、現在 `w = 1.0`、`lam = 0.1`、`m = 4` が既定値として記載されており、論文公開後の補足として線形補間との混合が説明されています。

両者の `w` は**同じ意味の数値ではありません**。Spectrum の `w` はチェビシェフの予測と線形補間を混ぜますが、ChebyCast の `w` は前述のニュートン差分商による外挿と混ぜます。同じ数値を入れても同じ結果にはなりません。

---

## Stage grouping を手動指定する場合

WebUI のステップカウンタからは望むステップ境界が取れない場合に備えて、境界を自分で指定できます。固定 4 段の解法であれば次のようになります。

```text
Stage grouping = 4
```

これはステップ幅が固定された解法に対する回避策であり、**ステップ幅を自動調整する解法への対応策ではありません**。自動調整型の解法では段数が可変で、ステップの棄却と再試行、誤差評価のための追加評価などが発生します。固定の数値ではこれらの境界を再現できません。

---

## 他の拡張機能との併用

ChebyCast は、既存の `model_function_wrapper` がある場合、**実際にモデルを実行するとき**にはそれを自分の内側で呼び出して維持できます。

しかし**予測に置き換えたとき**はモデルの評価そのものを行わないため、内側の `model_function_wrapper` は実行されません。

したがって、すべてのモデル呼び出しで自身の処理が走ることを必要とする拡張機能については、併用可否を個別に確認する必要があります。

---

## デバッグ出力

WebUI を起動する前に、PowerShell で共通のデバッグ変数を設定します。

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

レベル 1 では、選択された横軸の種類と、生成終了時のまとめが出力されます。

```powershell
$env:SD_WEBUI_SETI_DEBUG = "2"
```

レベル 2 では、これに加えて各ステップを実測にしたか予測にしたかが出力されます。

出力はモジュールロガーと stderr の両方に送られます。一部のバックエンドがモジュールレベルのロガー出力を抑制するためです。

---

## 検証方法

有効・無効を比較するときは、モデル、プロンプトとネガティブプロンプト、シード、サンプラー、スケジューラ、ステップ数、CFG、解像度をすべて固定してください。

確認する項目は次のとおりです。

- エラーなく生成が完了すること
- ログに NaN や Inf が出ないこと
- 深刻な色崩れが起きていないこと
- latent が発散していないこと
- 構図が破綻していないこと
- 実際のモデル呼び出し回数が減っていること
- 生成時間が実際に短縮されていること

[まず試す設定](#まず試す設定)に書いたとおり、ピクセル単位で一致することは目的ではなく、合否の基準にもなりません。

---

## 対応状況

| 対象 | 現在の状態 |
| --- | --- |
| reForge | 実装対象・実機検証が必要 |
| Forge Classic / Forge | 実装対象・実機検証が必要 |
| Forge Neo | 実装対象・実機検証が必要 |
| SDXL | 実装対象 |
| Anima / NextDiT | 構造上は対応可能・実機検証待ち |
| ステップ幅固定の単段サンプラー | 対応予定 |
| ステップ幅固定の多段 TDE / RK サンプラー | 対応予定 |
| ステップ幅可変（adaptive ODE）ソルバー | 非対応・未検証 |
| A1111 | 非対応 |

ChebyCast は初期実装です。上記の状態は、実機での検証が完了しだい更新します。

---

# License / Acknowledgements / References

## License / ライセンス

ChebyCast is released under the MIT License. See [`LICENSE`](LICENSE). Third-party attribution is recorded in [`NOTICE`](NOTICE).

本拡張機能は MIT License で公開しています。全文は [`LICENSE`](LICENSE) を参照してください。第三者の著作物に関する表示は [`NOTICE`](NOTICE) に記載しています。

## Acknowledgements / 謝辞

**Paper and official implementation / 論文および公式実装**

Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon
*Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*
CVPR 2026 / [arXiv:2603.01623](https://arxiv.org/abs/2603.01623)
Official implementation: [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)

ChebyCast is built on the idea presented in this paper. The official Spectrum repository is MIT licensed and credits TaylorSeer as an inspiration for part of its codebase.

ChebyCast は上記論文で示された考え方をもとにしています。公式リポジトリは MIT License で公開されており、コードの一部について TaylorSeer を着想元として挙げています。

**Scope of this implementation / 本実装の位置づけ**

ChebyCast was written as a separate, Forge-oriented implementation. It should not be described as an official Spectrum port, nor as a faithful reproduction of the official execution behaviour.

ChebyCast は Forge 向けに独自に書き起こした実装です。Spectrum の公式移植ではなく、公式実装の挙動を忠実に再現したものでもありません。

**Reference implementations / 参考実装**

The following community implementations were consulted for practical WebUI integration. ChebyCast does not derive its code from them.

WebUI への組み込み方を理解するにあたり、以下のコミュニティ実装を参考にさせていただきました。コードを流用したものではありません。

- [hirorohi03/sd-webui-forge-spectrum](https://github.com/hirorohi03/sd-webui-forge-spectrum)
- [hirorohi03/sd-forge-spectrum-faithful](https://github.com/hirorohi03/sd-forge-spectrum-faithful)
- [ruwwww/comfyui-spectrum-sdxl](https://github.com/ruwwww/comfyui-spectrum-sdxl)
- [judian17/ComfyUI-Spectrum](https://github.com/judian17/ComfyUI-Spectrum)

## References / 典拠

- Spectrum paper / 論文: [arXiv:2603.01623](https://arxiv.org/abs/2603.01623)
- Spectrum official repository / 公式リポジトリ: [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)
- TaylorSeer: [Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)
