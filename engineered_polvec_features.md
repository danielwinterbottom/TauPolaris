# Engineered visible-side features for polarimetric-vector regression

Status: **implemented, off by default.** Set `Data.engineered_polvec_features` in the
training config:

| value | meaning |
|---|---|
| `0` / `false` / absent | off — no columns added, nothing changes |
| `1` / `true` | the hadronic current for the leg's own reconstructed decay mode |
| `2` | level 1, plus the current under the alternative "the reconstructed π⁰ is spurious" hypothesis |

Existing prepared data and trained models are unaffected at level 0.

---

## Why

The direct polarimetric-vector model is well behind the neutrino-regression method
for 3-prong decays. Measured on the July29 model, mean Pearson *r* against the true
`ts_hh` per leg:

| leg DM | direct method | nu-method |
|---|---|---|
| 1 | 0.692 | 0.689 |
| 2 | 0.508 | 0.497 |
| **10** | **0.622** | **0.734** |
| **11** | **0.245** | **0.418** |

1-prong is at parity; the deficit is entirely 3-prong. The nu-method's advantage there
comes from a hard-coded piece of physics the direct model has to learn from data: the
exact per-decay-mode polarimetric formula, including the hadronic resonance structure.

These features hand that structure to the network directly.

## The factorisation this rests on

Every polarimetric-vector formula in the codebase — `polarimetric_vec_dm1`,
`PolarimetricA1_vectorised.PVC`, `Polarimetric3hpi0_vectorised.PVC` — splits cleanly:

```
h  =  f( hadronic current , visible 4-vector , tau 4-vector )
      \______ from the decay products alone ______/   \_ unknown _/
```

The neutrino enters only as `N = P_tau − visible`, and only *after* the current is
built. So the **hadronic current is the complete visible half** of the calculation:
everything about the hadronic system that `h` can possibly depend on, and nothing that
requires the tau.

That is what these features expose. The network is then left with the part that
genuinely needs the neutrino — contracting the current with `N` — instead of also
having to discover the resonance dynamics.

## What the current is, per decay mode

### DM1 / DM2 — tau → ρ ν, ρ → π π⁰

```
q = π − π⁰
h ∝ [ 2(q·N)q − q²N ] / [ 2(q·N)(q·P) − q²(N·P) ]
```

The visible system enters only through **q**. A single amplitude, so the current is
**real** — which is why `h` lies in the decay plane and the classic decay-plane method
works for these modes.

### DM10 — tau → a₁ ν, a₁ → π⁻π⁻π⁺

From `PolarimetricA1_vectorised.hadronic_current()`. With `q1, q2` the same-sign pions
and `q3` the opposite-sign pion:

```
a1   = q1 + q2 + q3
s1   = (q2+q3)²      opposite-sign/same-sign pair  →  ρ⁰ candidate
s2   = (q1+q3)²      opposite-sign/same-sign pair  →  ρ⁰ candidate
s3   = (q1+q2)²      same-sign pair                →  no resonance
vecᵢ = pairwise momentum differences, projected transverse to a1
Fᵢ   = F3PI(m²(a1), sᵢ, sⱼ)        complex ρ/ρ′/ρ″ Breit-Wigner form factors
HADCUR = vec1·F1 + vec2·F2 + vec3·F3
```

`vec1, vec2, vec3` are the direct analogue of DM1's `q = π − π⁰`, three times over.
In the a₁ rest frame the three pions are coplanar (verified: out-of-plane component
1.8e-10 GeV), so the current is a **complex vector in the decay plane**.

The Dalitz interpretation is confirmed on 31k reco DM10 legs — fraction of pairs within
150 MeV of m_ρ = 0.775:

| pair | median mass | near m_ρ |
|---|---|---|
| s1 (OS–SS) | 0.651 | 52% |
| s2 (OS–SS) | 0.731 | 66% |
| s3 (SS–SS) | 0.544 | 30% |
| m(a₁) | 1.112 | — |

### DM11 — tau → 3h + π⁰

Identical in structure, using the TAUOLA CLEO 4-pion current
(`Polarimetric3hpi0_vectorised.hadronic_current()`) in place of F3PI.

### DM0

No hadronic current exists — `h` is just the pion direction, so there is nothing to
pre-compute. 10 of the 11 columns are exactly zero; the eleventh, `hcur_m2vis`, is the
reco pion's own mass² (scatter around m²_π from mis-measurement, std 0.025 GeV²) and
carries no polarimetric information — it is also already derivable from the pion
four-vector the network receives.

**The DM0 token must therefore be masked**, exactly as the `pi2`/`pi3`/`pizero` tokens
already are when those particles are absent:

```python
pad_mask[:, hcur_idx] = ~(is3prong.bool() | haspizero.bool())   # no current for DM0
```

With the token masked, the block's contents for DM0 are irrelevant, so `hcur_m2vis` is
left populated rather than special-cased — it is a well-defined visible mass for every
decay mode, just not a useful one here.

## Why 3-prong is the hard case

For DM1 the current is real, so `h` sits in the decay plane and is largely determined
by visible quantities. For DM10/DM11 the interfering ρπ amplitudes carry different
Breit-Wigner phases, so the current has a genuine imaginary part and `h` picks up a
term set by the relative phase — accessible only by contracting with the neutrino.

Measured on reco events, |Im| / |Re| of the current:

| DM | \|Im\|/\|Re\| |
|---|---|
| 1, 2 | 0.00 (exactly real) |
| 10 | 1.01 |
| 11 | 1.27 |

That is the quantitative statement of why the visible-only decay-plane picture works
for ρ and fails for a₁.

## Level 2 — decay-mode misidentification

A leg reconstructed as DM11 may really be a DM10 with a spurious π⁰. Level 2 recomputes
the whole block with the π⁰ dropped, giving the network both hypotheses in a
`hcur_alt_` block.

**This is only computable in one direction.** Dropping a reconstructed π⁰ is easy;
*adding* one is not, because a leg reconstructed without a π⁰ has no π⁰ four-vector
stored (it is exactly zero). So DM11 → DM10 is available — and that is the case that
matters — while DM10 → DM11 is not.

Whether it is worth enabling depends entirely on the sample:

| reconstructed as DM11, what is it really? | Delphes (these samples) | full CMS (MVA DM) |
|---|---|---|
| truly DM11 | 98.1% | ~0.70 |
| truly DM10 | **1.1%** | **~0.14** |
| other | 0.8% | ~0.14 |

In the current Delphes samples level 2 does work for roughly one leg in a hundred and
is not worth the 11 extra columns per leg. In full CMS simulation the MVA decay-mode
purity for 3π±π⁰ is ~0.70 with ~0.14 leaking in from 3π±, so the alternative hypothesis
becomes a real handle. **Kept off by default until the switch to full CMS samples.**

Behaviour by decay mode (verified on 60k events):

| leg | alt block |
|---|---|
| DM11 | DM10 current from the 3 charged pions — non-zero for 100%, and genuinely different from the main block (max abs difference 1.75, so not degenerate) |
| DM10 | zero — no π⁰ to drop |
| DM1 / DM2 | zero — dropping the π⁰ gives DM0, which has no hadronic current |
| DM0 | zero |

The `hcur_alt_m2vis` column is the useful discriminator on its own: on reco-DM11 legs
its median is 0.869 GeV (the 3 charged pions) against 1.294 GeV for the full visible
system. A misidentified DM10 sits near m_a₁; a genuine DM11 does not.

## Columns added

Per tau leg, with the reconstruction prefix in use (`reco_taup_`, `reco_taun_`, or the
gen equivalents). Level 2 adds the same block again under `hcur_alt_`, so 11 columns
per leg at level 1 and 22 at level 2:

| column | count | content |
|---|---|---|
| `{leg}_hcur_re_{n,r,k}` | 3 | real part of the current, unit-normalised |
| `{leg}_hcur_im_{n,r,k}` | 3 | imaginary part, same normalisation |
| `{leg}_hcur_s1`, `_s2`, `_s3` | 3 | pair invariant masses squared |
| `{leg}_hcur_m2vis` | 1 | visible mass squared |

**10 numbers per leg.** Only the current's *direction* is passed — see below.

### Why the magnitude is not a feature

`CLVEC`/`CLAXI` are bilinear in (H, H\*) and `h` is a ratio of two such bilinears, so

```
h(lambda * H) = h(H)    for any complex lambda
```

exactly. Verified numerically: max |h(lambda H) − h(H)| = 2.8e-16 for lambda = 1000,
1e-4, 3e^{0.7i} and 0.01e^{-2.1i}. The magnitude carries no information about the
polarimetric vector at all.

It is redundant a second way too. In the a1 rest frame the current is purely spatial
(|H_t|/|H| < 1.5e-15, transversality) and its magnitude is exactly rotation-invariant
(1.7e-14 under a rigid rotation of all three pions), while 3-body kinematics in that
frame is fixed by the Dalitz variables up to rotation. So |H| is a function of
`m2vis, s1, s2`, which are already passed. A nearest-neighbour test confirms it: the
median relative |H| difference between events falls monotonically as they approach each
other in (m2vis, s1, s2) — 0.20, 0.084, 0.057, 0.025.

**A caveat on what remains.** The same bilinearity means the overall *phase* of the
current is also unphysical, so the six re/im components still carry one redundant degree
of freedom. That is left in deliberately: removing it by rotating to the principal axes
(R·I = 0) needs a discrete sign convention, which would introduce exactly the kind of
artificial seam that is worth more than the wasted dof. A redundant *input* direction is
mild — the network can learn to ignore it — unlike a redundant *output* direction, which
is the measure-zero `onorm` problem and is fatal.

### What each piece is actually for

`h = f(H, a1, tau)`, so given the current and the pion momenta the network already has,
**nothing else affects h**:

| feature | affects h? | why it is included |
|---|---|---|
| `re/im` | **yes** — the entire h-dependence | the Breit-Wigner form factors, genuinely hard to learn |
| `s1, s2, s3` | no — upstream of H, already consumed by the form factors | Dalitz position; plausibly useful for the tau-momentum output, and tells a density model where the form factors vary fast, i.e. how wide the conditional on H should be |
| `m2vis` | no | constrains the tau kinematics (theta_GJ^max depends on m_vis) |

Only `re/im` carries content that is hard for the network to reconstruct itself. The
scalars are cheap conveniences it could form from sums and dot products.

The current is computed in the **visible system's rest frame**, which makes it tau-free.
The form factors are Lorentz scalars so the frame choice does not affect them. The
`re`/`im` components are then projected onto the same per-tau visible-momentum
`(n,r,k)` basis as every other vector quantity, so they arrive in the coordinates the
model is trained in.

Note this reproduces the codebase's existing convention where the *vector* lives in one
frame and the *basis* is built from lab-frame visible momenta — the same mismatch
`ts_hh_*` already has. That is deliberate, so the features line up with the target.

## What this is and is not

**It adds no information.** Every one of these columns is a deterministic function of
the pion 4-momenta the network already receives. Nothing new enters.

Its value is purely as an **inductive bias**: pre-computing a function that is hard to
learn. `F3PI` is ρ/ρ′/ρ″ Breit-Wigners with relative phases evaluated across the Dalitz
plane, and the CLEO 4-pion current is worse. A 4-layer transformer feeding a
normalising flow is unlikely to reconstruct that from data.

Whether it helps is therefore an optimisation question, not an information one, and it
can only be settled by training. There is one independent reason to expect it might:
the direct model spends ~71% of its likelihood budget on the degenerate unit-norm
constraint in `onorm` coordinates, so it is capacity-limited — exactly the regime where
a pre-computed hard function should pay off.

## Validation

- `PVC()` output is **bit-identical** before and after the refactor that exposed
  `hadronic_current()` (checked to 10 d.p. on both DM10 and DM11).
- No NaN or inf in any feature across 100k events, all decay modes.
- Sanity: the current's real direction is aligned with the true polvec well above
  chance — ⟨|cos|⟩ = 0.64 (DM1), 0.61 (DM2), 0.61 (DM10), 0.53 (DM11), against 0.5 for
  no alignment.
- Edge case guarded: a high-momentum single pion can come out marginally spacelike from
  float cancellation in E²−p² (~0.06% of DM0 legs). Boosting to a spacelike vector's
  rest frame gives NaN, so those rows have their boost neutralised and their current
  zeroed.
- All three levels produce only finite values across 60k events; the level resolver
  maps `false/0 → 0`, `true/1 → 1`, `2 → 2`, absent → 0.
- **Level 0 is bit-compatible with existing checkpoints**: the July29 model loads with
  `strict=True`, `type_emb` keeps 13 rows and no `hcur_proj` is created.
- Levels 1 and 2 build and run a forward pass to a finite context vector (15 and 17
  `type_emb` rows, 22 and 44 extra input columns).
- Masking verified on a forward pass with one leg per decay mode: `hcur_` masked only
  for DM0, `hcur_alt_` masked for everything except DM11.

An aside worth recording: DM11's poor reconstruction is **not** a misidentification
effect. At 98.1% purity in these samples, r = 0.245 reflects the mode being
intrinsically hard — the CLEO 4-pion current has the most interfering amplitudes, so h
depends most strongly on the neutrino direction, which is the least well known input.

## Enabling it

```yaml
Data:
  engineered_polvec_features: 1     # or 2, once on full CMS samples
```

Data must be re-prepared (`prepare_inputs.py`) — the columns are written at prep time
in `DataProcessing._process_chunk`.

## Model side

Wired in, under the same option. `ParticleTransformerCondition` takes
`polvec_feature_level` and appends one extra token per hadronic leg per block, after the
mode's own tokens:

| level | extra tokens (leptonic_mode 0) | `type_emb` rows |
|---|---|---|
| 0 | 0 | 13 (unchanged) |
| 1 | 2 (`hcur_` for taup, taun) | 15 |
| 2 | 4 (+ `hcur_alt_`) | 17 |

For `leptonic_mode: 1` only the `tau2` leg gets tokens — `tau1` is leptonic by
construction and has no hadronic current, so `type_emb` goes 9 → 10 at level 1 and
9 → 11 at level 2.

### Semileptonic

Works without special-casing, but the naming shifts under you, so it is worth knowing
how. Prep writes `taup`/`taun`, because `convert_semileptonic_df` runs later — at the
train/val/test split, not at prep. That function renames by substring anywhere in the
column name, so `reco_taup_hcur_re_n` becomes `reco_tau1_hcur_re_n` along with
everything else, with tau1 always the leptonic leg and tau2 always the hadronic one.

The leptonic leg needs no special handling at prep time either: its pion columns are
zero, so the visible four-vector is not timelike and the guard zeroes the block. The
model never builds a token for it in any case.

To enable it, add the 11 (or 22 at level 2) `reco_tau2_hcur_*` names to
`input_features` in the semileptonic config.

Note `convert_semileptonic_df` shuffles rows before returning
(`sample(frac=1, random_state=42)`), so positional indexing across that call is
meaningless — carry an event id if you need to check row-wise behaviour.

The tokens are masked per decay mode, which matters as much as the values:

| leg | `hcur_` token | `hcur_alt_` token |
|---|---|---|
| DM0 | **masked** | masked |
| DM1 / DM2 | present | masked |
| DM10 | present | masked |
| DM11 | present | **present** |

An unmasked all-zero token is not neutral — the transformer would attend to it as a real
particle at the origin. Verified directly on a forward pass (see Validation).

A config asking for a level whose columns are not in `input_features` raises a clear
`ValueError` at model construction rather than silently ignoring the request.

`polvec_feature_level` is threaded through `load_model`, `setup_model_and_training`,
`train.py`, `evaluate.py` and `evaluate_polvec.py`, read from the config in each case, so
training and evaluation cannot disagree about the architecture.

The MSE `TransformerRegressor` baseline (`--useTransformerBaseline`) has its own
tokenisation and is **not** wired up — the option has no effect there.

## Code

| what | where |
|---|---|
| current + invariants, DM10 | `PolarimetricA1_vectorised.hadronic_current()` |
| current + invariants, DM11 | `Polarimetric3hpi0_vectorised.hadronic_current()` |
| feature construction | `acoplanarity_tools.hadronic_current_features()` |
| prep-time wiring | `DataProcessing._add_hadronic_current_features()` |
