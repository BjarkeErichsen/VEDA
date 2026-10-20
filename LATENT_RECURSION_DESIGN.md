# Latent recursion in VEDA — review of `7f51f5e` and a mechanism/training design

Scope: review of the `added latent recursion` commit against the Semla backbone in this
repo, and a concrete design for the conditioning mechanism and the training routine.
Written against `models/semla.py`, `models/fm.py`, `train.py` at `7f51f5e`.

Note: `docs` is listed in `.gitignore`, so the design docs referenced in the request are
not in this checkout and were not available for this review. Everything below is derived
from the code plus the thesis excerpt and the defence slides.

---

## 0. The one-paragraph summary

The commit is a correct, minimal **scalar-channel** latent recursion: the post-message-passing
node features are carried across denoising steps and injected residually at initialisation.
It is the VEDA analogue of the thesis's `s linear layer` mechanism, which the thesis found to
be *the weakest of the mechanisms in isolation*. The two channels the thesis identified as
carrying the effect — the **edge** channel (the largest single contributor to
self-conditioning, 90.9 of 92.8 points) and the **vector** channel — are both discarded here.
And unlike g3m, **Semla's edge representation is dynamic**, so the obstacle the thesis called
"Problem 1: the edge channel cannot be carried" does not exist in this architecture. That is
the single biggest opportunity in this port. There are also four concrete bugs, one of which
silently destroys the zero-initialisation that was put in for training stability.

---

## 1. Review of the commit

### 1.1 What is right

- `prev_node_latent` is plumbed cleanly through `MolecularCFM.forward` →
  `SemlaGenerator.forward`, and **all five** call sites of `self(...)` were updated to the
  new 5-tuple (`fm.py:786,812,820,826,1143`). `validation_step`/`predict_step` only go
  through `_generate`, so nothing was missed.
- The latent is taken **after** all message passing and after `feat_norm`
  (`semla.py:879`), which is the right analogue of the thesis's "node states after all
  message-passing layers have been applied".
- Injection is a **residual** add onto `atom_feats`, *after* `feat_proj` has already folded in
  the current `c_noise` time embedding (`semla.py:849-857`). This is exactly the point the
  thesis makes about not overwriting `s_t` and clobbering the timestep. Correct.
- `node_feats` in `EquiMessagePassingLayer` is a genuine residual stream
  (`semla.py:495,501`), so an injection at initialisation survives all 12 layers rather than
  being normalised away. The injection point is well chosen.
- The detach discipline in `_training_step` is right: `no_grad` for the K warm-up passes,
  explicit `.detach()` at the boundary, then `latrec_n` tracked passes, then the loss pass.
- `_generate` carries `node_latent` across integration steps (`fm.py:1143`), which is the
  actual inference-time mechanism, and the graph is index-aligned across steps because
  `edges_from_nodes` is called with `k=None` (fully connected) and atom ordering is fixed.
- The `map_location` and `.to(device)` changes in `evaluate.py`/`predict.py`/`scriptutil.py`
  are unrelated but good — they make the scripts CPU-runnable.
- `mask_rate_strategy` on the `eval_interpolant` (`train.py:299`) is a real consistency fix;
  validation was previously using `None` while training used `edm`.

### 1.2 Bugs

**B1 — `_init_params()` overwrites the zero-init of `latrec_mlp`. (confirmed, high impact)**

`SemlaGenerator.__init__` zero-initialises the last layer of `latrec_mlp`
(`semla.py:761-762`) so that the mechanism starts as an exact no-op. But
`MolecularCFM.__init__` ends with `self._init_params()` (`fm.py:710`), which is

```python
def _init_params(self):
    for p in self.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
```

`self.gen = gen` is set at `fm.py:581`, i.e. *before* line 710, so `latrec_mlp[-1].weight`
(dim 2) is re-initialised with xavier. Only the bias (dim 1) survives as zero. The
mechanism therefore starts by injecting a full-magnitude random projection of the carried
latent into the residual stream at step 0 of training. Given the defence note that
"latent recursion is harder to train — multiple runs collapsed and were restarted", this
is the first thing to fix.

Fix: tag the latent-recursion parameters and skip them in `_init_params`, then re-apply the
zero-init — or move the zero-init into a `reset_parameters()` called from `MolecularCFM`
*after* `_init_params()`. The gate-based design in §2.5 makes this robust by construction
(a scalar gate is `dim()==1`, so xavier never touches it).

**B2 — conditional parameter use breaks DDP with `latrec_transfer="linear"`. (high impact)**

`latrec_mlp` is used only when `prev_node_latent is not None`, i.e. only on the fraction
`latrec_p` of batches where the chain fires (`fm.py:808`). On a batch where the chain is
skipped, `latrec_mlp` receives no gradient. `L.Trainer` is constructed with default
`strategy="auto"` (`train.py:343`), so on any multi-GPU node this is DDP with
`find_unused_parameters=False`, which raises at the first skipped batch.

Fix: do not branch. Make the cold start a **zeros latent** rather than `None`, exactly as
self-conditioning already does with `_get_initial_cond_batch`. The transfer then always
runs, the graph is static (also required if you ever re-enable `torch.compile`), and the
dropout becomes a mask on the latent rather than a branch in the code. This is a
strictly better design, not just a DDP workaround — see §3.2.

**B3 — the mask multiply changes the no-recursion baseline. (low impact, breaks A/B parity)**

```python
atom_feats = atom_feats * atom_mask.unsqueeze(-1)   # semla.py:858
```

This runs unconditionally, outside the `if self.latent_recursion` guard. `feat_proj` has
biases, so before the commit padding atoms carried non-zero features into layer 0; now they
are zeroed. Padding atoms are excluded from attention by `adj_to_attn_mask`, so this is
probably harmless numerically — but it means a `--no_latent_recursion` run on this commit is
not bit-comparable with the pre-commit baseline, which matters when the whole point is an
ablation table. Move it inside the guard (or keep it and note the baseline was re-run).

**B4 — `latrec_k` is off by one relative to Algorithm 5. (cosmetic, but confounds the sweep)**

Thesis Algorithm 5 runs `for k = 1 to K-1` untracked passes, then `n` tracked, then the
final pass; so thesis `K` counts the *total* untracked-plus-final budget. Here the loop is
`for _ in range(self.latrec_k)` (`fm.py:811`), so `latrec_k=1, latrec_n=0` is thesis
`K=2, n=0`. Not wrong, but any number you quote against the thesis table will be shifted by
one. Either rename to `latrec_k_untracked` or subtract one in the loop.

### 1.3 Design gaps relative to the thesis and to this architecture

**G1 — the vector/coordinate channel is not carried.** Semla's analogue of `v` is the
`coords` tensor of shape `[B, n_coord_sets, N, 3]` that lives inside `EquiInvDynamics`
(`semla.py:650-669`). It is a residual stream like `node_feats`, and after
`final_ff_block` it is a perfectly good `v_{t-Δt}`. It is never returned, so it is thrown
away every step. `n_coord_sets=64` by default — this is a 64×3 per-atom equivariant state,
far wider than the single `cond_coords` set that self-conditioning carries.

Note the thesis's `v identity` baseline does **not** transplant: it worked because g3m
initialises `v_t` to zeros, so an identity copy was free. In Semla `coords` is initialised
from the *actual* noisy geometry via `coord_proj`, so an identity add would corrupt the
input. The correct analogue is a bias-free linear mix — see §2.3.

**G2 — the edge channel is not carried, and in Semla it can be.** This is the headline.
The thesis's "Problem 1: the edge channel cannot be carried" is a g3m property: edges there
are static pairwise distances built once. In Semla:

- `in_layer` consumes `edge_feats` (`d_edge_in=d_edge`, `semla.py:575-584`),
- the core layers drop edges entirely,
- `out_layer` *regenerates* them (`d_edge_out=d_edge`, `semla.py:585-594`),
- `BondRefine` refines them (`semla.py:673`), and `bond_norm` normalises them
  (`semla.py:680`).

So `pred_edges` inside `SemlaGenerator.forward` (`semla.py:874`) is already a learned,
normalised, dynamic `[B, N, N, d_edge=128]` edge representation. Carrying it across
denoising steps is a ~16k-parameter change. Compare what self-conditioning carries on that
route today: `cond_bonds`, the **softmax over ~5 bond types**. The latent route is a 128-dim
channel where self-conditioning has a 5-dim one, on the channel the ablation says matters
most. The defence's own "what comes next" slide names this as "the clearest piece of
unexploited headroom"; in VEDA it is also the cheapest.

This matters more in VEDA than in the thesis setting for a second reason: VEDA generates
types, bonds, charges *and* coordinates (masked-discrete + continuous), whereas the thesis
task was coordinates for a fixed formula. The bond channel is a generated output here, so a
wide edge latent carries committed discrete structure forward instead of re-deriving it.

**G3 — training never shows the model a *stale* latent.** In `_training_step` the latent is
produced by forward passes at the **same** `(interpolated, times)` that consume it. At
inference the latent consumed at `σ_t` was produced at `σ_{t-Δt} > σ_t` from a different
`x`. The thesis has the same mismatch and still got +3.5 points from BPTT, so this is not
fatal — but it is the most likely explanation for two of its observations: that recursion
is harder to train, and that multi-level inference (which *removes* the mismatch by
refining `z` at fixed `t`, matching training) was a **null result**. A model trained only
on same-`t` latents has no reason to benefit from extra same-`t` refinement at inference
beyond what it already does. §3.3 proposes fixing this at zero extra cost.

**G4 — the self-conditioning pass and the first recursion pass are not shared.** With
`self_condition=True` (the default, `train.py:484`) and `latent_recursion=True`,
`_training_step` runs `1` (self-cond) `+ latrec_k + latrec_n + 1` forward passes. With
defaults that is 3 passes where self-conditioning alone needs 2. But the self-conditioning
pass *already computes a node latent* and discards it (`fm.py:786`, the trailing `_`). Reuse
it and latent recursion at `K=1, n=0` costs exactly what self-conditioning costs. This is
worth having as the headline efficiency claim.

**G5 — no time-delta conditioning on the transfer.** The carried latent encodes the
*previous* step's `c_noise`. The model has no way to know how stale it is, which matters
because the inference schedule is non-uniform (`arcsin`, `max_sigma=80` down to
`min_sigma=1e-3`): the σ gap between consecutive steps varies by orders of magnitude along
the trajectory. Feeding `(c_noise(σ_t) − c_noise(σ_prev))` into the transfer is ~1k
parameters and pairs naturally with §3.3.

**G6 — Algorithm 9 (multi-level inference) is not implemented.** Given the thesis reports
it as a null result, this is the right call for now. Worth revisiting only *after* §3.3,
because the null result and the train/inference mismatch are plausibly the same fact.

### 1.4 Experiment-hygiene problems (these will cost you a sweep)

**H1 — every run overwrites the same checkpoint file.**

```python
checkpointing = ModelCheckpoint(
    dirpath="output/checkpoints",   # fixed, not per-run
    every_n_epochs=val_check_epochs,
    save_last=True,
    save_top_k=0,                   # no best-checkpoint tracking at all
)
```
(`train.py:328-333`)

Two problems. The path is a constant, so concurrent or sequential runs clobber each other's
`last.ckpt`. And `save_top_k=0` with no `monitor` means **no best checkpoint is ever
saved** — only the latest. The thesis selection protocol is "best validation checkpoint
inside the converged region is evaluated once on test"; that is not reproducible from
`last.ckpt`. For a mechanism sweep where several configs are expected to collapse and need
restarting from a pre-collapse checkpoint, this is the difference between a usable and an
unusable sweep.

Suggested:
```python
run_id = logger.experiment.id if logger is not None else "trial"
checkpointing = ModelCheckpoint(
    dirpath=f"output/checkpoints/{run_id}",
    monitor="val-molecule-stability",   # or val-validity, as before
    mode="max",
    save_top_k=3,
    save_last=True,
    every_n_epochs=val_check_epochs,
    filename="best-{epoch:02d}-{val-molecule-stability:.4f}",
)
```

**H2 — `latrec_n/k/p` are not surfaced at evaluation.** `evaluate.py` and `predict.py`
read `latent_recursion` and `latrec_transfer` but not the recursion depths. That is correct
today (inference carries one latent per step regardless), but it will silently break when
you add multi-level inference. Thread an explicit `--latrec_inference_steps` when you do.

**H3 — per-batch rather than per-sample dropout.** `torch.rand(1).item() < self.latrec_p`
(`fm.py:808`) makes one decision for the whole batch, so the effective number of
independent dropout draws per epoch is the number of *steps*, not the number of *samples*.
With `batch_cost`-based bucketing the batch size varies, so the effective dropout rate is
also weakly size-dependent. The pre-existing self-conditioning code has the same shape
(`fm.py:803`), so this is inherited rather than introduced — but it is cheap to fix and
reduces gradient variance. See §3.2.

---

## 2. Mechanism design

### 2.0 Channel inventory: g3m vs Semla

| thesis channel | g3m | Semla equivalent | shape | currently carried? |
|---|---|---|---|---|
| scalar `s` | node scalars | `pred_feats` after `feat_norm` | `[B, N, 384]` | **yes** |
| vector `v` | node vectors | `coords` after `final_ff_block` | `[B, 64, N, 3]` | no (G1) |
| edge `e` | static, cannot carry | `pred_edges` after `bond_norm` | `[B, N, N, 128]` | no (G2) |

All three are available at the same point in the forward pass. The design below carries all
three, each with a zero-initialised gate, so that any subset can be enabled by flag and the
ablation table from the thesis can be reproduced directly in VEDA.

### 2.1 Uniform injection form

Use the same shape for every channel: **normalise the carried latent, apply a transfer,
scale by a zero-initialised scalar gate, add residually at initialisation.**

```
h ← h + g_h · T_h( LN(z_h) ,  Δτ )
```

Properties this buys:
- **Cold start is exactly the baseline.** `g=0` at init, so step 0 of training is
  bit-identical to a non-recursive run. This removes B1's failure mode structurally: a
  scalar gate has `dim()==1` and `_init_params` only touches `dim()>1`.
- **The gate is a readable diagnostic.** Log `g_s, g_v, g_e` to wandb per step. The defence
  slide on learned temporal gates showed these are interpretable (the edge gate rising
  exactly where local geometry resolves); the same read is available here and it tells you
  within a few thousand steps whether a mechanism is being used at all, long before
  stability metrics move.
- **Scale is bounded.** Normalising `z` before injection is what keeps the carried state
  from drifting in magnitude over 100 integration steps. This is the main guard against the
  collapses the defence reported.

### 2.2 Scalar channel `s`

Keep the current injection point (`semla.py:849`), change the transfer:

```python
# in SemlaGenerator.__init__
self.latrec_s_norm = torch.nn.LayerNorm(d_model)
self.latrec_s_proj = torch.nn.Linear(d_model + d_tau, d_model)   # d_tau = time-delta embedding
self.latrec_s_gate = torch.nn.Parameter(torch.zeros(1))
```

Cost: `(384 + d_tau) × 384 + 384 ≈ 148k`.

Prefer this single linear map over the current `Linear(2d, d) → SiLU → Linear(d, d)` concat
MLP (443k params): the thesis's `s linear layer` is a single projection, the residual stream
already provides the mixing, and the concat form is three times the parameters for the
weakest channel. Make the MLP form an opt-in variant (`latrec_transfer="mlp"`) so you can
ablate it rather than paying for it by default.

### 2.3 Vector channel `v` — equivariance is the only constraint

`EquiInvDynamics.forward` builds the coord sets with a bias-free linear map over the set
index (`semla.py:649-650`):

```python
coords = torch.stack((coords, cond_coords)) if cond_coords is not None else coords.unsqueeze(0)
coords = self.coord_proj(coords.movedim(0, -1)).movedim(-1, 1)
```

A **bias-free linear combination over the set index** commutes with rotation and preserves
zero-CoM, so the equivariant carry is:

```python
# __init__
self.latrec_v_norm = CoordNorm(n_coord_sets, norm="length")
self.latrec_v_proj = torch.nn.Linear(n_coord_sets, n_coord_sets, bias=False)
self.latrec_v_gate = torch.nn.Parameter(torch.zeros(1))

# forward, right after coord_proj
if prev_coord_sets is not None:
    v = self.latrec_v_norm(prev_coord_sets, atom_mask)
    v = self.latrec_v_proj(v.transpose(1, -1)).transpose(1, -1)
    coords = coords + self.latrec_v_gate * v
```

Cost: `64 × 64 = 4,096` parameters.

Three things to get right:
1. **No bias, no non-linearity, no LayerNorm over the spatial axis.** `CoordNorm` is the
   only normalisation that is safe here — it scales by vector length, which is invariant.
   This is the same constraint the defence slide states for the self-conditioning vector
   channel ("a bias-free linear map with no non-linearity, which is what keeps it
   equivariant").
2. **Do not identity-add.** Unlike g3m, `coords` at this point *is* the geometry. The
   learned mix is what lets the model decide how much prior spatial state to admit.
3. **Zero-CoM is preserved.** `final_ff_block` returns `equivariant_mlp(coord_sets, ...)`,
   a bias-free linear combination of `CoordNorm`-normalised (hence zero-CoM) sets, so the
   carried `v` is zero-CoM; `CoordNorm` re-centres anyway.

To expose it, `EquiInvDynamics.forward` needs to return the post-`final_ff_block` `coords`
(`semla.py:666`) alongside its current outputs, and accept `prev_coord_sets`.

### 2.4 Edge channel `e` — the one to build first

`pred_edges` at `semla.py:874` is the `bond_norm`-normalised `[B, N, N, d_edge]`
representation. Carry it and inject residually after `edge_in_proj`:

```python
# __init__
self.latrec_e_norm = torch.nn.LayerNorm(d_edge)
self.latrec_e_proj = torch.nn.Linear(d_edge, d_edge)
self.latrec_e_gate = torch.nn.Parameter(torch.zeros(1))

# forward, after edge_feats = self.edge_in_proj(edge_feats)
if prev_edge_latent is not None:
    e = self.latrec_e_proj(self.latrec_e_norm(prev_edge_latent))
    edge_feats = edge_feats + self.latrec_e_gate * e
```

Cost: `128 × 128 + 128 = 16,512` parameters.

Why this is the highest-expected-value change:
- it is the channel the self-conditioning ablation says carries the effect (edge alone 90.9
  of 92.8);
- the thesis could only approximate it (`edge from v`, Algorithm 8) and the defence
  explicitly reports that reconstruction as *worse* than taking pairwise distances from
  `x̂₀` — i.e. the mechanism, not the information, was the bottleneck. Here there is no
  reconstruction: the representation itself is carried;
- self-conditioning's edge route in VEDA is only `n_edge_types` softmax values wide;
- it is index-aligned and dense (`k=None` in `edges_from_nodes`), so no graph-matching
  problem across steps;
- symmetry is already enforced downstream (`pred_edges + pred_edges.transpose(1,2)`), and
  carrying the pre-symmetrised tensor and symmetrising the carried copy is equivalent.

Memory: one extra `[B, N, N, 128]` tensor per tracked pass. At `N≈72` and float32 that is
~2.6 MB per molecule per pass. With BPTT at `n=1` this doubles; budget for it via
`--acc_batches`. If it bites, carrying `d_edge/2` channels via a projection is a cheap
fallback.

### 2.5 Cold start: zeros, not `None`

Replace the `prev_* is not None` branches with an always-on path fed either by the carried
latent or by a zeros tensor, mirroring `_get_initial_cond_batch`. Add

```python
def _get_initial_latent(self, interpolated):
    """Zero latent for the cold start, mirroring _get_initial_cond_batch."""
    ...
```

This fixes B2, makes the graph static for `torch.compile`, makes per-sample dropout a
simple multiply (§3.2), and makes "latent recursion off" and "latent recursion with a dropped
latent" numerically identical, which is what the dropout is supposed to mean.

### 2.6 Parameter budget

| channel | transfer | params |
|---|---|---|
| `s` | `LayerNorm` + `Linear(384+d_tau, 384)` + gate | ~148k |
| `v` | `CoordNorm` + `Linear(64, 64, bias=False)` + gate | ~4.1k |
| `e` | `LayerNorm` + `Linear(128, 128)` + gate | ~16.5k |
| `Δτ` | `Linear(1, d_tau)` + `SiLU` + `Linear(d_tau, d_tau)` | ~1k |
| **all three** | | **~170k** |

Against a ~20M-parameter 12-layer default configuration that is under 1%. The two channels
the thesis says matter are together ~21k parameters. This is a very favourable
headroom-to-cost ratio and worth stating in the paper.

---

## 3. Training design

### 3.1 Revised training step

```
procedure LR_TRAINING_STEP(x_t, x_0, σ, K, n, p_keep, p_shift)
  z ← zeros                                          # cold start, never None
  keep ~ Bernoulli(p_keep)^B                         # per-sample, shape [B,1,...]

  # one untracked pass gives BOTH the self-cond anchor and the first latent
  if self_condition or latent_recursion:
      σ', x' ← time_shift(x_t, σ, p_shift)           # §3.3
      with no_grad:
          x̂_0, z ← f_θ(x', σ', cond=zeros, z=zeros)
      cond ← {x̂_0, softmax(types), softmax(bonds)}   # as today
      z ← detach(z) * keep

  # remaining untracked recursion (K-1 further passes)
  with no_grad:
      repeat K-1 times:  _, z ← f_θ(x_t, σ, cond, z)
  z ← detach(z) * keep

  # BPTT
  repeat n times:        _, z ← f_θ(x_t, σ, cond, z)

  # loss pass
  x̂_0, _ ← f_θ(x_t, σ, cond, z)
  return L_diffusion(x̂_0, x_0, σ)
```

Differences from `_training_step` as it stands, in order of expected impact:

1. **Shared first pass** (G4). Fixes the 3-pass cost back to 2 at `K=1, n=0`.
2. **Time-shifted latent generation** (G3, §3.3).
3. **Per-sample dropout** via the `keep` mask instead of a batch-level branch (B2, H3).
4. **`p` renamed to a keep-probability** so it reads the same way as the thesis `p_drop`
   inverted; or keep `latrec_p` and document it as keep, but pick one and write it down —
   right now the flag help says "chain execution probability" while the thesis parameter is
   a *drop* probability, and that will get misread in a table.

### 3.2 Dropout as a mask

```python
keep = (torch.rand(batch_size, device=self.device) < self.latrec_p)
keep = keep.view(-1, *([1] * (z.dim() - 1))).float()
z = z.detach() * keep
```

One draw per molecule, per channel-shared. Keep it channel-shared rather than independent
per channel: independent per-channel dropout would let the model learn to rely on `e` alone
when `s` is dropped, which confounds the ablation you actually want to read.

Keep `p = 0.5` to match self-conditioning. Worth one sweep point at `p = 0.8`: the
argument for 0.5 is "never become dependent on it", but the depth-reduction result
(1 layer + recursion beating 8 layers vanilla) says dependence is exactly what you want
when the latent is always available at inference. A higher keep-probability is a cheap
thing to test and the thesis never tested it.

### 3.3 Time-shifted latent generation — the main training change

**The mismatch.** Training produces and consumes the latent at the same `(x_t, σ)`.
Inference consumes at `σ_t` a latent produced at `σ_{t-Δt} > σ_t` from a different `x`. So
the model is trained to *refine its own latent in place* and deployed to *interpret a stale
latent from a noisier state*. These are different functions.

**The fix, at zero extra compute.** `_interpolate_with_noise(curr, times, times_hat)`
already exists in `fm.py:1300` and is used by the sampler to move from a less noisy to a
more noisy state. Reuse it in training to synthesise `x_{σ'}` with `σ' > σ`, run the
untracked pass there, and consume the resulting latent at `(x_t, σ)`. The number of forward
passes is unchanged — only the *input* to the untracked pass changes.

Choosing `σ'`. Match the inference schedule rather than picking a constant, because the
`arcsin` schedule over `[min_sigma, max_sigma] = [1e-3, 80]` has wildly non-uniform σ gaps:

```python
# once, at setup
sched = self._create_time_schedule(self.integrator.steps, self.sampling_scheduler)  # descending σ
# per batch: for each sampled σ, find the neighbouring schedule points and take their ratio
i = torch.searchsorted(sched.flip(0), sigma)          # index of the bracketing step
ratio = sched[i - 1] / sched[i]                       # > 1
sigma_prev = sigma * ratio
```

This preserves the LogNormal marginal over training times (you are not resampling `σ`, only
deriving `σ'` from it) while making the staleness of the latent match what the sampler will
actually produce.

Expose it as `--latrec_t_shift {none, schedule, const}` with `--latrec_t_shift_const` for
the constant-ratio variant, and `--latrec_p_shift` for the probability of applying it (mix
shifted and unshifted so the model handles both, since the very first step of sampling has
no previous state at all).

**Why I expect this to matter.** It is the one structural difference between the training
distribution and the inference distribution of `z`, and it explains the thesis's multi-level
inference null result without any appeal to the mechanism being wrong. If the prediction is
right, two things should follow and are cheap to check: the time-shifted model should gain
*more* from extra integration steps (the defence already observed better inference scaling
for recursion than for vanilla), and multi-level inference should move from null to either
clearly positive or clearly negative rather than indistinguishable.

### 3.4 Time-delta conditioning (G5)

Embed `Δτ = c_noise(σ) − c_noise(σ')` (a scalar per molecule; `c_noise = log(σ)/4`, so this
is a log-ratio and naturally scaled) with a small MLP and concatenate it to the `s` transfer
input. Under `--latrec_t_shift none` this is constant-zero and the mechanism degenerates
gracefully. Under `schedule` it tells the transfer how stale the latent is, which is the
information it needs to decide how much to trust it.

### 3.5 BPTT depth

Thesis result: `n=1` gives ~+3.5 points on both mechanisms, `n=2` gives nothing further.
So:

- default `latrec_n = 1`, not `0`. The current default (`train.py:567`) makes the mechanism
  equivalent to "self-conditioning on latents", which the thesis showed sits *below*
  self-conditioning. Shipping `n=0` as the default means the headline configuration is the
  one that does not work.
- memory scales as `n+1` tracked passes. At `d_model=384`, 12 layers and dense
  `[B,N,N,128]` edges, expect roughly `2×` activation memory at `n=1`. Halve `--batch_cost`
  or double `--acc_batches` to keep the optimiser-step batch constant, and say so in the
  config — otherwise the BPTT ablation is confounded with an effective-batch-size change.
- if memory binds, `torch.utils.checkpoint` on `EquiMessagePassingLayer` is the right lever
  and trades ~30% step time for roughly linear activation savings.

### 3.6 Stability measures

The defence reports collapses and restarts. Concretely, for VEDA:

1. **Zero-init gates** (§2.1) — the mechanism cannot destabilise a run it has not yet
   learned to use.
2. **Normalise before injecting** — `LayerNorm` for `s` and `e`, `CoordNorm(norm="length")`
   for `v`. Without this, `z` can drift in magnitude across 100 steps at inference even if
   training was stable, which produces the characteristic failure where training looks fine
   and sampling degrades.
3. **Log the diagnostics that predict collapse early**: the three gate values, and
   `‖z_s‖`, `‖z_e‖`, `‖z_v‖` per integration step during the validation sample. A latent
   norm that grows monotonically along the sampling trajectory is the signature to watch
   for; it shows up in validation sampling long before `val-molecule-stability` moves.
4. **Fix H1 first.** The restart-from-pre-collapse-checkpoint protocol the thesis used is
   not possible with `save_top_k=0` and a shared `dirpath`.
5. `gradient_clip_val` is already wired through `train.py`; keep it non-`None` for every
   recursion run and record the value.

### 3.7 Interaction with EMA

`forward` uses `ema_gen` when `training=False` and `gen` otherwise (`fm.py:737-740`), and
`_training_step` calls everything with `training=True`. So the latent chain uses the live
weights during training and the EMA weights during sampling — consistent with how
self-conditioning already behaves, and correct. Worth one explicit note in the paper, since
"the latent was produced by a different parameter vector than the one consuming it" is a
reasonable reviewer question and the answer is "no, both are EMA at inference".

---

## 4. Inference

Keep what the commit does — carry the latent across integration steps, cold-start from
zeros at the first step — and add:

1. **`--latrec_inference_steps` (Algorithm 9)**, but only after §3.3 lands. With the
   same-`t` training distribution it is predicted to be null, and the thesis measured it as
   null; re-measuring it before changing training would just reproduce that.
2. **Latent-norm logging during sampling** (§3.6.3), cheap and diagnostic.
3. **Reset discipline**: the latent must be reset per sampling run, not reused across
   batches. `_generate` does this correctly (`node_latent = None` inside the function,
   `fm.py:1126`). Keep that invariant when adding the `v` and `e` carries.

---

## 5. Suggested order of work

Each step is independently measurable, and the order is by expected-value per unit of
compute.

| # | change | why first | cost |
|---|---|---|---|
| 0 | B1, B2, B3, H1 | B1 corrupts every recursion run; H1 makes the sweep unreadable | hours |
| 1 | edge latent (§2.4) | the channel the ablation says carries the effect; 16.5k params; blocked in g3m, free here | small |
| 2 | `latrec_n=1` default (§3.5) | thesis says the mechanism only passes self-conditioning *with* BPTT | config |
| 3 | shared self-cond/latrec pass (§2.5, G4) | makes the cost claim "same as self-conditioning" | small |
| 4 | gates + normalisation + logging (§2.1, §3.6) | turns collapses from mysterious into diagnosable | small |
| 5 | vector latent (§2.3) | 4.1k params; completes the thesis ablation grid in VEDA | small |
| 6 | time-shifted training (§3.3) | the one structural train/inference gap; reuses existing code | medium |
| 7 | multi-level inference (§4.1) | only interesting after 6 | small |
| 8 | depth/steps reduction grid | VEDA is ~3× the thesis model, so the 1-layer result should be even starker | compute |

### Ablation grid worth running

Reproduce the thesis's Table 6.4 structure so the two sets of numbers are comparable, then
extend into what is new in VEDA:

- channels: `{s}`, `{e}`, `{v}`, `{s,e}`, `{s,v}`, `{v,e}`, `{s,v,e}` — seven runs at
  `n=1`, against the self-conditioning and vanilla baselines. This directly tests whether
  the thesis's "edge dominates, `s` and `v` only matter jointly" finding transfers to a
  dynamic-edge architecture.
- the **matched-pairs table** (thesis Table 6.5) is especially sharp here: self-conditioning
  carries bonds as `n_edge_types` probabilities, latent recursion carries `d_edge=128`
  features through the same injection point. Same information route, 25× the width, and the
  readout bottleneck removed. That is the cleanest available test of the thesis's central
  claim that the readout is a real bottleneck.
- `n ∈ {0, 1, 2}` on the best channel set.
- `p_keep ∈ {0.5, 0.8}` on the best channel set (untested in the thesis).
- `t_shift ∈ {none, schedule}` on the best channel set.

One caveat to carry over from the defence's own scope slide: with one run per configuration
and ~20 comparisons against a shared reference, only large effects survive correction.
Budget two seeds for the final headline configuration and its baseline, even at the cost of
dropping a sweep point elsewhere.
