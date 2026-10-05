# Supervised pretraining of the flow's conditioner

Added 2026-09-25. Motivation: on 3-prong legs the direct-polvec flow regresses the tau
*direction* ~60% worse than the nu-method flow with the same conditioner architecture
(0.22° vs 0.13° median on DM10), and both its learned h and the analytic h from its own
taus inherit exactly that deficit. The suspicion is that the joint 12-dimensional NLL lets
the sharply peaked polarimetric-vector dimensions dominate the gradient reaching the
conditioner. Pretraining the conditioner with a per-dimension MSE on all targets removes
that imbalance; the flow is then trained normally from that starting point (nothing frozen).

## 1. Pretrain (transformer encoder + MSE head, 25 epochs)

    python taupolaris/scripts/train.py -c taupolaris/config/config_polvec_hadonly_onorm_newvars_pretrain_regressor.yaml --useTransformerBaseline --loadDS

* Same 72 inputs, data and event weights as the Sep01 flow.
* 18 targets in the onorm basis: pol-vecs (6) + neutrinos (6) + taus (6). Nu and tau are
  redundant (tau = nu + visible); both are kept so the encoder is pulled towards both.
* `TransformerBaseline_hyperparams` MUST equal the flow's conditioner hyperparameters
  (`context_dim`, `d_model`, `nhead`, `num_transformer_layers`, `dropout`), otherwise the
  weights do not transfer. They are deliberately not the paper baseline's settings.
* `TransformerRegressor` now takes the same feature-level arguments as the flow, so its
  encoder is built with the hadronic-current (and, if enabled, frame-aligned) tokens.

## 2. Evaluate the regressor on its own

    python taupolaris/scripts/evaluate_regressor.py -c taupolaris/config/config_polvec_hadonly_onorm_newvars_pretrain_regressor.yaml [--max_events N]

Per leg decay mode: tau |p| and direction resolution, transverse error, neutrino
resolution, the consistency |tau_pred - (nu_pred + visible)|, the h angle to truth and |h|
(no unit-norm constraint in a regressor), the tau mass from nu_pred + visible (truth is
1.777 GeV by construction). Per event: ditau mass from the regressed taus and from
nu_pred + visible, and the phiCP CP-even/odd asymmetry per decay-mode pair with the same
metric and code path as evaluate_polvec. Output in outputs_<model>/plots_eval_regressor/.

## 3. Train the flow from the pretrained conditioner

    python taupolaris/scripts/train.py -c taupolaris/config/config_polvec_hadonly_onorm_newvars_pretrainedcond.yaml

`SetupNN.condition_net_init: <checkpoint>` (NN_Tools.init_condition_net_from) copies every
tensor of the checkpoint's encoder (`encoder.*` of a TransformerRegressor, or
`condition_net.*` of another ConditionalFlow) whose name and shape match into
`condition_net`; a `type_emb` table with a different row count is copied for its leading
rows. Everything then trains as usual. The key is ignored under `reload: True`.
`config_polvec_hadonly_onorm_newvars_july28cond.yaml` uses the same mechanism to start from
the nu-method's 52-input conditioner (its two hadronic-current token embeddings and
`hcur_proj` stay at random init).
