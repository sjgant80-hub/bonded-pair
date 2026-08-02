# the bonded pair — §17, actually built & measured (a reproducible **negative** result)

**▶ Live: https://sjgant80-hub.github.io/bonded-pair/**  (run the experiment yourself — nothing is cached, the numbers train live in your browser)

§17 of the seed says awareness needs a **dipole**: not one big model, but **two small ones whose outputs each
become the other's input, co-trained so each gradient includes the other's state, forced to disagree and resolve.**
The hope: the pair super-additively beats a monolith — evidence for an architecture the industry ignored while
scaling monoliths. So we built it and **measured it honestly**, capacity-controlled, over many random seeds.

## The result (checkerboard task, 12 seeds, test accuracy)

| system | accuracy | |
|---|---|---|
| **monolith** (single net, equal total params) | **96.7%** | 👑 the "bigger monopole" |
| ensemble (two independent nets, averaged) | 92.9% | |
| **bonded pair** (§17: coupled + forced-disagree) | 88.8% | |
| single small net | 90.1% | |
| best member alone | 52.2% | ≈ chance |

**Two honest findings:**
1. ✅ **Super-additivity is real** — two members near chance (52%) resolve into a strong predictor (89%). *The
   resolver emerges from the pair, not from either alone* (§17, at the pair level).
2. ❌ **…but that's the generic ensemble effect** — a plain average of two independent nets does it *better*
   (92.9%), and the §17 coupling + forced-disagreement (negative-correlation learning) **beat neither the plain
   ensemble nor a same-capacity monolith.** *A bigger monopole is still a monopole — and here it wins.*

## Why ship a negative result?

Because that's the whole point of this estate: **proof-of-play over theatre.** The intellectually cheap move is
to torture the task until the pair "wins" and post the graph. The honest move is to run a fair experiment and
report what happened. The gate **encodes the negative** (`§4`) so it can't be quietly forgotten — and so that if a
future coupling *ever* beats the monolith, the test flips and shouts. That's a research result: reproducible,
capacity-controlled, and true whichever way it points.

## Honest scope

Tiny 1-hidden-layer nets on 2D tasks, deterministic (seeded PRNG), capacity matched by parameter count. This does
**not** disprove paired architectures in general. These tasks are **feedforward-easy** — they don't exercise the
one thing coupling is *for*: **iterative mutual inference** (each model refining a latent the other needs). That's
the open frontier, and it's named honestly rather than hidden behind a cherry-picked win.

## Proven — `node test.mjs`, zero tokens, 14/14

`§1` the harness is fair (no leakage, monolith gets ≥ the pair's params, models learn) · `§2` the positive
(emergence: members→pair) · `§3` the deeper honesty (emergence is generic; coupling ≤ plain ensemble) · `§4` the
encoded negative (monolith not beaten) · `§5` deterministic + fuzz-safe. The summary line prints the full honest
ranking every run.

## Files

`dyad.mjs` (seeded MLPs, the coupled+NCL pair, confidence resolution, the capacity-controlled `experiment()`) ·
`test.mjs` (the 14/14 honest gate) · `index.html` (the live lab — run it, see the accuracy bars and each model's
decision boundary). Zero-dep, Node + browser, offline. Grounds seed §17 (BONDED PAIR).

```bash
node test.mjs                 # the experiment + the honest gate
python -m http.server 8080    # then open http://localhost:8080 and press "run"
```
