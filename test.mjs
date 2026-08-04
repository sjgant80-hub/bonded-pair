// test.mjs — PROOF-OF-PLAY for THE BONDED PAIR (§17). Zero tokens. This is a RESEARCH experiment, so the gate's
// job is to (1) prove the harness is sound and fair, and (2) assert the TRUE empirical findings — the positive AND
// the negative — never a faked win. We measure whether two small coupled models, forced to disagree and resolved
// by confidence, beat either alone / an independent ensemble / a capacity-matched MONOLITH. The honest headline:
//   ✓ SUPER-ADDITIVITY / EMERGENCE is real — two ~chance members resolve into a strong pair (the resolver emerges
//     from the pair, not either alone).  ✗ the equal-capacity MONOLITH is NOT beaten — a bigger monopole still wins.
// Both are encoded as passing assertions so the result can't drift silently either way.
import { experiment, params, makeMLP, trainSingle, trainPair, makeTask, predict, predictPair, accuracy, rng, resolve } from './dyad.mjs';

let pass = 0, fail = 0;
const ok = (c, m) => { c ? pass++ : fail++; console.log((c ? '  ✓ ' : '  ✗ FAIL ') + m); };
const pct = x => (x * 100).toFixed(1);

// run the two headline experiments once, reuse across sections (deterministic).
// Sizes chosen so the WHOLE suite runs in a few seconds (fast enough for the witness mutation gate's per-run
// budget) WHILE the honest findings still hold with margin — the negative result is computed, not deleted.
const SEEDS = 6;                                       // deterministic seeds 1..SEEDS
const XOR = experiment({ seeds: SEEDS, kind: 'xor4', H: 4, nTrain: 100, epochs: 350, nTest: 250, lambda: 0.6 });
const MOD = experiment({ seeds: SEEDS, kind: 'modular', H: 4, nTrain: 90, epochs: 220, nTest: 250, lambda: 0.6 });

console.log('\n=== §1 · THE HARNESS IS SOUND & FAIR (no leakage, capacity-controlled, models learn) ===');
{
  ok(XOR.params.monolith >= XOR.params.member, `the monolith gets AT LEAST the pair's total parameters — a fair "bigger monopole" (mono ${XOR.params.monolith} ≥ pair ${XOR.params.member})`);
  ok(MOD.mean.single > 0.6 && MOD.mean.monolith > 0.6, `models actually learn the task (single ${pct(MOD.mean.single)}, monolith ${pct(MOD.mean.monolith)} > chance)`);
  // no train/test leakage: train seed s, test seed 10000+s — disjoint generators
  const tr = makeTask(50, 3, 'modular'), te = makeTask(50, 10003, 'modular');
  const overlap = tr.X.filter(a => te.X.some(b => b[0] === a[0] && b[1] === a[1])).length;
  ok(overlap === 0, 'train and test sets are disjoint (no leakage)');
}

console.log('\n=== §2 · THE POSITIVE — SUPER-ADDITIVITY / EMERGENCE (the resolver emerges from the pair) ===');
{
  // on the checkerboard, a single small member is near chance; the pair resolves into something far stronger.
  ok(XOR.mean.memberBest < 0.72, `alone, a small model is weak on the checkerboard (best member only ${pct(XOR.mean.memberBest)})`);
  ok(XOR.mean.bonded > XOR.mean.memberBest + 0.12, `but the BONDED PAIR is far stronger than either member — a real emergent jump (${pct(XOR.mean.memberBest)} → ${pct(XOR.mean.bonded)})`);
  ok(XOR.bondedBeatsBestMember >= SEEDS - 1, `the pair beats its own best member on ${XOR.bondedBeatsBestMember}/${SEEDS} seeds — "neither strand alone" (§17)`);
  ok(MOD.mean.bonded > MOD.mean.memberBest, `super-additivity holds on the modular task too (${pct(MOD.mean.memberBest)} → ${pct(MOD.mean.bonded)})`);
}

console.log('\n=== §3 · THE DEEPER HONESTY — the emergence is GENERIC, and the §17 coupling did NOT beat plain ensembling ===');
{
  // the crucial control: a plain INDEPENDENT ensemble ALSO turns two weak members into a strong predictor. So the
  // "emergence" in §2 is the generic two-heads effect — NOT proof that the special coupling did the work.
  ok(XOR.mean.ensemble > XOR.mean.memberBest + 0.12, `a plain ensemble ALSO emerges from the same members (${pct(XOR.mean.memberBest)} → ${pct(XOR.mean.ensemble)}) — emergence is generic, not the coupling's doing`);
  // and, uncomfortably for §17: forcing the pair to disagree (coupling + NCL) did NOT beat plain averaging here.
  ok(XOR.mean.ensemble >= XOR.mean.bonded, `the §17 coupling + "forced disagreement" did NOT beat a plain ensemble on the checkerboard (ensemble ${pct(XOR.mean.ensemble)} ≥ bonded ${pct(XOR.mean.bonded)}) — it slightly hurt`);
  console.log(`     bonded beats the ensemble on only ${XOR.bondedBeatsEnsemble}/${SEEDS} [xor4], ${MOD.bondedBeatsEnsemble}/${SEEDS} [modular] seeds — the coupling is not carrying its weight.`);
}

console.log('\n=== §4 · THE HONEST NEGATIVE — the equal-capacity MONOLITH is NOT beaten ("a bigger monopole is still a monopole") ===');
{
  // This is the result Simon hoped might flip. It did not, on these tasks. We ENCODE the negative so it can't be
  // quietly forgotten — and so that if a future coupling actually beats the monopole, this test flips and shouts.
  ok(XOR.mean.monolith > XOR.mean.bonded, `on the checkerboard, the monolith wins (monolith ${pct(XOR.mean.monolith)} > bonded ${pct(XOR.mean.bonded)}) — honest negative on §17's strong claim`);
  ok(MOD.mean.monolith >= MOD.mean.bonded, `on the modular task, the monolith is still ahead (monolith ${pct(MOD.mean.monolith)} ≥ bonded ${pct(MOD.mean.bonded)})`);
  console.log(`     verdict: the PAIR beats its PARTS (emergence, real) but not a same-size MONOLITH — the industry scales monoliths for a reason, shown here at small scale.`);
}

console.log('\n=== §5 · DETERMINISM + FUZZ ===');
{
  const a = experiment({ seeds: 3, kind: 'modular', H: 4, nTrain: 40, epochs: 60, nTest: 200 });
  const b = experiment({ seeds: 3, kind: 'modular', H: 4, nTrain: 40, epochs: 60, nTest: 200 });
  ok(a.mean.bonded === b.mean.bonded && a.mean.monolith === b.mean.monolith, 'the whole experiment is deterministic — same seeds, identical numbers (reproducible research)');
  let threw = false;
  try { const m = makeMLP(3, 2, rng(1)); predict(m, [NaN, Infinity, -5]); const p = trainPair([[0, 0], [1, 1]], [1, -1], { H: 2, epochs: 3, lr: .1, l2: 0, seed: 1, lambda: .5 }); predictPair(p.A, p.B, [9e9, -9e9]); } catch { threw = true; }
  ok(!threw, 'degenerate inputs / tiny data never throw');
  ok(Number.isFinite(XOR.mean.bonded) && XOR.mean.bonded >= 0 && XOR.mean.bonded <= 1, 'accuracies are finite and in [0,1]');
}

console.log('\n=== §6 · BOUNDARY KILLS — exported product functions pinned exactly at the mutated edge ===');
{
  // These assertions call the exported functions directly with inputs that sit ON the boundary a single-operator
  // flip would move, so the mutation gate can prove the harness logic is guarded (not just the statistical findings).

  // params() — the exact weight count of an MLP: H*(din+1) + (H+1). Flip either "+ 1" and it miscounts.
  ok(params(makeMLP(2, 4, rng(1))) === 17, `params() counts an MLP's weights exactly (din2→H4→1 has 4*(2+1)+(4+1)=17)`);
  // the experiment's capacity accounting is exact: the pair has 42 params, the capacity-matched monolith is sized to
  // Hbig=11 (45 params ≥ 42), and every seed is run. (kills the +1 in memberP / in the reported monolith size / the seed loop)
  ok(XOR.params.member === 42 && XOR.params.monolith === 45 && XOR.params.Hbig === 11, `capacity accounting is exact (member ${XOR.params.member}, monolith ${XOR.params.monolith} at Hbig ${XOR.params.Hbig})`);
  ok(XOR.seeds.length === SEEDS, `the experiment runs all ${SEEDS} seeds (the seed loop is inclusive, not one short)`);
  // the monolith-sizing loop `while(3*Hbig+(Hbig+1) < memberP) Hbig++` grows to the SMALLEST size reaching the pair's
  // params. Probe H=1 (memberP=12): the "+1" inside that ceiling moves the boundary to Hbig=3 (at H=4 both ±1 happen to
  // land on the same integer, so this smaller probe is what pins it).
  const probe = experiment({ seeds: 1, kind: 'modular', H: 1, nTrain: 20, epochs: 2, nTest: 50 });
  ok(probe.params.Hbig === 3, `the monolith is sized to the tight ceiling (H=1 → memberP 12 → Hbig ${probe.params.Hbig})`);
  // resolve()'s "+ 1e-9" is a divide-by-zero guard on the confidence denominator. With two tiny equal margins whose
  // weights sum to 8e-10, the guard keeps the denominator POSITIVE; flip it to "- 1e-9" and the denominator goes
  // NEGATIVE (8e-10 - 1e-9 < 0) so the blended output inverts sign.
  ok(resolve(4e-10, 4e-10) > 0, `resolve()'s zero-guard keeps the denominator positive on tiny margins (blend stays > 0)`);
  // trainSingle / trainPair with epochs=0 must do ZERO gradient steps — byte-identical to the fresh seeded init. An
  // off-by-one in the epoch loop (`e < epochs` → `e <= epochs`) would sneak in one hidden update.
  const dtask = makeTask(30, 7, 'modular');
  const s0 = trainSingle(dtask.X, dtask.Y, { H: 4, epochs: 0, lr: 0.15, l2: 2e-3, seed: 7 });
  ok(JSON.stringify(s0.W2) === JSON.stringify(makeMLP(2, 4, rng(7)).W2), `trainSingle(epochs=0) does zero steps (identical to init)`);
  const p0 = trainPair(dtask.X, dtask.Y, { H: 4, epochs: 0, lr: 0.15, l2: 2e-3, seed: 7, lambda: 0.6 });
  ok(JSON.stringify(p0.A.W2) === JSON.stringify(makeMLP(3, 4, rng(7 * 2 + 1)).W2), `trainPair(epochs=0) does zero steps (member A identical to init)`);
  // with epochs>0 the OUTPUT-BIAS weight (index H of W2) must actually move — the update loop must include j=H
  // (`j <= mlp.H`). Skip it and the bias never learns.
  const initB = makeMLP(2, 4, rng(7)), trB = trainSingle(dtask.X, dtask.Y, { H: 4, epochs: 50, lr: 0.15, l2: 2e-3, seed: 7 });
  ok(Math.abs(trB.W2[4] - initB.W2[4]) > 1e-9, `the output bias is actually trained (the W2 update loop covers j=0..H inclusive)`);
  // makeTask() is a deterministic generator: it returns exactly n points, and its seeded coordinates are exact (the
  // "+ 13" seed offset is load-bearing — change it and the whole dataset shifts).
  ok(makeTask(7, 1, 'modular').X.length === 7, `makeTask returns exactly n points (the fill loop runs k=0..n-1)`);
  ok(Math.abs(makeTask(2, 1, 'modular').X[0][0] - 0.6016551963984966) < 1e-12, `makeTask's seeded coordinates are exact (seed offset pinned)`);
  // the 'rings' task is a CLOSED annulus (0.35 < rad < 0.78) — it MUST contain points OUTSIDE the ring (label -1),
  // i.e. the "&&" is a real conjunction. Flip it to "||" and every point becomes +1 (no closed region at all).
  const rings = makeTask(200, 5, 'rings');
  ok(rings.Y.some(y => y === -1), `the rings task is a closed region — it has points outside the annulus (the && is a real AND)`);
}

const done = fail === 0;
console.log('\n' + (done
  ? `=== ✅ HONEST RESULT (ranking on the checkerboard): monolith ${pct(XOR.mean.monolith)} > ensemble ${pct(XOR.mean.ensemble)} > bonded-pair ${pct(XOR.mean.bonded)} > member ${pct(XOR.mean.memberBest)}. Two heads beat one (generic), but §17's coupling + forced-disagreement beat NEITHER a plain ensemble NOR a same-size monolith. A clean negative result, honestly obtained · ${pass}/${pass} · zero tokens ===`
  : `=== ❌ ${fail} FAILED / ${pass + fail} ===`));
process.exit(done ? 0 : 1);
