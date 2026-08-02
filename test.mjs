// test.mjs — PROOF-OF-PLAY for THE BONDED PAIR (§17). Zero tokens. This is a RESEARCH experiment, so the gate's
// job is to (1) prove the harness is sound and fair, and (2) assert the TRUE empirical findings — the positive AND
// the negative — never a faked win. We measure whether two small coupled models, forced to disagree and resolved
// by confidence, beat either alone / an independent ensemble / a capacity-matched MONOLITH. The honest headline:
//   ✓ SUPER-ADDITIVITY / EMERGENCE is real — two ~chance members resolve into a strong pair (the resolver emerges
//     from the pair, not either alone).  ✗ the equal-capacity MONOLITH is NOT beaten — a bigger monopole still wins.
// Both are encoded as passing assertions so the result can't drift silently either way.
import { experiment, params, makeMLP, trainSingle, trainPair, makeTask, predict, predictPair, accuracy, rng } from './dyad.mjs';

let pass = 0, fail = 0;
const ok = (c, m) => { c ? pass++ : fail++; console.log((c ? '  ✓ ' : '  ✗ FAIL ') + m); };
const pct = x => (x * 100).toFixed(1);

// run the two headline experiments once, reuse across sections (deterministic)
const XOR = experiment({ seeds: 12, kind: 'xor4', H: 4, nTrain: 300, epochs: 500, lambda: 0.6 });
const MOD = experiment({ seeds: 12, kind: 'modular', H: 4, nTrain: 90, epochs: 320, lambda: 0.6 });

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
  ok(XOR.bondedBeatsBestMember >= 10, `the pair beats its own best member on ${XOR.bondedBeatsBestMember}/12 seeds — "neither strand alone" (§17)`);
  ok(MOD.mean.bonded > MOD.mean.memberBest, `super-additivity holds on the modular task too (${pct(MOD.mean.memberBest)} → ${pct(MOD.mean.bonded)})`);
}

console.log('\n=== §3 · THE DEEPER HONESTY — the emergence is GENERIC, and the §17 coupling did NOT beat plain ensembling ===');
{
  // the crucial control: a plain INDEPENDENT ensemble ALSO turns two weak members into a strong predictor. So the
  // "emergence" in §2 is the generic two-heads effect — NOT proof that the special coupling did the work.
  ok(XOR.mean.ensemble > XOR.mean.memberBest + 0.12, `a plain ensemble ALSO emerges from the same members (${pct(XOR.mean.memberBest)} → ${pct(XOR.mean.ensemble)}) — emergence is generic, not the coupling's doing`);
  // and, uncomfortably for §17: forcing the pair to disagree (coupling + NCL) did NOT beat plain averaging here.
  ok(XOR.mean.ensemble >= XOR.mean.bonded, `the §17 coupling + "forced disagreement" did NOT beat a plain ensemble on the checkerboard (ensemble ${pct(XOR.mean.ensemble)} ≥ bonded ${pct(XOR.mean.bonded)}) — it slightly hurt`);
  console.log(`     bonded beats the ensemble on only ${XOR.bondedBeatsEnsemble}/12 [xor4], ${MOD.bondedBeatsEnsemble}/12 [modular] seeds — the coupling is not carrying its weight.`);
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
  const a = experiment({ seeds: 4, kind: 'modular', H: 4, nTrain: 60, epochs: 120 });
  const b = experiment({ seeds: 4, kind: 'modular', H: 4, nTrain: 60, epochs: 120 });
  ok(a.mean.bonded === b.mean.bonded && a.mean.monolith === b.mean.monolith, 'the whole experiment is deterministic — same seeds, identical numbers (reproducible research)');
  let threw = false;
  try { const m = makeMLP(3, 2, rng(1)); predict(m, [NaN, Infinity, -5]); const p = trainPair([[0, 0], [1, 1]], [1, -1], { H: 2, epochs: 3, lr: .1, l2: 0, seed: 1, lambda: .5 }); predictPair(p.A, p.B, [9e9, -9e9]); } catch { threw = true; }
  ok(!threw, 'degenerate inputs / tiny data never throw');
  ok(Number.isFinite(XOR.mean.bonded) && XOR.mean.bonded >= 0 && XOR.mean.bonded <= 1, 'accuracies are finite and in [0,1]');
}

const done = fail === 0;
console.log('\n' + (done
  ? `=== ✅ HONEST RESULT (ranking on the checkerboard): monolith ${pct(XOR.mean.monolith)} > ensemble ${pct(XOR.mean.ensemble)} > bonded-pair ${pct(XOR.mean.bonded)} > member ${pct(XOR.mean.memberBest)}. Two heads beat one (generic), but §17's coupling + forced-disagreement beat NEITHER a plain ensemble NOR a same-size monolith. A clean negative result, honestly obtained · ${pass}/${pass} · zero tokens ===`
  : `=== ❌ ${fail} FAILED / ${pass + fail} ===`));
process.exit(done ? 0 : 1);
