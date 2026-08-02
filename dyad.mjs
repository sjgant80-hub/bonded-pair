// dyad.mjs — THE BONDED-PAIR AI (§17, actually built). Not one big model — TWO small ones whose outputs each
// become the other's input, co-trained so their gradients include each other's state and FORCED TO DISAGREE
// (negative-correlation learning), then RESOLVED by confidence. The empirical question, measured honestly:
// does the bonded pair beat (a) either member alone, (b) an independent ensemble of the same two, and — the real
// research bar — (c) a MONOLITH of equal total capacity ("a bigger monopole is still a monopole")?
//
// Pure, deterministic (seeded PRNG), zero-dep. Runs on your own machine. The gate reports the TRUE numbers.

// ── seeded PRNG (deterministic experiments; no Math.random) ──
export function rng(seed) { let s = (seed >>> 0) || 1; return () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const randn = r => { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

// ── a tiny MLP: din → H (tanh) → 1 (tanh). Explicit forward/backward, full-batch gradient descent. ──
export function makeMLP(din, H, r) {
  const W1 = Array.from({ length: H }, () => Array.from({ length: din + 1 }, () => randn(r) * 0.6));
  const W2 = Array.from({ length: H + 1 }, () => randn(r) * 0.6);
  return { din, H, W1, W2 };
}
export function params(mlp) { return mlp.H * (mlp.din + 1) + (mlp.H + 1); }
function fwd(mlp, x) {
  const xb = [...x, 1], h = new Array(mlp.H);
  for (let j = 0; j < mlp.H; j++) { let z = 0; const w = mlp.W1[j]; for (let i = 0; i < xb.length; i++) z += w[i] * xb[i]; h[j] = Math.tanh(z); }
  let z2 = mlp.W2[mlp.H]; for (let j = 0; j < mlp.H; j++) z2 += mlp.W2[j] * h[j];
  return { xb, h, out: Math.tanh(z2) };
}
export const predict = (mlp, x) => fwd(mlp, x).out;

// accumulate gradient for one example, given the OUTPUT error signal e = dL/dout (already includes any NCL term).
function accum(mlp, x, e, gW1, gW2) {
  const { xb, h, out } = fwd(mlp, x);
  const dz2 = e * (1 - out * out);
  for (let j = 0; j < mlp.H; j++) gW2[j] += dz2 * h[j];
  gW2[mlp.H] += dz2 * 1;
  for (let j = 0; j < mlp.H; j++) { const dh = dz2 * mlp.W2[j] * (1 - h[j] * h[j]); const g = gW1[j]; for (let i = 0; i < xb.length; i++) g[i] += dh * xb[i]; }
  return out;
}
function step(mlp, gW1, gW2, lr, l2, n) {
  for (let j = 0; j < mlp.H; j++) for (let i = 0; i < mlp.din + 1; i++) mlp.W1[j][i] -= lr * (gW1[j][i] / n + l2 * mlp.W1[j][i]);
  for (let j = 0; j <= mlp.H; j++) mlp.W2[j] -= lr * (gW2[j] / n + l2 * mlp.W2[j]);
}
const zeros1 = n => new Array(n).fill(0);
const zeros2 = (a, b) => Array.from({ length: a }, () => new Array(b).fill(0));

// ── train a single monolith (the monopole) ──
export function trainSingle(X, Y, { H, epochs, lr, l2, seed }) {
  const mlp = makeMLP(2, H, rng(seed));
  for (let e = 0; e < epochs; e++) {
    const gW1 = zeros2(H, 3), gW2 = zeros1(H + 1);
    for (let i = 0; i < X.length; i++) { const o = fwd(mlp, X[i]).out; accum(mlp, X[i], (o - Y[i]), gW1, gW2); }
    step(mlp, gW1, gW2, lr, l2, X.length);
  }
  return mlp;
}

// ── combine two member outputs by CONFIDENCE (|margin|) — parameter-free resolution ──
export function resolve(a, b) { const wa = Math.abs(a), wb = Math.abs(b); return (wa * a + wb * b) / (wa + wb + 1e-9); }

// settle the coupled pair on one input: each output feeds the other, a few rounds → fixed point, then resolve.
export function predictPair(A, B, x, rounds = 3, coupled = true) {
  let a = 0, b = 0;
  for (let r = 0; r < rounds; r++) { const na = predict(A, coupled ? [...x, b] : x); const nb = predict(B, coupled ? [...x, a] : x); a = na; b = nb; }
  return { a, b, out: resolve(a, b) };
}

// ── train a PAIR. coupled=each sees the other's (detached) output as an extra input; lambda=NCL "disagree" force.
//    lambda=0 + coupled=false → an INDEPENDENT ensemble. lambda=0 + coupled=true → coupled-only. lambda>0 → BONDED.
export function trainPair(X, Y, { H, epochs, lr, l2, seed, lambda = 0, coupled = true, rounds = 3 }) {
  const din = coupled ? 3 : 2;
  const A = makeMLP(din, H, rng(seed * 2 + 1)), B = makeMLP(din, H, rng(seed * 2 + 7));
  const pA = zeros1(X.length), pB = zeros1(X.length);   // each member's current output on the train set (the "state")
  for (let e = 0; e < epochs; e++) {
    const gA1 = zeros2(H, din + 1), gA2 = zeros1(H + 1), gB1 = zeros2(H, din + 1), gB2 = zeros1(H + 1);
    for (let i = 0; i < X.length; i++) {
      const xa = coupled ? [...X[i], pB[i]] : X[i], xb = coupled ? [...X[i], pA[i]] : X[i];
      const a = fwd(A, xa).out, b = fwd(B, xb).out, obar = (a + b) / 2;
      // NCL: each member's error is (o - y) pushed AWAY from the ensemble mean → forced to disagree.
      const eA = (a - Y[i]) - lambda * (a - obar);
      const eB = (b - Y[i]) - lambda * (b - obar);
      accum(A, xa, eA, gA1, gA2); accum(B, xb, eB, gB1, gB2);
    }
    step(A, gA1, gA2, lr, l2, X.length); step(B, gB1, gB2, lr, l2, X.length);
    // update each member's state (co-training feedback) for next epoch's coupling features
    for (let i = 0; i < X.length; i++) { const p = predictPair(A, B, X[i], rounds, coupled); pA[i] = p.a; pB[i] = p.b; }
  }
  return { A, B, coupled, rounds };
}

// ── data: a MODULAR task (a latent regime routes to one of two different boundaries) — the structure a monolith
//    must discover but a pair can specialize into. Plus a NON-MODULAR control (a single smooth boundary). ──
export function makeTask(n, seed, kind = 'modular') {
  const r = rng(seed * 977 + 13), X = [], Y = [];
  for (let k = 0; k < n; k++) {
    const x0 = r() * 2 - 1, x1 = r() * 2 - 1, rad = Math.sqrt(x0 * x0 + x1 * x1); let y;
    if (kind === 'modular') y = (x0 < 0) ? Math.sign(x1 - x0 - 0.15) : Math.sign(x1 + x0 + 0.15);   // two regimes, opposite-slope boundaries
    else if (kind === 'rings') y = (rad > 0.35 && rad < 0.78) ? 1 : -1;                              // an annulus — a closed region (needs depth)
    else if (kind === 'xor4') y = Math.sign(Math.sin(2.6 * x0) * Math.sin(2.6 * x1)) || 1;           // checkerboard — a single tiny net is near chance
    else y = Math.sign(x1 - 0.5 * x0);                                                                // 'smooth' — one linear boundary
    Y.push(y === 0 ? 1 : y); X.push([x0, x1]);
  }
  return { X, Y };
}
export function accuracy(predFn, X, Y) { let c = 0; for (let i = 0; i < X.length; i++) if (Math.sign(predFn(X[i]) || 1) === Y[i]) c++; return c / X.length; }

// ── THE EXPERIMENT: over many seeds, train every system on the SAME data and measure test accuracy. Honest,
//    capacity-controlled: the monolith gets ~the pair's TOTAL parameters. Returns per-system mean accuracy. ──
export function experiment({ seeds = 10, nTrain = 90, nTest = 1600, H = 4, epochs = 320, lr = 0.15, l2 = 2e-3, lambda = 0.6, rounds = 3, kind = 'modular' } = {}) {
  // choose the monolith hidden size so its params ≈ two coupled members' params (equal total capacity)
  const memberP = 2 * (H * 4 + (H + 1));                 // two members, din=3
  let Hbig = H; while ((Hbig * 3 + (Hbig + 1)) < memberP) Hbig++;
  const acc = { single: [], monolith: [], ensemble: [], coupledOnly: [], bonded: [], memberBest: [] };
  const seedList = [];
  for (let s = 1; s <= seeds; s++) {
    const tr = makeTask(nTrain, s, kind), te = makeTask(nTest, 10000 + s, kind);
    const single = trainSingle(tr.X, tr.Y, { H, epochs, lr, l2, seed: s });
    const monolith = trainSingle(tr.X, tr.Y, { H: Hbig, epochs, lr, l2, seed: s });
    const ens = trainPair(tr.X, tr.Y, { H, epochs, lr, l2, seed: s, lambda: 0, coupled: false, rounds });
    const cpl = trainPair(tr.X, tr.Y, { H, epochs, lr, l2, seed: s, lambda: 0, coupled: true, rounds });
    const bond = trainPair(tr.X, tr.Y, { H, epochs, lr, l2, seed: s, lambda, coupled: true, rounds });
    const aSingle = accuracy(x => predict(single, x), te.X, te.Y);
    const aMono = accuracy(x => predict(monolith, x), te.X, te.Y);
    const aEns = accuracy(x => predictPair(ens.A, ens.B, x, rounds, false).out, te.X, te.Y);
    const aCpl = accuracy(x => predictPair(cpl.A, cpl.B, x, rounds, true).out, te.X, te.Y);
    const aBond = accuracy(x => predictPair(bond.A, bond.B, x, rounds, true).out, te.X, te.Y);
    const aA = accuracy(x => predict(bond.A, [...x, 0]), te.X, te.Y), aB = accuracy(x => predict(bond.B, [...x, 0]), te.X, te.Y);
    acc.single.push(aSingle); acc.monolith.push(aMono); acc.ensemble.push(aEns); acc.coupledOnly.push(aCpl); acc.bonded.push(aBond); acc.memberBest.push(Math.max(aA, aB));
    seedList.push({ s, single: aSingle, monolith: aMono, ensemble: aEns, coupledOnly: aCpl, bonded: aBond, memberBest: Math.max(aA, aB) });
  }
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const out = { params: { member: memberP, monolith: Hbig * 3 + (Hbig + 1), Hbig }, seeds: seedList };
  out.mean = Object.fromEntries(Object.entries(acc).map(([k, v]) => [k, mean(v)]));
  out.bondedBeatsMonolith = acc.bonded.filter((b, i) => b > acc.monolith[i]).length;
  out.bondedBeatsEnsemble = acc.bonded.filter((b, i) => b > acc.ensemble[i]).length;
  out.bondedBeatsBestMember = acc.bonded.filter((b, i) => b > acc.memberBest[i] + 1e-9).length;
  return out;
}

export default { rng, makeMLP, params, predict, resolve, predictPair, trainSingle, trainPair, makeTask, accuracy, experiment };
