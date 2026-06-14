/* chexvision-mini · Figure 2 — Architecture + hand-derived backprop
   The from-scratch story: the MLP layer stack with a forward pass and a
   reversed, hand-derived gradient pass. Same visual system as parent Fig 2/3. */
(function () {
  const W = 1680, H = 900;

  window.drawMiniFig2 = function (id) {
    const f = FL, C = f.C;
    let s = '';

    // ---- vertical feature-vector bar: height encodes dimensionality ----
    function barVec(cx, cy, h, lvl, dim) {
      const bw = 34, x = cx - bw / 2, y = cy - h / 2;
      const c = FL.RAMP[lvl];
      let r = `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="3" fill="${c.f}" stroke="${c.s}" stroke-width="1.2"/>`;
      for (let yy = y + 13; yy < y + h - 4; yy += 13)
        r += `<line x1="${x + 1.5}" y1="${yy}" x2="${x + bw - 1.5}" y2="${yy}" stroke="${c.s}" stroke-width="0.7" opacity="0.55"/>`;
      r += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="3" fill="none" stroke="${C.mut}" stroke-width="1.1"/>`;
      r += f.T(cx, 150, dim, { size: 13.5, mono: true, weight: 700, fill: C.ink2 });   // dim label, fixed row
      return r;
    }

    // ===== forward stack geometry =====
    const cy = 296;                 // bar midline
    const bars = [
      { cx: 326, h: 232, lvl: 1, dim: '4096' },
      { cx: 596, h: 176, lvl: 2, dim: '1024' },
      { cx: 866, h: 122, lvl: 3, dim: '256' },
      { cx: 1118, h: 78, lvl: 4, dim: '64' },
    ];

    // top bracket over the whole MLP
    s += f.bracketT(bars[0].cx - 40, 1560, 118,
      'Multilayer perceptron   ·   4096 → 1024 → 256 → 64 → 1   ·   ~4.5 M parameters   ·   no framework, no autograd',
      { rise: 9, size: 13, weight: 600, labelFill: C.ink2, color: C.hair });

    // ---- input image (left) ----
    const imX = 96, imW = 104, imY = cy - imW / 2;
    s += f.slab(imX, imY, imW, imW, 7, 0, { frontStripes: true, clip: 'm2img' });
    s += f.T(imX + imW / 2, 150, '64×64×1', { size: 12.5, mono: true, weight: 600, fill: C.mut });
    s += f.T(imX + imW / 2, cy + imW / 2 + 26, 'chest X-ray', { size: 12.5, fill: C.mut });
    s += f.T(imX + imW / 2, cy + imW / 2 + 44, 'grayscale', { size: 12.5, fill: C.mut });
    // flatten arrow to first bar
    s += f.arrow(imX + imW + 6, cy, bars[0].cx - 24, cy, { color: C.ink2, width: 1.8 });
    s += f.T((imX + imW + bars[0].cx) / 2 + 6, cy - 12, 'flatten', { size: 11.5, mono: true, fill: C.mut, weight: 500 });

    // ---- bars ----
    bars.forEach(b => { s += barVec(b.cx, cy, b.h, b.lvl, b.dim); });

    // ---- operation labels + forward step arrows between bars ----
    const ops = [
      ['Linear 4096→1024', 'ReLU', 'Dropout 0.3'],
      ['Linear 1024→256', 'ReLU', 'Dropout 0.3'],
      ['Linear 256→64', 'ReLU', 'Dropout 0.3'],
    ];
    for (let i = 0; i < bars.length - 1; i++) {
      const a = bars[i], b = bars[i + 1];
      const mid = (a.cx + b.cx) / 2;
      s += f.arrow(a.cx + 22, cy, b.cx - 22, cy, { color: C.ink2, width: 1.7 });
      s += f.lines(mid, cy - 52, ops[i].map(t => ({ t })), { lh: 18, size: 12.5, mono: true, fill: C.ink2, weight: 500 });
    }

    // ---- final Linear → logit node → sigmoid ----
    const last = bars[bars.length - 1];
    const zX = 1300;
    s += f.arrow(last.cx + 22, cy, zX - 26, cy, { color: C.ink2, width: 1.7 });
    s += f.T((last.cx + zX) / 2 + 2, cy - 30, 'Linear', { size: 12.5, mono: true, fill: C.ink2, weight: 500 });
    s += f.T((last.cx + zX) / 2 + 2, cy - 13, '64 → 1', { size: 12.5, mono: true, fill: C.ink2, weight: 500 });
    // logit z node
    s += `<rect x="${zX - 21}" y="${cy - 21}" width="42" height="42" rx="6" fill="${FL.RAMP[5].f}" stroke="${FL.RAMP[5].s}" stroke-width="1.2"/>`;
    s += f.T(zX, cy + 6, 'z', { size: 18, weight: 700, fill: '#fff' });
    s += f.T(zX, 150, '1', { size: 13.5, mono: true, weight: 700, fill: C.ink2 });
    s += f.T(zX, cy + 40, 'logit', { size: 12, mono: true, fill: C.mut, weight: 600 });
    // sigmoid → probability
    s += f.arrow(zX + 22, cy, zX + 92, cy, { color: C.ink2, width: 1.7 });
    s += f.T(zX + 57, cy - 12, 'σ', { size: 14, mono: true, fill: C.ink2, weight: 600 });
    // BCE loss panel (right)
    const lpX = 1400, lpW = 220, lpH = 116, lpY = cy - lpH / 2;
    s += f.box(lpX, lpY, lpW, lpH, { stroke: C.hair, accent: C.coral, fill: '#fffcfa', r: 11 });
    s += f.T(lpX + 20, lpY + 30, 'Loss', { anchor: 'start', size: 16, weight: 700, fill: C.ink });
    s += f.T(lpX + lpW - 18, lpY + 30, 'target y', { anchor: 'end', size: 11, mono: true, fill: C.mut, weight: 600 });
    s += f.T(lpX + 20, lpY + 56, 'BCE-with-logits', { anchor: 'start', size: 13, mono: true, fill: C.ink2 });
    s += f.T(lpX + 20, lpY + 80, 'σ(z) = P(abnormal)', { anchor: 'start', size: 12.5, mono: true, fill: C.mut });
    s += f.T(lpX + 20, lpY + 99, 'normal vs. abnormal', { anchor: 'start', size: 12, fill: C.mut });

    // ===== backward sweep (coral, right → left, below the bars) =====
    const bwY = 452;
    s += f.poly([[lpX + lpW / 2, lpY + lpH + 4], [lpX + lpW / 2, bwY], [bars[0].cx - 44, bwY]],
      { color: C.coral, width: 2, head: 11 });
    s += f.T(lpX - 24, bwY - 12, 'dL/dz = (σ(z) − y) / N', { anchor: 'end', size: 13.5, mono: true, weight: 600, fill: C.coral });
    s += f.T((bars[0].cx + bars[3].cx) / 2, bwY + 22, 'BACKWARD  ·  upstream gradient threaded layer-by-layer (chain rule, hand-derived)',
      { size: 12.5, mono: true, fill: C.coral, weight: 500 });

    // ===== INSET: the Linear layer, forward & backward =====
    const iy = 524;
    s += `<line x1="70" y1="${iy}" x2="${W - 70}" y2="${iy}" stroke="${C.hairsoft}" stroke-width="1.4"/>`;
    s += f.T(72, iy + 28, 'THE LINEAR LAYER', { anchor: 'start', size: 12.5, weight: 700, upper: true, ls: 1.6, fill: C.ink });
    s += f.T(258, iy + 28, '— derived and coded by hand; every gradient verified numerically', { anchor: 'start', size: 12.5, fill: C.mut });

    // -- left: x → [W·b] → y unit, forward on top, backward below --
    const ucy = iy + 196;
    const vIn = f.vector(150, ucy, { ncell: 7, w: 22, lvl: 1 });
    s += vIn.svg;
    s += f.T(161, ucy + 60, 'x  (batch in)', { size: 12, mono: true, fill: C.mut, weight: 600 });

    // Linear box owns its parameters W, b and their hand-derived gradients
    const wbX = 320, wbW = 232, wbH = 134, wbY = ucy - wbH / 2;
    s += f.box(wbX, wbY, wbW, wbH, { stroke: C.hair, fill: '#fbfcfe', r: 10 });
    const wcx = wbX + wbW / 2;
    s += f.T(wcx, wbY + 30, 'Linear', { size: 15.5, weight: 700, fill: C.ink });
    s += f.T(wcx, wbY + 54, 'y = xW + b', { size: 12.5, mono: true, fill: C.ink2 });
    s += `<line x1="${wbX + 18}" y1="${wbY + 70}" x2="${wbX + wbW - 18}" y2="${wbY + 70}" stroke="${C.hairsoft}" stroke-width="1"/>`;
    s += f.T(wcx, wbY + 92, 'dW = xᵀ · dout', { size: 13.5, mono: true, fill: C.coral, weight: 700 });
    s += f.T(wcx, wbY + 114, 'db = Σ dout', { size: 13, mono: true, fill: C.coral, weight: 600 });

    const vOut = f.vector(672, ucy, { ncell: 4, w: 22, lvl: 3 });
    s += vOut.svg;
    s += f.T(683, ucy + 60, 'y  (out)', { size: 12, mono: true, fill: C.mut, weight: 600 });

    // forward arrows (top, ink) — data flowing left → right
    s += f.arrow(vIn.right + 6, ucy, wbX - 6, ucy, { color: C.ink2, width: 1.7 });
    s += f.arrow(wbX + wbW + 6, ucy, 672 - 6, ucy, { color: C.ink2, width: 1.7 });
    s += f.poly([[vIn.right + 6, wbY - 18], [694, wbY - 18]], { color: C.mut, width: 1.2, noHead: true });
    s += f.T((vIn.right + 694) / 2, wbY - 26, 'forward  ·  data', { size: 12, mono: true, fill: C.mut, weight: 600 });

    // backward arrow (bottom, coral, right → left) — carries dx to the previous layer
    const dY = wbY + wbH + 30;
    s += f.poly([[694, dY], [vIn.right + 6, dY]], { color: C.coral, width: 1.8, head: 10 });
    s += f.T((vIn.right + 694) / 2, dY - 9, 'backward  ·  dx = dout · Wᵀ', { size: 13, mono: true, fill: C.coral, weight: 600 });

    // -- right: small cards (ReLU, Dropout, gradient-check seal) --
    const cX = 760, cW = 452, cardH = 64, gap = 14;
    let cYc = iy + 96;
    function card(yTop, accent, fill, title, body) {
      let r = f.box(cX, yTop, cW, cardH, { stroke: C.hair, accent: accent, fill: fill, r: 10 });
      r += f.T(cX + 24, yTop + 27, title, { anchor: 'start', size: 14, weight: 700, fill: C.ink });
      r += f.T(cX + 24, yTop + 49, body, { anchor: 'start', size: 12.5, mono: true, fill: C.ink2 });
      return r;
    }
    s += card(cYc, C.blue, '#fafcff', 'ReLU', 'backward:  dout ⊙ 1[x > 0]');
    cYc += cardH + gap;
    s += card(cYc, C.blue, '#fafcff', 'Dropout', 'backward:  dout ⊙ mask / (1 − p)');
    cYc += cardH + gap;
    // gradient-check seal (coral, the correctness proof)
    const sealH = 84;
    s += f.box(cX, cYc, cW, sealH, { stroke: C.coralLine, accent: C.coral, fill: '#fffaf7', r: 10 });
    s += f.T(cX + 24, cYc + 28, 'Gradient check  ✓', { anchor: 'start', size: 14.5, weight: 700, fill: C.coral });
    s += f.T(cX + 24, cYc + 51, 'analytic vs. central-difference  (L(w+ε) − L(w−ε)) / 2ε', { anchor: 'start', size: 12, mono: true, fill: C.ink2 });
    s += f.T(cX + 24, cYc + 70, 'max relative error  <  1e-6   →   backprop is correct', { anchor: 'start', size: 12, mono: true, fill: C.mut });

    // -- far right: forward/backward legend --
    const legX = 1264, legW = W - 70 - legX;
    s += f.T(legX, iy + 108, 'READING THE FIGURE', { anchor: 'start', size: 11, weight: 700, upper: true, ls: 1.2, fill: C.mut });
    s += f.arrow(legX, iy + 140, legX + 52, iy + 140, { color: C.ink2, width: 1.8 });
    s += f.T(legX + 64, iy + 145, 'forward pass', { anchor: 'start', size: 12.5, fill: C.ink2 });
    s += f.poly([[legX + 52, iy + 172], [legX, iy + 172]], { color: C.coral, width: 2, head: 10 });
    s += f.T(legX + 64, iy + 177, 'gradient (backward)', { anchor: 'start', size: 12.5, fill: C.coral, weight: 500 });
    s += f.T(legX, iy + 212, 'Sequential threads dout', { anchor: 'start', size: 12, fill: C.mut });
    s += f.T(legX, iy + 230, 'through layers in reverse —', { anchor: 'start', size: 12, fill: C.mut });
    s += f.T(legX, iy + 248, 'backprop in a few lines.', { anchor: 'start', size: 12, fill: C.mut });

    f.mount(id, W, H, s);
  };
})();
