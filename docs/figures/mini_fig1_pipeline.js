/* chexvision-mini · Figure 1 — End-to-end pipeline (pure NumPy, CPU)
   Two swimlanes, U-flow — same visual system as the parent CheXVision Fig 1. */
(function () {
  const W = 1680, H = 648;

  window.drawMiniFig1 = function (id) {
    const f = FL, C = f.C;
    let s = '';

    // ---------- lane bands ----------
    const topBand = { y: 96, h: 168 }, botBand = { y: 386, h: 168 };
    s += `<rect x="56" y="${topBand.y}" width="${W - 112}" height="${topBand.h}" rx="14" fill="#f7fafd" stroke="${C.hairsoft}" stroke-width="1.2"/>`;
    s += `<rect x="56" y="${botBand.y}" width="${W - 112}" height="${botBand.h}" rx="14" fill="#f8fbfd" stroke="${C.hairsoft}" stroke-width="1.2"/>`;
    // lane captions
    s += f.T(74, topBand.y - 12, 'INPUT PIPELINE', { anchor: 'start', size: 12.5, weight: 700, upper: true, ls: 1.4, fill: C.ink });
    s += f.T(222, topBand.y - 12, '— pure NumPy · streamed & downsampled on the fly', { anchor: 'start', size: 12.5, fill: C.mut });
    s += f.T(74, botBand.y - 12, 'TRAINING LOOP', { anchor: 'start', size: 12.5, weight: 700, upper: true, ls: 1.4, fill: C.ink });
    s += f.T(228, botBand.y - 12, '— per mini-batch · CPU only · no autograd', { anchor: 'start', size: 12.5, fill: C.mut });

    // ---------- node helper ----------
    function node(x, y, w, h, o) {
      let r = f.box(x, y, w, h, { stroke: C.hair, fill: '#ffffff', accent: o.accent, r: 12 });
      const cx = x + (o.accent ? (w + 10) / 2 + 5 : w / 2);
      r += f.T(cx, y + 30, o.title, { size: 16.5, weight: 700, fill: C.ink });
      r += f.lines(cx, y + 56, o.lines.map(t => ({ t })), { lh: 22, size: 13, mono: true, fill: C.ink2 });
      return r;
    }

    // =============== TOP LANE (L → R) ===============
    const tcy = topBand.y + 84;
    // dataset cylinder
    const cx = 92, cy = 112, cw = 232, ch = 140;
    s += f.cylinder(cx, cy, cw, ch, { fill: FL.RAMP[1].f, stroke: FL.RAMP[1].s, topFill: FL.RAMP[1].t, ry: 14 });
    s += f.T(cx + cw / 2, cy + 56, 'ChestX-ray14', { size: 13.5, weight: 700, mono: true, fill: C.blueDk });
    s += f.lines(cx + cw / 2, cy + 82, [
      { t: '320px parquet · HF' },
      { t: 'streamed, not stored' },
    ], { lh: 20, size: 12, mono: true, fill: C.ink2 });
    s += f.T(cx + cw / 2, cy + ch + 22, 'NIH chest X-rays', { size: 11.5, fill: C.mut });

    // downsample
    const dx = 520, dw = 300;
    s += node(dx, topBand.y + 26, dw, 116, {
      title: 'Downsample',
      lines: ['64 × 64 grayscale', 'flatten → 4096 floats'],
    });
    // standardize
    const stx = 980, stw = 320;
    s += node(stx, topBand.y + 26, stw, 116, {
      title: 'Standardize',
      lines: ['z = (x − μ) / σ   ·   train stats', 'light augment (flip · noise)'],
    });

    // top-lane arrows
    s += f.arrow(cx + cw + 6, tcy, dx - 8, tcy, { color: C.ink2, width: 1.9 });
    s += f.arrow(dx + dw + 6, tcy, stx - 8, tcy, { color: C.ink2, width: 1.9 });

    // =============== handoff (down-turn) ===============
    const fwCx = 1472;            // MLP-forward node centre x (defined below)
    const turnX = 1300;
    s += f.poly([[turnX, topBand.y + topBand.h], [turnX, 332], [fwCx, 332], [fwCx, botBand.y]], { color: C.blue, width: 1.9 });
    s += `<rect x="${(turnX + fwCx) / 2 - 90}" y="320" width="180" height="26" rx="13" fill="#fff" stroke="${FL.RAMP[1].s}" stroke-width="1.1"/>`;
    s += f.T((turnX + fwCx) / 2, 337, 'mini-batch · 256', { size: 12, mono: true, weight: 600, fill: C.blueDk });

    // =============== BOTTOM LANE (R → L) ===============
    const bcy = botBand.y + 84;
    // MLP forward (right)
    const mx = 1322, mw = 300;
    s += node(mx, botBand.y + 26, mw, 116, {
      title: 'MLP forward', accent: C.blue,
      lines: ['[Linear·ReLU·Dropout] ×3', '4096→1024→256→64→1'],
    });
    // BCE-with-logits
    const lx = 1004, lw = 300;
    s += node(lx, botBand.y + 26, lw, 116, {
      title: 'BCE-with-logits',
      lines: ['z·y − max(z,0) − log1p(e^−|z|)', 'fused sigmoid · stable'],
    });
    // backprop
    const bx = 686, bw = 300;
    s += node(bx, botBand.y + 26, bw, 116, {
      title: 'Backprop',
      lines: ['hand-derived ∇ · no autograd', 'dL/dz = (σ(z) − y) / N'],
    });
    // optimize
    const ox = 386, ow = 282;
    s += node(ox, botBand.y + 26, ow, 116, {
      title: 'Optimize',
      lines: ['Adam + cosine LR', 'L2 decay · dropout'],
    });
    // validate + select (left, terminal)
    const px = 60, pw = 308;
    s += node(px, botBand.y + 26, pw, 116, {
      title: 'Validate + select', accent: C.coral,
      lines: ['val AUC each epoch', 'best ckpt · Youden-J thresh'],
    });

    // bottom-lane arrows (leftward)
    s += f.arrow(mx - 6, bcy, lx + lw + 8, bcy, { color: C.ink2, width: 1.9 });
    s += f.arrow(lx - 6, bcy, bx + bw + 8, bcy, { color: C.ink2, width: 1.9 });
    s += f.arrow(bx - 6, bcy, ox + ow + 8, bcy, { color: C.ink2, width: 1.9 });
    s += f.arrow(ox - 6, bcy, px + pw + 8, bcy, { color: C.ink2, width: 1.9 });

    // =============== per-mini-batch loop (dashed return) ===============
    const loopY = 600;
    s += f.poly([[ox + ow / 2, botBand.y + botBand.h], [ox + ow / 2, loopY], [fwCx, loopY], [fwCx, botBand.y + botBand.h]],
      { color: C.mut, width: 1.5, dash: '5 5' });
    s += `<rect x="${(ox + ow / 2 + fwCx) / 2 - 242}" y="${loopY - 13}" width="484" height="26" rx="13" fill="#fff" stroke="${C.hair}" stroke-width="1.1"/>`;
    s += f.T((ox + ow / 2 + fwCx) / 2, loopY + 4.5, 'update weights  ·  repeat per mini-batch  ·  best epoch is kept', { size: 12, mono: true, fill: C.mut, weight: 500 });

    f.mount(id, W, H, s);
  };
})();
