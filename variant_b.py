#!/usr/bin/env python3
"""
Variant B: plaster-in Coanda valve with screw-in core, plus hole-marking template.
Needs coanda_valve.py (variant A) next to it.   Run:  python3 variant_b.py
z = 0 is the plasterboard face, +z up into the ceiling. mm.
Parts written:
  B_frame_no_line.stl / B_frame_deep_line.stl   spigot + bell with threaded bore, without / with reference line (print spigot end down)
  B_frame_deep_line_inlay.stl  optional: fills the reference line in the bell (same colour = no line, other colour = coloured line)
  B_brim.stl         perforated plaster-in ring                             (print flat)
  B_disc_carrier.stl deflector with hex peg, markings engraved underneath   (print flat side down)
  B_disc_markings_inlay.stl  optional: fills the engraving, load together with the carrier and give it another colour
  B_disc_cover.stl   pull-off cover disc (the visible face)                 (print face down)
  B_thread_ring.stl  threaded ring, glue onto the disc's hex peg            (print flat)
  template_ring.stl  plug + ring: trace around it to mark the hole          (print ring down)
  template_drill_guide.stl  plug with centre bushing for a hole saw's pilot drill (print plate down)
No screws: the disc + ring core screws into the thread in the frame's bore.
"""
import numpy as np
import coanda_valve as cv
mx = cv.mx

B = dict(
    duct_id     = 100.0,
    spigot_od   = 96.0, rib_od = 98.6, wall = 2.4,
    z_spigot_top= 75.0,
    cavity      = 40.5,    # board face -> concrete / duct mouth
    skim        = 2.5,     # bell lip stands this far below the board face (1 mm brim + 1.5 mm plaster)
    z_bell_top  = 34.0,
    bell_r_end  = 83.0,
    back_od     = 164.0,   # -> plasterboard hole 167
    stop_r      = 84.2,    # plaster stop edge radius
    brim_od     = 250.0,   # <= your bed size
    brim_t      = 1.0,
    brim_hole_d = 5.0,
    # big printed thread: core screws into the bore (adjust + remove by turning the disc)
    pitch       = 6.0,     # mm per turn, right-hand: clockwise seen from below = up
    thr_h       = 1.5,     # ridge height
    thr_crest   = 1.2,     # crest width (45 deg flanks)
    thr_z0      = 36.0, thr_z1 = 62.0,     # threaded length of the bore
    thr_clear   = 0.3,     # radial clearance, also sets ~0.6 mm axial play
    ring_t      = 1.6, ring_h = 10.0, ring_z0 = 42.5,   # threaded ring of the core (z0 = nominal, disc flush)
    hub_r       = 11.5, peg_af = 14.0, peg_clear = 0.3, arm_t = 2.4, n_arms = 3,
    pin_d       = 2.3,     # cross hole through hub + peg for a piece of 1.75 mm filament (locks ring to disc)
    # two-part disc: carrier (stays on the thread ring) + pull-off cover that hides the markings
    carrier_r   = 46.0, cover_t = 1.2, cover_clear = 0.2,
    sock_r      = 10.15, sock_depth = 12.0, bead_z = 8.5, bead_h = 0.4, groove_h = 0.5,
    collet_r    = 10.0, collet_t = 1.2, collet_len = 11.0, n_slits = 8, slit_w = 1.4,
    key_r       = 22.0, key_d = 5.0, key_h = 2.2,          # 2 drive pegs so turning the cover turns the core
    engrave     = 0.6,
    n_ticks     = 12,      # one tick = pitch / n_ticks = 0.5 mm of height
    mark        = True,    # False = frame without reference line
    mark_w      = 1.6, mark_depth = 0.6,                   # reference line engraved in the bell
    mark_r0     = 46.3, mark_r1 = 51.0,                    # radial extent: high in the throat, hidden behind the cover disc
    # marking template
    hole_d      = 167.0,   # cut line (both variants)
    pencil_offset = 0.5,   # pencil line lands this far outside the ring edge
    plug_od     = 99.5, plug_len = 65.0, tmpl_t = 2.4,
    # drill guide: plug with a centre bushing for the hole saw's pilot drill
    pilot_d     = 6.6,     # bushing bore: fits 6 mm and 1/4" (6.35) pilot drills; ream to size
    guide_len   = 50.0,    # plug length into the duct
    guide_lip_d = 106.0,   # stop lip at the board face, stays inside the hole saw's cup
    guide_t     = 3.0,     # base plate thickness
    bush_od     = 16.0, bush_len = 35.0,                 # bushing length measured from the room side
    n_ribs      = 4, rib_t = 3.0,
)

def frame_profile(p=B):
    ri = p['spigot_od']/2 - p['wall']; ro = p['spigot_od']/2; rr = p['rib_od']/2
    zt = p['z_spigot_top']; zb = p['z_bell_top']; zl = -p['skim']
    a = p['bell_r_end'] - ri; b = zb - zl
    th = np.radians(np.linspace(0, 90, 61))
    arc = np.c_[ri + a*(1-np.cos(th)), zb - b*np.sin(th)]
    rb = p['back_od']/2; rs = p['stop_r']; rm = p['duct_id']/2 - 0.5; zc = p['cavity']
    d = rr - ro
    rib = lambda c: [(ro, c-0.3-d), (rr, c-0.3), (rr, c+0.3), (ro, c+0.3+d)]
    zs = zl + 1.0                                   # top of the 1 mm plaster stop
    pts  = [(ri, zt)] + [tuple(q) for q in arc]
    pts += [(rs, zl), (rs, zs), (rb, zs + (rs-rb)), (rb, zc - (rb-rm)), (rm, zc), (ro, zc + (rm-ro))]
    pts += rib(46.0) + rib(70.5) + [(ro, zt-1.5), (ro-1.2, zt)]
    return np.array(pts), arc

def helix_d(phi, z, p, shift=0.0):
    """axial distance to the nearest thread crest centre line"""
    P = p['pitch']; u = z - P*(phi % (2*np.pi))/(2*np.pi) - shift
    return np.abs((u + P/2) % P - P/2)

def frame_field(prof, r, phi, z, p=B):
    ri = p['spigot_od']/2 - p['wall']; rc = ri - p['thr_h']
    d = helix_d(phi, z, p)
    ridge = mx(rc + np.maximum(d - p['thr_crest']/2, 0) - r, r - (ri + 1),
               rc + (p['thr_z0'] + p['thr_h'] - z) - r, rc + (z - (p['thr_z1'] - p['thr_h'])) - r)
    f = np.minimum(prof, ridge)
    return np.maximum(f, -ref_mark(prof, r, phi, z, p)) if p['mark'] else f + 0*phi

def ref_mark(prof, r, phi, z, p=B):
    """reference line in the bell at phi = 0, read against the ticks on the carrier"""
    ri = p['spigot_od']/2 - p['wall']; x = r*np.cos(phi); y = r*np.sin(phi)
    return mx(np.abs(y) - p['mark_w']/2, -x, -(prof + p['mark_depth']), z - (p['z_bell_top'] - 1),
              p['mark_r0'] - r, r - p['mark_r1'], r - (87 - z))

def frame_mark_inlay_field(prof, r, phi, z, p=B):
    return np.maximum(ref_mark(prof, r, phi, z, p), prof)

def ring_field(prof, r, phi, z, p=B):
    """threaded ring + arms + hub with hex socket; z measured from the ring's lower face"""
    x = r*np.cos(phi); y = r*np.sin(phi); h = p['ring_h']
    ri = p['spigot_od']/2 - p['wall']; r_root = ri - p['thr_h'] - p['thr_clear']; r_cr = ri - p['thr_clear']
    zz = mx(-z, z - h)
    d = helix_d(phi, z + p['ring_z0'], p, shift=p['pitch']/2)
    f = mx(r - r_root, r_root - p['ring_t'] - r, zz)
    ridge = mx(r - (r_cr - np.maximum(d - p['thr_crest']/2, 0)), r_root - 1 - r,
               r - (r_cr - (p['thr_h'] - z)), r - (r_cr - (z - (h - p['thr_h']))), zz)
    f = np.minimum(f, ridge)
    f = np.minimum(f, mx(r - p['hub_r'], zz))
    for k in range(p['n_arms']):
        a = 2*np.pi*k/p['n_arms']; xs = r*np.cos(phi-a); ys = r*np.sin(phi-a)
        f = np.minimum(f, mx(np.abs(ys) - p['arm_t']/2, p['hub_r']-1 - xs, xs - (r_root - 0.5), zz))
    f = np.maximum(f, -cv.hex_d(x, y, p['peg_af'] + p['peg_clear']))
    return np.maximum(f, -mx(np.hypot(x, z - h/2) - p['pin_d']/2, r - (p['hub_r'] + 1)))   # pin hole along y, between the arms

def _trumpet(p=B):
    _, cur = cv.defl_profile(cv.P); return cur            # rim -> tip, (r, z), r decreasing

def carrier_profile(p=B):
    cur = _trumpet(p); rc = p['carrier_r']; zc = np.interp(rc, cur[::-1, 0], cur[::-1, 1])
    inner = cur[cur[:, 0] < rc]
    return np.array([(-1, p['cover_t']), (rc, p['cover_t']), (rc, zc)] + [tuple(q) for q in inner] + [(-1, cv.P['tip_z'])])

def cover_profile(p=B):
    cur = _trumpet(p); rc = p['carrier_r'] + p['cover_clear']; zc = np.interp(rc, cur[::-1, 0], cur[::-1, 1])
    outer = cur[cur[:, 0] > rc]; R = cv.P['disc_d']/2
    return np.array([(-1, 0), (R-0.5, 0), (R, 0.5)] + [tuple(q) for q in outer] + [(rc, zc), (rc, p['cover_t']), (-1, p['cover_t'])])

_MARKS = None
def marks_sdf(p=B, ppm=20, half=50):
    """signed distance image (mm, negative inside the engraving) of the markings as seen from below"""
    global _MARKS
    if _MARKS is not None: return _MARKS
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Polygon
    from scipy.ndimage import distance_transform_edt as edt
    fig = plt.figure(figsize=(2*half/25.4, 2*half/25.4), dpi=25.4*ppm)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.axis('off')
    lw = lambda mm: mm/25.4*72
    n = p['n_ticks']; rc = p['carrier_r']
    for i in range(n):
        a = np.pi/2 - 2*np.pi*i/n; c, s_ = np.cos(a), np.sin(a)
        ax.plot([c*(rc-6.5), c*(rc-1.5)], [s_*(rc-6.5), s_*(rc-1.5)], 'k', lw=lw(1.5), solid_capstyle='butt')
        ax.text(c*(rc-11.5), s_*(rc-11.5), str(i), ha='center', va='center', fontsize=lw(6.2), fontweight='bold', family='DejaVu Sans')
    for txt, v, a0, a1 in (('OPEN', 20.5, 25, 150), ('CLOSE', -20.5, 210, 335)):
        ax.text(0, v, txt, ha='center', va='center', fontsize=lw(6.6), fontweight='bold', family='DejaVu Sans')
        ax.add_patch(Arc((0, 0), 29, 29, theta1=a0, theta2=a1, lw=lw(1.5), color='k'))
        e = np.radians(a1 if v > 0 else a0); sgn = 1 if v > 0 else -1          # arrow head, both travel right-to-left
        t = np.array([-np.sin(e), np.cos(e)])*sgn; nrm = np.array([np.cos(e), np.sin(e)]); c0 = 14.5*nrm
        ax.add_patch(Polygon([c0 + 4.5*t, c0 + 2.6*nrm, c0 - 2.6*nrm], color='k'))
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[:, :, 0] < 128; plt.close(fig)
    _MARKS = ((edt(~img) - edt(img))/ppm, ppm, half)
    return _MARKS

def marks_at(u, v, p=B):
    from scipy.ndimage import map_coordinates
    d, ppm, half = marks_sdf(p)
    return map_coordinates(d, [(half - v)*ppm - 0.5, (u + half)*ppm - 0.5], order=1, mode='nearest')

def _bead(r_base, z, zc, h, flat):
    """radius of a 45-degree-sided bead/groove profile centred at zc"""
    return r_base + np.maximum(0, h - np.maximum(np.abs(z - zc) - flat/2, 0))

def carrier_field(prof, r, phi, z, p=B):
    x = r*np.cos(phi); y = r*np.sin(phi); zt = cv.P['tip_z']; z0 = p['cover_t']
    f = np.minimum(prof, mx(cv.hex_d(x, y, p['peg_af']), zt - 1 - z, z - (zt + p['ring_h'])))
    f = np.maximum(f, -(np.hypot(x, z - (zt + p['ring_h']/2)) - p['pin_d']/2))
    f = np.maximum(f, -mx(r - _bead(p['sock_r'], z, z0 + p['bead_z'], p['groove_h'], 1.0), z - (z0 + p['sock_depth'])))
    for sx in (1, -1):
        f = np.maximum(f, -mx(np.hypot(x - sx*p['key_r'], y) - (p['key_d']/2 + 0.25), z - (z0 + p['key_h'] + 0.4)))
    xx, yy, _ = np.broadcast_arrays(x, y, z)
    m = marks_at(xx[:, :, 0], -yy[:, :, 0], p)[:, :, None]         # seen from below: u = x, v = -y
    return np.maximum(f, -mx(m, z - (z0 + p['engrave']), r - (p['carrier_r'] - 0.8)))

def inlay_field(prof, r, phi, z, p=B):
    x = r*np.cos(phi); y = r*np.sin(phi); z0 = p['cover_t']
    xx, yy, _ = np.broadcast_arrays(x, y, z)
    m = marks_at(xx[:, :, 0], -yy[:, :, 0], p)[:, :, None]
    return mx(m, z - (z0 + p['engrave']), z0 - z, r - (p['carrier_r'] - 0.8)) + 0*phi

def cover_field(prof, r, phi, z, p=B):
    x = r*np.cos(phi); y = r*np.sin(phi); z0 = p['cover_t']
    col = mx(r - _bead(p['collet_r'], z, z0 + p['bead_z'], p['bead_h'], 0.6), p['collet_r'] - p['collet_t'] - r,
             z0 - 0.5 - z, z - (z0 + p['collet_len']))
    st = 2*np.pi/p['n_slits']; a = (phi + st/2) % st - st/2
    col = np.maximum(col, -mx(np.abs(r*np.sin(a)) - p['slit_w']/2, (z0 + 0.6) - z))
    f = np.minimum(prof, col)
    for sx in (1, -1):
        f = np.minimum(f, mx(np.hypot(x - sx*p['key_r'], y) - p['key_d']/2, z0 - 0.5 - z, z - (z0 + p['key_h'])))
    return f

def brim_profile(p=B):
    rb = p['back_od']/2; t = p['brim_t']; ro = p['brim_od']/2
    zs = -p['skim'] + 1.0; r0 = rb + (zs + (p['stop_r']-rb)) + 0.1   # frame cone radius at z=0 + clearance
    return np.array([(r0, 0), (ro, 0), (ro, -0.4), (ro-12, -t), (r0 + t, -t)])

def brim_rows(p=B):
    r0 = p['stop_r'] + 5.5; r1 = p['brim_od']/2 - 5.5; pitch = p['brim_hole_d'] + 2.6
    n = max(1, int(round((r1 - r0)/pitch))); rows = np.linspace(r0, r1, n+1)
    return [(rj, int(2*np.pi*rj/(p['brim_hole_d'] + 4.5)), j) for j, rj in enumerate(rows)]

def brim_field(prof, r, phi, z, p=B):
    f = prof
    for rj, nj, j in brim_rows(p):
        st = 2*np.pi/nj; a = (phi + (j % 2)*st/2) % st - st/2
        f = np.maximum(f, -(np.hypot(r - rj, rj*a) - p['brim_hole_d']/2))
    return f

def tmpl_profile(p=B):
    """plug for the duct + flat ring; trace a pencil around the ring's edge"""
    R = p['plug_od']/2; t = p['tmpl_t']; L = p['plug_len']; Ro = p['hole_d']/2 - p['pencil_offset']
    return np.array([(R-2, -t), (Ro, -t), (Ro, 0), (R, 0), (R, L-1.5), (R-1, L), (R-2, L)])

def tmpl_field(prof, r, phi, z, p=B): return prof + 0*phi

def guide_field(prof, r, phi, z, p=B):
    """z = 0 is the room-side face of the base plate (it sits guide_t below the board face)"""
    R = p['plug_od']/2; t = p['guide_t']; L = t + p['guide_len']; top = p['bush_len']
    plate = mx(r - p['guide_lip_d']/2, -z, z - t)
    tube  = mx(r - R, (R - 2) - r, -z, z - L, r - (R - (z - (L - 1.5)).clip(0)))   # small lead-in chamfer at the top
    hub   = mx(r - p['bush_od']/2, -z, z - top)
    f = np.minimum(np.minimum(plate, tube), hub)
    for k in range(p['n_ribs']):
        a = 2*np.pi*k/p['n_ribs']; xs = r*np.cos(phi - a); ys = r*np.sin(phi - a)
        h = top - (xs - p['bush_od']/2)*(top - t - 8)/(R - 2 - p['bush_od']/2)          # rib slopes down to the tube
        f = np.minimum(f, mx(np.abs(ys) - p['rib_t']/2, p['bush_od']/2 - 1 - xs, xs - (R - 1), -z, z - h))
    bore = r - (p['pilot_d']/2 + np.maximum(0, 1.2 - z))                                 # 1.2 mm entry chamfer
    return np.maximum(f, -bore)

def throat(lift=0.0, p=B):
    _, arc = frame_profile(p); _, cur = cv.defl_profile(cv.P, lift - p['skim'])
    ri = p['spigot_od']/2 - p['wall']
    bell = np.r_[np.c_[np.full(50, ri), np.linspace(75, p['z_bell_top'], 50)], arc]
    t0 = np.linspace(0, 1, len(bell)); t1 = np.linspace(0, 1, 2000)
    bell = np.c_[np.interp(t1, t0, bell[:, 0]), np.interp(t1, t0, bell[:, 1])]
    d = np.linalg.norm(cur[:, None, :] - bell[None], axis=2); j = d.argmin(1); dm = d.min(1)
    A = np.pi*(cur[:, 0] + bell[j, 0])*dm
    return A.min(), dm[A.argmin()]

def flip(xyz, ztop):
    o = xyz.copy(); o[:, 2] = ztop - o[:, 2]; o[:, 0] *= -1; return o

if __name__ == '__main__':
    p = B; A = cv.axis; c = cv.P
    rep = lambda n, t, v: print('%s: %d tris, %.1f cm3, manifold errors %s' % (n, len(t), v/1000, cv.check_manifold(t)))
    for l in (4, 2, 0, -2, -4, -6):
        a, g = throat(l); print('disc %+d mm: throat %5.0f mm2, gap %.1f' % (l, a, g))
    ang = lambda extra: np.radians(np.unique(np.round(np.r_[np.arange(0, 360.01, 3.0), np.array(extra) % 360, [0, 360]], 4)))
    ri = p['spigot_od']/2 - p['wall']
    # frame
    fp, _ = frame_profile()
    r_ax = A(0, p['stop_r'] + 1.5, 1.2, list(fp[:, 0]) + list(np.arange(ri - p['thr_h'] - 0.3, ri + 0.4, 0.3))
             + list(np.arange(p['mark_r0'] - 0.3, p['mark_r1'] + 0.9, 0.3)))
    _, arc = frame_profile(); zm = np.interp([p['mark_r1'], p['mark_r0']], arc[:, 0], arc[:, 1])
    z_ax = A(-p['skim'] - 1, p['z_spigot_top'] + 1, 1.2, [q for q in fp[:, 1] if q > 0 or q < -1]
             + list(np.arange(p['thr_z0'] - 0.4, p['thr_z1'] + 0.4, 0.2)) + list(np.arange(zm[0] - 1.5, min(zm[1] + 1.5, p['thr_z0'] - 0.5), 0.3)), eps=0.02)
    fang = ang(list(np.arange(-2.5, 2.51, 0.2)))
    p['mark'] = False
    xyz, tri, vol = cv.mesh_polar(frame_field, fp, r_ax, fang, z_ax); rep('frame, no line', tri, vol)
    cv.write_stl('B_frame_no_line.stl', flip(xyz, p['z_spigot_top']), tri)
    p['mark'] = True
    xyz, tri, vol = cv.mesh_polar(frame_field, fp, r_ax, fang, z_ax); rep('frame, deep line', tri, vol)
    cv.write_stl('B_frame_deep_line.stl', flip(xyz, p['z_spigot_top']), tri)
    xyz, tri, vol = cv.mesh_polar(frame_mark_inlay_field, fp, r_ax, fang, z_ax); rep('deep line inlay', tri, vol)
    cv.write_stl('B_frame_deep_line_inlay.stl', flip(xyz, p['z_spigot_top']), tri)
    print('line spans z = %.1f .. %.1f mm above the board face' % (zm[0], zm[1]))
    # threaded ring (core, part 1)
    dummy = np.array([(200, 0), (201, 0), (201, 1)])
    r_ax = A(0, ri + 1, 0.9, [p['hub_r']] + list(np.arange(ri - 4, ri + 0.4, 0.3)))
    z_ax = A(-1, p['ring_h'] + 1, 0.4, [0, p['ring_h']])
    ex = []
    for k in range(p['n_arms']): ex += list(np.arange(360*k/p['n_arms']-9, 360*k/p['n_arms']+9.01, 0.5))
    ex += list(np.arange(72, 108.01, 0.75)) + list(np.arange(252, 288.01, 0.75))
    xyz, tri, vol = cv.mesh_polar(ring_field, dummy, r_ax, ang(ex), z_ax); rep('thread ring', tri, vol)
    cv.write_stl('B_thread_ring.stl', xyz, tri)
    # disc carrier (core, part 2) - fine grid so the engraved markings come out
    kp = carrier_profile(); z0 = p['cover_t']
    r_ax = A(0, p['carrier_r'] + 1, 0.45, [p['carrier_r'], p['sock_r'], p['peg_af']/2])
    z_ax = A(z0 - 1, c['tip_z'] + p['ring_h'] + 1, 1.0, [z0, z0 + p['engrave'], z0 + p['key_h'] + 0.4, z0 + p['sock_depth'],
             c['tip_z'], c['tip_z'] + p['ring_h']] + list(np.arange(z0 + p['bead_z'] - 1.4, z0 + p['bead_z'] + 1.5, 0.25))
             + list(np.arange(c['tip_z'] + p['ring_h']/2 - 1.6, c['tip_z'] + p['ring_h']/2 + 1.7, 0.25)))
    xyz, tri, vol = cv.mesh_polar(carrier_field, kp, r_ax, np.radians(np.arange(0, 360.01, 0.6)), z_ax); rep('disc carrier', tri, vol)
    xyz[:, 2] -= z0
    cv.write_stl('B_disc_carrier.stl', xyz, tri)
    # markings as a separate body that exactly fills the engraving (for multi-colour printing)
    xyz, tri, vol = cv.mesh_polar(inlay_field, kp, r_ax, np.radians(np.arange(0, 360.01, 0.6)), z_ax); rep('markings inlay', tri, vol)
    xyz[:, 2] -= z0
    cv.write_stl('B_disc_markings_inlay.stl', xyz, tri)
    # disc cover (pull-off)
    vp = cover_profile()
    r_ax = A(0, c['disc_d']/2 + 1, 0.8, [c['disc_d']/2, c['disc_d']/2-0.5, p['carrier_r'] + p['cover_clear'], p['collet_r'],
             p['collet_r'] - p['collet_t'], p['collet_r'] + p['bead_h']] + list(np.arange(p['key_r'] - 3, p['key_r'] + 3.1, 0.4)))
    z_ax = A(-1, z0 + p['collet_len'] + 1, 0.8, [0, 0.5, z0, z0 + 0.6, z0 + p['key_h'], z0 + p['collet_len']]
             + list(np.arange(z0 + p['bead_z'] - 1.2, z0 + p['bead_z'] + 1.3, 0.25)))
    xyz, tri, vol = cv.mesh_polar(cover_field, vp, r_ax, np.radians(np.arange(0, 360.01, 1.0)), z_ax); rep('disc cover', tri, vol)
    cv.write_stl('B_disc_cover.stl', xyz, tri)
    # brim
    bp = brim_profile()
    r_ax = A(bp[:, 0].min() - 1, p['brim_od']/2 + 1, 0.7, list(bp[:, 0]))
    z_ax = A(-p['brim_t'] - 0.6, 0.6, 0.3, [0, -0.4, -p['brim_t']])
    xyz, tri, vol = cv.mesh_polar(brim_field, bp, r_ax, np.radians(np.arange(0, 360.01, 0.4)), z_ax); rep('brim', tri, vol)
    xyz[:, 2] *= -1; xyz[:, 0] *= -1
    cv.write_stl('B_brim.stl', xyz, tri)
    # template
    tp = tmpl_profile()
    r_ax = A(0, tp[:, 0].max() + 1, 2.0, list(tp[:, 0]))
    z_ax = A(-p['tmpl_t'] - 1, p['plug_len'] + 1, 3.0, list(tp[:, 1]))
    xyz, tri, vol = cv.mesh_polar(tmpl_field, tp, r_ax, np.radians(np.arange(0, 360.01, 2.0)), z_ax); rep('template', tri, vol)
    xyz[:, 2] += p['tmpl_t']
    cv.write_stl('template_ring.stl', xyz, tri)
    # drill guide
    dummy = np.array([(400, 0), (401, 0), (401, 1)])
    R = p['plug_od']/2
    r_ax = A(0, p['guide_lip_d']/2 + 1, 0.5, [p['pilot_d']/2, p['pilot_d']/2 + 1.2, p['bush_od']/2, R - 2, R, p['guide_lip_d']/2])
    z_ax = A(-1, p['guide_t'] + p['guide_len'] + 1, 0.5, [0, 1.2, p['guide_t'], p['bush_len'], p['guide_t'] + p['guide_len']])
    ex = []
    for k in range(p['n_ribs']): ex += list(np.arange(360*k/p['n_ribs'] - 12, 360*k/p['n_ribs'] + 12.01, 0.5))
    xyz, tri, vol = cv.mesh_polar(guide_field, dummy, r_ax, ang(ex), z_ax); rep('drill guide', tri, vol)
    cv.write_stl('template_drill_guide.stl', xyz, tri)
