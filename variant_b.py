#!/usr/bin/env python3
"""
Variant B: plaster-in Coanda valve with screw-in core, plus hole-marking template.
Needs coanda_valve.py (variant A) next to it.   Run:  python3 variant_b.py
z = 0 is the plasterboard face, +z up into the ceiling. mm.
Parts written:
  B_frame.stl        spigot + bell with threaded bore, stays in the ceiling (print spigot end down)
  B_brim.stl         perforated plaster-in ring                             (print flat)
  B_disc.stl         deflector disc with hex peg                            (print face down)
  B_thread_ring.stl  threaded ring, glue onto the disc's hex peg            (print flat)
  template_ring.stl  plug + ring: trace around it to mark the hole          (print ring down)
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
    # marking template
    hole_d      = 167.0,   # cut line (both variants)
    pencil_offset = 0.5,   # pencil line lands this far outside the ring edge
    plug_od     = 99.5, plug_len = 65.0, tmpl_t = 2.4,
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
    return np.minimum(prof, ridge)

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
    return np.maximum(f, -cv.hex_d(x, y, p['peg_af'] + p['peg_clear']))

def disc_profile(p=B):
    pts, _ = cv.defl_profile(cv.P); return pts

def disc_field(prof, r, phi, z, p=B):
    x = r*np.cos(phi); y = r*np.sin(phi); zt = cv.P['tip_z']
    peg = mx(cv.hex_d(x, y, p['peg_af']), zt - 1 - z, z - (zt + p['ring_h']))
    return np.minimum(prof, peg)

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
    r_ax = A(0, p['stop_r'] + 1.5, 1.2, list(fp[:, 0]) + list(np.arange(ri - p['thr_h'] - 0.3, ri + 0.4, 0.3)))
    z_ax = A(-p['skim'] - 1, p['z_spigot_top'] + 1, 1.2, [q for q in fp[:, 1] if q > 0 or q < -1]
             + list(np.arange(p['thr_z0'] - 0.4, p['thr_z1'] + 0.4, 0.2)), eps=0.02)
    xyz, tri, vol = cv.mesh_polar(frame_field, fp, r_ax, ang([]), z_ax); rep('frame', tri, vol)
    cv.write_stl('B_frame.stl', flip(xyz, p['z_spigot_top']), tri)
    # threaded ring (core, part 1)
    dummy = np.array([(200, 0), (201, 0), (201, 1)])
    r_ax = A(0, ri + 1, 0.9, [p['hub_r']] + list(np.arange(ri - 4, ri + 0.4, 0.3)))
    z_ax = A(-1, p['ring_h'] + 1, 0.4, [0, p['ring_h']])
    ex = []
    for k in range(p['n_arms']): ex += list(np.arange(360*k/p['n_arms']-9, 360*k/p['n_arms']+9.01, 0.5))
    xyz, tri, vol = cv.mesh_polar(ring_field, dummy, r_ax, ang(ex), z_ax); rep('thread ring', tri, vol)
    cv.write_stl('B_thread_ring.stl', xyz, tri)
    # disc with hex peg (core, part 2)
    dp = disc_profile()
    r_ax = A(0, c['disc_d']/2 + 1, 1.0, [c['disc_d']/2, c['disc_d']/2-0.5, c['tip_r'], p['peg_af']/2])
    z_ax = A(-1, c['tip_z'] + p['ring_h'] + 1, 1.0, [0, 0.5, c['disc_rim_t'], c['tip_z'], c['tip_z'] + p['ring_h']])
    xyz, tri, vol = cv.mesh_polar(disc_field, dp, r_ax, np.radians(np.arange(0, 360.01, 2.0)), z_ax); rep('disc', tri, vol)
    cv.write_stl('B_disc.stl', xyz, tri)
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
