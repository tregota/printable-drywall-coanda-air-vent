#!/usr/bin/env python3
"""
Variant B: plaster-in Coanda valve with removable core, plus hole-marking template.
Needs coanda_valve.py (variant A) next to it.   Run:  python3 variant_b.py
z = 0 is the plasterboard face, +z up into the ceiling. mm.
Parts written:
  B_frame.stl      spigot + bell, stays in the ceiling          (print spigot end down)
  B_brim.stl       perforated plaster-in ring                   (print flat)
  B_core.stl       removable spider that carries the M6 screw   (print hub down)
  template.stl     plug + arm for marking the plasterboard hole (print arm down)
The deflector disc is the same part as variant A (coanda_valve_deflector.stl).
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
    # bayonet (3 J-slots in the bore, lugs on the core)
    n_lugs      = 3,
    groove_r    = 47.2,
    pocket_z    = 34.5, run_z = 36.5, groove_top = 41.0,
    entry_half  = 5.2, twist = 30.0,           # degrees
    lug_half_w  = 3.2, lug_r = 46.9, lug_z0 = 34.6, lug_h = 3.6,
    leg_r_out   = 45.2, leg_t = 2.0,
    hub_r = 9.5, hub_z0 = 59.5, hub_z1 = 72.5, arm_t = 2.4, arm_z0 = 61.5,
    screw_hole = 6.6, hex_af = 10.4, head_depth = 4.4,
    # template
    hole_d      = 167.0,   # cut line (both variants)
    mark_d      = (167.0, 190.0, 250.0),  # pencil holes: cut line, flange A, brim B
    plug_od     = 99.5, plug_len = 65.0, tmpl_t = 3.0, arm_w = 22.0, pencil = 3.2,
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

def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

def frame_field(prof, r, phi, z, p=B):
    f = prof
    ri = p['spigot_od']/2 - p['wall']
    zf = z - np.maximum(r - ri, 0)                  # 45 deg floors -> printable + self-centring
    eh = np.radians(p['entry_half']); tw = np.radians(p['twist'])
    for k in range(p['n_lugs']):
        a = wrap(phi - 2*np.pi*k/p['n_lugs'])
        rad = r - p['groove_r']; top = z - p['groove_top']
        entry  = mx(rad, (np.abs(a) - eh)*r, top)
        run    = mx(rad, (-eh - a)*r, (a - tw - eh)*r, p['run_z'] - zf, top)
        pocket = mx(rad, (tw - eh - a)*r, (a - tw - eh)*r, p['pocket_z'] - zf, top)
        f = np.maximum(f, -np.minimum(np.minimum(entry, run), pocket))
    return f

def core_field(prof, r, phi, z, p=B):
    x = r*np.cos(phi); y = r*np.sin(phi)
    f = mx(r - p['hub_r'], p['hub_z0'] - z, z - p['hub_z1'])
    ro = p['leg_r_out']
    for k in range(p['n_lugs']):
        a = 2*np.pi*k/p['n_lugs']; xs = r*np.cos(phi-a); ys = r*np.sin(phi-a)
        arm = mx(np.abs(ys) - p['arm_t']/2, p['hub_r']-1 - xs, xs - ro, p['arm_z0'] - z, z - p['hub_z1'])
        leg = mx(np.abs(ys) - p['lug_half_w'], ro - p['leg_t'] - xs, xs - ro, p['lug_z0'] - z, z - p['hub_z1'])
        e = np.maximum(r - ro, 0)
        lug = mx(np.abs(ys) - p['lug_half_w'], ro - p['leg_t'] - xs, r - p['lug_r'],
                 p['lug_z0'] + e - z, z - (p['lug_z0'] + p['lug_h'] - e))
        f = np.minimum(f, np.minimum(np.minimum(arm, leg), lug))
    f = np.maximum(f, -(r - p['screw_hole']/2))
    f = np.maximum(f, -mx(cv.hex_d(x, y, p['hex_af']), (p['hub_z1'] - p['head_depth']) - z))
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
    R = p['plug_od']/2; t = p['tmpl_t']; L = p['plug_len']
    return np.array([(30, -t), (R, -t), (R, L-1.5), (R-1, L), (R-2, L), (R-2, 0), (30, 0)])

def tmpl_field(prof, r, phi, z, p=B):
    x = r*np.cos(phi); y = r*np.sin(phi); t = p['tmpl_t']
    arm = mx(np.abs(y) - p['arm_w']/2, 40 - x, x - (max(p['mark_d'])/2 + 7), -t - z, z)
    f = np.minimum(prof, arm)
    for d in p['mark_d']:
        f = np.maximum(f, -(np.hypot(x - d/2, y) - p['pencil']/2))
    return f

def throat(lift=0.0, p=B):
    _, arc = frame_profile(p); _, cur = cv.defl_profile(cv.P, lift - p['skim'])
    ri = p['spigot_od']/2 - p['wall']
    bell = np.r_[np.c_[np.full(50, ri), np.linspace(75, p['z_bell_top'], 50)], arc]
    bell = np.c_[np.interp(np.linspace(0, 1, 2000), np.linspace(0, 1, len(bell)), bell[:, 0]),
                 np.interp(np.linspace(0, 1, 2000), np.linspace(0, 1, len(bell)), bell[:, 1])]
    d = np.linalg.norm(cur[:, None, :] - bell[None], axis=2); j = d.argmin(1); dm = d.min(1)
    A = np.pi*(cur[:, 0] + bell[j, 0])*dm
    return A.min(), dm[A.argmin()]

def flip(xyz, ztop):
    o = xyz.copy(); o[:, 2] = ztop - o[:, 2]; o[:, 0] *= -1; return o

if __name__ == '__main__':
    p = B; A = cv.axis; rep = lambda n, t, v: print('%s: %d tris, %.1f cm3, manifold errors %s' % (n, len(t), v/1000, cv.check_manifold(t)))
    for l in (4, 2, 0, -2, -4, -6):
        a, g = throat(l); print('disc %+d mm: throat %5.0f mm2, gap %.1f' % (l, a, g))
    ang = lambda extra: np.radians(np.unique(np.round(np.r_[np.arange(0, 360.01, 3.0), np.array(extra) % 360, [0, 360]], 4)))
    # frame
    fp, _ = frame_profile()
    r_ax = A(0, p['stop_r'] + 1.5, 1.2, list(fp[:, 0]) + [p['groove_r']])
    z_ax = A(-p['skim'] - 1, p['z_spigot_top'] + 1, 1.2, [q for q in fp[:, 1] if q > 0 or q < -1] + [p['pocket_z'], p['run_z'], p['groove_top']])
    ex = []
    for k in range(p['n_lugs']):
        c = 360*k/p['n_lugs']; eh = p['entry_half']; tw = p['twist']
        for b_ in (-eh, eh, tw-eh, tw+eh): ex += [c+b_-0.05, c+b_+0.05]
        ex += list(np.arange(c-eh, c+tw+eh, 1.0))
    xyz, tri, vol = cv.mesh_polar(frame_field, fp, r_ax, ang(ex), z_ax); rep('frame', tri, vol)
    cv.write_stl('B_frame.stl', flip(xyz, p['z_spigot_top']), tri)
    # core
    dummy = np.array([(200, 0), (201, 0), (201, 1)])
    r_ax = A(0, p['lug_r'] + 1, 0.9, [p['hub_r'], p['screw_hole']/2, p['leg_r_out'], p['leg_r_out']-p['leg_t'], p['lug_r']])
    z_ax = A(p['lug_z0'] - 1, p['hub_z1'] + 1, 0.9, [p['lug_z0'], p['lug_z0']+p['lug_h'], p['arm_z0'], p['hub_z0'], p['hub_z1'], p['hub_z1']-p['head_depth']])
    ex = []
    for k in range(p['n_lugs']): ex += list(np.arange(360*k/p['n_lugs']-9, 360*k/p['n_lugs']+9.01, 0.4))
    xyz, tri, vol = cv.mesh_polar(core_field, dummy, r_ax, ang(ex), z_ax); rep('core', tri, vol)
    cv.write_stl('B_core.stl', flip(xyz, p['hub_z1']), tri)
    # brim
    bp = brim_profile()
    r_ax = A(bp[:, 0].min() - 1, p['brim_od']/2 + 1, 0.7, list(bp[:, 0]))
    z_ax = A(-p['brim_t'] - 0.6, 0.6, 0.3, [0, -0.4, -p['brim_t']])
    xyz, tri, vol = cv.mesh_polar(brim_field, bp, r_ax, np.radians(np.arange(0, 360.01, 0.4)), z_ax); rep('brim', tri, vol)
    xyz[:, 2] *= -1; xyz[:, 0] *= -1
    cv.write_stl('B_brim.stl', xyz, tri)
    # template
    tp = tmpl_profile()
    r_ax = A(0, max(p['mark_d'])/2 + 8, 0.6, list(tp[:, 0]) + [40])
    z_ax = A(-p['tmpl_t'] - 1, p['plug_len'] + 1, 1.5, list(tp[:, 1]))
    phi = np.unique(np.round(np.r_[np.arange(0, 360.01, 3.0), np.arange(-18, 18.01, 0.3) % 360, [0, 360]], 4))
    xyz, tri, vol = cv.mesh_polar(tmpl_field, tp, r_ax, np.radians(phi), z_ax); rep('template', tri, vol)
    xyz[:, 2] += p['tmpl_t']
    cv.write_stl('template.stl', xyz, tri)
