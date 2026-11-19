#!/usr/bin/env python3
"""
Parametric Coanda supply-air valve for a 100 mm duct behind a lowered ceiling.
Generates two printable STL files (body + deflector disc).
Needs: numpy, scikit-image.   Run:  python3 coanda_valve.py

Coordinates (as installed): z = 0 is the visible plasterboard face, +z goes up
into the ceiling. All dimensions in mm.
"""
import numpy as np, struct, sys
from skimage import measure

# ------------------------------------------------------------------ parameters
P = dict(
    # --- duct / spigot
    spigot_od   = 96.0,   # spigot outer dia (duct ID 100 minus 2 x 2 mm for foam seal)
    rib_od      = 98.6,   # retaining ribs either side of the seal band
    wall        = 2.4,
    z_spigot_top= 75.0,   # top of spigot (40.5 cavity + 34.5 mm inside the duct)
    # --- bell (Coanda surface) inside the 40.5 mm cavity
    z_bell_top  = 38.0,   # where the straight spigot bore starts to flare
    bell_a      = 38.0,   # radial semi-axis of the flare
    back_start_z= 39.5,   # 45 deg back cone starts here (must be < 40.5 = concrete)
    back_od     = 164.0,  # largest dia behind the flange -> plasterboard hole 166-170
    flange_od   = 190.0,
    flange_t    = 2.0,
    # --- spider / hub that carries the M6 screw
    hub_r       = 9.5,
    hub_z0      = 62.0,
    strut_t     = 2.4,
    strut_z0    = 64.0,
    n_struts    = 3,
    screw_hole  = 6.6,    # M6 clearance
    hex_af      = 10.4,   # M6 head / nut across flats + clearance
    head_depth  = 4.4,
    # --- deflector disc
    disc_d      = 108.0,
    disc_rim_t  = 1.6,
    tip_z       = 45.0,
    tip_r       = 11.0,
    bore_floor  = 3.0,    # solid material left under the screw bore (visible face)
    nut_z0      = 35.5,   # side slot for captive M6 nut
    nut_h       = 5.4,
)

# ------------------------------------------------------------------ profiles
def body_profile(p=P):
    ri = p['spigot_od']/2 - p['wall']; ro = p['spigot_od']/2; rr = p['rib_od']/2
    zt = p['z_spigot_top']; zb = p['z_bell_top']; a = p['bell_a']; b = zb
    th = np.radians(np.linspace(0, 90, 61))
    arc = np.c_[ri + a - a*np.cos(th), b - b*np.sin(th)]
    rf = p['flange_od']/2; rb = p['back_od']/2; ft = p['flange_t']
    d = rr - ro
    def rib(zc):  # going upward
        return [(ro, zc-0.3-d), (rr, zc-0.3), (rr, zc+0.3), (ro, zc+0.3+d)]
    pts  = [(ri, zt)] + [tuple(q) for q in arc]
    pts += [(rf-0.6, 0), (rf, 0.6), (rf, ft), (rb, ft),
            (rb, p['back_start_z'] - (rb-ro)), (ro, p['back_start_z'])]
    pts += rib(46.0) + rib(70.5)
    pts += [(ro, zt-1.5), (ro-1.2, zt)]
    return np.array(pts), arc

def defl_profile(p=P, lift=0.0):
    R = p['disc_d']/2; zt = p['tip_z']; rt = p['tip_r']; t = p['disc_rim_t']
    a = R - rt; b = zt - t
    ph = np.radians(np.linspace(90, 0, 61))
    curve = np.c_[R - a*np.cos(ph), zt - b*np.sin(ph)]
    pts = [(-1, 0), (R-0.5, 0), (R, 0.5)] + [tuple(q) for q in curve] + [(-1, zt)]
    pts = np.array(pts); pts[:, 1] += lift; curve = curve + [0, lift]
    return pts, curve

# ------------------------------------------------------------------ SDF helpers
def poly_sdf(poly, r, z):
    """signed distance (neg. inside) from points (r,z) to closed polygon"""
    r = np.asarray(r, float); z = np.asarray(z, float)
    shp = np.broadcast(r, z).shape
    px = np.broadcast_to(r, shp).ravel(); py = np.broadcast_to(z, shp).ravel()
    d2 = np.full(px.shape, np.inf); inside = np.zeros(px.shape, bool)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]; bx, by = poly[(i+1) % n]
        ex, ey = bx-ax, by-ay
        wx, wy = px-ax, py-ay
        tt = np.clip((wx*ex + wy*ey)/(ex*ex+ey*ey+1e-30), 0, 1)
        d2 = np.minimum(d2, (wx-tt*ex)**2 + (wy-tt*ey)**2)
        c = ((ay > py) != (by > py)) & (px < (bx-ax)*(py-ay)/(by-ay+1e-30) + ax)
        inside ^= c
    d = np.sqrt(d2); d[inside] *= -1
    return d.reshape(shp)

def hex_d(x, y, af):
    d = np.abs(x)
    for k in (1, 2):
        c, s = np.cos(k*np.pi/3), np.sin(k*np.pi/3)
        d = np.maximum(d, np.abs(x*c + y*s))
    return d - af/2

def mx(*a):
    out = a[0]
    for q in a[1:]: out = np.maximum(out, q)
    return out

def body_field(prof, r, phi, z, p=P):
    x = r*np.cos(phi); y = r*np.sin(phi); zt = p['z_spigot_top']
    f = prof
    f = np.minimum(f, mx(r - p['hub_r'], p['hub_z0'] - z, z - zt))
    ri = p['spigot_od']/2 - p['wall']
    for k in range(p['n_struts']):
        a = 2*np.pi*k/p['n_struts']
        xs = r*np.cos(phi-a); ys = r*np.sin(phi-a)
        f = np.minimum(f, mx(np.abs(ys) - p['strut_t']/2, p['hub_r']-1 - xs,
                             xs - (ri+1), p['strut_z0'] - z, z - zt))
    f = np.maximum(f, -(r - p['screw_hole']/2))
    f = np.maximum(f, -mx(hex_d(x, y, p['hex_af']), (zt - p['head_depth']) - z))
    return f

def defl_field(prof, r, phi, z, p=P, lift=0.0):
    x = r*np.cos(phi); y = r*np.sin(phi); z = z - lift
    f = prof
    f = np.maximum(f, -mx(r - p['screw_hole']/2, p['bore_floor'] - z))
    f = np.maximum(f, -mx(np.abs(y) - p['hex_af']/2, -(p['hex_af']/2+1) - x,
                          p['nut_z0'] - z, z - (p['nut_z0'] + p['nut_h'])))
    return f

# ------------------------------------------------------------------ meshing
def axis(lo, hi, step, feats=(), eps=0.04):
    pts = list(np.arange(lo, hi + step/2, step))
    for c in feats:
        pts += [c-eps, c+eps]
    pts = np.unique(np.round([q for q in pts if lo <= q <= hi], 4))
    keep = [pts[0]]
    for q in pts[1:]:
        if q - keep[-1] > 0.015: keep.append(q)
    return np.array(keep)

def mesh_polar(field_fn, poly, r_ax, phi_ax, z_ax):
    prof = poly_sdf(poly, r_ax[:, None], z_ax[None, :])[:, None, :]
    R = r_ax[:, None, None]; PH = phi_ax[None, :, None]; Z = z_ax[None, None, :]
    f = field_fn(prof, R, PH, Z).astype(np.float32)
    f[np.abs(f) < 1e-5] = 1e-5
    v, tri, _, _ = measure.marching_cubes(f, 0.0)
    r = np.interp(v[:, 0], np.arange(len(r_ax)), r_ax)
    ph = np.interp(v[:, 1], np.arange(len(phi_ax)), phi_ax)
    z = np.interp(v[:, 2], np.arange(len(z_ax)), z_ax)
    xyz = np.c_[r*np.cos(ph), r*np.sin(ph), z]
    key = np.round(xyz, 4)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    xyz = xyz[idx]; tri = inv.reshape(-1)[tri]
    ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[ok]
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    vol = np.einsum('ij,ij->i', a, np.cross(b, c)).sum()/6
    if vol < 0: tri = tri[:, ::-1]; vol = -vol
    return xyz, tri, vol

def check_manifold(tri):
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    u, cnt = np.unique(np.sort(e, 1), axis=0, return_counts=True)
    d, dc = np.unique(e, axis=0, return_counts=True)
    return int((cnt != 2).sum()), int((dc != 1).sum())

def write_stl(fn, xyz, tri):
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    n = np.cross(b-a, c-a); n /= (np.linalg.norm(n, axis=1)[:, None] + 1e-30)
    rec = np.zeros(len(tri), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'] = n; rec['v'] = np.stack([a, b, c], 1)
    with open(fn, 'wb') as f:
        f.write(b'coanda valve'.ljust(80)); f.write(struct.pack('<I', len(tri))); rec.tofile(f)

def throat(p=P, lift=0.0):
    """minimum flow area between deflector and bell (mm2)"""
    _, arc = body_profile(p); _, cur = defl_profile(p, lift)
    th = np.radians(np.linspace(0, 90, 400)); ri = p['spigot_od']/2 - p['wall']
    a = p['bell_a']; b = p['z_bell_top']
    bell = np.c_[ri + a - a*np.cos(th), b - b*np.sin(th)]
    bell = np.r_[np.c_[np.full(50, ri), np.linspace(75, b, 50)], bell]
    d = np.linalg.norm(cur[:, None, :] - bell[None], axis=2)
    j = d.argmin(1); dm = d.min(1)
    area = np.pi*(cur[:, 0] + bell[j, 0])*dm
    return area.min(), dm[area.argmin()]

if __name__ == '__main__':
    p = P
    bp, _ = body_profile(p); dp, _ = defl_profile(p)
    print('duct bore area  %.0f mm2' % (np.pi*(p['spigot_od']/2-p['wall'])**2))
    for l in (6, 4, 2, 0, -2, -4, -6, -8, -10):
        A, g = throat(p, l)
        print('disc %+3d mm (%s): throat %5.0f mm2, gap %4.1f mm' %
              (l, 'recessed' if l > 0 else 'lowered ' if l < 0 else 'flush   ', A, g))
    # ---- body
    feats_r = list(bp[:, 0]) + [p['hub_r'], p['screw_hole']/2]
    feats_z = [q for q in bp[:, 1] if q > 0.7 or q == 0] + [p['hub_z0'], p['strut_z0'],
               p['z_spigot_top']-p['head_depth']]
    r_ax = axis(0, p['flange_od']/2 + 1, 1.2, feats_r)
    z_ax = axis(-1, p['z_spigot_top'] + 1, 1.2, feats_z)
    ph = list(np.arange(0, 360.01, 3.0))
    for k in range(p['n_struts']):
        c = 360*k/p['n_struts']
        ph += list(np.arange(c-8, c+8.01, 0.5) % 360)
    phi_ax = np.radians(np.unique(np.round(ph + [0, 360], 4)))
    xyz, tri, vol = mesh_polar(body_field, bp, r_ax, phi_ax, z_ax)
    print('body: %d tris, %.1f cm3 solid, manifold errors %s' % (len(tri), vol/1000, check_manifold(tri)))
    # flip for printing: spigot end on the bed
    xb = xyz.copy(); xb[:, 2] = p['z_spigot_top'] - xb[:, 2]; xb[:, 0] *= -1
    write_stl('coanda_valve_body.stl', xb, tri)
    # ---- deflector
    feats_r = [p['disc_d']/2, p['disc_d']/2-0.5, p['tip_r'], p['screw_hole']/2]
    feats_z = [0, 0.5, p['disc_rim_t'], p['tip_z'], p['bore_floor'], p['nut_z0'], p['nut_z0']+p['nut_h']]
    r_ax = axis(0, p['disc_d']/2 + 1, 1.0, feats_r)
    z_ax = axis(-1, p['tip_z'] + 1, 1.0, feats_z)
    phi_ax = np.radians(np.arange(0, 360.01, 2.0))
    xyz, tri, vol = mesh_polar(defl_field, dp, r_ax, phi_ax, z_ax)
    print('deflector: %d tris, %.1f cm3 solid, manifold errors %s' % (len(tri), vol/1000, check_manifold(tri)))
    write_stl('coanda_valve_deflector.stl', xyz, tri)
