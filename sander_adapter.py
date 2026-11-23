#!/usr/bin/env python3
"""
Sanding mandrel: screws into the back (spigot end) of B_frame and takes a hole-saw arbor
(5/8"-18 UNF, e.g. Bahco 3834-ARBR-11152 / -QC), so the frame can be spun in a drill.
Needs variant_b.py and coanda_valve.py next to it.  Run: python3 sander_adapter.py
Coordinates are the frame's: z = 0 board face, spigot top at z = 75. Printed with the arbor face (z = 79) on the bed.
"""
import numpy as np
import coanda_valve as cv, variant_b as vb
mx = cv.mx; B = vb.B

S = dict(
    plate_r   = 49.0,  plate_z0 = 75.0, plate_z1 = 79.0,   # cap that seats on the spigot end
    boss_r    = 21.0,  boss_z0 = 66.0,                     # centre boss carrying the arbor thread
    wall_t    = 2.0,   thr_lo = 46.0,  thr_hi = 60.0,       # threaded ring that engages the frame's bore thread
    pilot_gap = 0.2,   pilot_z0 = 62.4,                    # centring band in the plain bore above the thread
    n_ribs    = 6,     rib_t = 3.0,    rib_z0 = 66.0,
    # arbor thread: 5/8"-18 UNF, right hand
    unf_maj   = 15.875, unf_pitch = 25.4/18, unf_clear = 0.2,
    # drive-pin slots (radial, so any pin spacing fits)
    mount     = 'thread',  # 'thread' = printed 5/8"-18 UNF in the boss, 'clamp' = plain hole, held by a small hole saw
    clamp_t   = 3.0,       # centre plate thickness in clamp mode (the arbor stud must pass through it and still catch the saw)
    n_slots   = 4,     slot_w = 5.4,   slot_r0 = 10.0, slot_r1 = 19.0, slot_depth = 6.0,
)

def adapter_field(r, phi, z, p=B, s=S):
    x = r*np.cos(phi); y = r*np.sin(phi)
    ri = p['spigot_od']/2 - p['wall']; r_root = ri - p['thr_h'] - p['thr_clear']; r_cr = ri - p['thr_clear']
    # threaded ring (frame's bore thread, absolute helix, half a pitch out of phase = interlocking)
    d = vb.helix_d(phi, z, p, shift=p['pitch']/2)
    lo, hi = s['thr_lo'], s['thr_hi']
    ridge = mx(r - (r_cr - np.maximum(d - p['thr_crest']/2, 0)), r_root - 1 - r,
               r - (r_cr - (p['thr_h'] - (z - lo))), r - (r_cr - (z - (hi - p['thr_h']))), lo - z, z - hi)
    wall = mx(r - r_root, r_root - s['wall_t'] - r, lo - z, z - (s['plate_z0'] + 1))
    pilot = mx(r - (ri - s['pilot_gap']), r_root - 1 - r, s['pilot_z0'] - z, z - (s['plate_z0'] + 1))
    plate = mx(r - s['plate_r'], s['plate_z0'] - z, z - s['plate_z1'])
    boss = mx(r - s['boss_r'], (s['boss_z0'] if s['mount'] != 'clamp' else s['plate_z0']) - z, z - s['plate_z1'])
    from functools import reduce
    f = reduce(np.minimum, [ridge, wall, pilot, plate, boss])
    for k in range(s['n_ribs']):
        a = 2*np.pi*(k + 0.5)/s['n_ribs']; xs = r*np.cos(phi - a); ys = r*np.sin(phi - a)
        f = np.minimum(f, mx(np.abs(ys) - s['rib_t']/2, s['boss_r'] - 1 - xs, xs - (r_root - 1), s['rib_z0'] - z, z - (s['plate_z0'] + 1)))
    if s['mount'] == 'clamp':
        # thin centre plate with a plain hole; a small hole saw screwed on from inside clamps it to the arbor
        f = np.maximum(f, -mx(r - (s['boss_r'] - 1), z - (s['plate_z1'] - s['clamp_t'])))
        f = np.maximum(f, -(r - (s['unf_maj']/2 + s['unf_clear'] + 0.15)))
        depth_ok = s['clamp_t']
    else:
        depth_ok = s['slot_depth']
    # 5/8"-18 UNF internal thread through the boss (cut = the arbor stud grown by the clearance)
    P = s['unf_pitch']; Rm = s['unf_maj']/2; depth = 0.6134*P
    u = z - P*(phi % (2*np.pi))/(2*np.pi); dd = np.abs((u + P/2) % P - P/2)
    r_stud = np.maximum(Rm - depth, Rm - np.maximum(dd - P/16, 0)*np.sqrt(3))
    lead = np.maximum(0, 1.0 - (s['plate_z1'] - z)) + np.maximum(0, 1.0 - (z - s['boss_z0']))   # 45 deg lead-in at both ends
    if s['mount'] != 'clamp':
        f = np.maximum(f, -(r - (r_stud + s['unf_clear'] + lead)))
    # radial drive-pin slots in the arbor face
    for k in range(s['n_slots']):
        a = 2*np.pi*k/s['n_slots']; xs = r*np.cos(phi - a); ys = r*np.sin(phi - a)
        slot = mx(np.hypot(np.maximum(np.abs(xs - (s['slot_r0'] + s['slot_r1'])/2) - (s['slot_r1'] - s['slot_r0'])/2, 0), ys) - s['slot_w']/2,
                  (s['plate_z1'] - depth_ok - (0.5 if s['mount'] == 'clamp' else 0)) - z)
        f = np.maximum(f, -slot)
    return f

if __name__ == '__main__':
    A = cv.axis; s = S; p = B
    ri = p['spigot_od']/2 - p['wall']
    r_ax = A(0, s['plate_r'] + 1, 0.6, [s['boss_r'], s['plate_r'], s['slot_r0'], s['slot_r1'], ri - s['pilot_gap']]
             + list(np.arange(6.6, 8.6, 0.12)) + list(np.arange(ri - 4.5, ri + 0.2, 0.25)))
    z_ax = A(s['thr_lo'] - 1, s['plate_z1'] + 1, 0.5, [s['thr_lo'], s['thr_hi'], s['pilot_z0'], s['plate_z0'], s['plate_z1'],
             s['boss_z0'], s['plate_z1'] - s['slot_depth'], s['plate_z1'] - s['clamp_t'], s['plate_z1'] - s['clamp_t'] - 0.5] + list(np.arange(s['boss_z0'] - 0.2, s['plate_z1'] + 0.2, 0.15)), eps=0.02)
    ex = []
    for k in range(s['n_slots']): ex += list(np.arange(360*k/s['n_slots'] - 16, 360*k/s['n_slots'] + 16.01, 0.5))
    for k in range(s['n_ribs']): c = 360*(k + 0.5)/s['n_ribs']; ex += list(np.arange(c - 5, c + 5.01, 0.4))
    phi = np.radians(np.unique(np.round(np.r_[np.arange(0, 360.01, 2.0), np.array(ex) % 360, [0, 360]], 4)))
    for mode, fn in (('thread', 'B_sanding_adapter_threaded.stl'), ('clamp', 'B_sanding_adapter_clamp.stl')):
        s['mount'] = mode
        f = lambda prof, r, ph, z: adapter_field(r, ph, z)
        xyz, tri, vol = cv.mesh_polar(f, np.array([(400, 0), (401, 0), (401, 1)]), r_ax, phi, z_ax)
        print('sanding adapter (%s): %d tris, %.1f cm3, manifold errors %s' % (mode, len(tri), vol/1000, cv.check_manifold(tri)))
        xyz[:, 2] = s['plate_z1'] - xyz[:, 2]; xyz[:, 0] *= -1          # arbor face on the bed
        cv.write_stl(fn, xyz, tri)
