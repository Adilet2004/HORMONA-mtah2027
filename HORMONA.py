"""Frame-level ball possession for More than a Hack 2027 -- a rule-based state machine.

    python HORMONA.py                       # writes evaluation/results/HORMONA.csv
    python HORMONA.py --out somewhere/

Put this file in the root of the competition repository (next to games.py). numpy only.

Pipeline
--------
1. Read the tactical ATD (players + ball), convert to metres.
2. Fill short ball gaps by linear interpolation; long gaps stay unknown.
3. Dead ball (D):
     a. ball tracked outside the pitch,
     b. long low-activity stretches (everyone walking, ball still or lost),
     c. Smart Tagging set pieces: from the restart strike, walk back while the ball is still,
     d. kick-offs: from the half start (or the goal) to the kick-off strike.
4. Possession inside each live stretch: a state machine. The restarting team owns the ball;
   ownership changes only when the other team shows *control* -- one long control episode, or two
   separate touches (a pass) with no touch by the owner in between. The change is back-dated to
   the start of the new team's first control episode. Loose / contested balls stay with the owner.
5. Clean-up: drop very short dead stretches and very short live stretches between two stoppages.

All thresholds are football-motivated constants in PARAMS below. They were NOT fitted to the public
ground truth (the rules forbid that).
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = 25
H, A, NONE = 0, 1, -1

PARAMS = dict(
    # ball
    ball_gap_max=12,          # interpolate ball gaps up to this many frames (~0.5 s)
    ball_smooth=3,            # half-width of the moving average on ball position
    # control
    r_ctrl=1.6,               # metres: a player this close to the ball can control it
    margin=0.5,               # metres: and must be this much closer than the nearest opponent
    ep_gap=4,                 # frames of 'nobody' allowed inside one control episode
    ep_min=3,                 # frames: shorter episodes are noise (deflections, flicks)
    k_ctrl=12,                # frames: one episode this long is a controlled reception (~0.5 s)
    pass_window=4 * FPS,      # two new-team episodes within this span = a completed pass
    # dead ball
    out_margin=0.5,           # metres beyond the line before the ball counts as out
    out_min=3,                # consecutive frames out
    v_kick=6.0,               # m/s: ball speed of a restart strike / pass
    v_still=2.0,              # m/s: ball below this is 'still'
    act_window=2 * FPS,       # rolling window for team activity
    v_stop=1.1 / 3.6,         # m/s (= 1.1 km/h): mean player speed below this = stoppage (near standstill)
    stop_min=4 * FPS,         # low activity must last this long to count
    pre_dead_max=45 * FPS,    # walk back at most this far from a restart
    pre_dead_min=2 * FPS,     # a restart has at least this much dead ball before it
    restart_search=60 * FPS,  # look this far ahead of an out-of-play for the restart
    d_min=FPS,                # dead stretches shorter than this are dropped
    alive_min=int(1.5 * FPS), # live stretches shorter than this between two D become D
)


# ----------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------

def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([H if t == home_id else A for t in slots], dtype=np.int8)


def speed_scale(meta_path):
    """Factor that turns the ATD speed channel into m/s (the metadata declares km/h)."""
    for sensor in ET.parse(meta_path).getroot().iter("Sensor"):
        if sensor.get("id") == "speed":
            unit = (sensor.findtext(".//Unit") or "").strip().lower()
            if unit in ("kph", "km/h", "kmh"):
                return 1 / 3.6
    return 1.0


def read_atd(raw_path, n_slots, n_frames, pitch):
    """ball (N,2) metres, players (N,S,2) metres, speeds (N,S) m/s; NaN where missing."""
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)
    pos = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    spd = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                continue
            b = parts[2].split(",")
            if b[0][:1] != "N":
                ball[i, 0] = float(b[0]) * length
                ball[i, 1] = float(b[1]) * width
            for s, g in enumerate(parts[1].split(";")[:n_slots]):
                if not g or g[0] == "N":
                    continue
                v = g.split(",")
                pos[i, s, 0] = float(v[0]) * length
                pos[i, s, 1] = float(v[1]) * width
                if len(v) > 2 and v[2][:1] != "N":
                    spd[i, s] = float(v[2])
    return ball, pos, spd


def read_tags(path):
    out = []
    for inst in ET.parse(path).getroot().iter("instance"):
        tags = {}
        for lab in inst.findall("label"):
            tags[(lab.findtext("group") or "").strip().lower()] = (lab.findtext("text") or "").strip()
        out.append(dict(code=(inst.findtext("code") or "").strip().upper(),
                        start=int(round(float(inst.findtext("start")) * FPS)),
                        end=int(round(float(inst.findtext("end")) * FPS)),
                        team=team_from_tag(tags.get("team", "")),
                        type=tags.get("type", "").lower()))
    return out


def team_from_tag(text):
    t = text.lower()
    if "barc" in t:
        return H
    if "athl" in t or "atl" in t:
        return A
    return NONE


# ----------------------------------------------------------------------------------------------
# signal helpers
# ----------------------------------------------------------------------------------------------

def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def interpolate_ball(ball, max_gap, blocks):
    out = ball.copy()
    observed = ~np.isnan(ball[:, 0])
    for _, first, last in blocks:
        idx = np.arange(first, last + 1)
        obs = observed[first:last + 1]
        if obs.sum() < 2:
            continue
        for s, e in runs_of(~obs):
            s0, e0 = first + s, first + e
            if s0 == first or e0 > last or (e0 - s0) > max_gap:
                continue
            for k in range(2):
                out[s0:e0, k] = np.interp(idx[s:e], [s0 - 1, e0], [ball[s0 - 1, k], ball[e0, k]])
    return out


def moving_mean(x, w):
    """Centred moving mean over +/- w ignoring NaN; NaN where the window has no data."""
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out[cnt == 0] = np.nan
    out[~ok] = np.nan
    return out


def ball_speed(ball, w):
    sx, sy = moving_mean(ball[:, 0], w), moving_mean(ball[:, 1], w)
    v = np.full(len(ball), np.nan)
    v[1:-1] = np.hypot(sx[2:] - sx[:-2], sy[2:] - sy[:-2]) * FPS / 2.0
    return v


def nearest_by_team(ball, pos, team):
    """distance (N,2) of the nearest home / away player to the ball; inf where unknown."""
    d = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    return np.stack([d[:, team == H].min(axis=1), d[:, team == A].min(axis=1)], axis=1)


# ----------------------------------------------------------------------------------------------
# dead ball
# ----------------------------------------------------------------------------------------------

def find_strike(vb, near, lo, hi, p):
    """First frame in [lo, hi) where the ball accelerates past v_kick with a player near it."""
    lo, hi = max(lo, 1), min(hi, len(vb))
    for t in range(lo, hi):
        if vb[t] >= p["v_kick"] and (np.isnan(vb[t - 1]) or vb[t - 1] < p["v_kick"]):
            if near[max(0, t - 10):t + 1].min() < 2.5:
                return t
    return None


def walk_back_still(vb, t, floor, p):
    """From a restart at t, go back while the ball is still or lost. Returns the dead start."""
    lo = max(floor, t - p["pre_dead_max"])
    s = t - 1
    while s > lo and (np.isnan(vb[s]) or vb[s] < p["v_still"]):
        s -= 1
    return min(s + 1, t - p["pre_dead_min"])


def dead_mask(ball_obs, ball, vb, dist, spd, tags, blocks, pitch, p):
    n = len(ball)
    length, width = pitch
    near = dist.min(axis=1)
    dead = np.zeros(n, dtype=bool)
    restart_team = {}  # frame of restart -> team

    # (d) kick-offs and goals
    goals = sorted(t["end"] for t in tags if t["code"] == "GOALS")
    for t in tags:
        if t["code"] == "SET PIECES" and "kick off" in t["type"]:
            strike = find_strike(vb, near, t["start"], t["end"] + FPS, p) or t["start"]
            if "start" in t["type"]:
                floor = next((f for _, f, l in blocks if f <= strike <= l), strike)
            else:
                prev = [g for g in goals if g < strike]
                floor = prev[-1] if prev else strike - p["pre_dead_max"]
            dead[floor:strike] = True
            restart_team[strike] = t["team"]

    # (c) other set pieces
    for t in tags:
        if t["code"] == "SET PIECES" and "kick off" not in t["type"]:
            strike = find_strike(vb, near, t["start"], t["end"] + FPS, p)
            if strike is None:
                continue
            s = walk_back_still(vb, strike, 0, p)
            dead[s:strike] = True
            team = t["team"]
            if team == NONE:
                team = H if dist[strike, H] <= dist[strike, A] else A
            restart_team[strike] = team

    # (a) ball out of the pitch
    m = p["out_margin"]
    out = ((ball_obs[:, 0] < -m) | (ball_obs[:, 0] > length + m) |
           (ball_obs[:, 1] < -m) | (ball_obs[:, 1] > width + m))
    for s, e in runs_of(out):
        if e - s < p["out_min"]:
            continue
        strike = find_strike(vb, near, e, e + p["restart_search"], p)
        stop = strike if strike is not None else min(n, e + p["pre_dead_min"])
        dead[s:stop] = True
        if strike is not None and strike not in restart_team:
            restart_team[strike] = H if dist[strike, H] <= dist[strike, A] else A

    # (b) low activity
    mean_spd = np.nanmean(spd, axis=1)
    if not np.all(np.isnan(mean_spd)):
        act = moving_mean(mean_spd, p["act_window"])
        calm = (act < p["v_stop"]) & ~(vb >= p["v_kick"])
        for s, e in runs_of(calm):
            if e - s >= p["stop_min"]:
                strike = find_strike(vb, near, e - FPS, e + 10 * FPS, p)
                stop = strike if strike is not None else e
                dead[s:stop] = True

    return dead, restart_team


# ----------------------------------------------------------------------------------------------
# possession state machine
# ----------------------------------------------------------------------------------------------

def control_codes(dist, p):
    dh, da = dist[:, H], dist[:, A]
    c = np.full(len(dist), NONE, dtype=np.int8)
    c[(dh <= p["r_ctrl"]) & (dh + p["margin"] <= da)] = H
    c[(da <= p["r_ctrl"]) & (da + p["margin"] <= dh)] = A
    return c


def episodes(ctrl, s, e, p):
    """[(team, start, stop, n_frames)] control episodes within [s, e)."""
    out = []
    cur, st, last, cnt = NONE, None, None, 0
    for t in range(s, e):
        c = ctrl[t]
        if c == NONE:
            if cur != NONE and t - last > p["ep_gap"]:
                if cnt >= p["ep_min"]:
                    out.append((cur, st, last + 1, cnt))
                cur = NONE
            continue
        if c == cur:
            last, cnt = t, cnt + 1
        else:
            if cur != NONE and cnt >= p["ep_min"]:
                out.append((cur, st, last + 1, cnt))
            cur, st, last, cnt = c, t, t, 1
    if cur != NONE and cnt >= p["ep_min"]:
        out.append((cur, st, last + 1, cnt))
    return out


def possession(ctrl, s, e, start_team, p):
    """Labels for one live stretch [s, e)."""
    eps = episodes(ctrl, s, e, p)
    owner = start_team if start_team != NONE else (eps[0][0] if eps else NONE)
    labels = np.full(e - s, owner, dtype=np.int8)
    pending = None
    for team, st, sp, cnt in eps:
        if owner == NONE:
            owner = team
            labels[:] = owner
        if team == owner:
            pending = None
            continue
        change_at = None
        if cnt >= p["k_ctrl"]:
            change_at = pending[1] if pending else st
        elif pending is not None and st - pending[1] <= p["pass_window"]:
            change_at = pending[1]
        if change_at is not None:
            owner = team
            labels[change_at - s:] = owner
            pending = None
        else:
            pending = (team, st)
    return labels


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def clean_dead(dead, blocks, p):
    for _, first, last in blocks:
        seg = dead[first:last + 1]
        for s, e in runs_of(seg):
            if e - s < p["d_min"] and s > 0:
                seg[s:e] = False
        for s, e in runs_of(~seg):
            if e - s < p["alive_min"] and s > 0 and e < len(seg):
                seg[s:e] = True
    return dead


def solve(p=PARAMS, game=GAME, data_dir=None, verbose=True):
    meta, raw, tag_path = require(["atd_meta", "atd_raw", "smart_tagging"], game, data_dir)
    n = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)
    team = team_of_each_track(meta)

    if verbose:
        print("reading tracking ...")
    ball_obs, pos, spd = read_atd(raw, len(team), n, pitch)
    spd *= speed_scale(meta)
    ball = interpolate_ball(ball_obs, p["ball_gap_max"], blocks)
    vb = ball_speed(ball, p["ball_smooth"])
    dist = nearest_by_team(ball, pos, team)
    tags = read_tags(tag_path)
    if verbose:
        print("ball observed {:.1f}%, after interpolation {:.1f}%; {} tags".format(
            100 * np.mean(~np.isnan(ball_obs[:, 0])), 100 * np.mean(~np.isnan(ball[:, 0])),
            len(tags)))

    dead, restart_team = dead_mask(ball_obs, ball, vb, dist, spd, tags, blocks, pitch, p)
    dead = clean_dead(dead, blocks, p)
    ctrl = control_codes(dist, p)

    labels = np.full(n, "X", dtype="<U1")
    restarts = sorted(restart_team)
    for _, first, last in blocks:
        seg_dead = dead[first:last + 1]
        labels[first:last + 1][seg_dead] = "D"
        prev_owner = NONE
        for s, e in runs_of(~seg_dead):
            s0, e0 = first + s, first + e
            k = [r for r in restarts if abs(r - s0) <= FPS]
            start_team = restart_team[k[0]] if k else NONE
            if start_team == NONE:
                eps = episodes(ctrl, s0, min(e0, s0 + 3 * FPS), p)
                start_team = eps[0][0] if eps else prev_owner
            lab = possession(ctrl, s0, e0, start_team, p)
            labels[s0:e0] = np.where(lab == H, "H", np.where(lab == A, "A", "X"))
            prev_owner = lab[-1] if len(lab) else prev_owner

    if verbose:
        for half, first, last in blocks:
            seg = labels[first:last + 1]
            poss = len(runs_of(np.isin(seg, ["H", "A"])))  # live stretches
            changes = int(np.sum((seg[1:] != seg[:-1]) & np.isin(seg[1:], ["H", "A"])))
            print("half {}: H {:.1f}%  A {:.1f}%  D {:.1f}%  X {:.1f}%  live stretches {}  "
                  "possession starts {}".format(
                      half, *(100 * np.mean(seg == c) for c in "HADX"), poss, changes))
    return labels


def write_submission(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="HORMONA", help="file name = team name on the board")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = ap.parse_args()
    labels = solve()
    path = write_submission(os.path.join(args.out, args.team + ".csv"), labels)
    print("wrote", path)
    print("next:  python evaluation/validate_submission.py", path)
    print("       python evaluation/run_evaluation.py")
