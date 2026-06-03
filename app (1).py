#!/usr/bin/env python3
"""
=============================================================================
  College Smart Grid — AI Load Decision Agent
  A* Search-Based Energy Distribution Optimiser
  Single-file version: Python backend + HTML frontend, all in one.

  Run:  python app.py
  Open: http://127.0.0.1:5000
=============================================================================
"""

import heapq, json, time, os, sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
except ImportError:
    print("\n[ERROR] Flask not installed. Run:\n  pip install flask flask-cors\n")
    sys.exit(1)


class Priority(Enum):
    CRITICAL   = 1
    HIGH       = 2
    MEDIUM     = 3
    LOW        = 4
    NEGLIGIBLE = 5

PRIORITY_LABELS = {
    Priority.CRITICAL:   "Critical",
    Priority.HIGH:       "High",
    Priority.MEDIUM:     "Medium",
    Priority.LOW:        "Low",
    Priority.NEGLIGIBLE: "Negligible",
}

PRIORITY_COLORS = {
    Priority.CRITICAL:   "#ef4444",
    Priority.HIGH:       "#f97316",
    Priority.MEDIUM:     "#f59e0b",
    Priority.LOW:        "#22c55e",
    Priority.NEGLIGIBLE: "#718096",
}

PEAK_RATE    = 9.50
OFFPEAK_RATE = 5.00


@dataclass
class Zone:
    id: str
    name: str
    priority: Priority
    max_load_kw: float
    current_load_kw: float
    can_reduce: bool
    reduction_step_kw: float
    min_load_kw: float
    description: str = ""

    @property
    def reducible_kw(self) -> float:
        return max(0.0, self.current_load_kw - self.min_load_kw)


DEFAULT_ZONES: List[Zone] = [
    Zone("server_room",  "Server Room / Data Center",     Priority.CRITICAL,   15.0, 14.0, False, 0.0,  14.0, "UPS-protected, never shed"),
    Zone("main_lab",     "Computer / Science Labs",       Priority.CRITICAL,   20.0, 18.0, False, 0.0,  18.0, "Active experiments & sessions"),
    Zone("admin_block",  "Administration Block",          Priority.HIGH,       10.0,  8.0, True,  1.0,   4.0, "Office equipment, AC"),
    Zone("classrooms",   "Active Classrooms",             Priority.HIGH,       25.0, 20.0, True,  2.0,  10.0, "Projectors, AC, lighting"),
    Zone("library",      "Library",                       Priority.HIGH,       12.0, 10.0, True,  1.0,   5.0, "Computers, AC, lighting"),
    Zone("cafeteria",    "Cafeteria / Canteen",           Priority.MEDIUM,      8.0,  7.0, True,  1.5,   2.0, "Kitchen equipment, fans"),
    Zone("sports",       "Sports Complex / Gym",          Priority.MEDIUM,      6.0,  5.0, True,  1.0,   1.0, "Lighting, equipment"),
    Zone("corridors",    "Corridors & Common Areas",      Priority.LOW,         5.0,  4.0, True,  1.0,   1.0, "General lighting"),
    Zone("unused_rooms", "Unused / Vacant Classrooms",    Priority.LOW,         4.0,  3.5, True,  1.0,   0.0, "Standby loads"),
    Zone("deco_lighting","Decorative / Display Lights",   Priority.NEGLIGIBLE,  3.0,  2.5, True,  0.5,   0.0, "Aesthetic lighting, screens"),
]


@dataclass
class GridState:
    allocations: Dict[str, float]
    g_cost: float
    h_cost: float
    actions: List[str] = field(default_factory=list)
    solar_used_kw: float = 0.0

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: "GridState") -> bool:
        return self.f_cost < other.f_cost

    def total_load(self) -> float:
        return sum(self.allocations.values())

    def state_key(self) -> str:
        return json.dumps({k: round(v, 2) for k, v in sorted(self.allocations.items())})


class SmartGridAStar:
    def __init__(self, zones: List[Zone], grid_capacity_kw: float,
                 solar_available_kw: float = 0.0, is_peak_hour: bool = True):
        self.zones              = {z.id: z for z in zones}
        self.grid_capacity_kw   = grid_capacity_kw
        self.solar_available_kw = solar_available_kw
        self.is_peak_hour       = is_peak_hour
        self.rate               = PEAK_RATE if is_peak_hour else OFFPEAK_RATE
        self.visited: set       = set()
        self.iterations         = 0

    def _energy_cost(self, load_kw: float, solar_kw: float) -> float:
        return max(0.0, load_kw - solar_kw) * self.rate

    def _heuristic(self, allocations: Dict[str, float], solar_used: float) -> float:
        total     = sum(allocations.values())
        grid_draw = max(0.0, total - solar_used)
        return grid_draw * self.rate * 0.1

    def _overload_penalty(self, total_kw: float) -> float:
        if total_kw <= self.grid_capacity_kw:
            return 0.0
        return (total_kw - self.grid_capacity_kw) * self.rate * 50

    def _priority_violation_penalty(self, zone: Zone, allocated: float) -> float:
        if allocated < zone.min_load_kw:
            factor = {
                Priority.CRITICAL:   10000,
                Priority.HIGH:       1000,
                Priority.MEDIUM:     100,
                Priority.LOW:        10,
                Priority.NEGLIGIBLE: 1,
            }[zone.priority]
            return (zone.min_load_kw - allocated) * factor
        return 0.0

    def _generate_successors(self, state: GridState,
                              remaining_zones: List[str]) -> List[GridState]:
        if not remaining_zones:
            return []
        zone_id  = remaining_zones[0]
        zone     = self.zones[zone_id]
        rest     = remaining_zones[1:]

        if zone.can_reduce:
            candidates, level = [], zone.current_load_kw
            while level >= zone.min_load_kw - 0.001:
                candidates.append(round(level, 2))
                level -= zone.reduction_step_kw
            if zone.min_load_kw not in candidates:
                candidates.append(zone.min_load_kw)
        else:
            candidates = [zone.current_load_kw]

        successors = []
        for alloc in candidates:
            new_allocs          = dict(state.allocations)
            new_allocs[zone_id] = alloc
            total               = sum(new_allocs.values())
            solar_used          = min(self.solar_available_kw, total)
            cost = (self._energy_cost(total, solar_used)
                    + self._overload_penalty(total)
                    + self._priority_violation_penalty(zone, alloc))

            if alloc < zone.current_load_kw:
                desc = (f"Reduced <b>{zone.name}</b> by "
                        f"<span class='kw'>{zone.current_load_kw - alloc:.1f} kW</span> "
                        f"({zone.current_load_kw:.1f} → {alloc:.1f} kW)")
            else:
                desc = (f"Maintained <b>{zone.name}</b> at full load "
                        f"<span class='kw'>{alloc:.1f} kW</span>")

            successors.append(GridState(
                allocations   = new_allocs,
                g_cost        = cost,
                h_cost        = self._heuristic(new_allocs, solar_used),
                actions       = state.actions + [desc],
                solar_used_kw = solar_used,
            ))
        return successors

    def solve(self) -> Optional[GridState]:
        zone_order = sorted(self.zones.values(),
                            key=lambda z: (z.priority.value, -z.current_load_kw))
        zone_ids   = [z.id for z in zone_order]
        initial    = GridState(allocations={}, g_cost=0.0, h_cost=0.0)
        heap: List[Tuple[float, GridState]] = [(0.0, initial)]
        best: Optional[GridState] = None

        while heap:
            self.iterations += 1
            _, current = heapq.heappop(heap)
            key = current.state_key()
            if key in self.visited:
                continue
            self.visited.add(key)

            remaining = [z for z in zone_ids if z not in current.allocations]
            if not remaining:
                if best is None or current.f_cost < best.f_cost:
                    best = current
                continue

            for succ in self._generate_successors(current, remaining):
                if succ.state_key() not in self.visited:
                    heapq.heappush(heap, (succ.f_cost, succ))

            if len(heap) > 5000:
                heap = heapq.nsmallest(2000, heap)
                heapq.heapify(heap)

        return best


def build_result(state: GridState, zones: Dict[str, Zone],
                 grid_capacity_kw: float, solar_available_kw: float,
                 is_peak_hour: bool, iterations: int) -> dict:
    rate           = PEAK_RATE if is_peak_hour else OFFPEAK_RATE
    total_demand   = sum(z.current_load_kw for z in zones.values())
    total_alloc    = state.total_load()
    solar_used     = min(solar_available_kw, total_alloc)
    grid_draw      = max(0.0, total_alloc - solar_used)
    hourly_cost    = grid_draw * rate
    is_overload    = total_alloc > grid_capacity_kw
    load_pct       = (total_alloc / grid_capacity_kw) * 100

    zone_results = []
    for zid, zone in zones.items():
        alloc     = state.allocations.get(zid, zone.current_load_kw)
        reduction = zone.current_load_kw - alloc
        status    = "full"
        if reduction > 0.05:
            status = "minimal" if alloc <= zone.min_load_kw + 0.05 else "reduced"
        zone_results.append({
            "id":             zid,
            "name":           zone.name,
            "priority":       PRIORITY_LABELS[zone.priority],
            "priority_level": zone.priority.value,
            "priority_color": PRIORITY_COLORS[zone.priority],
            "requested_kw":   round(zone.current_load_kw, 2),
            "allocated_kw":   round(alloc, 2),
            "reduction_kw":   round(reduction, 2),
            "min_load_kw":    round(zone.min_load_kw, 2),
            "status":         status,
            "can_reduce":     zone.can_reduce,
            "description":    zone.description,
        })
    zone_results.sort(key=lambda z: z["priority_level"])

    saved = total_demand - total_alloc
    lp    = load_pct
    verdict_parts = []
    if lp > 95:
        verdict_parts.append("⚠️ Grid is operating near maximum capacity — immediate load shedding recommended.")
    elif lp > 80:
        verdict_parts.append("🟡 Grid load is elevated. Monitor closely and consider reducing non-essential zones.")
    else:
        verdict_parts.append("✅ Grid is operating within safe limits.")
    if saved > 0:
        verdict_parts.append(f"A* optimisation saved {saved:.1f} kW by redistributing non-critical loads.")
    if solar_used > 0:
        verdict_parts.append(f"Solar energy is contributing {solar_used:.1f} kW, reducing grid draw and cost.")
    verdict_parts.append(
        f"{'Peak-hour tariff (₹9.50/kWh)' if is_peak_hour else 'Off-peak tariff (₹5.00/kWh)'} is active.")

    return {
        "success": True,
        "summary": {
            "total_demand_kw":    round(total_demand, 2),
            "total_allocated_kw": round(total_alloc, 2),
            "total_saved_kw":     round(saved, 2),
            "grid_capacity_kw":   round(grid_capacity_kw, 2),
            "load_percentage":    round(load_pct, 1),
            "solar_available_kw": round(solar_available_kw, 2),
            "solar_used_kw":      round(solar_used, 2),
            "grid_draw_kw":       round(grid_draw, 2),
            "hourly_cost_inr":    round(hourly_cost, 2),
            "rate_per_kwh":       rate,
            "is_peak_hour":       is_peak_hour,
            "is_overload":        is_overload,
            "overload_kw":        round(max(0.0, total_alloc - grid_capacity_kw), 2),
            "astar_iterations":   iterations,
            "renewable_pct":      round((solar_used / total_alloc * 100) if total_alloc > 0 else 0, 1),
        },
        "zones":      zone_results,
        "actions":    state.actions,
        "ai_verdict": " ".join(verdict_parts),
    }


def run_optimization(payload: dict) -> dict:
    try:
        grid_cap  = float(payload["grid_capacity_kw"])
        solar_kw  = float(payload.get("solar_available_kw", 0))
        is_peak   = bool(payload.get("is_peak_hour", True))
        raw_zones = payload.get("zones", [])
        if not raw_zones:
            return {"success": False, "error": "No zones provided."}

        zones = []
        for z in raw_zones:
            zones.append(Zone(
                id                = z["id"],
                name              = z["name"],
                priority          = Priority(int(z["priority_level"])),
                max_load_kw       = float(z["max_load_kw"]),
                current_load_kw   = float(z["current_load_kw"]),
                can_reduce        = bool(z["can_reduce"]),
                reduction_step_kw = float(z.get("reduction_step_kw", 0.5)),
                min_load_kw       = float(z["min_load_kw"]),
                description       = z.get("description", ""),
            ))

        solver  = SmartGridAStar(zones, grid_cap, solar_kw, is_peak)
        t0      = time.time()
        result  = solver.solve()
        elapsed = round(time.time() - t0, 3)

        if result is None:
            return {"success": False, "error": "A* could not find a valid allocation."}

        out = build_result(result, {z.id: z for z in zones},
                           grid_cap, solar_kw, is_peak, solver.iterations)
        out["elapsed_seconds"] = elapsed
        return out
    except (KeyError, ValueError, TypeError) as e:
        return {"success": False, "error": str(e)}


def get_default_zones() -> list:
    return [
        {
            "id":                z.id,
            "name":              z.name,
            "priority_level":    z.priority.value,
            "priority_label":    PRIORITY_LABELS[z.priority],
            "max_load_kw":       z.max_load_kw,
            "current_load_kw":   z.current_load_kw,
            "can_reduce":        z.can_reduce,
            "reduction_step_kw": z.reduction_step_kw,
            "min_load_kw":       z.min_load_kw,
            "description":       z.description,
        }
        for z in DEFAULT_ZONES
    ]


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>College Smart Grid — AI Load Decision Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#0a0f1a;--bg2:#0f1729;--bg3:#141e33;--panel:#111827;
  --border:#1e2d4a;--border2:#243352;
  --teal:#00c9a7;--teal2:#00e5c3;--blue:#3b82f6;--indigo:#6366f1;
  --amber:#f59e0b;--red:#ef4444;--green:#22c55e;--orange:#f97316;
  --muted:#4a5a78;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --fh:'Syne',sans-serif;--fb:'DM Sans',sans-serif;--fm:'DM Mono',monospace;
  --r:12px;--r2:8px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--fb);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;z-index:0;
  background-image:linear-gradient(rgba(0,201,167,.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(0,201,167,.03) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none}
body::after{content:'';position:fixed;inset:0;z-index:0;
  background:radial-gradient(ellipse 80% 60% at 50% -10%,rgba(59,130,246,.08) 0%,transparent 60%);
  pointer-events:none}
.wrap{position:relative;z-index:1;max-width:1340px;margin:0 auto;padding:0 24px 60px}
header{padding:32px 0 28px;border-bottom:1px solid var(--border);margin-bottom:36px;
  display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:16px}
.logo-icon{width:52px;height:52px;background:linear-gradient(135deg,var(--teal),var(--blue));
  border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;
  box-shadow:0 0 20px rgba(0,201,167,.25);flex-shrink:0}
.logo h1{font-family:var(--fh);font-size:22px;font-weight:800;letter-spacing:-.5px;
  background:linear-gradient(90deg,#fff 0%,var(--teal2) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo p{font-size:12px;color:var(--text3);font-family:var(--fm);margin-top:2px}
.badges{display:flex;gap:8px;align-items:center}
.badge{padding:5px 12px;border-radius:20px;font-size:11px;font-family:var(--fm);font-weight:500;border:1px solid;letter-spacing:.5px}
.bt{border-color:var(--teal);color:var(--teal);background:rgba(0,201,167,.08)}
.bb{border-color:var(--blue);color:var(--blue);background:rgba(59,130,246,.08)}
#dot{width:8px;height:8px;border-radius:50%;background:var(--muted);margin-right:4px;display:inline-block;transition:background .3s}
#dot.live{background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.mg{display:grid;grid-template-columns:430px 1fr;gap:28px;align-items:start}
@media(max-width:1000px){.mg{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}
.ph{padding:18px 22px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.ph h2{font-family:var(--fh);font-size:14px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text2)}
.pb{padding:22px}
.fg{margin-bottom:18px}
.fg label{display:block;font-size:11px;font-family:var(--fm);font-weight:500;color:var(--text3);letter-spacing:.8px;text-transform:uppercase;margin-bottom:7px}
.fg label span{color:var(--teal);margin-left:4px}
.fr{display:grid;grid-template-columns:1fr 1fr;gap:14px}
input[type=number],input[type=text],select{width:100%;background:var(--bg2);border:1px solid var(--border2);border-radius:var(--r2);padding:10px 14px;font-family:var(--fm);font-size:13px;color:var(--text);outline:none;transition:border-color .2s,box-shadow .2s}
input[type=number]:focus,select:focus{border-color:var(--teal);box-shadow:0 0 0 3px rgba(0,201,167,.1)}
.trow{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:var(--bg2);border:1px solid var(--border2);border-radius:var(--r2)}
.trow span{font-size:13px;color:var(--text2)}
.tgl{position:relative;width:44px;height:24px;cursor:pointer}
.tgl input{opacity:0;width:0;height:0}
.ts{position:absolute;inset:0;background:var(--bg3);border-radius:12px;transition:.3s}
.ts::before{content:'';position:absolute;width:18px;height:18px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.3s}
.tgl input:checked+.ts{background:var(--teal)}
.tgl input:checked+.ts::before{transform:translateX(20px)}
.ztw{overflow-x:auto}
.zt{width:100%;border-collapse:collapse;font-size:12px}
.zt thead tr{background:var(--bg3)}
.zt th{padding:10px 12px;text-align:left;font-family:var(--fm);font-size:10px;font-weight:500;letter-spacing:.8px;text-transform:uppercase;color:var(--text3);border-bottom:1px solid var(--border);white-space:nowrap}
.zt td{padding:9px 12px;border-bottom:1px solid rgba(30,45,74,.5);vertical-align:middle;color:var(--text2)}
.zt tr:last-child td{border-bottom:none}
.zt tr:hover td{background:rgba(0,201,167,.03)}
.zt input[type=number]{padding:6px 8px;font-size:12px;width:75px;text-align:right}
.znc{min-width:160px}
.zn{font-weight:500;color:var(--text);font-size:12px}
.zd{font-size:10px;color:var(--text3);margin-top:1px}
.pp{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:20px;font-family:var(--fm);font-size:10px;font-weight:500;border:1px solid;white-space:nowrap}
.rtgl{width:14px;height:14px;accent-color:var(--teal);cursor:pointer}
.rbtn{width:100%;padding:14px 24px;background:linear-gradient(135deg,var(--teal),#00a884);border:none;border-radius:var(--r2);font-family:var(--fh);font-size:15px;font-weight:700;letter-spacing:.5px;color:#0a1a14;cursor:pointer;transition:transform .15s,box-shadow .15s,opacity .15s;box-shadow:0 4px 20px rgba(0,201,167,.3);margin-top:4px}
.rbtn:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,201,167,.4)}
.rbtn:active{transform:translateY(0)}
.rbtn:disabled{opacity:.5;cursor:not-allowed;transform:none}
#rp{display:none}
#rp.vis{display:block;animation:su .4s ease}
@keyframes su{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.kg{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:14px;margin-bottom:24px}
.kc{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);padding:16px;position:relative;overflow:hidden}
.kc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.kc.teal::before{background:var(--teal)}.kc.blue::before{background:var(--blue)}
.kc.amber::before{background:var(--amber)}.kc.green::before{background:var(--green)}
.kc.red::before{background:var(--red)}.kc.indigo::before{background:var(--indigo)}
.kl{font-family:var(--fm);font-size:10px;color:var(--text3);letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px}
.kv{font-family:var(--fh);font-size:22px;font-weight:700;color:var(--text)}
.kv.teal{color:var(--teal)}.kv.blue{color:var(--blue)}.kv.amber{color:var(--amber)}
.kv.green{color:var(--green)}.kv.red{color:var(--red)}.kv.indigo{color:var(--indigo)}
.ks{font-size:11px;color:var(--text3);margin-top:4px}
.lbw{margin-bottom:24px}
.lbh{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.lbl{font-family:var(--fm);font-size:11px;color:var(--text3)}
.lbp{font-family:var(--fh);font-size:18px;font-weight:700}
.lbb{width:100%;height:12px;background:var(--bg3);border-radius:6px;overflow:hidden}
.lbf{height:100%;border-radius:6px;transition:width 1s cubic-bezier(.4,0,.2,1);background:linear-gradient(90deg,var(--teal),var(--blue))}
.lbf.danger{background:linear-gradient(90deg,var(--amber),var(--red))}
.vb{background:var(--bg2);border:1px solid var(--border2);border-left:3px solid var(--teal);border-radius:var(--r2);padding:14px 18px;font-size:13px;line-height:1.65;color:var(--text2);margin-bottom:24px}
.vb.warning{border-left-color:var(--amber)}.vb.danger{border-left-color:var(--red)}
.zrg{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:14px;margin-bottom:24px}
.zrc{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);padding:16px;transition:border-color .2s,transform .15s;position:relative;overflow:hidden}
.zrc:hover{transform:translateY(-2px);border-color:var(--border2)}
.zrc.reduced{border-left:3px solid var(--amber)}
.zrc.minimal{border-left:3px solid var(--red)}
.zrc.full{border-left:3px solid var(--green)}
.zrh{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.zrn{font-family:var(--fh);font-size:13px;font-weight:700;color:var(--text)}
.zrs{font-family:var(--fm);font-size:10px;font-weight:500;padding:2px 8px;border-radius:10px;white-space:nowrap;flex-shrink:0}
.zrs.full{background:rgba(34,197,94,.15);color:var(--green)}
.zrs.reduced{background:rgba(245,158,11,.15);color:var(--amber)}
.zrs.minimal{background:rgba(239,68,68,.15);color:var(--red)}
.zbb{width:100%;height:6px;background:var(--bg3);border-radius:3px;overflow:hidden;margin:8px 0 6px}
.zbf{height:100%;border-radius:3px;transition:width .8s}
.znum{display:flex;justify-content:space-between;font-family:var(--fm);font-size:11px}
.za{color:var(--text);font-weight:500}.zq{color:var(--text3)}
.zred{margin-top:6px;font-size:11px;color:var(--amber);font-family:var(--fm)}
.alog{background:var(--bg);border:1px solid var(--border);border-radius:var(--r2);padding:16px 18px;font-family:var(--fm);font-size:12px;line-height:1.8;color:var(--text2);max-height:300px;overflow-y:auto}
.alog .ll{padding:3px 0;border-bottom:1px solid rgba(30,45,74,.4)}
.alog .ll:last-child{border-bottom:none}
.alog .kw{color:var(--teal);font-weight:500}
.lts{color:var(--text3);margin-right:8px}
.mr{display:flex;gap:16px;flex-wrap:wrap;padding:12px 16px;background:var(--bg3);border-radius:var(--r2);margin-top:20px}
.mi{font-family:var(--fm);font-size:11px;color:var(--text3)}
.mi span{color:var(--teal);margin-left:4px}
.st{font-family:var(--fh);font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text3);margin:20px 0 12px;display:flex;align-items:center;gap:8px}
.st::after{content:'';flex:1;height:1px;background:var(--border)}
.ph2{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:64px 32px;text-align:center;color:var(--text3)}
.phi{font-size:52px;margin-bottom:16px;opacity:.4}
.ph2 h3{font-family:var(--fh);font-size:18px;color:var(--text2);margin-bottom:8px}
.ph2 p{font-size:13px;max-width:320px;line-height:1.6}
.oa{background:rgba(239,68,68,.1);border:1px solid var(--red);border-radius:var(--r2);padding:12px 16px;font-size:13px;color:#fca5a5;margin-bottom:20px;display:flex;gap:10px;align-items:center}
.si{display:flex;gap:10px;align-items:center;padding:10px 14px;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:var(--r2);margin-bottom:16px;font-size:12px;color:var(--amber);font-family:var(--fm)}
.toast{position:fixed;bottom:28px;right:28px;z-index:9999;background:var(--panel);border:1px solid var(--red);color:var(--text);border-radius:var(--r2);padding:14px 20px;font-size:13px;box-shadow:0 4px 24px rgba(0,0,0,.4);transform:translateY(80px);opacity:0;transition:all .3s;max-width:360px}
.toast.show{transform:translateY(0);opacity:1}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
@media(max-width:640px){.kg{grid-template-columns:repeat(2,1fr)}.zrg{grid-template-columns:1fr}.fr{grid-template-columns:1fr}header{flex-direction:column;align-items:flex-start}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="logo">
    <div class="logo-icon">⚡</div>
    <div>
      <h1>College Smart Grid</h1>
      <p>AI-Based Load Decision Agent &nbsp;·&nbsp; A* Optimisation Engine</p>
    </div>
  </div>
  <div class="badges">
    <span class="badge bt">A* Search</span>
    <span class="badge bb"><span id="dot"></span>Ready</span>
  </div>
</header>

<div class="mg">
  <div>
    <div class="panel" style="margin-bottom:20px">
      <div class="ph"><span>🏫</span><h2>Grid Configuration</h2></div>
      <div class="pb">
        <div class="fr">
          <div class="fg">
            <label>Grid Capacity <span>(kW)</span></label>
            <input type="number" id="grid-cap" value="120" min="10" step="1"/>
          </div>
          <div class="fg">
            <label>Solar Available <span>(kW)</span></label>
            <input type="number" id="solar-kw" value="15" min="0" step="0.5"/>
          </div>
        </div>
        <div class="fg">
          <label>Time Mode</label>
          <div class="trow">
            <span id="peak-lbl">🌞 Peak Hour (₹9.50/kWh)</span>
            <label class="tgl">
              <input type="checkbox" id="peak-tog" checked/>
              <span class="ts"></span>
            </label>
          </div>
        </div>
        <button class="rbtn" id="run-btn" onclick="runOpt()">⚡ Run A* Optimisation</button>
      </div>
    </div>

    <div class="panel">
      <div class="ph"><span>🗂️</span><h2>College Zones — Input</h2></div>
      <div class="pb" style="padding:0">
        <div class="ztw">
          <table class="zt">
            <thead>
              <tr>
                <th>Zone</th>
                <th>Priority</th>
                <th>Load (kW)</th>
                <th>Min (kW)</th>
                <th>Reduce?</th>
              </tr>
            </thead>
            <tbody id="ztbody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div>
    <div class="panel" id="ph-panel">
      <div class="ph2">
        <div class="phi">🔋</div>
        <h3>Awaiting Optimisation</h3>
        <p>Configure your college zones and grid parameters, then click <strong>Run A* Optimisation</strong> to compute the ideal energy distribution strategy.</p>
      </div>
    </div>

    <div id="rp">
      <div class="oa" id="oa" style="display:none">⚠️ <span id="oa-msg"></span></div>
      <div class="si" id="si" style="display:none">☀️ <span id="si-msg"></span></div>
      <div class="kg" id="kg"></div>
      <div class="lbw">
        <div class="lbh">
          <span class="lbl">GRID UTILISATION</span>
          <span class="lbp" id="lbp">—</span>
        </div>
        <div class="lbb"><div class="lbf" id="lbf" style="width:0%"></div></div>
      </div>
      <div class="vb" id="vb"></div>
      <div class="st">Zone Allocation Results</div>
      <div class="zrg" id="zrg"></div>
      <div class="st">A* Decision Log</div>
      <div class="alog" id="alog"></div>
      <div class="mr" id="mr"></div>
    </div>
  </div>
</div>
</div>

<div class="toast" id="toast"></div>

<script>
const PRIORITY_META={
  1:{label:"Critical",  color:"#ef4444",bg:"rgba(239,68,68,0.12)"},
  2:{label:"High",      color:"#f97316",bg:"rgba(249,115,22,0.12)"},
  3:{label:"Medium",    color:"#f59e0b",bg:"rgba(245,158,11,0.12)"},
  4:{label:"Low",       color:"#22c55e",bg:"rgba(34,197,94,0.12)"},
  5:{label:"Negligible",color:"#718096",bg:"rgba(113,128,150,0.12)"},
};
let ZONES=[];

window.addEventListener("DOMContentLoaded",async()=>{
  document.getElementById("peak-tog").addEventListener("change",()=>{
    const on=document.getElementById("peak-tog").checked;
    document.getElementById("peak-lbl").textContent=on?"🌞 Peak Hour (₹9.50/kWh)":"🌙 Off-Peak (₹5.00/kWh)";
  });
  const res=await fetch("/api/defaults");
  const d=await res.json();
  ZONES=d.zones;
  renderTable(ZONES);
});

function renderTable(zones){
  const tb=document.getElementById("ztbody");
  tb.innerHTML="";
  zones.forEach(z=>{
    const pm=PRIORITY_META[z.priority_level];
    const tr=document.createElement("tr");
    tr.innerHTML=`
      <td class="znc">
        <div class="zn">${z.name}</div>
        <div class="zd">${z.description}</div>
      </td>
      <td><span class="pp" style="color:${pm.color};border-color:${pm.color};background:${pm.bg}">${pm.label}</span></td>
      <td><input type="number" class="zli" data-id="${z.id}" value="${z.current_load_kw}" min="${z.min_load_kw}" max="${z.max_load_kw}" step="0.5"/></td>
      <td><input type="number" class="zmi" data-id="${z.id}" value="${z.min_load_kw}" min="0" max="${z.max_load_kw}" step="0.5"/></td>
      <td style="text-align:center"><input type="checkbox" class="rtgl zri" data-id="${z.id}" ${z.can_reduce?"checked":""} ${z.priority_level<=2?"disabled":""}/></td>`;
    tb.appendChild(tr);
  });
}

function gatherPayload(){
  return{
    grid_capacity_kw:parseFloat(document.getElementById("grid-cap").value)||120,
    solar_available_kw:parseFloat(document.getElementById("solar-kw").value)||0,
    is_peak_hour:document.getElementById("peak-tog").checked,
    zones:ZONES.map(z=>({
      id:z.id,name:z.name,priority_level:z.priority_level,
      max_load_kw:z.max_load_kw,
      current_load_kw:parseFloat(document.querySelector(`.zli[data-id="${z.id}"]`)?.value??z.current_load_kw),
      min_load_kw:parseFloat(document.querySelector(`.zmi[data-id="${z.id}"]`)?.value??z.min_load_kw),
      can_reduce:document.querySelector(`.zri[data-id="${z.id}"]`)?.checked??z.can_reduce,
      reduction_step_kw:z.reduction_step_kw,description:z.description,
    })),
  };
}

async function runOpt(){
  const btn=document.getElementById("run-btn");
  const dot=document.getElementById("dot");
  btn.disabled=true;btn.textContent="⏳ Running A* Search…";dot.className="";
  try{
    const r=await fetch("/api/optimize",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(gatherPayload())});
    const d=await r.json();
    if(!d.success)throw new Error(d.error||"Optimisation failed");
    renderResults(d);dot.className="live";
  }catch(e){showToast("Error: "+e.message);}
  finally{btn.disabled=false;btn.textContent="⚡ Run A* Optimisation";}
}

function renderResults(data){
  const s=data.summary;
  document.getElementById("ph-panel").style.display="none";
  const rp=document.getElementById("rp");rp.classList.add("vis");

  const oa=document.getElementById("oa");
  if(s.is_overload){oa.style.display="flex";document.getElementById("oa-msg").textContent=`Grid overload: ${s.overload_kw} kW above capacity (${s.grid_capacity_kw} kW). Immediate load shedding required.`;}
  else oa.style.display="none";

  const si=document.getElementById("si");
  if(s.solar_used_kw>0){si.style.display="flex";document.getElementById("si-msg").textContent=`Solar contributing ${s.solar_used_kw} kW — saving ₹${(s.solar_used_kw*s.rate_per_kwh).toFixed(2)}/hr on grid draw`;}
  else si.style.display="none";

  const kpis=[
    {l:"Total Allocated",v:`${s.total_allocated_kw} kW`,s:`of ${s.total_demand_kw} kW demanded`,c:"teal"},
    {l:"Hourly Cost",v:`₹${s.hourly_cost_inr}`,s:`@ ₹${s.rate_per_kwh}/kWh (${s.is_peak_hour?"peak":"off-peak"})`,c:"amber"},
    {l:"Load Saved",v:`${s.total_saved_kw} kW`,s:"via A* redistribution",c:"green"},
    {l:"Grid Draw",v:`${s.grid_draw_kw} kW`,s:`${s.renewable_pct}% renewable`,c:s.is_overload?"red":"blue"},
    {l:"Solar Used",v:`${s.solar_used_kw} kW`,s:`of ${s.solar_available_kw} kW avail.`,c:"indigo"},
    {l:"A* Iterations",v:s.astar_iterations,s:`solved in ${data.elapsed_seconds}s`,c:"teal"},
  ];
  document.getElementById("kg").innerHTML=kpis.map(k=>`<div class="kc ${k.c}"><div class="kl">${k.l}</div><div class="kv ${k.c}">${k.v}</div><div class="ks">${k.s}</div></div>`).join("");

  const pct=Math.min(s.load_percentage,100);
  const bar=document.getElementById("lbf");
  bar.style.width="0%";
  setTimeout(()=>bar.style.width=pct+"%",50);
  bar.className="lbf"+(s.load_percentage>90?" danger":"");
  const lbp=document.getElementById("lbp");
  lbp.textContent=s.load_percentage+"%";
  lbp.style.color=s.load_percentage>90?"var(--red)":s.load_percentage>75?"var(--amber)":"var(--teal)";

  const vb=document.getElementById("vb");
  vb.innerHTML=data.ai_verdict;
  vb.className="vb"+(s.load_percentage>90?" danger":s.load_percentage>75?" warning":"");

  document.getElementById("zrg").innerHTML=data.zones.map(z=>{
    const pm=PRIORITY_META[z.priority_level];
    const ap=z.requested_kw>0?Math.round(z.allocated_kw/z.requested_kw*100):100;
    const bc=z.status==="full"?"#22c55e":z.status==="reduced"?"#f59e0b":"#ef4444";
    return`<div class="zrc ${z.status}">
      <div class="zrh">
        <div>
          <span class="pp" style="color:${pm.color};border-color:${pm.color};background:${pm.bg};font-size:9px">${pm.label}</span>
          <div class="zrn" style="margin-top:6px">${z.name}</div>
        </div>
        <span class="zrs ${z.status}">${z.status.toUpperCase()}</span>
      </div>
      <div class="zbb"><div class="zbf" style="width:${ap}%;background:${bc}"></div></div>
      <div class="znum"><span class="za">${z.allocated_kw} kW allocated</span><span class="zq">${z.requested_kw} kW requested</span></div>
      ${z.reduction_kw>0?`<div class="zred">↓ Reduced by ${z.reduction_kw} kW</div>`:""}
    </div>`;
  }).join("");

  const ts=new Date().toLocaleTimeString();
  document.getElementById("alog").innerHTML=data.actions.length
    ?data.actions.map(a=>`<div class="ll"><span class="lts">[${ts}]</span>${a}</div>`).join("")
    :`<div class="ll" style="color:var(--green)">[${ts}] ✅ All zones at full capacity — no redistribution needed.</div>`;

  document.getElementById("mr").innerHTML=`
    <span class="mi">Algorithm: <span>A* Search</span></span>
    <span class="mi">Zones: <span>${data.zones.length}</span></span>
    <span class="mi">Capacity: <span>${s.grid_capacity_kw} kW</span></span>
    <span class="mi">Solar: <span>${s.solar_available_kw} kW</span></span>
    <span class="mi">Tariff: <span>₹${s.rate_per_kwh}/kWh</span></span>
    <span class="mi">Mode: <span>${s.is_peak_hour?"Peak":"Off-Peak"}</span></span>`;
}

function showToast(msg){
  const t=document.getElementById("toast");t.textContent=msg;t.classList.add("show");
  setTimeout(()=>t.classList.remove("show"),4000);
}
</script>
</body>
</html>"""


app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/api/defaults")
def api_defaults():
    return jsonify({"zones": get_default_zones()})

@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"success": False, "error": "Empty request body"}), 400
    result = run_optimization(payload)
    return jsonify(result), (200 if result.get("success") else 422)

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  College Smart Grid — AI Load Decision Agent")
    print("  A* Search-Based Energy Distribution Optimiser")
    print()
    print("  Open your browser at:  http://127.0.0.1:5000")
    print("=" * 60)
    print()
    app.run(debug=False, port=5000)
