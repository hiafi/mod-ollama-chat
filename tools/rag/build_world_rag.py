#!/usr/bin/env python3
"""Generate mod-ollama-chat RAG JSON (quests and NPC services) from an AzerothCore world database export.

Standard library only. See README.md for the export queries that produce the input directory.

    build_world_rag.py --inputs <dir> --out <dir> [--max-level 60] [--maps 0,1] [--faction both]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

MAP_FILE_NAMES = {0: "eastern_kingdoms", 1: "kalimdor"}

# ChrRaces bit (1 << (race - 1)) masks as used by quest_template.AllowableRaces.
ALLIANCE_MASK = 1 | 4 | 8 | 64 | 1024      # human, dwarf, night elf, gnome, draenei
HORDE_MASK = 2 | 16 | 32 | 128 | 512       # orc, undead, tauren, troll, blood elf

NPCFLAG_TRAINER_CLASS = 32
NPCFLAG_TRAINER_PROFESSION = 64
NPCFLAG_FLIGHTMASTER = 8192
NPCFLAG_INNKEEPER = 65536

AREA_FLAG_CAPITAL = 0x200000     # AreaTable.dbc flags
PREFER_SLACK = 0.15             # zone_of: a known zone this close to the deepest box still wins
DEPTH_SLACK = 0.1               # zone_of: a capital this close to the deepest box still wins

SUBZONE_RADIUS_YARDS = 200.0   # a point is 'in' a learned subzone with two of its quest givers this close
SUBZONE_MIN_QUESTS = 3
SUBZONE_MAX_MEDIAN_SPREAD = 300.0
SUBZONE_MAX_P90_SPREAD = 700.0

NEARBY_YARDS = 1500.0          # a quest target counts as "near" the giver inside this radius
LANDMARK_NEAR_YARDS = 400.0    # "near X" up to here
LANDMARK_MAX_YARDS = 800.0     # "about N yards <dir> of X" up to here, nothing beyond
SPAWN_CLUSTER_YARDS = 250.0
NPC_CLUSTER_YARDS = 300.0
RIGHT_NEXT_TO_YARDS = 60.0
SECOND_CLUSTER_RATIO = 0.6
MAX_SOURCES = 3

COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]

JUNK_TITLE = re.compile(r"deprecated|^<|^\[|\bunused\b|^zz|\btest\b|\bdnd\b|\(old\)|\bold\b$", re.IGNORECASE)
PLACE_PREFIX = re.compile(r"^(near|in|about \d+ yards [a-z-]+ of) ")
JUNK_NODE = re.compile(r"^(quest|transport|test|generic|flavor|development)\b|->| - ", re.IGNORECASE)
JUNK_TARGET = re.compile(r"credit|\btrigger\b|\bdnd\b|\bvisual\b|\(.*\bqc\b.*\)", re.IGNORECASE)

# ---------------------------------------------------------------------------------------------
# Reading inputs
# ---------------------------------------------------------------------------------------------

_UNESCAPE = {"t": "\t", "n": "\n", "\\": "\\", "0": "\0", "b": "\b", "r": "\r", "Z": "\x1a"}


def unescape_field(value: str) -> str | None:
    """Undo `mysql -B` escaping. A bare NULL is None; everything else is a string."""
    if value == "NULL":
        return None
    if "\\" not in value:
        return value
    return re.sub(r"\\(.)", lambda m: _UNESCAPE.get(m.group(1), m.group(1)), value)


def parse_tsv(text: str) -> list[dict[str, str | None]]:
    """Parse a `mysql -B` dump with a header row into a list of dicts."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) < len(header):
            cells += ["NULL"] * (len(header) - len(cells))
        rows.append({h: unescape_field(c) for h, c in zip(header, cells)})
    return rows


def read_tsv(path: Path) -> list[dict[str, str | None]]:
    return parse_tsv(path.read_text(encoding="utf-8", errors="replace"))


def as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Dbc:
    """A WDBC file: fixed-width uint32 fields and a string block."""

    def __init__(self, data: bytes):
        magic, self.count, self.fields, self.record_size, string_size = struct.unpack_from("<4s4I", data, 0)
        if magic != b"WDBC":
            raise ValueError("not a WDBC file")
        if self.record_size != self.fields * 4:
            raise ValueError("unexpected record size")
        self._data = data
        self._records_at = 20
        self._strings_at = 20 + self.count * self.record_size
        if self._strings_at + string_size > len(data):
            raise ValueError("truncated DBC")

    def record(self, index: int) -> tuple[int, ...]:
        return struct.unpack_from("<%dI" % self.fields, self._data, self._records_at + index * self.record_size)

    def records(self):
        for i in range(self.count):
            yield self.record(i)

    def string(self, offset: int) -> str:
        start = self._strings_at + offset
        end = self._data.index(b"\0", start)
        return self._data[start:end].decode("utf-8", errors="replace")


def u2f(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


# ---------------------------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------------------------

# 3.3.5a layouts (src/server/shared/DataStores/DBCfmt.h):
#   AreaTable     niiiixxxxxissssssssssssssssxiiiiixxx  -> 0 ID, 1 ContinentID, 2 ParentAreaID, 4 Flags, 11 AreaName (enUS)
#   WorldMapArea  xinxffffixx  -> 1 MapID, 2 AreaID, 3 AreaName offset, 4 LocLeft, 5 LocRight, 6 LocTop, 7 LocBottom
#   TaxiNodes     nifffssssssssssssssssxii  -> 0 ID, 1 ContinentID, 2..4 X Y Z, 5 Name (enUS), 22/23 mount creature
# World X is north and Y is west. WorldMapArea LocTop/LocBottom bound X and LocLeft/LocRight bound Y.


class Geography:
    def __init__(self, area_dbc: Dbc, wma_dbc: Dbc, taxi_dbc: Dbc, maps: set[int]):
        self.area_names: dict[int, str] = {}
        self.capital_areas: set[int] = set()
        self.parent: dict[int, int] = {}
        for r in area_dbc.records():
            if r[2]:
                self.parent[r[0]] = r[2]
            name = area_dbc.string(r[11])
            if name:
                self.area_names[r[0]] = name
                if r[4] & AREA_FLAG_CAPITAL:
                    self.capital_areas.add(r[0])
        self.boxes: dict[int, list[tuple[float, int, float, float, float, float]]] = defaultdict(list)
        for r in wma_dbc.records():
            map_id, area_id = r[1], r[2]
            if map_id not in maps or area_id == 0:
                continue                      # area 0 is the continent-wide map
            left, right, top, bottom = (u2f(v) for v in r[4:8])
            size = (top - bottom) * (left - right)
            if size > 0 and area_id in self.area_names:
                self.boxes[map_id].append((size, area_id, top, bottom, left, right))
        for lst in self.boxes.values():
            lst.sort()
        self.nodes: dict[int, list[tuple[str, float, float, int | None]]] = defaultdict(list)
        zone_ids = {a for lst in self.boxes.values() for _s, a, *_r in lst}
        by_name = {self.area_names[a].lower(): a for a in zone_ids}
        for r in taxi_dbc.records():
            if r[1] not in maps:
                continue
            name = taxi_dbc.string(r[5])
            x, y = u2f(r[2]), u2f(r[3])
            if (not name or JUNK_NODE.search(name) or (r[22] == 0 and r[23] == 0)
                    or abs(x) > 17000 or abs(y) > 17000):
                continue                      # no mount for either faction: a test or retired node
            self.nodes[r[1]].append((landmark_name(name), x, y, _zone_hint(name, by_name)))

    def keep_nodes_near(self, spawns_by_map: dict[int, list[Point]], radius: float = 150.0) -> None:
        """Keep only taxi nodes with a flight master spawn within `radius`.

        TaxiNodes.dbc also holds test, transport and quest-flight nodes (and a hidden one in Goldshire called
        'Filming'); a real flight point has a flight master standing at it.
        """
        for map_id in list(self.nodes):
            pts = spawns_by_map.get(map_id, [])
            self.nodes[map_id] = [n for n in self.nodes[map_id]
                                  if any(math.hypot(n[1] - x, n[2] - y) <= radius for x, y in pts)]

    def set_subzones(self, points: dict[int, list[tuple[float, float, int]]]) -> None:
        self.subzone_points = points

    def subzone_at(self, map_id: int, x: float, y: float) -> tuple[int, str] | None:
        """A named subzone the point is inside, learned from quest givers (see learn_subzones)."""
        near = Counter()
        for px, py, area_id in getattr(self, "subzone_points", {}).get(map_id, ()):
            if math.hypot(x - px, y - py) <= SUBZONE_RADIUS_YARDS:
                near[area_id] += 1
        if not near:
            return None
        area_id, n = near.most_common(1)[0]
        if n < 2 or n * 2 < sum(near.values()):
            return None
        return area_id, self.area_names[area_id]

    def ancestors(self, area_id: int) -> list[int]:
        """The area and its parents, innermost first (a subzone's parent is its zone)."""
        chain = []
        while area_id and area_id not in chain:
            chain.append(area_id)
            area_id = self.parent.get(area_id, 0)
        return chain

    def zone_of(self, map_id: int, x: float, y: float, prefer: int | None = None) -> tuple[int, str] | None:
        """The zone whose map box the point sits deepest inside.

        Boxes are map-render rectangles: neighbouring zones overlap and capital boxes reach into the zone around
        them (Northshire Abbey is 14 yd inside the Stormwind City box; the Crossroads is inside Mulgore's box).
        'Smallest box' therefore misnames border points. Depth is the distance to the nearest box edge as a
        fraction of that axis' size, so a point near the edge of one box and in the middle of another goes to
        the second. A capital city's box within DEPTH_SLACK of the deepest one wins (a city sits inside a zone's
        box). `prefer` is an area id (or a subzone of one) known from other data, such as a quest's sort zone:
        when the point is inside that zone's box and not much shallower than the deepest, it decides.
        Points within a few hundred yards of a border can still be wrong.
        """
        hits = []
        for size, area_id, top, bottom, left, right in self.boxes.get(map_id, ()):
            if bottom <= x <= top and right <= y <= left:
                depth = min((top - x) / (top - bottom), (x - bottom) / (top - bottom),
                            (left - y) / (left - right), (y - right) / (left - right))
                hits.append((depth, size, area_id))
        if not hits:
            return None
        best_depth = max(h[0] for h in hits)
        # A capital inside a zone's box wins unless the point is clearly nearer the edge of the capital's box.
        capitals = [h for h in hits if h[2] in self.capital_areas and h[0] >= best_depth - DEPTH_SLACK]
        if capitals:
            depth, size, area_id = min(capitals, key=lambda h: (h[1], h[2]))
            return area_id, self.area_names[area_id]
        if prefer:
            wanted = set(self.ancestors(prefer))
            known = [h for h in hits if h[2] in wanted and h[0] >= best_depth - PREFER_SLACK]
            if known:
                area_id = max(known, key=lambda h: (h[0], -h[1], -h[2]))[2]
                return area_id, self.area_names[area_id]
        depth, size, area_id = max(hits, key=lambda h: (h[0], -h[1], -h[2]))
        return area_id, self.area_names[area_id]

    def landmark_near(self, map_id: int, x: float, y: float) -> tuple[str, float, float, float, int | None] | None:
        """Nearest flight-path town: (name, distance, its x, its y, its zone id or None), within LANDMARK_MAX_YARDS."""
        best = None
        for name, nx, ny, zone in self.nodes.get(map_id, ()):
            d = math.hypot(x - nx, y - ny)
            if best is None or d < best[1]:
                best = (name, d, nx, ny, zone)
        if best is not None and best[1] <= LANDMARK_MAX_YARDS:
            return best
        return None


def _zone_hint(taxi_name: str, zones_by_name: dict[str, int]) -> int | None:
    """'Darkshire, Duskwood' -> the Duskwood area id; 'Booty Bay, Stranglethorn' -> Stranglethorn Vale (unique
    prefix). None when the node name has no zone part or it matches no (or several) zones."""
    if "," not in taxi_name:
        return None
    hint = taxi_name.split(",", 1)[1].strip().lower()
    if hint in zones_by_name:
        return zones_by_name[hint]
    found = {a for n, a in zones_by_name.items() if hint and n.startswith(hint)}
    return found.pop() if len(found) == 1 else None


def landmark_name(taxi_name: str) -> str:
    """'Sentinel Hill, Westfall' -> 'Sentinel Hill'."""
    return taxi_name.split(",")[0].strip()


def compass(dx: float, dy: float) -> str:
    """8-point compass of the vector (dx north, dy west)."""
    bearing = math.degrees(math.atan2(-dy, dx)) % 360.0     # east = -dy
    return COMPASS[int((bearing + 22.5) // 45) % 8]


def round_distance(d: float) -> int:
    return int(math.floor(d / 50.0 + 0.5)) * 50


def relative_phrase(frm: tuple[float, float], to: tuple[float, float], label: str) -> str:
    """'about 300 yards south-east of the quest giver', or 'right next to the quest giver'."""
    d = math.hypot(to[0] - frm[0], to[1] - frm[1])
    if d < RIGHT_NEXT_TO_YARDS:
        return f"right next to {label}"
    return f"about {max(round_distance(d), 50)} yards {compass(to[0] - frm[0], to[1] - frm[1])} of {label}"


def landmark_phrase(geo: Geography, map_id: int, x: float, y: float, zone_name: str | None = None,
                    near_only: bool = False) -> str | None:
    """'near Sentinel Hill' / 'about 650 yards west of Lakeshire' / None. Never guesses a subzone."""
    lm = geo.landmark_near(map_id, x, y)
    if lm is None or (near_only and lm[1] > LANDMARK_NEAR_YARDS):
        return None
    name, d, nx, ny, _zone = lm
    if zone_name and (name.lower() in zone_name.lower() or zone_name.lower() in name.lower()):
        return None                               # 'Stormwind City, near Stormwind' says nothing
    if d <= LANDMARK_NEAR_YARDS:
        return f"near {name}"
    return relative_phrase((nx, ny), (x, y), name)


# ---------------------------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------------------------

Point = tuple[float, float]


def cluster_points(points: list[Point], radius: float = SPAWN_CLUSTER_YARDS) -> list[tuple[Point, int]]:
    """Greedy density clustering. Returns [(centroid, size)] sorted by size, largest first.

    Repeatedly takes the unassigned point with the most unassigned neighbours inside `radius` (first one on a
    tie, so the result is deterministic for a given input order) and claims every unassigned point inside it.
    """
    pts = list(points)
    alive = [True] * len(pts)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x, y) in enumerate(pts):
        grid[(int(x // radius), int(y // radius))].append(i)

    def neighbours(i: int) -> list[int]:
        cx, cy = int(pts[i][0] // radius), int(pts[i][1] // radius)
        out = []
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), ()):
                    if alive[j] and math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) <= radius:
                        out.append(j)
        return out

    clusters = []
    remaining = len(pts)
    while remaining:
        best_i, best_n = -1, []
        for i in range(len(pts)):
            if not alive[i]:
                continue
            n = neighbours(i)
            if len(n) > len(best_n):
                best_i, best_n = i, n
        members = best_n
        for j in members:
            alive[j] = False
        remaining -= len(members)
        clusters.append(((sum(pts[j][0] for j in members) / len(members),
                          sum(pts[j][1] for j in members) / len(members)), len(members)))
    clusters.sort(key=lambda c: -c[1])      # stable: ties keep discovery order
    return clusters


# ---------------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------------

def clean(text: str | None) -> str:
    """One line, no double quotes (the module's config tooling forbids them), single spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace('"', "'")).strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def level_band(level: int) -> str:
    lo = ((max(level, 1) - 1) // 10) * 10 + 1
    return f"lvl{lo}-{lo + 9}"


def side_of_mask(mask: int) -> str:
    a, h = bool(mask & ALLIANCE_MASK), bool(mask & HORDE_MASK)
    if a and not h:
        return "alliance"
    if h and not a:
        return "horde"
    return "both"


def join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def words_keywords(*texts: str | None) -> list[str]:
    seen, out = set(), []
    for t in texts:
        t = clean(t)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


# The module scores bag-of-words cosine with raw term frequency, so a question like
# 'where is the warrior trainer in northshire' matches every trainer entry on is/in/trainer and the one
# distinctive word counts once. Repeating the distinctive terms (places, names) in `keywords` raises their weight
# in the entry vector; KEYWORD_REPEAT is how many copies of them are added (measured: 1 is best).
KEYWORD_REPEAT = 1                  # --keyword-repeat


def boosted(*terms: str | None) -> list[str]:
    out = []
    for t in terms:
        t = clean(t)
        if t:
            out.extend([t] * KEYWORD_REPEAT)
    return out


def plural(word: str) -> str:
    """Plain English plural of one word: wolf -> wolves, candy -> candies, miner -> miners. Players type plurals
    ('the kobolds', 'wolves') and the module matches whole words only, with no stemming."""
    w = word.lower()
    if len(w) < 3 or not w.isalpha() or w.endswith("s"):
        return word
    if w.endswith(("x", "z", "ch", "sh")):
        return word + "es"
    if w.endswith("y") and w[-2] not in "aeiou":
        return word[:-1] + "ies"
    if w.endswith("f"):
        return word[:-1] + "ves"
    if w.endswith("fe"):
        return word[:-2] + "ves"
    return word + "s"


def plural_terms(name: str) -> str:
    """'Kobold Miner' -> 'Kobolds Miners' (a keyword, so the plural a player types counts as a match)."""
    return " ".join(plural(w) for w in name.split() if w.isalpha() and len(w) >= 3)


def assert_no_double_quotes(entries: list[dict]) -> None:
    for e in entries:
        for key in ("id", "title", "content"):
            if '"' in e[key]:
                raise ValueError(f"double quote in {e['id']}.{key}")
        for kw in e["keywords"] + e["tags"]:
            if '"' in kw:
                raise ValueError(f"double quote in {e['id']} keywords/tags")


# ---------------------------------------------------------------------------------------------
# World data
# ---------------------------------------------------------------------------------------------

class World:
    def __init__(self, inputs: Path, maps: set[int]):
        self.maps = maps
        self.creature_name: dict[int, str] = {}
        self.creature_row: dict[int, dict] = {}
        for r in read_tsv(inputs / "creature_template.tsv"):
            e = as_int(r["entry"])
            self.creature_row[e] = r
            self.creature_name[e] = clean(r["name"])
        self.go_name = {as_int(r["entry"]): clean(r["name"]) for r in read_tsv(inputs / "gameobject_template.tsv")}
        self.item_name = {as_int(r["entry"]): clean(r["name"]) for r in read_tsv(inputs / "item_template.tsv")}
        self.creature_spawns: dict[int, dict[int, list[Point]]] = defaultdict(lambda: defaultdict(list))
        for r in read_tsv(inputs / "creature.tsv"):
            m = as_int(r["map"])
            if m in maps:
                self.creature_spawns[as_int(r["id"])][m].append((as_float(r["position_x"]), as_float(r["position_y"])))
        self.go_spawns: dict[int, dict[int, list[Point]]] = defaultdict(lambda: defaultdict(list))
        for r in read_tsv(inputs / "gameobject.tsv"):
            m = as_int(r["map"])
            if m in maps:
                self.go_spawns[as_int(r["id"])][m].append((as_float(r["position_x"]), as_float(r["position_y"])))
        self.quest_rows = {as_int(r["ID"]): r for r in read_tsv(inputs / "quest_template.tsv")}
        self.quest_addon = {as_int(r["ID"]): r for r in read_tsv(inputs / "quest_template_addon.tsv")}
        self.starters = _pairs(read_tsv(inputs / "creature_queststarter.tsv"))
        self.go_starters = _pairs(read_tsv(inputs / "gameobject_queststarter.tsv"))
        self.enders = _pairs(read_tsv(inputs / "creature_questender.tsv"))
        self._inputs = inputs
        self._loot = None

    def flight_master_spawns(self) -> dict[int, list[Point]]:
        out: dict[int, list[Point]] = defaultdict(list)
        for entry, row in self.creature_row.items():
            if as_int(row["npcflag"]) & NPCFLAG_FLIGHTMASTER:
                for m, pts in self.creature_spawns.get(entry, {}).items():
                    out[m].extend(pts)
        return out

    def loot_sources(self) -> dict[int, dict[int, tuple[int, float]]]:
        """item -> {creature entry: (quest_required, chance)}. Direct rows plus one level of reference."""
        if self._loot is not None:
            return self._loot
        ref_items: dict[int, dict[int, tuple[int, float]]] = defaultdict(dict)   # ref entry -> item -> (qr, chance)
        for r in read_tsv(self._inputs / "reference_loot_template.tsv"):
            item = as_int(r["Item"])
            if item:
                ref_items[as_int(r["Entry"])][item] = (as_int(r["QuestRequired"]), abs(as_float(r["Chance"])))
        out: dict[int, dict[int, tuple[int, float]]] = defaultdict(dict)

        def add(item, creature, qr, chance):
            cur = out[item].get(creature)
            if cur is None or (qr, chance) > cur:
                out[item][creature] = (qr, chance)

        for r in read_tsv(self._inputs / "creature_loot_template.tsv"):
            creature = as_int(r["Entry"])
            if creature not in self.creature_spawns:
                continue
            qr, chance, ref = as_int(r["QuestRequired"]), abs(as_float(r["Chance"])), as_int(r["Reference"])
            if ref:
                for item, (rqr, rchance) in ref_items.get(ref, {}).items():
                    add(item, creature, qr or rqr, rchance or chance)
            else:
                add(as_int(r["Item"]), creature, qr, chance)
        self._loot = out
        return out


def _pairs(rows) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        out[as_int(r["quest"])].append(as_int(r["id"]))
    return out


def learn_creature_sides(world: World) -> dict[int, str]:
    """creature_template.faction -> 'alliance' | 'horde', learned from race-restricted quest givers/enders.

    A faction template counts as a side only with at least 3 votes and 90% agreement. No FactionTemplate.dbc is
    needed, and a faction that cannot be settled stays unlabeled rather than guessed.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for quest, row in world.quest_rows.items():
        side = side_of_mask(as_int(row["AllowableRaces"]))
        if as_int(row["AllowableRaces"]) == 0 or side == "both":
            continue
        for creature in world.starters.get(quest, []) + world.enders.get(quest, []):
            cr = world.creature_row.get(creature)
            if cr:
                votes[as_int(cr["faction"])][side] += 1
    sides = {}
    for faction, c in votes.items():
        total = sum(c.values())
        side, n = c.most_common(1)[0]
        if total >= 3 and n / total >= 0.9:
            sides[faction] = side
    return sides


def learn_subzones(world: World, geo: Geography) -> dict[int, list[tuple[float, float, int]]]:
    """Subzone names from quest sort ids: points of quest givers whose quests are sorted into a small subzone.

    No client file maps a point to a subzone, but QuestSortID names one for many quests (Northshire Valley,
    Deathknell, Coldridge Valley, Valley of Trials, ...). A sort id counts only if it is a child area, at least
    3 quests and 2 givers use it, and the givers sit close together (median distance from their centre
    <= 300 yd, 90th percentile <= 700 yd); instances and battlegrounds spread over thousands of yards and drop out.
    """
    zones = {a for lst in geo.boxes.values() for _s, a, *_r in lst}
    per_area: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))    # area -> map -> givers
    quests_of: Counter = Counter()
    for qid, row in world.quest_rows.items():
        sid = as_int(row["QuestSortID"])
        if sid <= 0 or sid not in geo.area_names or sid in zones or not geo.parent.get(sid):
            continue
        for giver in world.starters.get(qid, []):
            spawns = world.creature_spawns.get(giver, {})
            if spawns and sum(len(p) for p in spawns.values()) <= 3:
                quests_of[sid] += 1
                for m in spawns:
                    per_area[sid][m].add(giver)
    out: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    for sid, by_map in sorted(per_area.items()):
        if quests_of[sid] < SUBZONE_MIN_QUESTS:
            continue
        for m, givers in sorted(by_map.items()):
            pts = [pt for g in sorted(givers) for pt in world.creature_spawns[g][m]]
            if len(givers) < 2 or len(pts) < 2:
                continue
            cx, cy = sorted(p[0] for p in pts)[len(pts) // 2], sorted(p[1] for p in pts)[len(pts) // 2]
            dist = sorted(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
            if dist[len(dist) // 2] <= SUBZONE_MAX_MEDIAN_SPREAD and dist[int(len(dist) * 0.9)] <= SUBZONE_MAX_P90_SPREAD:
                out[m].extend((x, y, sid) for x, y in pts)
    return out


# ---------------------------------------------------------------------------------------------
# Place descriptions
# ---------------------------------------------------------------------------------------------

def describe_place(geo: Geography, map_id: int, pt: Point,
                   prefer: int | None = None) -> tuple[int | None, str | None, str | None]:
    """(zone id, zone name, place phrase).

    The phrase is `in <subzone>` when quest givers around the point name one, else `near <flight town>` or
    `about N yards <dir> of <flight town>`, else None. `prefer` is a zone id known from other data (a quest's sort
    zone); failing that the zone of a flight-path town within 400 yd of the point is the hint.
    """
    lm = geo.landmark_near(map_id, pt[0], pt[1])
    hint = prefer or (lm[4] if lm and lm[1] <= LANDMARK_NEAR_YARDS else None)
    zone = geo.zone_of(map_id, pt[0], pt[1], hint)
    zone_id, zone_name = (zone[0], zone[1]) if zone else (None, None)
    sub = geo.subzone_at(map_id, pt[0], pt[1])
    if sub:
        chain = geo.ancestors(sub[0])
        if zone_id is None or zone_id in chain:
            if zone_id is None and len(chain) > 1:
                zone_id = chain[-1]
                zone_name = geo.area_names.get(zone_id)
            return zone_id, zone_name, f"in {sub[1]}"
    return zone_id, zone_name, landmark_phrase(geo, map_id, pt[0], pt[1], zone_name)


def main_cluster(points: list[Point]) -> tuple[Point, int, int]:
    clusters = cluster_points(points)
    return clusters[0][0], clusters[0][1], len(clusters)


def target_phrase(geo: Geography, map_id: int, giver: Point, spawns: list[Point]) -> str | None:
    """Where a target is, relative to the giver. None if there is no spawn at all on this map."""
    if not spawns:
        return None
    near = [p for p in spawns if math.hypot(p[0] - giver[0], p[1] - giver[1]) <= NEARBY_YARDS]
    if not near:
        nearest = min(spawns, key=lambda p: math.hypot(p[0] - giver[0], p[1] - giver[1]))
        zone = geo.zone_of(map_id, nearest[0], nearest[1])
        return f"further away, nearest in {zone[1]}" if zone else "further away"
    clusters = cluster_points(near)
    parts = []
    for i, (centroid, size) in enumerate(clusters[:2]):
        if i == 1 and size < SECOND_CLUSTER_RATIO * clusters[0][1]:
            break
        phrase = relative_phrase(giver, centroid, "the quest giver")
        lm = landmark_phrase(geo, map_id, centroid[0], centroid[1], near_only=True)
        if lm and "right next to" not in phrase:
            phrase += f" ({lm})"
        parts.append(phrase)
    return "; also ".join(parts)


# ---------------------------------------------------------------------------------------------
# Quest entries
# ---------------------------------------------------------------------------------------------

def quest_available_level(row: dict) -> int:
    min_level = as_int(row["MinLevel"])
    return min_level if min_level > 0 else max(as_int(row["QuestLevel"]), 1)


def build_quests(world: World, geo: Geography, sides: dict[int, str], max_level: int, min_level: int,
                 faction: str) -> list[dict]:
    loot = world.loot_sources()
    entries = []
    for qid in sorted(world.quest_rows):
        row = world.quest_rows[qid]
        title = clean(row["LogTitle"])
        if not title or JUNK_TITLE.search(title):
            continue
        level_req = quest_available_level(row)
        if level_req > max_level or level_req < min_level:
            continue

        # Givers with a spawn on the selected maps, creature first, in id order.
        givers = []
        for gid in sorted(world.starters.get(qid, [])):
            for m, pts in sorted(world.creature_spawns.get(gid, {}).items()):
                givers.append(("creature", gid, m, pts))
                break
        for gid in sorted(world.go_starters.get(qid, [])):
            for m, pts in sorted(world.go_spawns.get(gid, {}).items()):
                givers.append(("object", gid, m, pts))
                break
        givers = [g for g in givers if (world.creature_name if g[0] == "creature" else world.go_name).get(g[1])]
        if not givers:
            continue

        mask = as_int(row["AllowableRaces"])
        side = side_of_mask(mask) if mask else "both"
        if side == "both" and mask == 0:
            giver_sides = {sides.get(as_int(world.creature_row[g[1]]["faction"])) for g in givers
                           if g[0] == "creature" and g[1] in world.creature_row}
            if len(giver_sides) == 1 and None not in giver_sides:
                side = giver_sides.pop()           # a Horde-only NPC cannot give the quest to the Alliance
        if faction != "both" and side not in ("both", faction):
            continue

        giver_kind, giver_id, map_id, giver_pts = givers[0]
        giver_pt, _n, _clusters = main_cluster(giver_pts)
        names = world.creature_name if giver_kind == "creature" else world.go_name
        giver_name = names[giver_id]
        sort_id = as_int(row["QuestSortID"])
        sort_id = sort_id if sort_id in geo.area_names else 0
        zone_id, zone_name, giver_lm = describe_place(geo, map_id, giver_pt, sort_id or None)
        place = zone_name
        if zone_id is None and sort_id:
            zone_name = place = geo.area_names[sort_id]
        elif zone_id and sort_id and sort_id != zone_id and zone_id in geo.ancestors(sort_id):
            place = f"{geo.area_names[sort_id]}, {zone_name}"       # the quest's own subzone, e.g. Northshire Valley

        level = as_int(row["QuestLevel"])
        level = level if level > 0 else level_req
        head = f"level {level}"
        if place:
            head += f", {place}"
        if side != "both":
            head += f", {side.capitalize()}"
        sentences = []
        given = f"Given by {giver_name}"
        if giver_lm and not (giver_lm.startswith("in ") and giver_lm[3:] in (place or "")):
            given += f", {giver_lm}"
        if len(givers) > 1:
            other = [clean((world.creature_name if g[0] == "creature" else world.go_name)[g[1]]) for g in givers[1:3]
                     if (world.creature_name if g[0] == "creature" else world.go_name)[g[1]] != giver_name]
            if other:
                given += f" (also {join_names(other)})"
        sentences.append(given + ".")

        keywords = [title, giver_name]
        place_terms = []
        if place:
            place_terms.extend(place.split(", "))
        if giver_lm:
            place_terms.append(re.sub(PLACE_PREFIX, "", giver_lm))
        place_terms.append(title)

        # Kill / interact objectives.
        for i in range(1, 5):
            target = as_int(row[f"RequiredNpcOrGo{i}"])
            count = max(as_int(row[f"RequiredNpcOrGoCount{i}"]), 1)
            if not target:
                continue
            if target > 0:
                name, spawns = world.creature_name.get(target, ""), world.creature_spawns.get(target, {}).get(map_id, [])
                verb = "Kill"
            else:
                name, spawns = world.go_name.get(-target, ""), world.go_spawns.get(-target, {}).get(map_id, [])
                verb = "Interact with"
            if not name or JUNK_TARGET.search(name):
                continue
            where = target_phrase(geo, map_id, giver_pt, spawns)
            sentences.append(f"{verb} {count} {name}" + (f", found {where}." if where else "."))
            keywords.extend([name, plural_terms(name)])

        # Collection objectives.
        for i in range(1, 7):
            item = as_int(row[f"RequiredItemId{i}"])
            count = max(as_int(row[f"RequiredItemCount{i}"]), 1)
            name = world.item_name.get(item, "")
            if not item or not name:
                continue
            keywords.extend([name, plural_terms(name)])
            sources = loot.get(item, {})
            cands = []
            for cid, (qr, chance) in sources.items():
                spawns = world.creature_spawns.get(cid, {}).get(map_id)
                cname = world.creature_name.get(cid, "")
                if spawns and cname and not JUNK_TARGET.search(cname):
                    nearby = [p for p in spawns if math.hypot(p[0] - giver_pt[0], p[1] - giver_pt[1]) <= NEARBY_YARDS]
                    cands.append((cid, cname, qr, chance, nearby, spawns, as_int(world.creature_row[cid]['rank'])))
            near_cands = [c for c in cands if c[4]]
            pool = near_cands or cands
            pool.sort(key=lambda c: (-c[2], -round(c[3] / 10), c[6], -len(c[4]), c[1], c[0]))   # quest-required, drop chance, normal mobs, common
            chosen, seen = [], set()
            for c in pool:
                if c[1] not in seen:
                    seen.add(c[1])
                    chosen.append(c)
                if len(chosen) == MAX_SOURCES:
                    break
            sentence = f"Collect {count} {name}"
            if chosen:
                sentence += f", dropped by {join_names([c[1] for c in chosen])}"
                spawn_union = [p for c in chosen for p in (c[4] if near_cands else c[5])]
                where = target_phrase(geo, map_id, giver_pt, spawn_union)
                if where:
                    sentence += f", found {where}"
                keywords.extend(x for c in chosen for x in (c[1], plural_terms(c[1])))
            sentences.append(sentence + ".")

        # Turn-in.
        for eid in sorted(world.enders.get(qid, [])):
            ename = world.creature_name.get(eid)
            espawns = world.creature_spawns.get(eid, {}).get(map_id)
            if not ename or not espawns or (giver_kind == "creature" and eid == giver_id):
                continue
            ept, _n, _c = main_cluster(espawns)
            if math.hypot(ept[0] - giver_pt[0], ept[1] - giver_pt[1]) < RIGHT_NEXT_TO_YARDS:
                phrase = f"right next to {giver_name}"
            else:
                _zid, ezone, elm = describe_place(geo, map_id, ept)
                phrase = relative_phrase(giver_pt, ept, "the quest giver")
                if math.hypot(ept[0] - giver_pt[0], ept[1] - giver_pt[1]) > NEARBY_YARDS and ezone:
                    phrase = f"in {ezone}" + (f", {elm}" if elm else "")
                elif elm:
                    phrase += f", {elm}"
            sentences.append(f"Turn in to {ename}, {phrase}.")
            keywords.append(ename)
            break

        prev = abs(as_int(world.quest_addon.get(qid, {}).get("PrevQuestID")))
        if prev and prev in world.quest_rows and clean(world.quest_rows[prev]["LogTitle"]) not in ("", title):
            sentences.append(f"Requires finishing {clean(world.quest_rows[prev]['LogTitle'])} first.")

        tags = ["quest", slug(zone_name) if zone_name else "unknown-zone", side, level_band(level)]
        if place and place != zone_name:
            tags.insert(2, slug(place.split(", ")[0]))                      # the subzone
        entries.append({
            "id": f"quest_{qid}",
            "title": f"Quest {title} ({head})",
            "content": " ".join(sentences),
            "keywords": words_keywords(*keywords) + boosted(*place_terms),
            "tags": [t for t in tags if t],
            "_map": map_id,
        })
    return entries


# ---------------------------------------------------------------------------------------------
# NPC service entries
# ---------------------------------------------------------------------------------------------

def npc_label(row: dict) -> str | None:
    flags = as_int(row["npcflag"])
    sub = clean(row["subname"])
    if flags & NPCFLAG_FLIGHTMASTER:
        return "Flight Master"
    if flags & NPCFLAG_INNKEEPER:
        return "Innkeeper"
    if flags & (NPCFLAG_TRAINER_CLASS | NPCFLAG_TRAINER_PROFESSION):
        return sub if sub.lower().endswith(("trainer", "master", "smith", "crafter")) or "trainer" in sub.lower() else None
    return None


def npc_kind(row: dict) -> str:
    flags = as_int(row["npcflag"])
    if flags & NPCFLAG_FLIGHTMASTER:
        return "flightmaster"
    if flags & NPCFLAG_INNKEEPER:
        return "innkeeper"
    return "class-trainer" if flags & NPCFLAG_TRAINER_CLASS else "profession-trainer"


NPC_KINDS = ("class-trainer", "profession-trainer", "innkeeper", "flightmaster")


def build_npcs(world: World, geo: Geography, sides: dict[int, str], faction: str,
               kinds: tuple[str, ...] = NPC_KINDS) -> list[dict]:
    entries = []
    for entry in sorted(world.creature_row):
        row = world.creature_row[entry]
        name, label = world.creature_name.get(entry), npc_label(row)
        if not name or not label or JUNK_TARGET.search(name) or JUNK_TITLE.search(name):
            continue
        if npc_kind(row) not in kinds:
            continue
        side = sides.get(as_int(row["faction"]), "both")
        if faction != "both" and side not in ("both", faction):
            continue
        for map_id, pts in sorted(world.creature_spawns.get(entry, {}).items()):
            for idx, (centroid, size) in enumerate(cluster_points(pts, NPC_CLUSTER_YARDS)):
                _zid, zone_name, lm = describe_place(geo, map_id, centroid)
                if not lm and not zone_name:
                    continue                  # nothing we can say without guessing
                full = f"{label} {name}" if label.lower() not in name.lower() else name
                tail = ", ".join([lm, zone_name] if lm and zone_name else [lm] if lm else [f"in {zone_name}"])
                if side != "both":
                    tail += f", {side.capitalize()}"
                tags = ["npc", npc_kind(row), slug(zone_name) if zone_name else "unknown-zone", side]
                entries.append({
                    "id": f"npc_{entry}_{idx}",
                    "title": full,
                    "content": f"{full} is {tail}.",
                    "keywords": words_keywords(label) + boosted(zone_name, re.sub(PLACE_PREFIX, "", lm or "")),
                    "tags": tags,
                    "_map": map_id,
                })
    return entries


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

def write_outputs(entries: list[dict], kind: str, out: Path) -> list[tuple[Path, int, int]]:
    """Writes <kind>_<continent>.json per map. Returns [(path, entries, bytes)]."""
    by_map: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        by_map[e["_map"]].append(e)
    written = []
    for map_id, items in sorted(by_map.items()):
        items = sorted(items, key=lambda e: (e["id"].split("_")[0], [int(p) for p in e["id"].split("_")[1:]]))
        payload = [{k: v for k, v in e.items() if not k.startswith("_")} for e in items]
        assert_no_double_quotes(payload)
        path = out / f"{kind}_{MAP_FILE_NAMES.get(map_id, 'map_' + str(map_id))}.json"
        text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8")
        written.append((path, len(payload), len(text.encode("utf-8"))))
    return written


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, type=Path, help="directory with the TSV exports and dbc/")
    ap.add_argument("--out", required=True, type=Path, help="output directory for the RAG JSON files")
    ap.add_argument("--max-level", type=int, default=60, help="highest quest level requirement to include")
    ap.add_argument("--min-level", type=int, default=1, help="lowest quest level requirement to include")
    ap.add_argument("--maps", default="0,1", help="comma separated map ids (0 and 1 have continent file names)")
    ap.add_argument("--faction", choices=("both", "alliance", "horde"), default="both")
    ap.add_argument("--npc-kinds", default=",".join(NPC_KINDS),
                    help="comma separated subset of: " + ", ".join(NPC_KINDS))
    ap.add_argument("--keyword-repeat", type=int, default=KEYWORD_REPEAT,
                    help="copies of the place, name and title terms put in `keywords`, to weight them in the module's "
                         "term-frequency cosine (default %(default)s)")
    ap.add_argument("--no-quests", action="store_true")
    ap.add_argument("--no-npcs", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    global KEYWORD_REPEAT
    KEYWORD_REPEAT = max(args.keyword_repeat, 0)
    maps = {int(m) for m in args.maps.split(",") if m.strip()}
    dbc = args.inputs / "dbc"
    geo = Geography(Dbc((dbc / "AreaTable.dbc").read_bytes()), Dbc((dbc / "WorldMapArea.dbc").read_bytes()),
                    Dbc((dbc / "TaxiNodes.dbc").read_bytes()), maps)
    world = World(args.inputs, maps)
    geo.keep_nodes_near(world.flight_master_spawns())
    geo.set_subzones(learn_subzones(world, geo))
    kinds = tuple(k.strip() for k in args.npc_kinds.split(",") if k.strip())
    unknown = set(kinds) - set(NPC_KINDS)
    if unknown:
        raise SystemExit(f"unknown --npc-kinds: {', '.join(sorted(unknown))}")
    sides = learn_creature_sides(world)
    args.out.mkdir(parents=True, exist_ok=True)
    total_entries = total_bytes = 0
    for kind, build in (("quests", lambda: build_quests(world, geo, sides, args.max_level, args.min_level, args.faction)),
                        ("npcs", lambda: build_npcs(world, geo, sides, args.faction, kinds))):
        if (kind == "quests" and args.no_quests) or (kind == "npcs" and args.no_npcs):
            continue
        for path, n, size in write_outputs(build(), kind, args.out):
            print(f"{path.name}: {n} entries, {size} bytes")
            total_entries += n
            total_bytes += size
    print(f"total: {total_entries} entries, {total_bytes} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
