import json
import struct
import tempfile
import unittest
from pathlib import Path

import build_world_rag as b


def make_dbc(records: list[tuple], strings: bytes = b"\0") -> bytes:
    fields = len(records[0])
    body = b"".join(struct.pack("<%dI" % fields, *r) for r in records)
    return struct.pack("<4s4I", b"WDBC", len(records), fields, fields * 4, len(strings)) + body + strings


def strtab(*names: str) -> tuple[bytes, list[int]]:
    """A DBC string block and the offset of each name in it."""
    block, offsets = b"\0", []
    for n in names:
        offsets.append(len(block))
        block += n.encode() + b"\0"
    return block, offsets


def f2u(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


class DbcTest(unittest.TestCase):
    def test_reads_records_strings_and_floats(self):
        strings = b"\0Elwynn Forest\0"
        dbc = b.Dbc(make_dbc([(12, 0, 1), (13, f2u(1.5), 0)], strings))
        self.assertEqual(dbc.count, 2)
        self.assertEqual(dbc.record(0), (12, 0, 1))
        self.assertEqual(dbc.string(1), "Elwynn Forest")
        self.assertEqual(b.u2f(dbc.record(1)[1]), 1.5)

    def test_rejects_other_files(self):
        with self.assertRaises(ValueError):
            b.Dbc(b"WDB2" + bytes(16))

    def test_rejects_truncated_file(self):
        data = make_dbc([(1, 2)], b"abc\0")
        with self.assertRaises(ValueError):
            b.Dbc(data[:-2])


class GeographyTest(unittest.TestCase):
    def setUp(self):
        # Area table: only the name (field 11) matters to Geography.
        names, (elwynn, stormwind, sentinel, fake) = strtab("Elwynn Forest", "Stormwind City",
                                                             "Sentinel Hill, Westfall", "Fake Node")
        area = [[0] * 12 for _ in range(3)]
        for row, (aid, off) in zip(area, ((12, elwynn), (1519, stormwind), (40, 0))):
            row[0], row[11] = aid, off
        area[1][4] = 0x200000                                   # capital city flag
        self.area = b.Dbc(make_dbc([tuple(r) for r in area], names))
        # WorldMapArea: x, map, area, name, left, right, top, bottom, ...  (Elwynn box from the real DBC)
        def wma(map_id, area_id, left, right, top, bottom):
            return (0, map_id, area_id, 0, f2u(left), f2u(right), f2u(top), f2u(bottom), 0, 0, 0)
        self.wma = b.Dbc(make_dbc([
            wma(0, 12, 1535.4, -1935.4, -7939.6, -10254.2),     # Elwynn
            wma(0, 1519, 1722.9, -14.6, -7995.8, -9154.2),      # Stormwind (smaller, inside)
            wma(0, 0, 18171.0, -22569.0, 11176.0, -15973.0),    # continent map: skipped
            wma(530, 12, 100.0, -100.0, 100.0, -100.0),         # other map: skipped
        ]))
        # TaxiNodes: id, map, x, y, z, name, ..., mount[2]
        def node(map_id, x, y, name_off, mount):
            r = [0] * 24
            r[1], r[2], r[3], r[5], r[22] = map_id, f2u(x), f2u(y), name_off, mount
            return tuple(r)
        self.taxi = b.Dbc(make_dbc([node(0, -10629.0, 1037.0, sentinel, 541), node(0, 0.0, 0.0, fake, 0)], names))
        self.geo = b.Geography(self.area, self.wma, self.taxi, {0, 1})

    def test_axis_mapping_northshire_is_in_elwynn(self):
        # Northshire Abbey flight node: x=-8889 (north/south), y=-0.5 (east/west).
        zone = self.geo.zone_of(0, -8889.0, -0.5)
        self.assertEqual(zone, (12, "Elwynn Forest"))
        # Swapped axes must not land in Elwynn.
        self.assertIsNone(self.geo.zone_of(0, -0.5, -8889.0))

    def test_deepest_box_wins_over_smallest(self):
        # 14 yd inside the Stormwind box edge (Northshire Abbey): still Elwynn, although the city box is smaller.
        self.assertEqual(self.geo.zone_of(0, -8889.0, -0.5)[1], "Elwynn Forest")

    def test_capital_in_zone_box_wins_and_continent_box_is_skipped(self):
        self.assertEqual(self.geo.zone_of(0, -8800.0, 500.0), (1519, "Stormwind City"))
        self.assertIsNone(self.geo.zone_of(0, 5000.0, 5000.0))

    def test_landmark_uses_name_before_comma_and_ignores_unmounted_nodes(self):
        near = self.geo.landmark_near(0, -10600.0, 1000.0)
        self.assertEqual(near[0], "Sentinel Hill")
        self.assertIsNone(self.geo.landmark_near(0, 10.0, 10.0))      # the mount-less node at 0,0 was dropped

    def test_landmark_phrase_tiers(self):
        self.assertEqual(b.landmark_phrase(self.geo, 0, -10600.0, 1000.0), "near Sentinel Hill")
        # 600 yd north of the node -> distance form, not 'near'
        self.assertEqual(b.landmark_phrase(self.geo, 0, -10029.0, 1037.0), "about 600 yards north of Sentinel Hill")
        self.assertIsNone(b.landmark_phrase(self.geo, 0, -8000.0, 1037.0))     # 2600 yd away: omitted

    def test_keep_nodes_near_flight_masters(self):
        self.geo.keep_nodes_near({0: [(-10600.0, 1000.0)]})
        self.assertEqual(len(self.geo.nodes[0]), 1)
        self.geo.keep_nodes_near({0: [(5000.0, 5000.0)]})
        self.assertEqual(self.geo.nodes[0], [])


class ZoneHintTest(unittest.TestCase):
    """Two overlapping zone boxes: A (deep at x=50) and B (edge at x=50), a subzone S of B, a node named after B."""

    def setUp(self):
        names, (a, b_, s, n) = strtab("Zone A", "Zone B", "Sub S", "Hamlet, Zone B")
        rows = [[0] * 12 for _ in range(3)]
        for r, (aid, off, parent) in zip(rows, ((1, a, 0), (2, b_, 0), (3, s, 2))):
            r[0], r[2], r[11] = aid, parent, off
        area = b.Dbc(make_dbc([tuple(r) for r in rows], names))

        def wma(area_id, left, right, top, bottom):
            return (0, 0, area_id, 0, f2u(left), f2u(right), f2u(top), f2u(bottom), 0, 0, 0)
        wm = b.Dbc(make_dbc([wma(1, 100.0, -100.0, 100.0, -100.0), wma(2, 100.0, -100.0, 200.0, 40.0)]))
        node = [0] * 24
        node[1], node[2], node[3], node[5], node[22] = 0, f2u(60.0), f2u(0.0), n, 541
        self.geo = b.Geography(area, wm, b.Dbc(make_dbc([tuple(node)], names)), {0})

    def test_deepest_without_hint(self):
        self.assertEqual(self.geo.zone_of(0, 60.0, 0.0)[1], "Zone A")

    def test_hint_wins_inside_slack_and_loses_outside_it(self):
        self.assertEqual(self.geo.zone_of(0, 60.0, 0.0, 2)[1], "Zone B")      # depth 0.125 vs 0.2: inside the slack
        self.assertEqual(self.geo.zone_of(0, 45.0, 0.0, 2)[1], "Zone A")      # depth 0.03 vs 0.275: too shallow
        self.assertEqual(self.geo.zone_of(0, -50.0, 0.0, 2)[1], "Zone A")     # outside B's box: hint ignored

    def test_subzone_hint_resolves_to_its_zone(self):
        self.assertEqual(self.geo.zone_of(0, 60.0, 0.0, 3)[1], "Zone B")

    def test_node_zone_hint_from_the_name(self):
        self.assertEqual(self.geo.landmark_near(0, 60.0, 0.0)[4], 2)
        zid, name, lm = b.describe_place(self.geo, 0, (55.0, 0.0))
        self.assertEqual((zid, name, lm), (2, "Zone B", "near Hamlet"))


    def test_subzone_from_nearby_givers(self):
        self.geo.set_subzones({0: [(50.0, 10.0, 3), (52.0, -10.0, 3), (-900.0, 900.0, 3)]})
        self.assertEqual(self.geo.subzone_at(0, 51.0, 0.0), (3, "Sub S"))
        self.assertIsNone(self.geo.subzone_at(0, 500.0, 0.0))                # nothing within 200 yd
        self.assertIsNone(self.geo.subzone_at(0, -900.0, 900.0))               # a single giver is not enough
        zid, name, place = b.describe_place(self.geo, 0, (55.0, 0.0))
        self.assertEqual((zid, name, place), (2, "Zone B", "in Sub S"))

    def test_learn_subzones_keeps_tight_clusters_only(self):
        from types import SimpleNamespace
        quests = {i: {"QuestSortID": "3"} for i in range(1, 5)}                 # 4 quests sorted into subzone 3
        quests.update({i: {"QuestSortID": "2"} for i in range(5, 9)})            # zone-level sort: ignored
        spawns = {g: {0: [(50.0 + g, 0.0)]} for g in range(1, 5)}
        spawns.update({g: {0: [(5000.0 * g, 0.0)]} for g in range(5, 9)})
        world = SimpleNamespace(quest_rows=quests, starters={q: [q] for q in range(1, 9)}, creature_spawns=spawns)
        got = b.learn_subzones(world, self.geo)
        self.assertEqual(sorted({a for _x, _y, a in got[0]}), [3])
        self.assertEqual(len(got[0]), 4)
        spawns.update({g: {0: [(5000.0 * g, 0.0)]} for g in range(1, 5)})        # same quests, spread far apart
        self.assertEqual(b.learn_subzones(world, self.geo), {})


class CompassTest(unittest.TestCase):
    def test_eight_points_with_plus_x_north_and_plus_y_west(self):
        cases = {(100, 0): "north", (100, -100): "north-east", (0, -100): "east", (-100, -100): "south-east",
                 (-100, 0): "south", (-100, 100): "south-west", (0, 100): "west", (100, 100): "north-west"}
        for (dx, dy), expected in cases.items():
            self.assertEqual(b.compass(dx, dy), expected, (dx, dy))

    def test_real_pair_northshire_is_east_of_stormwind(self):
        self.assertEqual(b.compass(-8889 - -8841, -0.5 - 490), "east")

    def test_distance_rounding_and_next_to(self):
        self.assertEqual(b.round_distance(324.0), 300)
        self.assertEqual(b.round_distance(326.0), 350)
        self.assertEqual(b.relative_phrase((0, 0), (59, 0), "the quest giver"), "right next to the quest giver")
        self.assertEqual(b.relative_phrase((0, 0), (-212, -212), "the quest giver"),
                         "about 300 yards south-east of the quest giver")
        self.assertEqual(b.relative_phrase((0, 0), (61, 0), "X"), "about 50 yards north of X")


class ClusterTest(unittest.TestCase):
    def test_densest_cluster_first_with_centroid(self):
        pts = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (2000.0, 0.0), (2010.0, 0.0)]
        clusters = b.cluster_points(pts, 250.0)
        self.assertEqual([n for _c, n in clusters], [3, 2])
        self.assertAlmostEqual(clusters[0][0][0], 10.0)
        self.assertAlmostEqual(clusters[1][0][0], 2005.0)

    def test_is_deterministic_and_covers_every_point(self):
        pts = [(float(i * 37 % 900), float(i * 91 % 700)) for i in range(60)]
        a, c = b.cluster_points(pts, 200.0), b.cluster_points(list(pts), 200.0)
        self.assertEqual(a, c)
        self.assertEqual(sum(n for _c, n in a), 60)

    def test_empty(self):
        self.assertEqual(b.cluster_points([]), [])

    def test_target_phrase_second_cluster_only_when_comparable(self):
        geo = _NoGeo()
        giver = (0.0, 0.0)
        big = [(300.0 + i, 0.0) for i in range(10)]
        small = [(0.0, 1000.0)]
        self.assertEqual(b.target_phrase(geo, 0, giver, big + small),
                         "about 300 yards north of the quest giver")
        comparable = [(0.0, -1000.0 + i) for i in range(8)]
        self.assertEqual(b.target_phrase(geo, 0, giver, big + comparable),
                         "about 300 yards north of the quest giver; also about 1000 yards east of the quest giver")

    def test_target_phrase_far_away_and_none(self):
        geo = _NoGeo()
        self.assertIsNone(b.target_phrase(geo, 0, (0.0, 0.0), []))
        self.assertEqual(b.target_phrase(geo, 0, (0.0, 0.0), [(9000.0, 0.0)]), "further away")


class _NoGeo:
    def zone_of(self, *a):
        return None

    def landmark_near(self, *a):
        return None


class TsvTest(unittest.TestCase):
    def test_unescape(self):
        self.assertEqual(b.unescape_field("a\\tb\\nc\\\\d"), "a\tb\nc\\d")
        self.assertIsNone(b.unescape_field("NULL"))
        self.assertEqual(b.unescape_field("NULLS"), "NULLS")
        self.assertEqual(b.unescape_field("plain"), "plain")

    def test_parse_with_header_and_short_rows(self):
        rows = b.parse_tsv("a\tb\tc\n1\tx\\ty\tNULL\n2\n")
        self.assertEqual(rows[0], {"a": "1", "b": "x\ty", "c": None})
        self.assertEqual(rows[1], {"a": "2", "b": None, "c": None})

    def test_escaped_tab_does_not_split_a_field(self):
        rows = b.parse_tsv("name\tid\nTwo\\tTabs\t7\n")
        self.assertEqual(rows, [{"name": "Two\tTabs", "id": "7"}])


QUEST_COLS = ["ID", "QuestLevel", "MinLevel", "QuestSortID", "AllowableRaces", "LogTitle", "RequiredNpcOrGo1",
              "RequiredNpcOrGo2", "RequiredNpcOrGo3", "RequiredNpcOrGo4", "RequiredNpcOrGoCount1",
              "RequiredNpcOrGoCount2", "RequiredNpcOrGoCount3", "RequiredNpcOrGoCount4"] + \
    [f"RequiredItemId{i}" for i in range(1, 7)] + [f"RequiredItemCount{i}" for i in range(1, 7)]


def tsv(cols, rows):
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, 0)) for c in cols))
    return "\n".join(lines) + "\n"


class SyntheticWorldTest(unittest.TestCase):
    """A one-quest world: Remy gives 'Kobold Candles' (collect 8 Large Candle from Kobolds south-east of him)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "dbc").mkdir()
        names, (elwynn, goldshire) = strtab("Elwynn Forest", "Goldshire, Elwynn")
        area = [tuple([12] + [0] * 10 + [elwynn] + [0] * 24)]
        (d / "dbc/AreaTable.dbc").write_bytes(make_dbc(area, names))
        wma = (0, 0, 12, 0, f2u(1535.4), f2u(-1935.4), f2u(-7939.6), f2u(-10254.2), 0, 0, 0)
        (d / "dbc/WorldMapArea.dbc").write_bytes(make_dbc([wma]))
        node = [0] * 24
        node[1], node[2], node[3], node[5], node[22] = 0, f2u(-9500.0), f2u(0.0), goldshire, 541
        (d / "dbc/TaxiNodes.dbc").write_bytes(make_dbc([tuple(node)], names))
        (d / "creature_template.tsv").write_text(tsv(
            ["entry", "name", "subname", "minlevel", "maxlevel", "rank", "faction", "npcflag", "type"],
            [dict(entry=1, name='Remy "Two Times"', subname="NULL", faction=11, npcflag=3, rank=0),
             dict(entry=2, name="Kobold Miner", subname="NULL", faction=7, npcflag=0, rank=0),
             dict(entry=3, name="Kobold Tunneler", subname="NULL", faction=7, npcflag=0, rank=0),
             dict(entry=4, name="Hurt Bob", subname="Flight Master", faction=11, npcflag=8192, rank=0),
             dict(entry=5, name="Lyria", subname="Warrior Trainer", faction=11, npcflag=48, rank=0)]))
        (d / "creature.tsv").write_text(tsv(
            ["guid", "id", "map", "zoneId", "areaId", "position_x", "position_y", "position_z", "spawntimesecs"],
            [dict(guid=1, id=1, map=0, position_x=-9500.0, position_y=0.0),
             dict(guid=2, id=2, map=0, position_x=-9650.0, position_y=-150.0),
             dict(guid=3, id=2, map=0, position_x=-9660.0, position_y=-140.0),
             dict(guid=4, id=3, map=0, position_x=-9640.0, position_y=-160.0),
             dict(guid=5, id=4, map=0, position_x=-9490.0, position_y=10.0),
             dict(guid=6, id=5, map=0, position_x=-9450.0, position_y=30.0)]))
        (d / "creature_loot_template.tsv").write_text(tsv(
            ["Entry", "Item", "Reference", "Chance", "QuestRequired", "LootMode", "GroupId", "MinCount", "MaxCount",
             "Comment"],
            [dict(Entry=2, Item=100, Reference=0, Chance=60, QuestRequired=1),
             dict(Entry=3, Item=0, Reference=900, Chance=100, QuestRequired=0)]))
        (d / "reference_loot_template.tsv").write_text(tsv(
            ["Entry", "Item", "Reference", "Chance", "QuestRequired", "LootMode", "GroupId", "MinCount", "MaxCount",
             "Comment"], [dict(Entry=900, Item=100, Reference=0, Chance=20, QuestRequired=1)]))
        (d / "item_template.tsv").write_text(tsv(["entry", "name", "Quality"], [dict(entry=100, name="Large Candle")]))
        (d / "gameobject_template.tsv").write_text(tsv(["entry", "type", "name"], []))
        (d / "gameobject.tsv").write_text(tsv(["guid", "id", "map", "position_x", "position_y"], []))
        (d / "gameobject_queststarter.tsv").write_text("id\tquest\n")
        (d / "creature_queststarter.tsv").write_text("id\tquest\n1\t7\n1\t8\n")
        (d / "creature_questender.tsv").write_text("id\tquest\n1\t7\n")
        (d / "quest_template_addon.tsv").write_text("ID\tPrevQuestID\n7\t0\n")
        q = dict(ID=7, QuestLevel=7, MinLevel=5, QuestSortID=12, AllowableRaces=1101, LogTitle="Kobold Candles",
                 RequiredItemId1=100, RequiredItemCount1=8)
        junk = dict(ID=8, QuestLevel=5, MinLevel=5, QuestSortID=12, AllowableRaces=0, LogTitle="DEPRECATED Candles")
        (d / "quest_template.tsv").write_text(tsv(QUEST_COLS, [q, junk]))
        self.dir = d
        self.geo = b.Geography(b.Dbc((d / "dbc/AreaTable.dbc").read_bytes()), b.Dbc((d / "dbc/WorldMapArea.dbc").read_bytes()),
                               b.Dbc((d / "dbc/TaxiNodes.dbc").read_bytes()), {0})
        self.world = b.World(d, {0})
        self.geo.keep_nodes_near(self.world.flight_master_spawns())

    def tearDown(self):
        self.tmp.cleanup()

    def test_quest_prose(self):
        entries = b.build_quests(self.world, self.geo, {}, 60, 1, "both")
        self.assertEqual([e["id"] for e in entries], ["quest_7"])
        e = entries[0]
        self.assertEqual(e["title"], "Quest Kobold Candles (level 7, Elwynn Forest, Alliance)")
        self.assertEqual(
            e["content"],
            "Given by Remy 'Two Times', near Goldshire. Collect 8 Large Candle, dropped by Kobold Miner and "
            "Kobold Tunneler, found about 200 yards south-east of the quest giver (near Goldshire).")
        self.assertIn("Large Candle", e["keywords"])
        self.assertIn("Larges Candles", e["keywords"])
        self.assertIn("Kobolds Miners", e["keywords"])
        self.assertEqual(e["tags"], ["quest", "elwynn-forest", "alliance", "lvl1-10"])

    def test_level_and_faction_filters(self):
        self.assertEqual(b.build_quests(self.world, self.geo, {}, 4, 1, "both"), [])
        self.assertEqual(b.build_quests(self.world, self.geo, {}, 60, 1, "horde"), [])
        self.assertEqual(len(b.build_quests(self.world, self.geo, {}, 60, 1, "alliance")), 1)

    def test_npc_prose(self):
        entries = b.build_npcs(self.world, self.geo, {11: "alliance"}, "both")
        by_id = {e["id"]: e for e in entries}
        self.assertEqual(by_id["npc_5_0"]["content"], "Warrior Trainer Lyria is near Goldshire, Elwynn Forest, Alliance.")
        self.assertEqual(by_id["npc_4_0"]["content"], "Flight Master Hurt Bob is near Goldshire, Elwynn Forest, Alliance.")

    def test_keyword_repeat_weights_place_and_title_terms(self):
        b.KEYWORD_REPEAT = 2
        try:
            e = b.build_quests(self.world, self.geo, {}, 60, 1, "both")[0]
        finally:
            b.KEYWORD_REPEAT = 1
        self.assertEqual(e["keywords"].count("Elwynn Forest"), 2)           # the place terms, repeated
        self.assertEqual(e["keywords"].count("Kobold Candles"), 3)          # the title once plain, then repeated

    def test_no_double_quotes_anywhere(self):
        entries = b.build_quests(self.world, self.geo, {}, 60, 1, "both") + b.build_npcs(self.world, self.geo, {}, "both")
        b.assert_no_double_quotes([{k: v for k, v in e.items() if not k.startswith("_")} for e in entries])
        self.assertNotIn('"', b.clean('say "hi" to "me"'))

    def test_double_quote_check_rejects(self):
        bad = {"id": "x", "title": 't"', "content": "c", "keywords": [], "tags": []}
        with self.assertRaises(ValueError):
            b.assert_no_double_quotes([bad])

    def test_main_writes_sorted_deterministic_files(self):
        out1, out2 = self.dir / "o1", self.dir / "o2"
        for out in (out1, out2):
            self.assertEqual(b.main(["--inputs", str(self.dir), "--out", str(out), "--maps", "0"]), 0)
        for name in ("quests_eastern_kingdoms.json", "npcs_eastern_kingdoms.json"):
            self.assertEqual((out1 / name).read_text(), (out2 / name).read_text())
        data = json.loads((out1 / "quests_eastern_kingdoms.json").read_text())
        self.assertEqual(set(data[0]), {"id", "title", "content", "keywords", "tags"})

    def test_learn_sides_needs_agreement(self):
        w = self.world
        w.quest_rows = {i: {"AllowableRaces": "1101"} for i in range(1, 4)}
        w.quest_rows[9] = {"AllowableRaces": "690"}
        w.starters = {1: [1], 2: [1], 3: [1], 9: [1]}     # creature 1 (faction 11) gives 3 Alliance and 1 Horde quest
        w.enders = {}
        self.assertEqual(b.learn_creature_sides(w), {})        # 75% agreement: not settled
        w.starters[9] = []
        self.assertEqual(b.learn_creature_sides(w), {11: "alliance"})


class PluralTest(unittest.TestCase):
    def test_plain_plurals(self):
        self.assertEqual([b.plural(w) for w in ("Kobold", "Wolf", "Candy", "Boss", "Fox", "Day", "Wife", "Leaf")],
                         ["Kobolds", "Wolves", "Candies", "Boss", "Foxes", "Days", "Wives", "Leaves"])

    def test_plural_terms_skips_short_and_non_alpha_words(self):
        self.assertEqual(b.plural_terms("Kobold Miner"), "Kobolds Miners")
        self.assertEqual(b.plural_terms("Fizzle's Claw of X"), "Claws")
        self.assertEqual(b.plural_terms("Large Candle"), "Larges Candles")


class TextTest(unittest.TestCase):
    def test_level_band(self):
        self.assertEqual([b.level_band(n) for n in (1, 10, 11, 60)], ["lvl1-10", "lvl1-10", "lvl11-20", "lvl51-60"])

    def test_side_of_mask(self):
        self.assertEqual(b.side_of_mask(1101), "alliance")
        self.assertEqual(b.side_of_mask(690), "horde")
        self.assertEqual(b.side_of_mask(1791), "both")

    def test_join_names(self):
        self.assertEqual(b.join_names(["a"]), "a")
        self.assertEqual(b.join_names(["a", "b", "c"]), "a, b and c")


if __name__ == "__main__":
    unittest.main()
