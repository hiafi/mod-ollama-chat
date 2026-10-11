# RAG from a world database

`build_world_rag.py` writes RAG JSON files (the schema in `data/rag/README.md`) with facts derived from an
AzerothCore world database export, so a bot can say where a quest giver is, what to kill or collect for a quest,
and where trainers, innkeepers and flight masters stand. Python 3.11+, standard library only. It does not
touch the database; it reads TSV exports and three client DBC files.

```
python3 tools/rag/build_world_rag.py --inputs <dir> --out <dir> [--max-level 60] [--min-level 1]
                                     [--maps 0,1] [--faction both|alliance|horde]
                                     [--npc-kinds class-trainer,profession-trainer,innkeeper,flightmaster]
                                     [--keyword-repeat 1] [--no-quests] [--no-npcs]
python3 -m unittest discover -s tools/rag      # tests
```

Output, per continent (`0` = Eastern Kingdoms, `1` = Kalimdor; other maps are named `map_<id>`):
`quests_<continent>.json` and `npcs_<continent>.json`. Entries are sorted by id and the output is
deterministic, so a regeneration gives a reviewable diff.

## Inputs

`--inputs` is a directory holding the TSVs below (`mysql -B`: header row, tab separated; `\t`, `\n`, `\\` are
escaped and NULL is written `NULL`; the script undoes that) and a `dbc/` folder with `AreaTable.dbc`,
`WorldMapArea.dbc` and `TaxiNodes.dbc` from the 3.3.5a client (`DBFilesClient`, or the server's `dbc/` folder).

```sh
q() { mysql -B acore_world -e "$2" > "$1.tsv"; }
q quest_template            "SELECT * FROM quest_template"
q quest_template_addon      "SELECT * FROM quest_template_addon"
q creature_queststarter     "SELECT id, quest FROM creature_queststarter"
q creature_questender       "SELECT id, quest FROM creature_questender"
q gameobject_queststarter   "SELECT id, quest FROM gameobject_queststarter"
q creature                  "SELECT guid, id, map, zoneId, areaId, position_x, position_y, position_z, spawntimesecs FROM creature WHERE map IN (0,1)"
q creature_template         "SELECT entry, name, subname, minlevel, maxlevel, \`rank\`, faction, npcflag, \`type\` FROM creature_template"
q creature_loot_template    "SELECT * FROM creature_loot_template"
q reference_loot_template   "SELECT * FROM reference_loot_template"
q gameobject                "SELECT guid, id, map, zoneId, areaId, position_x, position_y, position_z FROM gameobject WHERE map IN (0,1)"
q gameobject_template       "SELECT entry, \`type\`, name FROM gameobject_template"
q item_template             "SELECT entry, name, Quality, class, subclass, ItemLevel, RequiredLevel, startquest FROM item_template"
```

Only the columns the script reads matter; extra columns are ignored. (On a schema where `creature.id` is split into
`id1`/`id2`/`id3`, select `id1 AS id`.)

## What is derived

**Geography** (`WorldMapArea.dbc`, `AreaTable.dbc`, `TaxiNodes.dbc`). World X is north and Y is west, in yards.

* *Zone of a point.* `WorldMapArea` gives each zone a rectangle (`LocTop`/`LocBottom` bound X,
  `LocLeft`/`LocRight` bound Y). The continent-wide rectangle is skipped. Rectangles are map-render extents and
  overlap their neighbours, so the script takes the rectangle the point sits deepest inside (distance to the nearest
  edge as a fraction of that axis), except that a capital city's rectangle wins when its depth is within 0.1 of the
  deepest. Zone names come from `AreaTable`.
* *Zone hints.* A quest's `QuestSortID` (and, for NPCs, the zone named in the nearest flight node's name, such as
  `Darkshire, Duskwood`) breaks ties between overlapping boxes when the point is inside that zone's box and not much
  shallower than the deepest. A sort id that is a subzone (Northshire Valley) hints its parent zone and is shown as
  the quest's subzone.
* *Subzones.* No client file maps a point to a subzone. They are learned from quest givers: a child area used as
  `QuestSortID` by at least 3 quests and 2 givers whose spawns sit within 300 yd (median) and 700 yd (90th
  percentile) of their centre is a subzone; a point with two of its givers inside 200 yd is `in <subzone>`. This
  finds the starting valleys (Northshire Valley, Coldridge Valley, Shadowglen, Deathknell, Valley of Trials, Red
  Cloud Mesa) and a few towns, never a dungeon.
* *Named landmark.* The nearest flight-path town, with a flight master standing within 150 yd of the node (this
  removes the test, transport, quest-flight and hidden nodes `TaxiNodes.dbc` also holds). Within 400 yd: `near X`.
  400 to 800 yd: `about 650 yards west of X`. Further: omitted. Never an invented subzone.
* *Direction and distance.* 8-point compass, distance rounded to 50 yd, under 60 yd `right next to`.
* *Spawn clusters.* Greedy density clustering (250 yd for quest targets, 300 yd for NPC spawn groups). A quest target
  is looked up among its spawns within 1500 yd of the quest giver; the densest cluster is reported, a second one only
  when it has at least 60% as many spawns. With none inside 1500 yd the text says `further away, nearest in <zone>`.

**Quest entries** (`quest_<id>`): every quest available at `--min-level`..`--max-level` that has a creature or
gameobject giver with a spawn on `--maps`. Content: who gives it and a landmark, then for each kill/interact
objective the target and where it is, for each collect objective the item, up to 3 sources (creature loot, direct rows
and one level of `reference_loot_template`; quest-required and high-chance sources first, normal mobs before
elites, common before rare) and where they are, the turn-in NPC when it differs, and the previous quest of a chain.
The zone is `QuestSortID` when it is a zone, else the giver's zone. Faction is from `AllowableRaces`; when that is 0
and every giver belongs to a creature faction that is clearly one side, the quest is tagged with that side.

**NPC entries** (`npc_<entry>_<group>`): class trainers, profession trainers, innkeepers and flight masters, one
per spawn group, with landmark and zone. The side of a creature faction is learned from the race-restricted quests
its members give (at least 3 and 90% agreement); a faction that cannot be settled gets no side.

`keywords` hold the title, giver, zone, landmark, targets and items, and the place, name and title terms
are repeated `--keyword-repeat` times (the module's cosine uses raw term frequency, so a repeated distinctive word
outweighs the is/in/the every question shares; 1 measured best); `tags` hold `quest`/`npc`, the zone slug, the
side and the level band. No string contains a double quote (the module's config tooling forbids them); they are
turned into single quotes.

## What is not derived, and known limits

* **Subzones.** Only the ones learned from quest givers (above) are named; towns without a flight master and
  without enough sorted quests (Goldshire, Razor Hill) are only named relative to a flight point.
* **Zone near borders.** A point within a few hundred yards of a zone border can get the neighbour's name.
* **Loot.** `creature_loot_template.Entry` is taken to be the creature entry (`lootid` is not exported), and
  gameobject loot is not used at all (a chest's loot id is `Data1`, not necessarily its entry). Vendor and
  profession sources of quest items are not known, so such items get no source.
* **Object givers and enders.** Only creature quest enders are used; objects that start quests are named but
  `gameobject_questender` is not read.
* **Dungeons.** Targets inside instance maps are not on `--maps`, so dungeon quests have no locations.
* **Quest requirements** other than the previous quest (skill, reputation, class) are not described.
* **Scoring cost.** The module scores every entry per reply. With the stock dense scorer the cost is entries x
  vocabulary: about 1.6 s for the full output (4300 entries). See `feat/rag-sparse-similarity` in the same module for
  a sparse scorer that gives identical scores in about 3 ms. Without it, trim with `--max-level`, `--min-level`,
  `--npc-kinds` or `--faction`.
