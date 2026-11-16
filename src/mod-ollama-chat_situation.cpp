#include "mod-ollama-chat_situation.h"

#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat-utilities.h"

#include "AiFactory.h"
#include "Creature.h"
#include "Group.h"
#include "GroupReference.h"
#include "Log.h"
#include "Map.h"
#include "Player.h"
#include "PlayerbotAI.h"
#include "SpellAuraEffects.h"
#include "SpellInfo.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;

    struct SituationEvent
    {
        Clock::time_point when;
        std::string       text;
    };

    // One record per group, keyed by the group's guid. Strings and times only.
    struct GroupRecord
    {
        bool                      hasCombatEnd = false;
        Clock::time_point         lastCombatEnd{};
        Clock::time_point         touched{};
        std::deque<SituationEvent> events;
    };

    std::mutex                                g_trackerMutex;
    std::unordered_map<uint64_t, GroupRecord> g_groupRecords;

    constexpr size_t   kMaxRecords        = 64;
    constexpr uint32_t kRecordLifetimeSec = 900;

    double SecondsSince(Clock::time_point t, Clock::time_point now)
    {
        return std::chrono::duration<double>(now - t).count();
    }

    // Caller holds g_trackerMutex.
    GroupRecord& RecordFor(uint64_t key, Clock::time_point now)
    {
        if (g_groupRecords.size() >= kMaxRecords && g_groupRecords.find(key) == g_groupRecords.end())
        {
            for (auto it = g_groupRecords.begin(); it != g_groupRecords.end();)
            {
                if (SecondsSince(it->second.touched, now) > kRecordLifetimeSec)
                    it = g_groupRecords.erase(it);
                else
                    ++it;
            }

            // Still full of recent groups: drop the least recently touched
            // ones so the map really stays at kMaxRecords.
            while (g_groupRecords.size() >= kMaxRecords)
            {
                auto oldest = g_groupRecords.begin();
                for (auto it = g_groupRecords.begin(); it != g_groupRecords.end(); ++it)
                    if (it->second.touched < oldest->second.touched)
                        oldest = it;
                g_groupRecords.erase(oldest);
            }
        }

        GroupRecord& rec = g_groupRecords[key];
        rec.touched = now;
        return rec;
    }

    void NoteEvent(Group* group, std::string text)
    {
        if (!group || !g_SituationEnabled || g_SituationEventWindowSeconds == 0)
            return;

        const auto now = Clock::now();
        std::lock_guard<std::mutex> lock(g_trackerMutex);
        GroupRecord& rec = RecordFor(group->GetGUID().GetRawValue(), now);
        rec.events.push_back({ now, std::move(text) });
        while (rec.events.size() > 3)
            rec.events.pop_front();
    }

    // --- describing people ------------------------------------------------

    // Tank, healer or dps. PlayerbotAI::IsTank/IsHeal read the strategy of a
    // bot and the talent spec of anyone else, so one call works for both.
    // (A Bestial Fury druid reads as a tank here; known and left alone.)
    const char* RoleOf(Player* p)
    {
        if (PlayerbotAI::IsTank(p))
            return "tank";
        if (PlayerbotAI::IsHeal(p))
            return "healer";
        return "dps";
    }

    bool UsesMana(Player* p)
    {
        return p->GetMaxPower(POWER_MANA) > 0;
    }

    uint32_t HealthPct(Player* p)
    {
        const uint32_t max = std::max<uint32_t>(1, p->GetMaxHealth());
        return static_cast<uint32_t>((uint64_t(p->GetHealth()) * 100 + max / 2) / max);
    }

    uint32_t ManaPct(Player* p)
    {
        const uint32_t max = std::max<uint32_t>(1, p->GetMaxPower(POWER_MANA));
        return static_cast<uint32_t>((uint64_t(p->GetPower(POWER_MANA)) * 100 + max / 2) / max);
    }

    // True when the player carries an aura of this type that only lasts while
    // seated (food and drink).
    bool HasSeatedAura(Player* p, AuraType type)
    {
        for (AuraEffect const* eff : p->GetAuraEffectsByType(type))
        {
            if (SpellInfo const* info = eff->GetSpellInfo())
                if (info->AuraInterruptFlags & AURA_INTERRUPT_FLAG_NOT_SEATED)
                    return true;
        }
        return false;
    }

    // "healer, 100% HP, 8% mana, drinking"; "dps, dead".
    std::string Describe(Player* p)
    {
        std::string out = RoleOf(p);

        if (!p->IsAlive())
            return out + ", dead";

        out += SafeFormat(", {}% HP", HealthPct(p));
        if (UsesMana(p))
            out += SafeFormat(", {}% mana", ManaPct(p));

        // Eating and drinking are seated regeneration auras: food is
        // SPELL_AURA_MOD_REGEN and drink SPELL_AURA_MOD_POWER_REGEN in the
        // 3.3.5 Spell.dbc (the OBS_MOD_* types are only a handful of odd
        // items). Requiring the not-seated interrupt flag keeps an unrelated
        // regeneration buff on a sitting player from counting.
        if (p->getStandState() == UNIT_STAND_STATE_SIT)
        {
            const bool drinking = HasSeatedAura(p, SPELL_AURA_MOD_POWER_REGEN) ||
                                  HasSeatedAura(p, SPELL_AURA_OBS_MOD_POWER);
            const bool eating   = HasSeatedAura(p, SPELL_AURA_MOD_REGEN) ||
                                  HasSeatedAura(p, SPELL_AURA_OBS_MOD_HEALTH);
            if (drinking && eating)
                out += ", eating and drinking";
            else if (drinking)
                out += ", drinking";
            else if (eating)
                out += ", eating";
        }

        return out;
    }

    // --- sections ---------------------------------------------------------

    std::string CombatPart(Player* bot, Player* speaker, Group* group)
    {
        bool inCombat = bot->IsInCombat() || (speaker && speaker->IsInCombat());
        if (group && !inCombat)
        {
            for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
            {
                Player* member = ref->GetSource();
                if (member && member->IsInWorld() && member->IsInCombat())
                {
                    inCombat = true;
                    break;
                }
            }
        }

        if (inCombat)
            return "in combat";

        if (group && g_SituationBetweenPullsSeconds > 0)
        {
            const auto now = Clock::now();
            std::lock_guard<std::mutex> lock(g_trackerMutex);
            auto it = g_groupRecords.find(group->GetGUID().GetRawValue());
            if (it != g_groupRecords.end() && it->second.hasCombatEnd &&
                SecondsSince(it->second.lastCombatEnd, now) <= g_SituationBetweenPullsSeconds)
                return "between pulls";
        }

        return "not in combat";
    }

    // "just now: Garn died, killed Edwin VanCleef", or empty.
    std::string EventsPart(Group* group)
    {
        if (!group || g_SituationEventWindowSeconds == 0)
            return "";

        const auto now = Clock::now();
        std::string out;
        std::lock_guard<std::mutex> lock(g_trackerMutex);
        auto it = g_groupRecords.find(group->GetGUID().GetRawValue());
        if (it == g_groupRecords.end())
            return "";

        for (SituationEvent const& ev : it->second.events)
        {
            if (SecondsSince(ev.when, now) > g_SituationEventWindowSeconds)
                continue;
            out += out.empty() ? "just now: " : ", ";
            out += ev.text;
        }
        return out;
    }

    // The other members: each one in turn for a party, a head count for a big raid.
    std::string GroupPart(Player* bot, Player* speaker, Group* group)
    {
        if (!group)
            return "";

        const uint32_t size = group->GetMembersCount();

        if (size > g_SituationRaidSummaryAbove)
        {
            uint32_t dead = 0;
            std::vector<uint32_t> healerMana;
            for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
            {
                Player* member = ref->GetSource();
                if (!member || member == bot || member == speaker || !member->IsInWorld())
                    continue;
                if (!member->IsAlive())
                {
                    ++dead;
                    continue;
                }
                if (UsesMana(member) && PlayerbotAI::IsHeal(member))
                    healerMana.push_back(ManaPct(member));
            }

            std::string out = SafeFormat("raid of {}", size);
            if (dead)
                out += SafeFormat(", {} dead", dead);
            if (!healerMana.empty())
            {
                std::sort(healerMana.begin(), healerMana.end());
                if (healerMana.size() > 5)
                    healerMana.resize(5);
                out += ", healers at ";
                for (size_t i = 0; i < healerMana.size(); ++i)
                    out += (i ? "/" : "") + std::to_string(healerMana[i]);
                out += "% mana";
            }
            return out;
        }

        std::string list;
        for (GroupReference* ref = group->GetFirstMember(); ref; ref = ref->next())
        {
            Player* member = ref->GetSource();
            if (!member || member == bot || member == speaker || !member->IsInWorld())
                continue;
            if (!list.empty())
                list += ", ";
            list += SafeFormat("{} ({})", member->GetName(), Describe(member));
        }
        if (list.empty())
            return "";
        return std::string(group->isRaidGroup() ? "raid: " : "party: ") + list;
    }
}

// --------------------------------------------------------------------------

std::string Situation_Build(Player* bot, Player* speaker)
{
    if (!g_SituationEnabled || !bot)
        return "";

    Group* group = bot->GetGroup();

    // Segments, joined with "; ". Only what is real: nothing is printed for
    // something that could not be worked out.
    std::vector<std::string> parts;

    // The open world already has zone and area in {extra_info}; an instance
    // is the part worth naming.
    if (Map* map = bot->GetMap())
    {
        if (map->IsDungeon())
            parts.push_back(std::string("in ") + map->GetMapName());
    }

    parts.push_back(CombatPart(bot, speaker, group));

    if (std::string events = EventsPart(group); !events.empty())
        parts.push_back(std::move(events));

    if (speaker && speaker != bot)
        parts.push_back(SafeFormat("speaker {}: {}", speaker->GetName(), Describe(speaker)));

    parts.push_back(SafeFormat("you: {}", Describe(bot)));

    if (std::string members = GroupPart(bot, speaker, group); !members.empty())
        parts.push_back(std::move(members));

    std::string details;
    for (std::string const& part : parts)
    {
        if (part.empty())
            continue;
        if (!details.empty())
            details += "; ";
        details += part;
    }

    if (details.empty())
        return "";

    return SafeFormat(g_SituationTemplate, fmt::arg("details", details));
}

// --------------------------------------------------------------------------
// Tracker hooks
// --------------------------------------------------------------------------

OllamaSituationTracker::OllamaSituationTracker()
    : PlayerScript("OllamaSituationTracker", {
          PLAYERHOOK_ON_PLAYER_LEAVE_COMBAT,
          PLAYERHOOK_ON_PLAYER_JUST_DIED,
          PLAYERHOOK_ON_CREATURE_KILL_CREDIT,
      }) { }

void OllamaSituationTracker::OnPlayerLeaveCombat(Player* player)
{
    if (!player || !g_SituationEnabled || g_SituationBetweenPullsSeconds == 0)
        return;

    Group* group = player->GetGroup();
    if (!group)
        return;

    const auto now = Clock::now();
    std::lock_guard<std::mutex> lock(g_trackerMutex);
    GroupRecord& rec = RecordFor(group->GetGUID().GetRawValue(), now);
    rec.hasCombatEnd  = true;
    rec.lastCombatEnd = now;
}

void OllamaSituationTracker::OnPlayerJustDied(Player* player)
{
    if (player)
        NoteEvent(player->GetGroup(), player->GetName() + " died");
}

void OllamaSituationTracker::OnPlayerCreatureKillCredit(Player* player, Creature* killed)
{
    if (!player || !killed)
        return;

    if (killed->isWorldBoss() || killed->IsDungeonBoss())
        NoteEvent(player->GetGroup(), "killed " + killed->GetName());
}
