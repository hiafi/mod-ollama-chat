#ifndef MOD_OLLAMA_CHAT_SITUATION_H
#define MOD_OLLAMA_CHAT_SITUATION_H

#include "ScriptMgr.h"
#include <string>

class Player;

// --------------------------------------------------------------------------
// The {situation} line of a reply prompt: where the bot is, whether the group
// is fighting, how the speaker and the bot are doing, and who else is in the
// group. It is what lets a bot tell 'mana' said by a healer at 8% mana from a
// stray word.
//
// Built from live game state, so WORLD THREAD ONLY. Empty when
// OllamaChat.SituationTemplate is empty, which is the default.
// --------------------------------------------------------------------------

// `speaker` is whoever the reply answers (a real player or a bot).
std::string Situation_Build(Player* bot, Player* speaker);

// Keeps the little per-group history the line needs and the game does not
// keep: when the group last left combat, and the last few things that
// happened (a death, a boss kill).
//
// These hooks run on map update threads, so the state they write is behind a
// mutex and holds only strings and times, never a Player pointer.
class OllamaSituationTracker : public PlayerScript
{
public:
    OllamaSituationTracker();
    void OnPlayerLeaveCombat(Player* player) override;
    void OnPlayerJustDied(Player* player) override;
    void OnPlayerCreatureKillCredit(Player* player, Creature* killed) override;
};

#endif // MOD_OLLAMA_CHAT_SITUATION_H
