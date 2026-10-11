"""Emote names the module resolves in a reply (``EmoteNameTable`` in src/mod-ollama-chat_expression.cpp).

Only the names matter to the evaluator: ProcessLlmResponse decides whether a
``*word*`` span or a bare ``/word`` is a gesture by whether the word resolves.
Regenerate this list if upstream adds emotes.
"""

EMOTE_NAMES = frozenset([
    "wave", "hello", "hi", "greet", "hail", "welcome", "bow", "curtsey", "salute", "thank", "bye", "farewell",
    "introduce", "congratulate", "grats", "praise", "commend", "highfive", "goodluck", "encourage", "toast",
    "nod", "agree", "yes", "no", "disagree", "shake", "shrug", "veto", "doubt", "ready", "laugh", "chuckle",
    "giggle", "cackle", "guffaw", "rofl", "snicker", "smirk", "grin", "smile", "joke", "jk", "tease", "wink",
    "happy", "cheer", "applaud", "clap", "golfclap", "dance", "flex", "victory", "gloat", "cry", "sigh",
    "mourn", "groan", "whine", "moan", "panic", "cower", "cringe", "plead", "beg", "grovel", "apologize",
    "sorry", "comfort", "soothe", "pity", "scared", "surrender", "helpme", "faint", "facepalm", "blame",
    "fail", "embarrass", "blush", "bashful", "shy", "angry", "glare", "growl", "snarl", "roar", "threaten",
    "taunt", "mock", "insult", "rude", "spit", "snub", "challenge", "brandish", "enemy", "charge", "slap",
    "shoo", "stare", "frown", "glower", "ponder", "think", "confused", "puzzle", "boggle", "curious", "peer",
    "listen", "point", "beckon", "talk", "amaze", "awe", "gasp", "surprised", "eyebrow", "lost", "serious",
    "badfeeling", "tired", "yawn", "sleep", "bored", "hungry", "thirsty", "cold", "drink", "eat", "oom",
    "healme", "work", "pray", "kneel", "stand", "sit", "calm", "brb", "wait", "follow", "flee", "incoming",
    "cough", "whistle", "shiver", "scratch", "crack", "raise", "ding", "rolleyes", "eyeroll", "scoff",
    "snort", "grunt", "scowl", "pout", "crossarms", "foldarms", "shakefist", "warn", "scold", "idea",
    "search", "look", "suspicious", "shifty", "blank", "gaze", "blink", "tap", "sneak", "pat", "pet",
    "cuddle", "love", "promise", "offer", "truce", "mercy", "proud", "yw", "luck", "sad", "regret", "nervous",
    "sweat", "headache", "shudder", "jealous", "sneeze", "hiccup", "sniff", "sing", "shout", "silence",
    "hurry", "duck", "bounce", "fidget", "twiddle", "mutter",
])
