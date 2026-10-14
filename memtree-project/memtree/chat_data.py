"""Natural fine-tune data: conversations replayed through the real runtime.

Each dialogue is a run of turns between "me" and Phi:
  fact     you tell her something ("My dog's name is Biscuit.")
  filler   ordinary questions, answered in Phi's own words (self-distilled)
  question you ask about an earlier fact in natural wording

Turns are replayed with the exact memory machinery the chat runtime uses (log
gate, watchdog, content search, same ingest order, same system block), so the
prompts she trains on are the prompts she will see. For a question:
  * fact still in fresh memory   -> answer directly, in a sentence
  * fact further back            -> <|mem_ref|>plain words<|/mem_ref|>, the
                                    watchdog's real delivery, then the answer
  * never told                   -> reference, "not found", honest answer
Filler turns teach her to answer normally without referencing anything.
"""
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .runtime import (MEM_REF, MEM_REF_END, clean_reply, file_user_turn, render_delivery,
                      render_memory, whisper_query, window_budgets)
from .store import LogGate, MemoryStore
from .watchdog import Watchdog

SLOTS = [
    {"name": "dog", "values": ["Biscuit", "Pepper", "Mochi", "Rufus", "Juniper", "Waffles", "Ziggy", "Clover"],
     "fact": ["My dog's name is {v}.", "I have a dog called {v}.", "Our dog is named {v}, she's a sweetheart."],
     "q": ["What's my dog called?", "Do you remember my dog's name?", "What did I say my dog's name was?"],
     "ref": ["dog name", "dog"],
     "a": ["Your dog is called {v}.", "Your dog's name is {v}."]},
    {"name": "job", "values": ["welder", "nurse", "electrician", "teacher", "line cook", "truck driver",
                               "software tester", "carpenter"],
     "fact": ["I work as a {v}.", "My job? I'm a {v}.", "I've been working as a {v} for about six years now."],
     "q": ["What do I do for work?", "Do you remember my job?", "What's my line of work again?"],
     "ref": ["work job", "job work"],
     "a": ["You work as a {v}.", "You told me you're a {v}."]},
    {"name": "city", "values": ["Tulsa", "Duluth", "Tacoma", "Savannah", "Boise", "Lisbon", "Leeds", "Cork"],
     "fact": ["I live in {v}.", "I'm based in {v}.", "Home for me is {v}."],
     "q": ["Where do I live?", "Which city am I in, do you remember?", "Where did I say I'm based?"],
     "ref": ["live based home", "home live"],
     "a": ["You live in {v}.", "You're based in {v}."]},
    {"name": "sister", "values": ["Lisbon", "Denver", "Osaka", "Glasgow", "Austin", "Montreal", "Perth", "Seville"],
     "fact": ["My sister just moved to {v}.", "My sister relocated to {v} last month.",
              "Big news, my sister moved to {v}."],
     "q": ["Where did my sister move?", "Which city did my sister move to?", "Remind me where my sister lives now?"],
     "ref": ["sister moved", "sister"],
     "a": ["Your sister moved to {v}.", "She moved to {v}."]},
    {"name": "food", "values": ["pho", "lasagna", "tacos al pastor", "chicken curry", "sushi", "shakshuka",
                                "pierogi", "gumbo"],
     "fact": ["My favorite food is {v}.", "I could eat {v} every day.", "Nothing beats {v}, it's my favorite food."],
     "q": ["What's my favorite food?", "Do you remember what food I love?", "What did I say I could eat every day?"],
     "ref": ["favorite food eat", "food favorite eat"],
     "a": ["Your favorite food is {v}.", "You love {v}."]},
    {"name": "allergy", "values": ["peanuts", "shellfish", "penicillin", "cats", "bee stings", "latex",
                                   "pollen", "dairy"],
     "fact": ["I'm allergic to {v}.", "Heads up, I have a {v} allergy.", "I can't be around {v}, I'm allergic."],
     "q": ["What am I allergic to?", "Do you remember my allergy?", "What was I allergic to again?"],
     "ref": ["allergic allergy", "allergy"],
     "a": ["You're allergic to {v}.", "Your allergy is {v}."]},
    {"name": "car", "values": ["a 2009 Civic", "an old Ford Ranger", "a blue Corolla", "a Subaru Outback",
                               "a beat-up Jeep", "a white Prius", "a Mazda 3", "a red Silverado"],
     "fact": ["I drive {v}.", "My car is {v}.", "I've been driving {v} for years."],
     "q": ["What car do I drive?", "Do you remember my car?", "What did I say I drive?"],
     "ref": ["car drive", "drive car"],
     "a": ["You drive {v}.", "Your car is {v}."]},
    {"name": "hobby", "values": ["rock climbing", "woodworking", "chess", "baking bread", "fishing",
                                 "playing bass", "birdwatching", "pottery"],
     "fact": ["In my free time I'm into {v}.", "My hobby is {v}.", "Lately I've been really into {v}."],
     "q": ["What's my hobby?", "What do I do in my free time?", "Do you remember what I'm into lately?"],
     "ref": ["hobby free time lately", "hobby lately free"],
     "a": ["You're into {v}.", "Your hobby is {v}."]},
    {"name": "kid", "values": ["Mateo", "Ava", "Noah", "Lily", "Kai", "Nora", "Eli", "Zoe"],
     "fact": ["My son's name is {v}.", "My kid {v} just turned seven.", "I have a son named {v}."],
     "q": ["What's my son's name?", "Do you remember my kid's name?", "What did I say my son is called?"],
     "ref": ["son kid", "kid son name"],
     "a": ["Your son's name is {v}.", "Your kid is {v}."]},
    {"name": "birthday", "values": ["March", "July", "October", "January", "May", "September", "December", "April"],
     "fact": ["My birthday is in {v}.", "I was born in {v}.", "My birthday's coming up in {v}."],
     "q": ["When's my birthday?", "What month is my birthday?", "Do you remember when I was born?"],
     "ref": ["birthday born", "born birthday"],
     "a": ["Your birthday is in {v}.", "You were born in {v}."]},
]

NEVER_TOLD = ["I don't think you've told me that yet.",
              "I don't have that in my memory. Want to tell me?",
              "You haven't mentioned that to me, as far as I can find."]

ACKS = ["Got it, I'll remember that.", "Noted! Thanks for telling me.", "Good to know, I'll keep that in mind.",
        "Thanks for sharing that.", "Okay, I've got that down."]

# Fallback filler (used only when no self-distilled answers are supplied).
CANNED_FILLER = [
    ("What's a good way to cook rice?", "Rinse the rice, then simmer it with about 1.5 to 2 parts water per part rice, covered, on low heat for around 15 minutes. Let it rest off the heat for 5 minutes before fluffing."),
    ("Tell me a fun fact about octopuses.", "Octopuses have three hearts: two pump blood through the gills and one through the rest of the body."),
    ("How far is the moon?", "On average the Moon is about 384,400 kilometers (238,900 miles) from Earth."),
    ("Give me a one-line joke.", "I told my wife she was drawing her eyebrows too high. She looked surprised."),
    ("Why is the sky blue?", "Sunlight scatters off air molecules, and shorter blue wavelengths scatter much more than red ones, so blue light reaches your eyes from every direction."),
    ("How do I boil an egg?", "Place the egg in boiling water for about 7 minutes for a jammy yolk or 10 minutes for hard-boiled, then cool it in cold water."),
    ("What's the capital of Australia?", "The capital of Australia is Canberra."),
    ("Explain how a rainbow forms.", "Sunlight enters raindrops, bends, reflects off the back of the drop and bends again on the way out. Different colors bend by different amounts, which spreads the light into a band of colors."),
]


def _fmt(t: str, v: str) -> str:
    return t.replace("{v}", v)


class DialogueSim:
    """Mirror of MemoryRuntime.chat without a model: same overflow filing, same
    whispers, same system block, same reference call."""

    def __init__(self, peer: str = "me", deliver_chars: int = 400, window: int = 256):
        self.store = MemoryStore()
        self.gate = LogGate(self.store)
        self.dog = Watchdog(self.store)
        self.peer, self.deliver_chars, self.turn = peer, deliver_chars, 0
        self.budget = window_budgets(window)
        self.system = self.shown = self.tail = ""

    def turn_example(self, user: str, target_fn) -> Dict:
        self.turn += 1
        shown, qkey, tail = file_user_turn(self.gate, self.peer, self.turn, user,
                                           self.budget["tail_words"], self.budget["span_words"])
        whispers = self.dog.whispers(self.peer, whisper_query(tail), exclude={qkey})
        self.system = render_memory(self.dog, self.peer, entry_chars=self.budget["entry_chars"],
                                    whispers=whispers)
        self.shown = shown
        self.tail = tail
        target = target_fn(qkey)
        if target is None:
            return None
        self.gate.ingest(self.peer, f"turn{self.turn}:assistant", clean_reply(target))
        self.dog.sweep(self.peer, budget=8)
        return {"system": self.system, "user": shown, "assistant": target}

    def visible(self, value: str) -> bool:
        """Is the answer sharp in front of Phi: fresh memory, a whisper, or the
        part of the message inside her window? (The whole message is in her
        prompt, but past the window she only has the tree's faint summary.)"""
        return value in self.system or value in getattr(self, "tail", "")

    def in_context(self, key: str) -> bool:
        return any(n.key == key for n in self.dog.context_view(self.peer))

    def reference(self, words: str, qkey: str):
        return self.dog.reference(self.peer, words, budget_chars=self.deliver_chars,
                                  expand=True, exclude={qkey})


def _target(sim, stats, key, v, words, ans):
    def target(qkey):
        if sim.visible(v):  # in fresh memory or whispered: trust it and answer
            stats["direct"] += 1
            return ans
        d = sim.reference(words, qkey)
        if not (d.found and v in str(d.value)):  # search found the wrong memory: don't train on it
            stats["dropped"] += 1
            return None
        stats["reference"] += 1
        return f"{MEM_REF}{words}{MEM_REF_END}{render_delivery(d)}{ans}"
    return target


def build_dialogue(rng: random.Random, filler: Sequence[Tuple[str, str]],
                   n_facts=(2, 4), gap=(0, 6), never_told_p: float = 0.15,
                   keep_filler: float = 0.3, ask_now_p: float = 0.25) -> Tuple[List[Dict], Dict]:
    sim = DialogueSim()
    stats = {"fact": 0, "filler": 0, "direct": 0, "reference": 0, "never_told": 0, "dropped": 0}
    slots = rng.sample(SLOTS, rng.randint(*n_facts) + 1)
    told, untold = slots[:-1], slots[-1]
    facts = {}  # slot name -> (turn key, value)
    examples: List[Dict] = []
    pending = list(told)
    asked = []

    def add(ex):
        if ex is not None:
            examples.append(ex)

    def filler_turns(k):
        for _ in range(k):
            u, a = rng.choice(filler)
            ex = sim.turn_example(u, lambda q: a)  # always replayed, so memory evolves
            if rng.random() < keep_filler:  # but only some are trained on
                add(ex)
                stats["filler"] += 1

    while pending or len(asked) < len(told):
        if pending and (not facts or rng.random() < 0.6):
            s = pending.pop()
            v = rng.choice(s["values"])
            text = _fmt(rng.choice(s["fact"]), v)
            add(sim.turn_example(text, lambda q, s=s, v=v: rng.choice(ACKS)))
            facts[s["name"]] = (f"turn{sim.turn}:user", v, s)
            stats["fact"] += 1
            if rng.random() < ask_now_p:  # ask right away: still in fresh memory
                continue_ask = s["name"]
            else:
                continue_ask = None
                filler_turns(rng.randint(*gap))
            if continue_ask is None:
                continue
        if 'continue_ask' in locals() and continue_ask and continue_ask not in asked:
            name = continue_ask
            continue_ask = None
            asked.append(name)
            key, v, s = facts[name]
            q, words, ans = rng.choice(s["q"]), rng.choice(s["ref"]), _fmt(rng.choice(s["a"]), v)
            add(sim.turn_example(q, _target(sim, stats, key, v, words, ans)))
            filler_turns(rng.randint(*gap))
            continue
        remaining = [n for n in facts if n not in asked]
        if not remaining:
            break
        name = rng.choice(remaining)
        asked.append(name)
        key, v, s = facts[name]
        q, words, ans = rng.choice(s["q"]), rng.choice(s["ref"]), _fmt(rng.choice(s["a"]), v)
        add(sim.turn_example(q, _target(sim, stats, key, v, words, ans)))
        filler_turns(rng.randint(*gap))
    if rng.random() < never_told_p:
        q, words = rng.choice(untold["q"]), rng.choice(untold["ref"])

        def nt(qkey):
            d = sim.reference(words, qkey)
            if d.found:  # something matched by accident: skip rather than teach a wrong answer
                stats["dropped"] += 1
                return None
            stats["never_told"] += 1
            return f"{MEM_REF}{words}{MEM_REF_END}{render_delivery(d)}{rng.choice(NEVER_TOLD)}"
        add(sim.turn_example(q, nt))
    return examples, stats


# Names for planted codes in training. eval_recall uses a different list.
TRAIN_NAMES = ["Harlan", "Imogen", "Bastian", "Callista", "Dmitri", "Esperanza", "Fenwick", "Gwendolyn",
               "Horatio", "Isolde", "Jasper", "Kalinda", "Leopold", "Magdalena", "Nikolai", "Ottoline"]
PASSAGE_ACKS = ["Thanks, I've read through it.", "Got it, that's an interesting read.",
                "Okay, I've taken that in.", "Thanks for sharing the passage."]


def _fallback_passages(filler) -> List[str]:
    return [" ".join(a for _, a in filler)] * 4


def _cut(rng, passage: str, n_words: int) -> str:
    words = passage.split()
    if len(words) <= n_words:
        return passage
    start = rng.randrange(0, len(words) - n_words)
    return " ".join(words[start:start + n_words])


def _plant(text: str, needle: str, depth: float) -> str:
    import re as _re
    pos = int(len(text) * depth)
    m = _re.compile(r"[.!?][\"')\]]?\s").search(text, pos)
    cut = m.end() if m else len(text)
    return text[:cut] + needle + " " + text[cut:]


def build_passage_dialogue(rng: random.Random, filler, passages: Sequence[str],
                           words=(150, 2500), same_message_p: float = 0.5) -> Tuple[List[Dict], Dict]:
    """A long pasted passage with one planted fact, then a question about it,
    either at the end of the same message or a few turns later. Overflow filing
    sends most of a long passage out to the watchdog, exactly as in chat."""
    sim = DialogueSim()
    stats = {"fact": 0, "filler": 0, "direct": 0, "reference": 0, "never_told": 0, "dropped": 0,
             "passage": 1}
    examples: List[Dict] = []
    body = _cut(rng, rng.choice(passages), rng.randint(*words))
    if rng.random() < 0.6:
        name, code = rng.choice(TRAIN_NAMES), str(rng.randint(10000, 99999))
        needle, v = f"The secret code for {name} is {code}.", code
        q = rng.choice([f"What is the secret code for {name}?", f"What was {name}'s secret code?",
                        f"Remind me, what's the code for {name}?"])
        words_ref, ans = f"secret code {name}", f"The secret code for {name} is {code}."
    else:
        slot = rng.choice(SLOTS)
        v = rng.choice(slot["values"])
        needle = _fmt(rng.choice(slot["fact"]), v)
        q, words_ref, ans = rng.choice(slot["q"]), rng.choice(slot["ref"]), _fmt(rng.choice(slot["a"]), v)
    text = _plant(body, needle, rng.uniform(0.05, 0.95))
    if rng.random() < same_message_p:
        ex = sim.turn_example(f"{text}\n\n{q}", _target(sim, stats, None, v, words_ref, ans))
        if ex is not None:
            examples.append(ex)
        return examples, stats
    ack = sim.turn_example(text, lambda qk: rng.choice(PASSAGE_ACKS))
    if rng.random() < 0.3:  # train on a few acks only, so she doesn't answer every paste with one
        examples.append(ack)
    for _ in range(rng.randint(0, 3)):
        u, a = rng.choice(filler)
        sim.turn_example(u, lambda qk, a=a: a)
    ex = sim.turn_example(q, _target(sim, stats, None, v, words_ref, ans))
    if ex is not None:
        examples.append(ex)
    return examples, stats


def build_dataset(n_dialogues: int, filler: Optional[Sequence[Tuple[str, str]]] = None,
                  seed: int = 0, passages: Optional[Sequence[str]] = None,
                  passage_frac: float = 0.4, **kw) -> Tuple[List[Dict], Dict]:
    """Mix of personal-fact chats and long-passage dialogues."""
    rng = random.Random(seed)
    filler = list(filler or CANNED_FILLER)
    passages = list(passages or _fallback_passages(filler))
    out: List[Dict] = []
    totals: Dict[str, int] = {}
    for _ in range(n_dialogues):
        if rng.random() < passage_frac:
            ex, st = build_passage_dialogue(rng, filler, passages)
        else:
            ex, st = build_dialogue(rng, filler, **kw)
        out += ex
        for k, v in st.items():
            totals[k] = totals.get(k, 0) + v
    return out, totals
