"""Episodic memory: a time-ordered record of events and the world state
derived from it (who is where, who holds what, what was given to whom).

Like the hippocampus, it stores each episode once, in order, and answers
questions by replaying the record rather than by storing conclusions.  It
also learns simple motivations: when an agent in a state (hungry, tired)
goes somewhere, the pairing is counted, so later it can say why someone went
to the kitchen or where a hungry person will go.
"""
from collections import Counter, defaultdict, deque


class World:
    def __init__(self):
        self.t = 0
        self.events = []                 # (t, kind, agent, obj, place, recipient)
        self.loc = defaultdict(list)     # entity -> [(t, place or None, certainty)]
        self.holder = defaultdict(list)  # object -> [(t, agent or None)]
        self.feelings = defaultdict(list)  # agent -> [(t, state)]
        self.motive = defaultdict(Counter)  # state -> Counter(place)
        self.directions = {}             # place -> {direction: neighbour place}

    # -- perception --------------------------------------------------------
    def observe(self, ev):
        self.t += 1
        t = self.t
        self.events.append((t, ev.kind, ev.agent, ev.obj, ev.place, ev.recipient))
        if ev.kind == "move":
            self.loc[ev.agent].append((t, ev.place, 1.0))
            for st in self.current_feelings(ev.agent):
                self.motive[st][ev.place] += 1
        elif ev.kind == "be_at":
            self.loc[ev.agent].append((t, ev.place, 1.0))
        elif ev.kind == "either_at":
            self.loc[ev.agent].append((t, tuple(ev.place), 0.5))
        elif ev.kind == "not_at":
            self.loc[ev.agent].append((t, ("not", ev.place), 0.0))
        elif ev.kind == "take":
            self.holder[ev.obj].append((t, ev.agent))
        elif ev.kind == "drop":
            place = self.where(ev.agent)[0]
            self.holder[ev.obj].append((t, None))
            if isinstance(place, str):
                self.loc[ev.obj].append((t, place, 1.0))
        elif ev.kind == "give":
            self.holder[ev.obj].append((t, ev.recipient))
        elif ev.kind == "feel":
            self.feelings[ev.agent].append((t, ev.obj))

    def knows(self, x):
        return x in self.loc or x in self.holder or x in self.feelings

    def add_direction(self, a, rel, b):
        """a is <rel> of b, e.g. kitchen north_of hallway."""
        d = rel[:-3]
        opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}[d]
        self.directions.setdefault(b, {})[d] = a
        self.directions.setdefault(a, {})[opposite] = b

    # -- state ---------------------------------------------------------------
    def current_holder(self, obj, t=None):
        h = [x for x in self.holder.get(obj, []) if t is None or x[0] <= t]
        return h[-1][1] if h else None

    def where(self, x, t=None, _depth=0):
        """Location of x at time t (default now): (place, supporting events)."""
        t = self.t if t is None else t
        if _depth > 5:
            return None, []
        holder_hist = [h for h in self.holder.get(x, []) if h[0] <= t]
        own = [l for l in self.loc.get(x, []) if l[0] <= t]
        if holder_hist and holder_hist[-1][1] is not None and \
                (not own or holder_hist[-1][0] >= own[-1][0]):
            agent = holder_hist[-1][1]
            th = holder_hist[-1][0]
            # the object goes wherever its holder goes after picking it up
            place, sup = self.where(agent, t, _depth + 1)
            return place, sup + [self._ev(th)]
        if not own:
            return None, []
        tl, place, cert = own[-1]
        return place, [self._ev(tl)]

    def _ev(self, t):
        for e in self.events:
            if e[0] == t:
                return e
        return None

    def location_history(self, x):
        """Successive places of x (following holders for objects)."""
        hist = []
        for t in range(1, self.t + 1):
            p, _ = self.where(x, t)
            if isinstance(p, str) and (not hist or hist[-1] != p):
                hist.append(p)
        return hist

    def where_before(self, x, place):
        hist = self.location_history(x)
        for i in range(len(hist) - 1, 0, -1):
            if hist[i] == place:
                return hist[i - 1]
        return None

    def is_at(self, x, place):
        """'yes', 'no' or 'maybe'."""
        p, _ = self.where(x)
        if p is None:
            return None
        if isinstance(p, tuple) and p and p[0] == "not":
            return "no" if p[1] == place else "maybe"
        if isinstance(p, tuple):
            return "maybe" if place in p else "no"
        return "yes" if p == place else "no"

    def carrying(self, agent):
        return [o for o in self.holder if self.current_holder(o) == agent]

    def who_gave(self, obj, recipient=None):
        for e in reversed(self.events):
            if e[1] == "give" and e[3] == obj and (recipient is None or e[5] == recipient):
                return e[2]
        return None

    def what_gave(self, agent, recipient):
        for e in reversed(self.events):
            if e[1] == "give" and e[2] == agent and e[5] == recipient:
                return e[3]
        return None

    def who_received(self, obj, giver=None):
        for e in reversed(self.events):
            if e[1] == "give" and e[3] == obj and (giver is None or e[2] == giver):
                return e[5]
        return None

    def current_feelings(self, agent):
        f = self.feelings.get(agent, [])
        return [f[-1][1]] if f else []

    def why_went(self, agent, place):
        """The agent's state before going to place, preferring states whose
        learned association with that place is strongest."""
        cands = [st for _, st in self.feelings.get(agent, [])]
        if not cands:
            return None
        return max(reversed(cands), key=lambda st: self.motive[st][place])

    def will_go(self, agent):
        for st in reversed([s for _, s in self.feelings.get(agent, [])]):
            if self.motive[st]:
                return self.motive[st].most_common(1)[0][0], st
        return None, None

    def path(self, a, b):
        """Shortest route between places using known directions (BFS)."""
        q = deque([(a, [])])
        seen = {a}
        while q:
            x, p = q.popleft()
            if x == b:
                return p
            for d, y in self.directions.get(x, {}).items():
                if y not in seen:
                    seen.add(y)
                    q.append((y, p + [d]))
        return None
