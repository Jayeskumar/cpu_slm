"""Saying it the reader's way: how people actually ask, mapped onto the forms
the reader knows.

People rarely ask in canonical form.  They wrap questions ("could you tell me
whether dolphins are mammals"), embed them ("tell me what a giraffe is"), ask
for checks of a statement ("is it true that bats are birds"), shout, drop the
verb ("capital of Spain?"), and expect a mind to do sums.  `rephrase` undoes
these one rule at a time, keeping the speaker's capitals for the names, and
reports whether the answer must be turned round ("is it true that penguins
can't fly" is asked as "can penguins fly", and the verdict flipped).
`arithmetic` answers sums exactly.
"""
import math
import re

from . import language as L

_I = re.I

# leading words that carry no content
_FILLER = re.compile(r"^(?:please|hey|hi|hello|ok(?:ay)?|so|well|um+|uh+|mnemo)\b[\s,]*", _I)
# statements to learn: 'remember that X'
_TEACH = re.compile(r"^(?:remember|note|learn|keep in mind|bear in mind|did you know|"
                    r"you should know|fyi|for your information)(?: that)?[\s,:]+", _I)
# a statement to check: 'is it true that X'
_CHECK = re.compile(r"^(?:is it true|is it the case|would you say|do you think|do you believe|"
                    r"would you agree)(?: that)?\s+", _I)
# an embedded question: 'could you tell me whether/what ...'
_ASK = re.compile(r"^(?:(?:can|could|would|will) you (?:please )?(?:tell me|say|explain)|"
                  r"tell me|do you know|i (?:wonder|want to know|would like to know|'d like to "
                  r"know)|i am curious|i'm curious|explain)\s+", _I)
_WH = r"what|where|who|whom|which|when|how many|how much|why|how"

_SPELLED = [(r"\bwhats\b", "what is"), (r"\bwhat're\b", "what are"), (r"\bwheres\b", "where is"),
            (r"\bwhos\b", "who is"), (r"\bhows\b", "how is"), (r"\bwats\b", "what is"),
            (r"\bu\b", "you"), (r"\br\b", "are")]

_RELS = r"capital|currency|language|languages|opposite|colour|color|population|continent"


def _question(text):
    text = text.strip().rstrip(".!?").strip()
    return text[:1].upper() + text[1:] + "?"


def _invert(stmt):
    """'bats are birds' -> 'are bats birds'; 'cats eat mice' -> 'do cats eat
    mice'.  Returns (question, negated) or None."""
    s = stmt.strip().rstrip(".!?").strip()
    neg = False
    s2 = re.sub(r"\b(can't|cannot|can not)\b", "can", s, flags=_I)
    s2 = re.sub(r"\b(isn't|is not)\b", "is", s2, flags=_I)
    s2 = re.sub(r"\b(aren't|are not)\b", "are", s2, flags=_I)
    s2 = re.sub(r"\b(doesn't|does not|don't|do not)\s+", "", s2, flags=_I)
    s2 = re.sub(r"\b(has|have) no\b", r"\1", s2, flags=_I)
    if s2 != s:
        neg, s = True, s2
    m = re.match(r"^(.+?)\s+(is|are|was|were|can|could|will|should)\s+(.+)$", s, _I)
    if m:
        return f"{m.group(2)} {m.group(1)} {m.group(3)}", neg
    m = re.match(r"^(.+?)\s+(has|have)\s+(.+)$", s, _I)
    if m:
        aux = "do" if m.group(2).lower() == "have" else "does"
        return f"{aux} {m.group(1)} have {m.group(3)}", neg
    svo = L._split_svo(s.lower())
    if svo:
        subj = s[:len(svo[0])]
        aux = "do" if L.np_is_plural(svo[0]) or svo[0] in ("i", "you", "we", "they") else "does"
        return f"{aux} {subj} {svo[1]} {s[len(s) - len(svo[2]):]}", neg
    return None


def _unembed(rest):
    """'what a giraffe is' -> 'what is a giraffe'; 'whether dolphins are
    mammals' -> 'are dolphins mammals'."""
    m = re.match(r"^(?:whether|if)\s+(.+)$", rest, _I)
    if m:
        return _invert(m.group(1))
    m = re.match(rf"^({_WH})\s+(.+?)\s+(is|are|was|were)$", rest, _I)
    if m:
        return f"{m.group(1)} {m.group(3)} {m.group(2)}", False
    m = re.match(r"^(how many \w+)\s+(.+?)\s+(has|have)$", rest, _I)
    if m:
        aux = "do" if m.group(3).lower() == "have" else "does"
        return f"{m.group(1)} {aux} {m.group(2)} have", False
    m = re.match(rf"^({_WH})\s+(.+)$", rest, _I)
    if m:
        svo = L._split_svo(m.group(2).lower() + " it")     # 'what cows eat' (+ dummy object)
        if svo and svo[1] not in ("be", "have") and not re.match(
                r"^(is|are|was|were|can|could|do|does|did|will)\b", m.group(2), _I):
            subj = m.group(2)[:len(svo[0])]
            aux = "do" if L.np_is_plural(svo[0]) else "does"
            return f"{m.group(1)} {aux} {subj} {svo[1]}", False
        return rest, False
    return None


def rephrase(text):
    """(sentence the reader understands, flip) for one sentence."""
    s = text.strip()
    letters = re.sub(r"[^A-Za-z]", "", s)
    if len(letters) > 3 and letters.isupper():
        s = s.lower()                                    # 'IS A WHALE A FISH?'
    q = s.endswith("?")
    for _ in range(3):
        s2 = _FILLER.sub("", s)
        if s2 == s:
            break
        s = s2
    for pat, rep in _SPELLED:
        s = re.sub(pat, rep, s, flags=_I)
    m = _TEACH.match(s)
    if m:
        return s[m.end():], False
    m = _CHECK.match(s)
    if m:
        inv = _invert(s[m.end():])
        if inv:
            return _question(inv[0]), inv[1]
    m = _ASK.match(s)
    if m:
        rest = s[m.end():].strip().rstrip(".!?")
        if re.match(r"^(?:about|some|a few|several|an? examples?|examples|the names)\b", rest, _I):
            return "Tell me " + rest + ".", False          # 'tell me some birds
        un = _unembed(rest)
        if un:
            return _question(un[0]), un[1]
        if rest:                                           # 'tell me the capital of Kenya'
            return _question("what is " + rest), False
    body = s.rstrip(".!?").strip()
    # bare fragments: 'capital of Spain?', 'the currency of Japan'
    if re.match(rf"^(?:the )?(?:{_RELS}) (?:city )?of\b", body, _I) and \
            not re.search(r"\b(?:is|are|was|were|has|have)\b", body, _I):
        body = re.sub(r"^the ", "", body, flags=_I)
        return _question("what is the " + body[:1].lower() + body[1:]), False
    # 'how many days are (there) in a week' -> 'how many days does a week have'
    m = re.match(r"^how many (.+?) (?:are (?:there )?|is (?:there )?)?in (?:a |an |one |the )?(.+)$",
                 body, _I)
    if m and " have" not in body.lower():
        return _question(f"how many {m.group(1)} does a {m.group(2)} have"), False
    # 'what language do they speak in Brazil', 'what currency does Japan use'
    m = re.match(r"^(?:what|which) (?:language|languages) (?:do (?:they|people|you) speak|"
                 r"is spoken|are spoken|is used) in (.+)$", body, _I) or \
        re.match(r"^(?:what|which) (?:language|languages) (?:does|do) (.+?) speak$", body, _I)
    if m:
        return _question("what is the language of " + m.group(1)), False
    m = re.match(r"^(?:what|which) (?:currency|money) (?:does|do) (.+?) use$", body, _I) or \
        re.match(r"^(?:what|which) (?:currency|money) (?:do|does) (?:they|people|you|one) "
                 r"(?:use|pay with|spend) in (.+)$", body, _I) or \
        re.match(r"^(?:what|which) (?:is the )?(?:currency|money) (?:is )?(?:used )?in (.+)$",
                 body, _I) or \
        re.match(r"^what is the currency in (.+)$", body, _I)
    if m:
        return _question("what is the currency of " + m.group(1)), False
    m = re.match(r"^(?:what|which) (?:city|town) is the capital of (.+)$", body, _I) or \
        re.match(r"^what is the capital city of (.+)$", body, _I)
    if m:
        return _question("what is the capital of " + m.group(1)), False
    return (s if not q or s.endswith("?") else s + "?"), False


# ---------------------------------------------------------------- arithmetic

_NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUM_WORDS.update({w: 10 * i for i, w in enumerate(
    "_ _ _ thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"})
_SCALE = {"hundred": 100, "thousand": 1000, "million": 10 ** 6}
_OPS = [(r"\bmultiplied by\b|\btimes\b|×|\bx\b", "*"), (r"\bdivided by\b|\bover\b|÷", "/"),
        (r"\bplus\b|\badded to\b", "+"), (r"\bminus\b|\btake away\b", "-"),
        (r"\b(?:raised )?to (?:the )?power of\b|\*\*", "^"), (r"\bsquared\b", "^2"),
        (r"\bcubed\b", "^3"), (r"\bsquare root of\b|√", " sqrt "), (r"\bpercent of\b|% of", "% ")]


def _numbers(text):
    """Number words to digits ('twenty one' -> '21', 'three hundred' -> '300')."""
    out, cur, have = [], 0, False
    for tok in text.split():
        if tok in _NUM_WORDS:
            cur += _NUM_WORDS[tok]
            have = True
        elif tok in _SCALE and have:
            cur *= _SCALE[tok]
        elif tok == "and" and have:
            continue
        else:
            if have:
                out.append(str(cur))
                cur, have = 0, False
            out.append(tok)
    if have:
        out.append(str(cur))
    return " ".join(out)


def _tokens(expr):
    toks = re.findall(r"\d+(?:\.\d+)?|sqrt|[-+*/^()%]", expr)
    if re.sub(r"\d+(?:\.\d+)?|sqrt|[-+*/^()%\s]", "", expr):
        return None                                      # something that is not a sum
    return toks


def _evaluate(toks):
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def expr():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v

    def term():
        v = power()
        while peek() in ("*", "/", "%"):
            op = take()
            r = power()
            v = v * r if op == "*" else v * r / 100 if op == "%" else v / r
        return v

    def power():
        v = unary()
        if peek() == "^":
            take()
            return v ** power()
        return v

    def unary():
        if peek() == "-":
            take()
            return -unary()
        if peek() == "sqrt":
            take()
            return math.sqrt(unary())
        return atom()

    def atom():
        t = take()
        if t == "(":
            v = expr()
            if take() != ")":
                raise ValueError
            return v
        return float(t)

    v = expr()
    if pos != len(toks):
        raise ValueError
    return v


def arithmetic(text):
    """'What is 12 times 7?' -> '12 × 7 = 84.'; None when it is not a sum."""
    t = text.strip().lower().rstrip("?.!= ")
    t = re.sub(r"^(?:what is|what's|whats|how much is|how much are|calculate|compute|"
               r"evaluate|work out|solve|tell me)\s+", "", t)
    t = re.sub(r"\bthe\b", " ", t)
    if not re.search(r"\d|\b(?:" + "|".join(_NUM_WORDS) + r")\b", t):
        return None
    t = _numbers(t)
    for pat, rep in _OPS:
        t = re.sub(pat, rep, t)
    toks = _tokens(t)
    if not toks or not any(op in toks for op in ("+", "-", "*", "/", "^", "sqrt", "%")):
        return None
    try:
        v = _evaluate(toks)
    except ZeroDivisionError:
        return "I can't divide by zero."
    except (ValueError, IndexError, OverflowError):
        return None
    shown = " ".join(toks).replace("*", "×").replace("/", "÷").replace("( ", "(") \
        .replace(" )", ")").replace("sqrt ", "√").replace(" % ", "% of ")
    if abs(v) < 1e15 and float(v).is_integer():
        val = str(int(v))
    else:
        val = f"{v:.6g}"
    return f"{shown} = {val}."
