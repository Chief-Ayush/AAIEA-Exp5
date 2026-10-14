"""Rule-based natural-language -> Requirements parser (the deterministic core of the Planner).

Handles Indian price formats (₹70,000 / 70k / 1.2 lakh / Rs. 50000), RAM vs storage disambiguation, TB, star ratings,
battery, brand include/exclude, feature synonyms, ranking priorities and 'compare A and B'.  Deliberately *not* an LLM:
it is fast, free, deterministic and testable.  An optional LangChain LLM planner (see llm.py) can augment it.
Known limitations (documented in the README): spelled-out numbers ("thirty thousand"), implicit needs, typos.
"""
from __future__ import annotations

import re

from .schemas import Requirements

CATEGORY_SYNONYMS = {
    "Laptop": ["laptop", "laptops", "notebook", "notebooks", "ultrabook", "macbook-like", "lappy"],
    "Smartphone": ["smartphone", "smartphones", "phone", "phones", "mobile", "mobiles", "cellphone", "handset"],
    "Tablet": ["tablet", "tablets", "tab", "tabs", "ipad-like"],
    "Headphones": ["headphone", "headphones", "earbuds", "earbud", "earphones", "earphone", "headset", "headsets", "airpods-like", "tws"],
    "Smartwatch": ["smartwatch", "smartwatches", "smart watch", "watch", "watches", "fitness band", "fitness tracker"],
}
FEATURE_SYNONYMS = {
    r"\b5g\b": "5G", r"\banc\b|noise[- ]?cancel+(?:l?ing|ation)": "ANC", r"\bbacklit\b": "Backlit Keyboard",
    r"wi-?fi ?6\b": "WiFi 6", r"fingerprint": "Fingerprint", r"touch ?screen": "Touchscreen", r"\boled\b": "OLED",
    r"\bamoled\b": "AMOLED", r"fast[- ]charg": "Fast Charging", r"wireless charg": "Wireless Charging",
    r"\bgps\b": "GPS", r"\bnfc\b": "NFC", r"water[- ]?(proof|resist)|\bip6\d\b|\bip68\b": "Water Resistant",
    r"spatial audio": "Spatial Audio", r"120 ?hz": "120Hz Display", r"dual[- ]sim": "Dual SIM", r"thunderbolt": "Thunderbolt",
    r"numpad": "Numpad", r"bluetooth": "Bluetooth", r"stylus": "Stylus Support", r"multipoint": "Multipoint",
    r"\bspo2\b|blood oxygen": "SpO2", r"heart[- ]?rate": "Heart Rate", r"dedicated gpu|graphics card|discrete gpu|\bgaming\b": "Dedicated GPU",
}
PRIORITY_SYNONYMS = {
    "battery": r"long[- ]?(lasting )?battery|good battery|great battery|best battery|battery life|battery backup|all[- ]day battery",
    "price": r"\bcheap(est)?\b|\baffordable\b|\bbest value\b|value for money|\bbudget[- ]friendly\b|\beconomical\b|\bbang for (the )?buck\b",
    "rating": r"best[- ]rated|highest[- ]rated|top[- ]rated|highly rated|best rating|well[- ]reviewed",
    "performance": r"\bfast\b(?![- ]?charg)|\bpowerful\b|\bsnappy\b|high performance|\bcoding\b|programming|video editing|\bediting\b|\bgaming\b|multitask",
    "portability": r"light ?weight|\bportable\b|\bultra[- ]?light\b|\bslim\b|travel",
    "popularity": r"most popular|best[- ]?sell|popular|most reviewed",
}
NEG = r"(?:not|no|except|excluding|exclude|without|avoid|other than|anything but|besides)"

_NUM = r"(\d+(?:\.\d+)?)"
_MULT = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5, "crore": 1e7}


def _clean(text: str) -> str:
    t = text.lower().replace("₹", " inr ").replace("rs.", " inr ").replace("rupees", " inr ").replace("rupee", " inr ")
    t = re.sub(r"\brs\b", " inr ", t)
    for _ in range(3):
        t = re.sub(r"(\d),(\d{2,3})\b", r"\1\2", t)           # 70,000 / 1,00,000
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _amount(num: str, mult: str | None) -> float:
    v = float(num)
    if mult:
        v *= _MULT.get(mult, 1)
    return v


UNIT_AFTER = r"(?!\s*(?:gb|tb|g\b|gig|mb|hours?|hrs?|h\b|mah|inch|in\b|\"|hz|mp|stars?|star\b|days?|kg|%|cores?|nits|w\b))"
PRICE_RX = re.compile(rf"(?:inr\s*)?{_NUM}\s*(k|thousand|lakhs?|lacs?|crore)?{UNIT_AFTER}")


def _price_values(t: str) -> list[tuple[float, int, int, bool]]:
    """All price-like numbers: (value, start, end, has_currency_or_suffix)."""
    out = []
    for m in PRICE_RX.finditer(t):
        num, mult = m.group(1), m.group(2)
        v = _amount(num, mult)
        pre = t[max(0, m.start() - 6):m.start()]
        has_cur = "inr" in m.group(0) or "inr" in pre
        if mult == "k" and re.match(r"\s*(display|tv|video|resolution|camera|uhd)", t[m.end():]):
            continue
        if v >= 100:
            out.append((v, m.start(), m.end(), bool(has_cur or mult or v >= 3000)))
    return out


def parse_requirements(query: str, categories: list[str], brands: list[str], product_names: dict[str, str]) -> dict:
    """Returns {"requirements": Requirements, "notes": [...], "conflicts": [...]}."""
    t = _clean(query)
    notes, conflicts = [], []
    req: dict = {}

    # ------------------------------------------------------------------ compare intent by product name
    lower_names = {n.lower(): pid for n, pid in product_names.items()}
    targets = [pid for n, pid in sorted(lower_names.items(), key=lambda x: -len(x[0])) if re.search(rf"\b{re.escape(n)}\b", t)]
    # drop names that are substrings of longer matched names
    if re.search(r"\b(compare|versus|vs\.?|difference between|which is better)\b", t) and len(targets) >= 2:
        req["intent"] = "compare"
        req["compare_targets"] = targets[:6]

    # ------------------------------------------------------------------ category
    cat, first = None, 10 ** 6
    for c in categories:
        syns = CATEGORY_SYNONYMS.get(c, [c.lower()]) + [c.lower()]
        for s in syns:
            m = re.search(rf"\b{re.escape(s)}\b", t)
            if m and m.start() < first:
                cat, first = c, m.start()
    if cat is None:                                         # typo tolerance: "lptop", "samrtphone"
        import difflib
        pool = {syn: c for c in categories for syn in CATEGORY_SYNONYMS.get(c, [c.lower()]) + [c.lower()] if len(syn) >= 5}
        for tok in re.findall(r"[a-z]{5,}", t):
            m = difflib.get_close_matches(tok, list(pool), n=1, cutoff=0.82)
            if m:
                cat = pool[m[0]]
                break
    req["category"] = cat

    # ------------------------------------------------------------------ price
    maxp = []
    minp = None
    rng = re.search(rf"\bbetween\s+(?:inr\s*)?{_NUM}\s*(k|thousand|lakhs?|lacs?)?\s*(?:and|to|-)\s*(?:inr\s*)?{_NUM}\s*(k|thousand|lakhs?|lacs?)?", t) \
        or re.search(rf"(?:inr\s*)?{_NUM}\s*(k|thousand|lakhs?|lacs?)?\s*(?:to|-)\s*(?:inr\s*)?{_NUM}\s*(k|thousand|lakhs?|lacs?)\b", t)
    consumed: list[tuple[int, int]] = []
    if rng:
        a, am, b, bm = rng.group(1), rng.group(2), rng.group(3), rng.group(4)
        if am is None and bm:      # "50 to 70k"
            am = bm
        va, vb = _amount(a, am), _amount(b, bm)
        if va >= 500 and vb >= 500:
            minp, maxp = min(va, vb), [max(va, vb)]
            consumed.append(rng.span())
    for v, s, e, strong in _price_values(t):
        if any(s >= a and e <= b for a, b in consumed):
            continue
        ctx = t[max(0, s - 28):s]
        post = t[e:e + 12]
        if re.search(r"(under|below|less than|within|upto|up to|max(?:imum)?|not more than|no more than|at most|cheaper than|budget(?: of| is| around| about)?|around|about|approx(?:imately)?|~|<=?|for|in)\s*(inr\s*)?$", ctx) or re.search(r"^\s*(or less|or below|max|maximum|budget|and below|and under)", post):
            maxp.append(v)
        elif re.search(r"(above|over|more than|at least|minimum|min|starting|from|>=?)\s*(inr\s*)?$", ctx):
            minp = v if minp is None else max(minp, v)
        elif strong and not maxp and re.search(r"(budget|price|cost)", t):
            maxp.append(v)
        elif strong and not maxp and minp is None:
            maxp.append(v)             # "laptop 60k" -> treat as ceiling
    if len(set(maxp)) > 1:
        conflicts.append({"field": "max_price", "values": sorted(set(maxp)), "resolution": "most restrictive value used"})
        notes.append(f"Conflicting budgets {sorted(set(maxp))}; the most restrictive ({min(maxp):,.0f}) is used.")
    if maxp:
        req["max_price"] = min(maxp)
    if minp is not None:
        req["min_price"] = minp

    # ------------------------------------------------------------------ RAM / storage
    ram = stor = None
    m = re.search(rf"{_NUM}\s*(?:gb|g|gig|gigs)?\s*(?:of\s*)?(?:ddr\d\s*)?(?:ram|memory)\b", t)
    if m:
        ram = float(m.group(1))
    else:
        m = re.search(rf"\bram\s*(?:of|:|is|>=?)?\s*{_NUM}\s*(?:gb|g|gig)?", t)
        if m:
            ram = float(m.group(1))
    for rx in (rf"{_NUM}\s*(gb|tb)\s*(?:of\s*)?(?:ssd|storage|hdd|rom|disk|nvme|internal|flash)\b",
               rf"\b(?:ssd|storage|hdd|rom|disk|nvme)\s*(?:of|:|is|>=?)?\s*{_NUM}\s*(gb|tb)\b"):
        m = re.search(rx, t)
        if m:
            stor = float(m.group(1)) * (1024 if m.group(2) == "tb" else 1)
            break
    m = re.search(rf"\b{_NUM}\s*(?:gb)?\s*/\s*{_NUM}\s*(gb|tb)\b", t)          # 8gb/256gb  or 8/256
    if m and ram is None and stor is None and float(m.group(1)) <= 32:
        ram = float(m.group(1))
        stor = float(m.group(2)) * (1024 if m.group(3) == "tb" else 1)
    if stor is None:
        m = re.search(rf"\b{_NUM}\s*tb\b", t)
        if m:
            stor = float(m.group(1)) * 1024
    if stor is None:                                                           # bare "512 gb"
        for m in re.finditer(rf"\b{_NUM}\s*gb\b", t):
            v = float(m.group(1))
            if ram is not None and v == ram and not re.search(rf"{_NUM}\s*gb[^,.]*{_NUM}\s*gb", t):
                continue
            if v >= 64 and (cat in (None, "Laptop", "Smartphone", "Tablet")) and v != ram:
                stor = v
                break
    if ram is None:
        for m in re.finditer(rf"\b{_NUM}\s*(?:gb|gig|gigs)\b", t):
            v = float(m.group(1))
            if v <= 32 and v != stor and not re.search(r"(ssd|storage|hdd|rom|disk|nvme)", t[m.end():m.end() + 10]):
                ram = v                         # a lone small "N GB" next to no storage keyword is RAM
                break
    if ram is not None:
        req["min_ram_gb"] = ram
    if stor is not None:
        req["min_storage_gb"] = stor

    # ------------------------------------------------------------------ rating
    rating = None
    for rx in (rf"rat(?:ing|ed)\s*(?:of|is|:|>=?|above|over|at least|minimum|min|greater than|more than|higher than|atleast)?\s*{_NUM}",
               rf"{_NUM}\s*\+?\s*(?:stars?|★|rating)",
               rf"(?:at least|above|over|minimum|min)\s*{_NUM}\s*(?:stars?|★)"):
        m = re.search(rx, t)
        if m and 1 <= float(m.group(1)) <= 5:
            rating = float(m.group(1))
            break
    if rating is not None:
        req["min_rating"] = rating

    # ------------------------------------------------------------------ battery
    bat = None
    for rx in (rf"{_NUM}\s*(hours?|hrs?|h|days?)\s*(?:of\s*)?(?:battery|backup|playtime|playback|battery life)",
               rf"battery\s*(?:life|backup)?\s*(?:of|:|is|>=?|above|over|at least|minimum|min)?\s*{_NUM}\s*(hours?|hrs?|h|days?)"):
        m = re.search(rx, t)
        if m:
            bat = float(m.group(1)) * (24 if m.group(2).startswith("day") else 1)
            break
    if bat is not None:
        req["min_battery_hours"] = bat

    # ------------------------------------------------------------------ brands
    inc, exc = [], []
    for b in brands:
        for m in re.finditer(rf"\b{re.escape(b.lower())}\b", t):
            pre = t[max(0, m.start() - 22):m.start()]
            if re.search(rf"\b{NEG}\b[^,.;]*$", pre) or re.search(r"\bnot from\b", pre):
                exc.append(b)
            else:
                inc.append(b)
            break
    if req.get("intent") == "compare":
        inc = []   # brand words inside product names are not brand filters
    req["include_brands"] = [b for b in inc if b not in exc]
    req["exclude_brands"] = exc

    # ------------------------------------------------------------------ features & priorities
    feats = []
    for rx, name in FEATURE_SYNONYMS.items():
        if re.search(rx, t):
            feats.append(name)
    req["required_features"] = feats
    pri = [p for p, rx in PRIORITY_SYNONYMS.items() if re.search(rx, t)]
    req["priorities"] = pri

    # ------------------------------------------------------------------ top-n
    n = None
    for rx in (r"\btop\s*(\d+)\b", r"\bbest\s*(\d+)\b", r"\b(\d+)\s*(?:best|top)\b", r"\b(\d+)\s*(?:options|choices|recommendations|suggestions|picks)\b", r"\bshow (?:me )?(\d+)\b"):
        m = re.search(rx, t)
        if m and 1 <= int(m.group(1)) <= 20:
            n = int(m.group(1))
            break
    if n:
        req["top_n"] = n

    return {"requirements": req, "notes": notes, "conflicts": conflicts}


def to_requirements(parsed: dict) -> Requirements:
    return Requirements(**{k: v for k, v in parsed["requirements"].items() if v is not None})
