"""Kazakh Cyrillic -> Latin transliteration for the four evaluated alphabet proposals.

Five script conditions are supported: the original Cyrillic (identity) and the
2017, 2018, 2019 and 2021 Latin proposals.

The transformation is deterministic and character/rule based:
  * one table of single-character mappings per Latin variant (Appendix A / Table 7),
  * one sequence rule (`сц`, which would otherwise produce `sts`),
  * one context-sensitive rule (`У` in the 2019 variant).

Entry points:
    transliterate(text, version)   -> str
    mapping_table_text(version)    -> str   (used for the `mapping` prompt condition)
    LATIN_VERSIONS, SCRIPT_CONDITIONS
"""

# Each entry is  Cyrillic-uppercase: (Latin-uppercase, Latin-lowercase).
# Keeping the pair together makes upper/lower mismatches impossible.

_TABLE_2017 = {
    "А": ("A", "a"),      "Ә": ("A'", "a'"),   "Б": ("B", "b"),     "В": ("V", "v"),
    "Г": ("G", "g"),      "Ғ": ("G'", "g'"),   "Д": ("D", "d"),     "Е": ("E", "e"),
    "Ё": ("I'o", "i'o"),  "Ж": ("J", "j"),     "З": ("Z", "z"),     "И": ("I'", "i'"),
    "Й": ("I'", "i'"),    "К": ("K", "k"),     "Қ": ("Q", "q"),     "Л": ("L", "l"),
    "М": ("M", "m"),      "Н": ("N", "n"),     "Ң": ("N'", "n'"),   "О": ("O", "o"),
    "Ө": ("O'", "o'"),    "П": ("P", "p"),     "Р": ("R", "r"),     "С": ("S", "s"),
    "Т": ("T", "t"),      "У": ("Y'", "y'"),   "Ұ": ("U", "u"),     "Ү": ("U'", "u'"),
    "Ф": ("F", "f"),      "Х": ("H", "h"),     "Һ": ("H", "h"),     "Ц": ("Ts", "ts"),
    "Ч": ("Ch", "ch"),    "Ш": ("S'", "s'"),   "Щ": ("Sch", "sch"), "Ъ": ("", ""),
    "Ы": ("Y", "y"),      "І": ("I", "i"),     "Ь": ("", ""),       "Э": ("E", "e"),
    "Ю": ("I'y'", "i'y'"), "Я": ("I'a", "i'a"),
}

_TABLE_2018 = {
    "А": ("A", "a"),      "Ә": ("Á", "á"),     "Б": ("B", "b"),     "В": ("V", "v"),
    "Г": ("G", "g"),      "Ғ": ("Ǵ", "ǵ"),     "Д": ("D", "d"),     "Е": ("E", "e"),
    "Ё": ("Io", "ıo"),    "Ж": ("J", "j"),     "З": ("Z", "z"),     "И": ("I", "ı"),
    "Й": ("I", "ı"),      "К": ("K", "k"),     "Қ": ("Q", "q"),     "Л": ("L", "l"),
    "М": ("M", "m"),      "Н": ("N", "n"),     "Ң": ("Ń", "ń"),     "О": ("O", "o"),
    "Ө": ("Ó", "ó"),      "П": ("P", "p"),     "Р": ("R", "r"),     "С": ("S", "s"),
    "Т": ("T", "t"),      "У": ("Ý", "ý"),     "Ұ": ("U", "u"),     "Ү": ("Ú", "ú"),
    "Ф": ("F", "f"),      "Х": ("H", "h"),     "Һ": ("H", "h"),     "Ц": ("Ts", "ts"),
    "Ч": ("Ch", "ch"),    "Ш": ("Sh", "sh"),   "Щ": ("Sch", "sch"), "Ъ": ("", ""),
    "Ы": ("Y", "y"),      "І": ("İ", "i"),     "Ь": ("", ""),       "Э": ("E", "e"),
    "Ю": ("Iý", "ıý"),    "Я": ("Ia", "ıa"),
}

# `У` is deliberately absent from the 2019 table: it is produced by _u_2019().
_TABLE_2019 = {
    "А": ("A", "a"),      "Ә": ("Ä", "ä"),     "Б": ("B", "b"),     "В": ("V", "v"),
    "Г": ("G", "g"),      "Ғ": ("Ğ", "ğ"),     "Д": ("D", "d"),     "Е": ("E", "e"),
    "Ё": ("Io", "ıo"),    "Ж": ("J", "j"),     "З": ("Z", "z"),     "И": ("I", "ı"),
    "Й": ("I", "ı"),      "К": ("K", "k"),     "Қ": ("Q", "q"),     "Л": ("L", "l"),
    "М": ("M", "m"),      "Н": ("N", "n"),     "Ң": ("Ñ", "ñ"),     "О": ("O", "o"),
    "Ө": ("Ö", "ö"),      "П": ("P", "p"),     "Р": ("R", "r"),     "С": ("S", "s"),
    "Т": ("T", "t"),      "Ұ": ("U", "u"),     "Ү": ("Ü", "ü"),     "Ф": ("F", "f"),
    "Х": ("H", "h"),      "Һ": ("H", "h"),     "Ц": ("Ts", "ts"),   "Ч": ("Ç", "ç"),
    "Ш": ("Ş", "ş"),      "Щ": ("Sch", "sch"), "Ъ": ("", ""),       "Ы": ("Y", "y"),
    "І": ("İ", "i"),      "Ь": ("", ""),       "Э": ("E", "e"),     "Ю": ("Iu", "ıu"),
    "Я": ("Ia", "ıa"),
}

_TABLE_2021 = {
    "А": ("A", "a"),      "Ә": ("Ä", "ä"),     "Б": ("B", "b"),     "В": ("V", "v"),
    "Г": ("G", "g"),      "Ғ": ("Ğ", "ğ"),     "Д": ("D", "d"),     "Е": ("E", "e"),
    "Ё": ("İo", "io"),    "Ж": ("J", "j"),     "З": ("Z", "z"),     "И": ("İ", "i"),
    "Й": ("İ", "i"),      "К": ("K", "k"),     "Қ": ("Q", "q"),     "Л": ("L", "l"),
    "М": ("M", "m"),      "Н": ("N", "n"),     "Ң": ("Ñ", "ñ"),     "О": ("O", "o"),
    "Ө": ("Ö", "ö"),      "П": ("P", "p"),     "Р": ("R", "r"),     "С": ("S", "s"),
    "Т": ("T", "t"),      "У": ("U", "u"),     "Ұ": ("Ū", "ū"),     "Ү": ("Ü", "ü"),
    "Ф": ("F", "f"),      "Х": ("H", "h"),     "Һ": ("H", "h"),     "Ц": ("Ts", "ts"),
    "Ч": ("Ch", "ch"),    "Ш": ("Ş", "ş"),     "Щ": ("Sch", "sch"), "Ъ": ("", ""),
    "Ы": ("Y", "y"),      "І": ("I", "ı"),     "Ь": ("", ""),       "Э": ("E", "e"),
    "Ю": ("İu", "iu"),    "Я": ("İa", "ia"),
}

PAIR_TABLES = {
    "2017": _TABLE_2017,
    "2018": _TABLE_2018,
    "2019": _TABLE_2019,
    "2021": _TABLE_2021,
}

LATIN_VERSIONS = ["2017", "2018", "2019", "2021"]
SCRIPT_CONDITIONS = ["cyrillic"] + LATIN_VERSIONS

# Variants whose lowercase `i` is the dotted letter, i.e. whose uppercase forms
# are Turkish-style (i -> İ, ı -> I).
_DOTTED_I_VERSIONS = {"2018", "2019", "2021"}

# `сц` would otherwise surface as `sts`.
_SEQUENCES = {"сц": "s"}

# Vowel-harmony classes used by the 2019 `У` rule.
_VOWELS_FRONT = set("еәөүіэ")
_VOWELS_BACK = set("аоұыяёю")
# Vowels for the "adjacent to a vowel" test (`у` itself is excluded).
_VOWELS_ADJACENT = _VOWELS_FRONT | _VOWELS_BACK | set("и")


def _char_table(version):
    table = {}
    for cyr_upper, (lat_upper, lat_lower) in PAIR_TABLES[version].items():
        table[cyr_upper] = lat_upper
        table[cyr_upper.lower()] = lat_lower
    return table


CHAR_TABLES = {v: _char_table(v) for v in LATIN_VERSIONS}


def _upper(text, version):
    """Uppercase a transliterated string, respecting the dotted/dotless distinction."""
    if version in _DOTTED_I_VERSIONS:
        text = text.replace("i", "İ").replace("ı", "I")
    return text.upper()


def _harmony(word, i):
    """front / back / None, from the nearest preceding vowel (harmony is progressive)."""
    for j in range(i - 1, -1, -1):
        c = word[j].lower()
        if c in _VOWELS_FRONT:
            return "front"
        if c in _VOWELS_BACK:
            return "back"
    return None


def _u_2019(word, i):
    """2019 `У`.

    `w` when adjacent to a vowel; otherwise `ü` under front harmony, `ū` under
    back harmony, and `u` when no preceding vowel fixes the harmony class.
    """
    prev_char = word[i - 1].lower() if i > 0 else ""
    next_char = word[i + 1].lower() if i + 1 < len(word) else ""
    if prev_char in _VOWELS_ADJACENT or next_char in _VOWELS_ADJACENT:
        out = "w"
    else:
        harmony = _harmony(word, i)
        out = {"front": "ü", "back": "ū"}.get(harmony, "u")
    return out.upper() if word[i].isupper() else out


_CONTEXT_RULES = {"2019": {"у": _u_2019}}


def _transliterate_word(word, version):
    table = CHAR_TABLES[version]
    rules = _CONTEXT_RULES.get(version, {})
    out = []
    i = 0
    while i < len(word):
        pair = word[i:i + 2].lower()
        if len(pair) == 2 and pair in _SEQUENCES:
            replacement = _SEQUENCES[pair]
            out.append(replacement.capitalize() if word[i].isupper() else replacement)
            i += 2
            continue
        rule = rules.get(word[i].lower())
        if rule is not None:
            out.append(rule(word, i))
            i += 1
            continue
        out.append(table.get(word[i], word[i]))
        i += 1
    return "".join(out)


def _has_cyrillic(word):
    return any(ch in CHAR_TABLES["2021"] for ch in word)


def _transliterate_latin(text, version):
    if not isinstance(text, str):
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        if not text[i].isalpha():
            out.append(text[i])
            i += 1
            continue
        j = i
        while j < n and text[j].isalpha():
            j += 1
        word = text[i:j]
        if not _has_cyrillic(word):
            out.append(word)                      # leave Latin/English words alone
        elif len(word) > 1 and word.isupper():
            out.append(_upper(_transliterate_word(word.lower(), version), version))
        else:
            out.append(_transliterate_word(word, version))
        i = j
    return "".join(out)


def transliterate(text, version):
    """Transliterate `text` into `version`. `cyrillic` returns the text unchanged."""
    if version == "cyrillic":
        return text
    if version not in PAIR_TABLES:
        raise ValueError(f"unknown script condition: {version!r}; "
                         f"expected one of {SCRIPT_CONDITIONS}")
    return _transliterate_latin(text, version)


def mapping_table_text(version):
    """The complete Cyrillic-to-Latin mapping for `version`, as prompt text."""
    if version not in PAIR_TABLES:
        return ""
    lines = []
    for cyr_upper, (lat_upper, lat_lower) in PAIR_TABLES[version].items():
        cyr = f"{cyr_upper} {cyr_upper.lower()}"
        if lat_upper == "" and lat_lower == "":
            lines.append(f"{cyr} = (omitted)")
        else:
            lines.append(f"{cyr} = {lat_upper} {lat_lower}")
    if version == "2019":
        lines.append("У у = W w next to a vowel, otherwise U u / Ū ū / Ü ü "
                     "according to vowel harmony")
    lines.append("сц = s (the sequence is not written as sts)")
    return "\n".join(lines)


if __name__ == "__main__":
    sample = "Қазақстанның астанасы — Астана қаласы."
    for v in LATIN_VERSIONS:
        print(f"{v}  {transliterate(sample, v)}")
