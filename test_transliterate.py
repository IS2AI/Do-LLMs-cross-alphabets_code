"""Unit tests for the Cyrillic-to-Latin transliteration.

Covers the letters listed in the validation checklist, the nine Russian-origin
letters that the four proposals do not define, and the context-sensitive 2019 `У`.

Run:  python test_transliterate.py
"""
import sys

from transliterate import (LATIN_VERSIONS, PAIR_TABLES, SCRIPT_CONDITIONS,
                           transliterate)

FAILURES = []


def check(label, got, expected):
    if got != expected:
        FAILURES.append(f"{label}: got {got!r}, expected {expected!r}")


def check_letter(cyrillic, expected_upper, expected_lower=None):
    """`expected_lower` is only needed where the case pair is not a plain .lower()."""
    for idx, version in enumerate(LATIN_VERSIONS):
        upper, lower = PAIR_TABLES[version][cyrillic]
        want_upper = expected_upper[idx]
        want_lower = (expected_lower[idx] if expected_lower is not None
                      else want_upper.lower())
        check(f"{cyrillic} [{version}] upper", upper, want_upper)
        check(f"{cyrillic} [{version}] lower", lower, want_lower)


def check_word(word, expected_per_version):
    for version, expected in zip(LATIN_VERSIONS, expected_per_version):
        check(f"{word!r} [{version}]", transliterate(word, version), expected)


# ---- letters called out in the checklist  (2017 / 2018 / 2019 / 2021) ----

check_letter("Ә", ["A'", "Á", "Ä", "Ä"])
check_letter("Ғ", ["G'", "Ǵ", "Ğ", "Ğ"])
check_letter("Ң", ["N'", "Ń", "Ñ", "Ñ"])
check_letter("Ө", ["O'", "Ó", "Ö", "Ö"])
check_letter("Ү", ["U'", "Ú", "Ü", "Ü"])
check_letter("Қ", ["Q", "Q", "Q", "Q"])
check_letter("Ш", ["S'", "Sh", "Ş", "Ş"])
check_letter("Х", ["H", "H", "H", "H"])
check_letter("Һ", ["H", "H", "H", "H"])
check_letter("Ь", ["", "", "", ""])
check_letter("Ъ", ["", "", "", ""])

# ---- the nine letters the proposals do not cover ----

check_letter("Ч", ["Ch", "Ch", "Ç", "Ch"])
check_letter("Щ", ["Sch", "Sch", "Sch", "Sch"])
check_letter("Ц", ["Ts", "Ts", "Ts", "Ts"])
check_letter("Э", ["E", "E", "E", "E"])

check_word("чай", ["chai'", "chaı", "çaı", "chai"])
check_word("щетка", ["schetka", "schetka", "schetka", "schetka"])
check_word("цирк", ["tsi'rk", "tsırk", "tsırk", "tsirk"])
check_word("экран", ["ekran", "ekran", "ekran", "ekran"])
check_word("объект", ["obekt", "obekt", "obekt", "obekt"])
check_word("съезд", ["sezd", "sezd", "sezd", "sezd"])
check_word("фильм", ["fi'lm", "fılm", "fılm", "film"])
check_word("ёлка", ["i'olka", "ıolka", "ıolka", "iolka"])
check_word("Ялта", ["I'alta", "Ialta", "Ialta", "İalta"])
check_word("Юпитер", ["I'y'pi'ter", "Iýpıter", "Iupıter", "İupiter"])

# `сц` must not surface as `sts`, in either case
check_word("сцена", ["sena", "sena", "sena", "sena"])
check_word("СЦЕНА", ["SENA", "SENA", "SENA", "SENA"])

# ---- dotted / dotless I ----

check_letter("И", ["I'", "I", "I", "İ"], ["i'", "ı", "ı", "i"])
check_letter("Й", ["I'", "I", "I", "İ"], ["i'", "ı", "ı", "i"])
check_letter("І", ["I", "İ", "İ", "I"], ["i", "i", "i", "ı"])
check_word("Ірі", ["Iri", "İri", "İri", "Irı"])
check_word("институт", ["i'nsti'ty't", "ınstıtýt", "ınstıtut", "institut"])

# ---- 2019 `У`: `w` next to a vowel, else u / ū / ü by harmony ----

check("тау [2019]", transliterate("тау", "2019"), "taw")
check("жауап [2019]", transliterate("жауап", "2019"), "jawap")
check("ауыл [2019]", transliterate("ауыл", "2019"), "awyl")
check("су [2019]", transliterate("су", "2019"), "su")
check("бару [2019]", transliterate("бару", "2019"), "barū")
check("біту [2019]", transliterate("біту", "2019"), "bitü")
check("университет [2019]", transliterate("университет", "2019"), "unıversıtet")
check("Университет [2019]", transliterate("Университет", "2019"), "Unıversıtet")
# the other three variants map `У` statically
check_word("тау", ["tay'", "taý", "taw", "tau"])

# ---- casing ----

check_word("ШЫҒЫС", ["S'YG'YS", "SHYǴYS", "ŞYĞYS", "ŞYĞYS"])
check_word("Шығыс", ["S'yg'ys", "Shyǵys", "Şyğys", "Şyğys"])
check_word("Ұлы", ["Uly", "Uly", "Uly", "Ūly"])

# ---- geminates are preserved (character-level mapping) ----

check_word("касса", ["kassa", "kassa", "kassa", "kassa"])
check_word("Қазақстанның", ["Qazaqstannyn'", "Qazaqstannyń", "Qazaqstannyñ", "Qazaqstannyñ"])

# ---- scope, determinism, idempotence ----

for version in LATIN_VERSIONS:
    check(f"cyrillic passthrough [{version}]",
          transliterate("Астана", "cyrillic"), "Астана")
    text = 'Question 12: "Астана" — A. 1,500 ₸'
    once = transliterate(text, version)
    check(f"ASCII untouched [{version}]", once.count('"'), 2)
    check(f"digits untouched [{version}]", "1,500" in once, True)
    check(f"english untouched [{version}]", "Question 12" in once, True)
    check(f"answer label untouched [{version}]", "A." in once, True)
    check(f"idempotent [{version}]", transliterate(once, version), once)
    check(f"deterministic [{version}]", transliterate(text, version), once)

check("script conditions", SCRIPT_CONDITIONS,
      ["cyrillic", "2017", "2018", "2019", "2021"])

if FAILURES:
    print(f"FAILED ({len(FAILURES)})")
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all transliteration tests passed")
