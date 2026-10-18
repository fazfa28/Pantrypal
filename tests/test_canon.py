from core.canon import get_canon, normalize


def test_normalize_plurals():
    assert normalize("Cherry Tomatoes!") == "cherry tomato"
    assert normalize("Berries") == "berry"


def test_exact_alias():
    m = get_canon().match("scallions")
    assert m.id == "spring_onion" and m.method == "exact"


def test_fuzzy_extra_words():
    assert get_canon().match("Farm fresh whole milk 1L").id == "milk"


def test_longer_alias_wins_on_tie():
    assert get_canon().match("tomato sauce jar").id == "tomato_sauce"


def test_unknown_food():
    assert get_canon().match("xylophone").id is None
