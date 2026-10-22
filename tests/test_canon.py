from core.canon import get_canon, normalize


def test_normalize_plurals():
    assert normalize("Cherry Tomatoes!") == "cherry tomato"
    assert normalize("Berries") == "berry"


def test_exact_alias():
    m = get_canon().match("scallions")
    assert m.id == "spring_onion" and m.method == "exact"


def test_fuzzy_catches_typos_only():
    assert get_canon().match("whole mlik").id == "milk"
    assert get_canon().match("scalions").id == "spring_onion"


def test_unknown_food():
    assert get_canon().match("xylophone").id is None


def test_extra_words_do_not_match_a_shorter_id():
    # Regression: token_set_ratio scored these 100 against the shorter name.
    canon = get_canon()
    for text, wrong in [("fish sauce", "fish"), ("coconut milk", "milk"), ("baking soda", "soda"),
                        ("chicken stock", "chicken"), ("tomato paste", "tomato"), ("lemon juice", "juice"),
                        ("garlic powder", "garlic"), ("beef broth", "beef"), ("sour cream", "cream"),
                        ("cream cheese", "cheese")]:
        assert canon.match(text).id != wrong, text


def test_unseen_phrase_is_unmatched_not_guessed():
    assert get_canon().match("farm fresh whole milk 1L").id is None
