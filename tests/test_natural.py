"""Natural phrasings: wrappers, embedded questions, sums, and the question
types people ask (kinds, lists, comparisons, choices, why)."""
import conftest  # noqa: F401
from mnemo.mind import Mind
from mnemo.rephrase import arithmetic, rephrase


def test_arithmetic_is_exact_and_knows_what_is_not_a_sum():
    assert arithmetic("What is 12 times 7?") == "12 × 7 = 84."
    assert arithmetic("What's 3 plus 4 times 2?").endswith("= 11.")
    assert arithmetic("How much is 100 divided by 4?").endswith("= 25.")
    assert arithmetic("What is 2 to the power of 10?").endswith("= 1024.")
    assert arithmetic("what is the square root of 81").endswith("= 9.")
    assert arithmetic("What is twenty one plus three hundred?").endswith("= 321.")
    assert arithmetic("What is 7 divided by 0?") == "I can't divide by zero."
    for not_a_sum in ("What is a whale?", "How many legs does a spider have?",
                      "Mary has 3 apples."):
        assert arithmetic(not_a_sum) is None


def test_rephrase_undoes_wrappers_and_embedding():
    assert rephrase("Could you tell me whether dolphins are mammals?") == \
        ("Are dolphins mammals?", False)
    assert rephrase("Please tell me what a giraffe is.") == ("What is a giraffe?", False)
    assert rephrase("Do you know what cows eat?") == ("What do cows eat?", False)
    assert rephrase("Is it true that penguins can't fly?") == ("Can penguins fly?", True)
    assert rephrase("Remember that Max is a cat.") == ("Max is a cat.", False)
    assert rephrase("capital of spain") == ("What is the capital of spain?", False)
    assert rephrase("How many days are in a week?") == ("How many days does a week have?", False)
    assert rephrase("IS A WHALE A FISH?") == ("is a whale a fish?", False)
    assert rephrase("The currency of the USA is the dollar.")[0] == \
        "The currency of the USA is the dollar."     # a statement stays a statement


def _mind():
    m = Mind(learn_language=False)
    for s in ["Whales are mammals.", "Mammals are animals.", "Fish are animals.",
              "Birds are animals.", "Robins are birds.", "Sparrows are birds.",
              "Penguins are birds.", "Birds can fly.", "Penguins cannot fly.",
              "Birds have two legs.", "Elephants are bigger than mice.", "Tweety is a bird."]:
        m.hear(s)
    return m


def test_kinds_lists_and_which():
    m = _mind()
    assert m.hear("What kind of animal is a whale?") == "A mammal."
    birds = m.hear("Name some birds.").lower()
    assert "robins" in birds and "sparrows" in birds
    flyers = m.hear("Which birds can't fly?")
    assert "Penguins" in flyers and "Robins" not in flyers


def test_comparisons_choices_counts_and_checks():
    m = _mind()
    assert m.hear("Which is bigger, an elephant or a mouse?").startswith("An elephant.")
    assert m.hear("Is a whale a fish or a mammal?").startswith("A mammal.")
    assert m.hear("Do birds have four legs?") == "No. Birds have two legs."
    assert m.hear("Do birds have two legs?") == "Yes. Birds have two legs."
    assert m.hear("Is it true that penguins can't fly?").startswith("Yes")


def test_why_questions_give_reasons_or_correct_the_question():
    m = _mind()
    assert m.hear("Why can Tweety fly?") == "Because Tweety is a bird, and birds can fly."
    assert m.hear("Why can a penguin fly?") == "Actually, penguins cannot fly."


def test_learning_on_the_way():
    m = Mind(learn_language=False)
    assert m.hear("Remember that Max is a cat.") == "OK."
    assert m.hear("is max a cat") == "Yes."
    assert m.hear("Note that Bob likes pizza.") == "OK."
    assert "pizza" in m.hear("What does Bob like?").lower()
