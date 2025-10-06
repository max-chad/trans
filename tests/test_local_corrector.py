from nlp.local_corrector import LocalTextCorrector


def test_rule_based_deterministic_normalization():
    config = {"backend": "rules", "max_input_len": 128}
    with LocalTextCorrector(config) as corrector:
        text = "  Hello ,  world !!  "
        result1 = corrector.correct(text)
        result2 = corrector.correct(text)
    assert result1 == result2 == "Hello, world!"
