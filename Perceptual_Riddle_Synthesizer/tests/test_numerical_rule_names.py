from synth_engine.format_normalizer import _normalize_numerical_rule


def test_numerical_rule_names_do_not_use_full_question_descriptions():
    assert _normalize_numerical_rule({"rule": "题目特征是 'curve'，模式 E-O-E-O，正确答案为偶数 (4)"}) == "even_odd_even_odd"
    assert _normalize_numerical_rule({"rule": "题目特征是 'cart'，模式 O-E-O-E，正确答案为奇数 (3)"}) == "odd_even_odd_even"
    assert _normalize_numerical_rule({"rule": "递增2"}) == "increasing_2"
    assert _normalize_numerical_rule({"rule": "递减3"}) == "decreasing_3"
