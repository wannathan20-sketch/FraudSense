from train.supervised import make_lgb_classifier, make_xgb_classifier


def test_smote_tree_classifiers_do_not_also_set_class_weighting():
    xgb = make_xgb_classifier()
    lgb = make_lgb_classifier()

    assert xgb.get_params().get("scale_pos_weight") is None
    assert lgb.get_params().get("scale_pos_weight") is None
