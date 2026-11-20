import pytest

from clickguide.model import create_engine, parse_qwen_point


def test_unknown_backend_does_not_silently_fall_back():
    with pytest.raises(ValueError):
        create_engine("unavailable-backend")


def test_selectable_backends_remain_lazy_and_distinct():
    showui = create_engine("showui")
    qwen = create_engine("qwen")
    assert showui._model is None and qwen._model is None
    assert showui.model_id != qwen.model_id


@pytest.mark.parametrize(
    "raw", ['{"point_2d":[500,1000]}', '[{"point_2d":[500,1000]}]', "[500,1000]"]
)
def test_qwen_native_coordinates_are_explicitly_scaled(raw):
    assert parse_qwen_point(raw) == (0.5, 1.0)


@pytest.mark.parametrize(
    "raw",
    [
        "[0.5,0.2]",
        "[true,200]",
        "[1001,0]",
        "[-1,0]",
        "[10,20,30,40]",
        '[{"point_2d":[1,2]},{"point_2d":[3,4]}]',
    ],
)
def test_qwen_rejects_ambiguous_units_and_multiple_targets(raw):
    with pytest.raises(ValueError):
        parse_qwen_point(raw)


def test_qwen_accepts_one_complete_json_code_fence():
    assert parse_qwen_point('```json\n{"point_2d":[851,874]}\n```') == (0.851, 0.874)
    for raw in ['```json\n{"point_2d":[851,874]}', "Here: ```json\n[851,874]\n```"]:
        with pytest.raises(ValueError):
            parse_qwen_point(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"point_2d":[100,200],"point_2d":[800,900]}',
        '{"point_2d":[100,200],"bbox_2d":[1,2,3,4]}',
        '{"point_2d":[100,200],"other_target":[800,900]}',
    ],
)
def test_qwen_rejects_duplicate_keys_and_additional_predictions(raw):
    with pytest.raises(ValueError):
        parse_qwen_point(raw)
