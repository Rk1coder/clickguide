import pytest

from clickguide.model import create_engine


def test_unknown_backend_does_not_silently_fall_back():
    with pytest.raises(ValueError):
        create_engine("unavailable-backend")


def test_selectable_backends_remain_lazy_and_distinct():
    showui = create_engine("showui")
    qwen = create_engine("qwen")
    assert showui._model is None and qwen._model is None
    assert showui.model_id != qwen.model_id
