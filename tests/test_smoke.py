from src.autograd import Value  # noqa: F401


def test_import():
    assert Value is not None
