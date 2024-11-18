from main import add


def test_add():
    """Testing add function"""
    assert add(2, 2) == 4
    assert add(3, 2) == 5


def test_add2():
    """Testing add function"""
    assert add(2, 2) == 4
    assert add(3, 2) == 5
    assert add(1, 2) == 3


if __name__ == "__main__":
    test_add()
