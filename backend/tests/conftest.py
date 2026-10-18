import pytest

from app.mission import TrustedConstraints


def make_constraints(width=6, height=6, start=(0, 0), destination=(3, 0), obstacles=(), restricted=(),
                     max_steps=20, return_to_base=True) -> TrustedConstraints:
    return TrustedConstraints(width, height, start, destination, frozenset(obstacles), frozenset(restricted),
                              max_steps, return_to_base)


@pytest.fixture
def open_grid() -> TrustedConstraints:
    return make_constraints()


@pytest.fixture
def walled_grid() -> TrustedConstraints:
    """Restricted column at x=2 for y in 0..3; the only way past is via y >= 4."""
    return make_constraints(restricted=[(2, 0), (2, 1), (2, 2), (2, 3)], max_steps=30)
