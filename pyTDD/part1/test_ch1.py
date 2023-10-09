from dataclasses import dataclass


@dataclass
class Dollar:
    amount: float

    def times(self, x):
        return Dollar(self.amount * x)


def test_multiplication():
    five = Dollar(5)
    product = five.times(2)
    assert product.amount == 10
    product = five.times(3)
    assert product.amount == 15
