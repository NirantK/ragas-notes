from __future__ import annotations

from dataclasses import dataclass


class Money:
    def __init__(self, amount):
        self.amount = amount

    def __eq__(self, x: Money) -> bool:
        return self.amount == x.amount and type(self) == type(x)


class Dollar(Money):
    def __mul__(self, x):
        # Value Objects
        return Dollar(self.amount * x)


class Franc(Money):
    def __mul__(self, x):
        # Value Objects
        return Franc(self.amount * x)


def test_multiplication():
    five = Dollar(5)
    assert five * 2 == Dollar(10)
    assert five * 3 == Dollar(15)


def test_equity():
    assert Dollar(5) == Dollar(5)
    assert not Dollar(5) == Dollar(6)

    assert Franc(5) == Franc(5)
    assert not Franc(5) == Franc(6)

    assert Franc(5) != Dollar(5)


def test_franc_multiplication():
    five = Franc(5)
    assert Franc(10) == five * 2
    assert Franc(15) == five * 3
