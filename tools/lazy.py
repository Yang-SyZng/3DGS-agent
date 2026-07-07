from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


class LazyToolList(Sequence[T]):
    def __init__(self, factory: Callable[[], Sequence[T]]):
        self._factory = factory
        self._items: Sequence[T] | None = None

    def _load(self) -> Sequence[T]:
        if self._items is None:
            self._items = self._factory()
        return self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def __getitem__(self, index):
        return self._load()[index]
