from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Any, Tuple


@dataclass(frozen=True)
class ParamRef:
    slice_id: int
    peak_id: int
    name: str             # "pos" | "amp" | "lor" | "gauss"

@dataclass(frozen=True)
class ParamBounds:
    """
    Simple bounds container.
    lo/hi are inclusive; None means unset.
    """
    lo: Optional[float] = None
    hi: Optional[float] = None

    def is_set(self) -> bool:
        return (self.lo is not None) or (self.hi is not None)