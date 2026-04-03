from typing import NewType

import numpy as np
import numpy.typing as npt

RawName = NewType("RawName", str)
type RawEncoding = npt.NDArray[np.float32]
type RawFrame = npt.NDArray[np.uint8]
type RawBoundingBox = tuple[int, int, int, int]
