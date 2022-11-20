import math
from typing import List, Tuple

import numpy as np

class WaypointsExtender:
  def __init__(self, k: float = 1.0):
    """
    Args:
      k (int, optional): tells how many steps per matrix pixel
    """
    BOUNDARIES = [(21.9430, -67.5), (55.7765, -135)]
    M_WIDTH = abs(BOUNDARIES[0][1] - BOUNDARIES[1][1])
    M_HEIGHT = abs(BOUNDARIES[0][0] - BOUNDARIES[1][0])
    P_WIDTH = 5120
    P_HEIGHT = 2566
    WIDTH_STEP = M_WIDTH / P_WIDTH
    HEIGHT_STEP = M_HEIGHT / P_HEIGHT
    self.step = (WIDTH_STEP + HEIGHT_STEP) / 2 / k

  def _create_between(self,
                      p1: Tuple[float, float], p2: Tuple[float, float],
                      verbose: bool = False) -> List[Tuple[float, float]]:
    dist = math.sqrt(abs(p2[0] - p1[0])**2 + abs(p2[1] - p1[1])**2)
    if dist < self.step:
      return [p1, p2]
    n = math.ceil(dist / self.step)
    
    if verbose:
      print(f"dist={dist}, n={n}")
    x_space = np.linspace(start = p1[0], stop = p2[0], num = n)
    y_space = np.linspace(start = p1[1], stop = p2[1], num = n)
    
    return [t for t in zip(x_space, y_space)]

  def __call__(self, points: List[Tuple[float, float]], verbose: bool = False) -> List[Tuple[float, float]]:
    new_points = []
    for idx in range(len(points) - 1):
      i, j = idx, idx + 1
      between = self._create_between(points[i], points[j], verbose)
      if i != 0:
        new_points.extend(between[1:])
      else:
        new_points.extend(between)
    return new_points

    