import os
import datetime as dt
import numpy as np
from math import floor
from scipy import sparse


class WeatherCollector:
    BOUNDARIES = [(21.9430, -67.5), (55.7765, -135)]

    def __init__(self) -> None:
        self._data_path = 'data\\VIL_merc'

    def get_total_weather(self, route: list, timestamp: str) -> list:
        matrix = self._load_matrix(timestamp)
        if len(matrix) == 1 and matrix[0] == -1.0:
            return [-1.0]
        values = []
        for point in route:
            y, x = point
            y_mat, x_mat = self._get_y(
                y, matrix.shape), self._get_x(x, matrix.shape)
            values.append(self._interpolate(matrix, x_mat, y_mat))
        return values

    def _load_matrix(self, timestamp: str) -> np.array:
        date_str = self._parse_date(timestamp)
        file_name = f'VIL-{date_str}Z.npz'
        file_path = os.path.join(self._data_path, file_name)
        if os.path.exists(file_path):
            return sparse.load_npz(file_path).toarray()
        return -1.0 * np.ones((1, 1))

    def _parse_date(self, timestamp: str) -> str:
        def rounder(t):
            return (t.replace(minute=0, hour=t.hour) + dt.timedelta(hours=t.minute // 30))

        date = dt.datetime.strptime(
            timestamp.split('+')[0], '%Y-%m-%d %H:%M:%S')
        date = rounder(date)
        return f'{date.year}-{date.month:02d}-{date.day:02d}-{date.hour:02d}_{date.minute:02d}'

    def _get_x(self, point: float, shape: tuple) -> float:
        return self._get_index(point, dimension=1, matrix_size=shape[1])

    def _get_y(self, point: float, shape: tuple) -> float:
        return self._get_index(point, dimension=0, matrix_size=shape[0])

    def _get_index(self, point: float, dimension: int, matrix_size: int) -> float:
        size = np.abs(self.BOUNDARIES[1][dimension] -
                      self.BOUNDARIES[0][dimension])
        return np.abs((point - self.BOUNDARIES[0][dimension])) / size * matrix_size

    def _interpolate(self, matrix: np.array, x: float, y: float) -> float:
        ''' Copied from stackoverflow -- do not change! '''
        x_low, x_high = floor(x), floor(x+1)
        y_low, y_high = floor(y), floor(y+1)
        points = [(x_low, y_low, matrix[y_low, x_low]), (x_high, y_low, matrix[y_high, x_low]),
                  (x_low, y_high, matrix[y_low, x_high]), (x_high, y_high, matrix[y_high, x_high])]
        points = sorted(points)
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1) + 0.0)
