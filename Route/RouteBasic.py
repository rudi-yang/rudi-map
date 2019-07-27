import numpy as np

from Map.MapBasic import RudiMapBasic


# desc 测试路径规划算法
class MapEntity(RudiMapBasic):
    def __init__(self, x, y):
        super(MapEntity, self).__init__(x, y)
        # add barrier
