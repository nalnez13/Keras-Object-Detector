from AnchorUtil import LayerConfigs

default_config = [
    LayerConfigs(8, 0.2, 0.375, [1.0, 2., 3., 1 / 2., 1 / 3., 1.0]),
    LayerConfigs(16, 0.375, 0.55, [1.0, 2., 3., 1 / 2., 1 / 3., 1.0]),
    LayerConfigs(32, 0.55, 0.725, [1.0, 2., 3., 1 / 2., 1 / 3., 1.0]),
    LayerConfigs(64, 0.725, 0.9, [1.0, 2., 3., 1 / 2., 1 / 3., 1.0]),
    LayerConfigs(128, 0.9, 1.0, [1.0, 2., 3., 1 / 2., 1 / 3., 1.0]),
]
