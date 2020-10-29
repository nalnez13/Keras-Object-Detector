import math
import tensorflow as tf
import json

cluster_configs = {
    'num_anchors_per_layer': 3,
    'anchor_configs':
        [

            {
                'layer_width': 16, 'layer_height': 16, 'min_size': 25.0,
                'max_size': 51.0, 'aspect_ratios': [1.0, 1.4, 1.0]
            },
            {
                'layer_width': 8, 'layer_height': 8, 'min_size': 50.0,
                'max_size': 110.0, 'aspect_ratios': [1.0, 1.4, 1.0]
            },
            {
                'layer_width': 4, 'layer_height': 4, 'min_size': 110.0,
                'max_size': 170.0, 'aspect_ratios': [1.0, 1.4, 1.0]
            },
            {
                'layer_width': 2, 'layer_height': 2, 'min_size': 170.0,
                'max_size': 230.0, 'aspect_ratios': [1.0, 1.4, 1.0]
            },
        ]
}

# if __name__ == '__main__':
#
