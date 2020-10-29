import keras
from keras_radam import RAdam
import tensorflow as tf
from models import LossFunc
from models import Backbones
from models import Head


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)

        return frozen_graph


if __name__ == '__main__':
    keras.backend.set_learning_phase(0)
    backbone, s1, s2, s3, s4, s5, s6 = Backbones.v4_tiny(input_shape=(256, 256, 3))
    model = Head.SubNet(backbone, s4, s6, num_classes=2, num_anchors_per_layer=3)
    model.load_weights('saved_models/v4_subnet-00735.h5')
    frozen_graph = freeze_session(keras.backend.get_session(),
                                  output_names=[out.op.name for out in model.outputs])

    tf.train.write_graph(frozen_graph, "frozen_model", "v4.pb", False)
