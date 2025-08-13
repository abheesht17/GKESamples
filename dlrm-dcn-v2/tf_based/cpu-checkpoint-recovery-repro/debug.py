import tensorflow as tf
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.core.tpu.kernels import sparse_core_layout_pb2

def get_layouts(reader):
  sc_layouts_key = "_sparse_core_table_layouts"
  sparsecore_layouts_str = ''
  for name in reader.get_variable_to_dtype_map():
     if sc_layouts_key in name:
      sparsecore_layouts_str = reader.get_tensor(name)
      break
  if sparsecore_layouts_str:
    sc_layouts = sparse_core_layout_pb2.SparseCoreTableLayouts()
    sc_layouts.ParseFromString(sparsecore_layouts_str)
    print("==== # Total layouts: ====", len(sc_layouts.tables))
    return {l.table_name: l for l in sc_layouts.tables}
  else:
    print("No layouts found")

print("Checkpoint for shuffled:")
path = "gs://chavoshi-dlrm-training/checkpoints-repro/tpu-v6e-trainer-shuffle"

reader = tf.train.load_checkpoint(path)
layouts_dict = get_layouts(reader)

print(layouts_dict) 

#List Variables
variables = tf.train.list_variables(path)
for var, shape in variables:
    if '_tpu_embedding' in var:
      print("variable: ", var, "shape: ", shape)

variables = tf.train.list_variables(path)
for var, shape in variables:
    if 'embedding' in var:
      print("variable: ", var, "shape: ", shape)

print(reader.get_tensor("model/embedding_layer/_tpu_embedding/movie_id_user_id/.ATTRIBUTES/VARIABLE_VALUE"))

print("Checkpoint for Device Assignment:")
path = "gs://chavoshi-dlrm-training/checkpoints-repro/tpu-v6e-trainer-da"

reader = tf.train.load_checkpoint(path)
layouts_dict = get_layouts(reader)

print(layouts_dict) 

#List Variables
variables = tf.train.list_variables(path)
for var, shape in variables:
    if '_tpu_embedding' in var:
      print("variable: ", var, "shape: ", shape)

variables = tf.train.list_variables(path)
for var, shape in variables:
    if 'embedding' in var:
      print("variable: ", var, "shape: ", shape)


print(reader.get_tensor("model/embedding_layer/_tpu_embedding/movie_id_user_id/.ATTRIBUTES/VARIABLE_VALUE"))
