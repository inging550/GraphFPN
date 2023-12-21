import dgl
import torch

def heterograph(name_n_feature, dim_n_feature, nb_nodes = 2, is_birect = True):
  graph_data = {
      ('n', 'contextual', 'n'): (torch.LongTensor([0]), torch.LongTensor([1])),
      ('n', 'hierarchical', 'n'): (torch.LongTensor([0]), torch.LongTensor([1]))
      }
  g = dgl.heterograph(graph_data, num_nodes_dict = {'n': nb_nodes}).int()
  g.nodes['n'].data[name_n_feature] = torch.zeros([g.num_nodes(), dim_n_feature])
  if is_birect:
    g = dgl.to_bidirected(g, copy_ndata = True)
  return g

def build_edges(g, c3_shape = 80, c4_shape = 40, c5_shape = 20):
  c3_size, c4_size , c5_size= c3_shape * c3_shape, c4_shape * c4_shape, c5_shape * c5_shape
  c3 = torch.arange(0, c3_size)
  c4 = torch.arange(c3_size, c3_size + c4_size)   
  c5 = torch.arange(c3_size + c4_size, c3_size + c4_size + c5_size)
  
  # build contextual edges
  for i in range(c3_shape - 1):
    g = hetero_add_edges(g, c3[i*c3_shape : (i+1)*c3_shape], c3[(i+1)*c3_shape : (i+2)*c3_shape], 'contextual')          # build edges between different rows (27 * 28 = 756)
    g = hetero_add_edges(g, c3[i : (c3_size+i) : c3_shape], c3[i+1 : (c3_size+i+1) : c3_shape], 'contextual')            # build edges between different column (27 * 28 = 756)
  for i in range(c4_shape - 1):
    g = hetero_add_edges(g, c4[i*c4_shape : (i+1)*c4_shape], c4[(i+1)*c4_shape : (i+2)*c4_shape], 'contextual')          # 14 * 13 = 182 
    g = hetero_add_edges(g, c4[i : (c4_size+i) : c4_shape], c4[i+1 : (c4_size+i+1) : c4_shape], 'contextual') 
      # g = hetero_add_edges(g, c4[i*c4_shape : (i+1)*c4_shape], c3)
  for i in range(c5_shape - 1):
    g = hetero_add_edges(g, c5[i*c5_shape : (i+1)*c5_shape], c5[(i+1)*c5_shape : (i+2)*c5_shape], 'contextual')          # 6 * 7 = 42
    g = hetero_add_edges(g, c5[i : (c5_size+i) : c5_shape], c5[i+1 : (c5_size+i+1) : c5_shape], 'contextual') 
  
  # build hierarchical edges
  c3_stride = torch.reshape(c3, (c3_shape, c3_shape))[2:c3_shape:2, 2:c3_shape:2]  # Get pixel indices in C3 for build hierarchical edges
  c4_stride = torch.reshape(c4, (c4_shape, c4_shape))[2:c4_shape:2, 2:c4_shape:2]
  c5_stride = torch.reshape(c3, (c3_shape, c3_shape))[2:c3_shape-4:4, 2:c3_shape-4:4]
  
  # Edges between c3 and c4
  counter = 1
  for i in torch.reshape(c3_stride, [-1]).numpy():
    c3_9 = neighbor_9(i, c3_shape)
    g = hetero_add_edges(g, c3_9, c4[counter], 'hierarchical') 
    if counter % (c4_shape-1) == 0 :
      counter += 2 
    else:
      counter += 1

  # Edges between c4 and c5
  counter = 1
  for i in torch.reshape(c4_stride, [-1]).numpy():
    c4_9 = neighbor_9(i, c4_shape)
    g = hetero_add_edges(g, c4_9, c5[counter], 'hierarchical') 
    if counter % (c5_shape-1) == 0 :
      counter += 2 
    else:
      counter += 1
  
  # Edges between c3 and c5
  counter = 1
  for i in torch.reshape(c5_stride, [-1]).numpy():
    c5_9 = neighbor_25(i, c3_shape)
    g = hetero_add_edges(g, c5_9, c5[counter], 'hierarchical') 
    if counter % (c5_shape-1) == 0 :
      counter += 2 
    else:
      counter += 1
  return g

def hetero_add_edges(g, u, v, edges):
  if isinstance(u,int):
    g.add_edges(torch.Tensor([u]), torch.Tensor([v]), etype = edges)
  elif isinstance(u,list):
    g.add_edges(torch.Tensor(u), torch.Tensor(v), etype = edges)
  else:
    g.add_edges(u.int(), v.int(), etype = edges)
  return g

def neighbor_9(i, c_shape):
  return torch.Tensor([i-c_shape-1, i-c_shape, i-c_shape+1, i-1, i, i+1, i+c_shape-1, i+c_shape, i+c_shape+1])


def neighbor_25(i, c_shape):
  return torch.Tensor([i-2*c_shape-2, i-2*c_shape-1, i-2*c_shape, i-2*c_shape+1, i-2*c_shape+2,
                      i-c_shape-2, i-c_shape-1, i-c_shape, i-c_shape+1, i-c_shape+2, 
                      i-2, i-1, i, i+1, i+2, 
                      i+c_shape-2, i+c_shape-1, i+c_shape, i+c_shape+1, i+c_shape+2,
                      i+2*c_shape-2, i+2*c_shape-1, i+2*c_shape, i+2*c_shape+1, i+2*c_shape+2])

def simple_graph(g):
  g = g.to("cpu")
  g = dgl.to_simple(g, copy_ndata = True)
  g = g.to("cuda")
  return g

def simple_birected(g):
  g = g.to("cpu")
  g = dgl.to_simple(g, copy_ndata = True)
  g = dgl.to_bidirected(g, copy_ndata = True)
  g = g.to("cuda")
  return g

def hetero_subgraph(g, edges):
  return dgl.edge_type_subgraph(g, [edges])

