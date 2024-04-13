# include GCN, GAttnNet
# reimplement of Image Generation from Scene Graphs by Jinbin Bai
#

import torch
import torch.nn as nn

# Sample data for GConv, GCN, GAttn and GAttnNet
# {
#     "objects": [
#       "car", "car", "cage", "person", "grass", "tree", "playingfield", "person"
#     ],
#     "relationships": [
#       [0, "left of", 1],
#       [0, "above", 6],
#       [1, "above", 4],
#       [3, "left of", 7],
#       [5, "above", 7],
#       [6, "below", 2],
#       [6, "below", 3],
#       [7, "left of", 4]
#     ]
# }

class GConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """
    def __init__(self, input_dim, output_dim=None, hidden_dim=384, pooling='avg', mlp_normalization='none'):
        super(GConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        self.gconv1 = self._build_mlp([3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim], batch_norm=mlp_normalization)
        self.gconv1.apply(self._init_weights)

        self.gconv2 = self._build_mlp([hidden_dim, hidden_dim, output_dim], batch_norm=mlp_normalization)
        self.gconv2.apply(self._init_weights)
    
    def _init_weights(self,module):
        if hasattr(module, 'weight'):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def _build_mlp(self, dim_list, activation='relu',batch_norm='none',dropout=0,final_nonlinearity=True):
        layers = []
        for i in range(len(dim_list) - 1):
            input_dim, output_dim = dim_list[i], dim_list[i + 1]
            layers.append(nn.Linear(input_dim, output_dim))
            final_layer = (i == len(dim_list) - 2)
            if not final_layer or final_nonlinearity:
                if batch_norm == 'batch':
                    layers.append(nn.BatchNorm1d(output_dim))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leakyrelu':
                    layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

        
    def forward(self, obj_vecs, pred_vecs, edges):
        """
            Inputs:
            - obj_vecs: FloatTensor of shape (B, O, D) giving vectors for all objects
            - pred_vecs: FloatTensor of shape (B, T, D) giving vectors for all predicates
            - edges: LongTensor of shape (B, T, 2) where edges[k] = [i, j] indicates the presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
            
            Outputs:
            - new_obj_vecs: FloatTensor of shape (B, O, D) giving new vectors for objects
            - new_pred_vecs: FloatTensor of shape (B, T, D) giving new vectors for predicates
        """
        B, _, D = obj_vecs.shape
        B, T, _ = edges.shape
        assert T == pred_vecs.shape[1]
        assert D == pred_vecs.shape[2]
        Din, Hidden, Dout = self.input_dim, self.hidden_dim, self.output_dim
        # print("===================obj_vecs edges pred_vecs", obj_vecs.shape, edges.shape, pred_vecs.shape)

        # Break apart the edges into subject and object indices
        subjects_idx = edges[:, :, 0].contiguous() # (T,)
        objects_idx = edges[:, :, 1].contiguous() # (T,)
        # print("===================objects_idx subjects_idx", objects_idx.shape, subjects_idx.shape) # torch.Size([80, 6006]) torch.Size([80, 6006])
        # print(objects_idx)
        # Get the subject and object vectors
        current_subject_vecs = torch.gather(obj_vecs, 1, subjects_idx.unsqueeze(2).expand(-1,-1,384).long()) # (B, T, D)
        current_object_vecs = torch.gather(obj_vecs, 1, objects_idx.unsqueeze(2).expand(-1,-1,384).long())  # (B, T, D)
        # current_subject_vecs = obj_vecs[subjects_idx] # (T, D)
        # current_object_vecs = obj_vecs[objects_idx] # (T, D)
        # print("===================current_subject_vecs current_object_vecs", current_subject_vecs.shape, current_object_vecs.shape)


        # Concatenate the subject, predicate, and object vectors
        current_triple_vecs = torch.cat([current_subject_vecs, pred_vecs, current_object_vecs], dim=-1) # (T, 3*Din)
        # print("=============================current_triple_vecs", current_triple_vecs.shape)
        new_triple_vecs = self.gconv1(current_triple_vecs) # (T, 2*Hidden + Dout)

        # Split the new triple vectors into new subject, predicate, and object vectors
        new_subject_vecs = new_triple_vecs[:, :, :Hidden] # (B, T, Hidden)
        new_pred_vecs = new_triple_vecs[:, :, Hidden:(Hidden + Dout)] # (B, T, Dout)
        new_object_vecs = new_triple_vecs[:, :, (Hidden + Dout):] # (B, T, Hidden)
        # Aggregate the new subject and object vectors
        new_obj_vecs = self._aggregate(new_subject_vecs, new_object_vecs, subjects_idx, objects_idx, obj_vecs.shape[1]) # (O, Dout)
        
        # Apply a second MLP to the new object vectors
        new_obj_vecs = self.gconv2(new_obj_vecs) # (O, Dout)

        return new_obj_vecs, new_pred_vecs

    def _aggregate(self, new_s_vecs, new_o_vecs, s_idx, o_idx, Obj_vecs_shape_0):
        
        # print("================================new_s_vecs, new_o_vecs, s_idx, o_idx, Obj_vecs_shape_0", new_s_vecs.shape, new_o_vecs.shape, s_idx.shape, o_idx.shape, Obj_vecs_shape_0)
        # torch.Size([80, 6006, 512]) torch.Size([80, 6006, 512]) torch.Size([80, 6006]) torch.Size([80, 6006]) 78
        dtype, device = new_s_vecs.dtype, new_s_vecs.device
        B, _, H = new_s_vecs.shape

        # Allocate space for pooled object vectors of shape (B, O, H)
        # pooled_obj_vecs = torch.zeros(Obj_vecs_shape_0, H, dtype=dtype, device=device)
        pooled_obj_vecs = torch.zeros(B, Obj_vecs_shape_0, H, dtype=dtype, device=device)

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (B, T, D)
        s_idx_exp = s_idx.unsqueeze(2).expand_as(new_s_vecs).long()
        o_idx_exp = o_idx.unsqueeze(2).expand_as(new_o_vecs).long()
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(1, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(1, o_idx_exp, new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs / pooled_obj_vecs.norm()
        # print("==================pooled_obj_vecs", pooled_obj_vecs.shape)

        # if self.pooling == 'avg':
        #     # Figure out how many times each object has appeared
        #     obj_counts = torch.zeros(B, Obj_vecs_shape_0, dtype=dtype, device=device)
        #     obj_counts = obj_counts.scatter_add(1, s_idx.long(), torch.ones_like(s_idx, dtype=dtype, device=device))
        #     obj_counts = obj_counts.scatter_add(1, o_idx.long(), torch.ones_like(o_idx, dtype=dtype, device=device))
            # Divide by the number of times each object appeared
            # print("=======================", obj_counts.shape)
            # pooled_obj_vecs = pooled_obj_vecs / obj_counts.unsqueeze(2)

        return pooled_obj_vecs


class GCN(nn.Module):
    """
    A sequence of scene graph convolutions layers.
    """
    def __init__(self, input_dim, hidden_dim=384, num_layers=5, pooling='avg', mlp_normalization='none'):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dim = input_dim
            else:
                input_dim = hidden_dim
            self.convs.append(GConv(input_dim, hidden_dim, hidden_dim, pooling, mlp_normalization))

    def forward(self, obj_vecs, pred_vecs, edges):
        """
            Inputs:
            - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
            - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
            - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
            
            Outputs:
            - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
            - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
        """
        for i in range(self.num_layers):
            obj_vecs, pred_vecs = self.convs[i](obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs
    