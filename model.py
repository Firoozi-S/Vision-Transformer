import torch
import torch.nn as nn

import globals

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(in_features = globals.EMBEDDING_DIM, out_features = 2 * globals.EMBEDDING_DIM, bias = False)
        self.linear_2 = nn.Linear(in_features = 2 * globals.EMBEDDING_DIM, out_features = globals.EMBEDDING_DIM, bias = False)

    def forward(self, data):
        layer_1 = self.linear_1(data)
        layer_1 = nn.GELU()(layer_1)

        return self.linear_2(layer_1)


class MultiHeadedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(in_features = globals.EMBEDDING_DIM, out_features =  globals.EMBEDDING_DIM, bias = False)
        self.query = nn.Linear(in_features = globals.EMBEDDING_DIM, out_features =  globals.EMBEDDING_DIM, bias = False)
        self.value = nn.Linear(in_features = globals.EMBEDDING_DIM, out_features =  globals.EMBEDDING_DIM, bias = False)

        self.out = nn.Linear(in_features = globals.EMBEDDING_DIM, out_features = globals.NUM_HEADS, bias = False)

    def forward(self, data):
        batch_size, num_patches, embed_dim = data.shape

        key = self.key(data) # batch_size, num_patches, embed_dim
        query = self.query(data) # batch_size, num_patches, embed_dim
        value = self.value(data) # batch_size, num_patches, embed_dim

        key = key.reshape( batch_size, num_patches, globals.NUM_LAYERS, globals.NUM_HEADS).transpose(1,2) # batch_size, num_layers, num_patches, num_heads
        query = query.reshape( batch_size, num_patches, globals.NUM_LAYERS, globals.NUM_HEADS).transpose(1,2) # batch_size, num_layers, num_patches, num_heads
        value = value.reshape( batch_size, num_patches, globals.NUM_LAYERS, globals.NUM_HEADS).transpose(1,2) # batch_size, num_layers, num_patches, num_heads

        attention_scores = ( query @ torch.transpose(input = key, dim0 = -2, dim1 = -1) ) * (globals.NUM_HEADS)**-0.5 # batch_size, num_layers, num_patches, num_patches

        attention = torch.softmax(input = attention_scores, dim = -1)# batch_size, num_layers, num_patches, num_patches

        # (attention @ value) --> batch_size, num_layers, num_patches, num_heads
        #  torch.transpose(input = (attention @ value), dim0 = 1, dim1 = 2) ---> batch_size, num_patches, num_layers, num_heads
        weighted_avg = torch.transpose(input = (attention @ value), dim0 = 1, dim1 = 2).reshape(batch_size, num_patches, embed_dim) 

        output = self.out(weighted_avg)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_headed_attention = nn.ModuleList([MultiHeadedAttention() for _ in range(globals.NUM_LAYERS)])
        self.mlp = MLP()
        self.norm_layer_1 = nn.LayerNorm(normalized_shape = (globals.BATCH_SIZE, globals.NUM_PATCHS + 1, globals.EMBEDDING_DIM))
        self.norm_layer_2 = nn.LayerNorm(normalized_shape = (globals.BATCH_SIZE, globals.NUM_PATCHS + 1, globals.EMBEDDING_DIM))
    
    def forward(self, embedded_data):

        multi_headed_attention = torch.cat(tensors = [h(embedded_data) for h in self.multi_headed_attention], dim = -1)

        multi_headed_attention_output = embedded_data + self.norm_layer_1(multi_headed_attention)

        encoder_output = multi_headed_attention_output + self.mlp(self.norm_layer_2(multi_headed_attention_output))

        return encoder_output



class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding = nn.Parameter(data = torch.zeros(size = (1, globals.NUM_PATCHS + 1, globals.EMBEDDING_DIM)))
        self.class_token = nn.Parameter(data = torch.zeros(size = (globals.BATCH_SIZE, 1, globals.EMBEDDING_DIM)))
        self.patch_embedding = nn.Conv2d(in_channels = globals.CHANNELS, out_channels = globals.EMBEDDING_DIM, kernel_size = globals.PATCH_SIZE, stride = globals.PATCH_SIZE, bias = False)
        self.transformer_encoder_blocks = nn.ModuleList([TransformerEncoder() for _ in range(globals.NUM_LAYERS)])
        self.mlp_head = nn.Linear(in_features = globals.EMBEDDING_DIM, out_features = globals.NUM_CLASSES)



    def forward(self, data):
        patch_embedding = self.patch_embedding(data)
        patch_embedding = nn.Flatten(start_dim = 2, end_dim = -1)(patch_embedding)
        patch_embedding = torch.transpose(input = patch_embedding, dim0 = 1, dim1 = -1)

        patch_class_embedding = torch.cat(tensors = ( self.class_token, patch_embedding ), dim = 1)

        embedded_data = patch_class_embedding + self.position_embedding


        for block in self.transformer_encoder_blocks:
            encoder_output = block(embedded_data)

        class_output = encoder_output[:, 0]

        prediction = self.mlp_head(class_output)
        
        return prediction