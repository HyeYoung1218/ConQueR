import torch
import torch.nn as nn

activations = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU
}

class SentencePairModel(nn.Module):
    def __init__(self, 
                backbone,
                hidden_size=768,
                dropout=0.0,
                fc_layers=[],
                act='tanh',
                **args):
        super(SentencePairModel, self).__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc_layers = fc_layers if isinstance(fc_layers, list) else list(fc_layers)
        self.act = act

        self.classifier = self.build_classifiers()
    
    def build_classifiers(self):
        dims = [self.hidden_size] + self.fc_layers + [1]
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if (i+1) != len(dims) - 1:
                layers.append(activations[self.act]())
        classifiers = nn.Sequential(*layers)

        self.pooler_layer = nn.Linear(self.hidden_size, self.hidden_size)
        torch.nn.init.xavier_uniform_(self.pooler_layer.weight)
        self.tanh = nn.Tanh()
        self.pooler = torch.nn.Sequential(self.pooler_layer, self.tanh)

        return classifiers
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids, pooling=False):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        output = self.backbone(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.long().to(token_ids.device))
        
        if self.backbone.__module__.split('.')[0] == 'etribert':
            if isinstance(output, tuple):
                all_seq_output, pooled_output = output
                sequence_output = all_seq_output[-1]    # last layer
            else:
                sequence_output = output['last_hidden_state']
                pooled_output = output['pooler_output']
        else:
            sequence_output = output[0] ## (b, length, embedding_dim)
            pooled_output = sequence_output[:, 0, :]
            if pooling:
                pooled_output = self.pooler(pooled_output)

        if self.dropout:
            out = self.dropout(pooled_output)
        else:
            out = pooled_output
        

        cls_out = self.classifier(out).squeeze(-1)
        
        ret = {'cls_out': cls_out}
        
        return ret