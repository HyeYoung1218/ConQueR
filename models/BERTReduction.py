import torch
import torch.nn as nn
import torch.nn.functional as F

activations = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU
}

class BERTReduction(nn.Module):
    def __init__(self, 
                backbone,
                hidden_size=768,
                dropout=0.0,
                fc_layers=[],
                act='tanh',
                cls_enhanced=False,
                pred_num_core=False,
                contrastive=False,
                **args):
        super(BERTReduction, self).__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc_layers = fc_layers if isinstance(fc_layers, list) else list(fc_layers)
        self.act = act
        self.cls_enhanced = cls_enhanced
        self.pred_num_core = pred_num_core
        self.contrastive = contrastive

        self.classifier, self.cls_linear, self.num_core_linear = self.build_classifiers()
    
    def build_classifiers(self):
        token_classifier, cls_linear, num_core_linear = None, None, None
        

        ## out_dim = 2로 변경함. (ES)
        dims = [self.hidden_size] + self.fc_layers + [1] #  [768, 1]
        token_classifier = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            token_classifier.append(nn.Linear(in_dim, out_dim))
            if (i+1) != len(dims) - 1:
                token_classifier.append(activations[self.act]())

        if self.cls_enhanced:
            cls_linear = nn.Linear(self.hidden_size * 2, dims[1])


            
        if self.pred_num_core:
            num_core_linear = nn.Linear(self.hidden_size, 1)
        
        if self.contrastive:
            cls_linear = nn.Linear(self.hidden_size * 2, dims[1])
            
        self.pooler_layer = nn.Linear(self.hidden_size, self.hidden_size)
        torch.nn.init.xavier_uniform_(self.pooler_layer.weight)
        self.tanh = nn.Tanh()
        self.pooler = torch.nn.Sequential(self.pooler_layer, self.tanh)


        
        return token_classifier, cls_linear, num_core_linear
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        # Extract contextual embeddings from pre-trained BERT: (# batch, length, dim)
        # all_seq_output: list of embeddings from all layers: list of (# batch, length, dim)
        # pooled_output: [CLS] embedding (# batch, dim)
        
        output = self.backbone(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.long().to(token_ids.device))
        
        if self.backbone.__module__.split('.')[0] == 'etribert':
            all_seq_output, pooled_output = output
            sequence_output = all_seq_output[-1]    # last layer
        else:
            sequence_output = output[0]
            pooled_output = sequence_output[:, 0, :]
            cls_output = self.pooler(pooled_output)
        
        if self.dropout:
            out = self.dropout(sequence_output)
        else:
            out = sequence_output
        
        # out [out;pooled_output]

        # Token-wise classifier: out - (# batch, length, dim) => [CLS] concat (# batch, length, dim*2) -> (# batch, length, 1)
        
        for i, layer in enumerate(self.classifier):
            
            if i == 0 and self.cls_enhanced:
                
                p_output = pooled_output.unsqueeze(dim=1)
                x = p_output.expand(-1,32,-1)
                y = torch.cat([x,out],dim=2)
                # out = layer(out)
                # cls_out = self.cls_linear(pooled_output).unsqueeze(1)
                # out = out+cls_out
                out = self.cls_linear(y)

                # from IPython import embed; embed(); exit(1)
            elif self.contrastive:
                pooled_output = F.normalize(pooled_output, p=2, dim=1)
                token_out = F.normalize(out, p=2, dim=2)
                cos_sim = torch.matmul(pooled_output.unsqueeze(1), token_out.transpose(1,2))
                out = layer(out)
            else:
                out = layer(out)
        
        # Reshape output
        # token_out: (# batch, length)
        ret = {'token_out': out.squeeze(-1)}
        ret['cls_out'] = cls_output

        if self.contrastive:
            ret['cos_sim'] = cos_sim
        if self.pred_num_core:
            num_core_out = self.num_core_linear(pooled_output).squeeze(-1)
            ret['num_core_out'] = num_core_out
        
        return ret