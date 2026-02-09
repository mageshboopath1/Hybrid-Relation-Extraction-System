import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
from transformers import DistilBertModel, DistilBertTokenizerFast
from torch_geometric.nn import GATConv


class DistilBERTGATModel(nn.Module):
    def __init__(self, num_classes=19, gat_dim=256, gat_heads=8, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size
        self.gat1 = GATConv(self.hidden_size,gat_dim,heads = gat_heads,dropout = dropout,add_self_loops=True)
        self.gat2 = GATConv(gat_dim*gat_heads,gat_dim,heads=1,dropout=dropout,add_self_loops=True)
        combined_dim = self.hidden_size+gat_dim+self.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(dropout),nn.Linear(combined_dim,512),nn.ReLU(),nn.Dropout(dropout),nn.Linear(512,num_classes))

    def forward(self,input_ids,attention_mask,edge_index,batch_data,return_attention=False):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_out.last_hidden_state
        cls_tokens = embeddings[:, 0, :] 
        e1_indices = batch_data.get('e1_idx', [0]*input_ids.shape[0])
        e2_indices = batch_data.get('e2_idx', [0]*input_ids.shape[0])
        batch_size = input_ids.shape[0]
        gat_features = []
        raw_entity_features = []
        attention_weight_list = []
        for i in range(batch_size):
            valid_len = attention_mask[i].sum().item()
            x = embeddings[i, :valid_len, :]

            idx1 = min(e1_indices[i], valid_len-1)
            idx2 = min(e2_indices[i], valid_len-1)

            raw_ent = torch.stack([x[idx1], x[idx2]]).mean(dim=0)
            raw_entity_features.append(raw_ent)

            edges = edge_index[i].to(device) if isinstance(edge_index[i], torch.Tensor) else edge_index[i]
            if edges.numel() > 0:
                mask = (edges[0] < valid_len) & (edges[1] < valid_len)
                edges = edges[:, mask]
            
            if return_attention:
                x_gat,(edge_index_attn,alpha) = self.gat1(x,edges,return_attention_weights=True)
                attention_weight_list.append((edge_index_attn,alpha))
            else:
                x_gat = self.gat1(x,edges)

            x_gat = F.relu(x_gat)
            x_gat = F.dropout(x_gat, p=0.3, training=self.training)
            x_gat = self.gat2(x_gat, edges)            
            gat_ent = torch.stack([x_gat[idx1], x_gat[idx2]]).mean(dim=0)
            gat_features.append(gat_ent)
            
        combined = torch.cat([cls_tokens, torch.stack(gat_features), torch.stack(raw_entity_features)], dim=1)
        logits = self.classifier(combined)

        if return_attention:
            return logits,attention_weight_list
        return logits

def extract_entities(sentence):
    """Parses <e1> and <e2> tags from the raw string."""
    e1_start = sentence.find('<e1>') + 4
    e1_end = sentence.find('</e1>')
    e1 = sentence[e1_start:e1_end]
    
    e2_start = sentence.find('<e2>') + 4
    e2_end = sentence.find('</e2>')
    e2 = sentence[e2_start:e2_end]
    
    clean = sentence.replace('<e1>', '').replace('</e1>', '')
    clean = clean.replace('<e2>', '').replace('</e2>', '')
    return clean, e1, e2

def build_dependency_graph(sentence):
    doc = nlp(sentence)
    edges = []
    for token in doc:
        if token.head.i != token.i:
            edges.append([token.i, token.head.i])
            edges.append([token.head.i, token.i])
            
    if len(edges) == 0:
        for i in range(len(doc)):
            for j in range(len(doc)):
                if i != j:
                    edges.append([i, j])
                    
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    return edges

def predict_relation(tagged_sentence):
    model.eval()
    
    clean_sent, e1_text, e2_text = extract_entities(tagged_sentence)
    
    encoded = tokenizer(clean_sent, max_length=128, truncation=True, padding='max_length', return_tensors='pt', return_offsets_mapping=True)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    offset_mapping = encoded['offset_mapping'].squeeze(0)
    
    e1_start_char = clean_sent.lower().find(e1_text.lower())
    e2_start_char = clean_sent.lower().find(e2_text.lower())
    
    e1_token_id = 1
    e2_token_id = 2
    
    for i, (start, end) in enumerate(offset_mapping):
        if start <= e1_start_char < end:
            e1_token_id = i
        if start <= e2_start_char < end:
            e2_token_id = i

    edge_index = build_dependency_graph(clean_sent)
    edge_index_list = [edge_index.to(device)]
    
    batch_data = {
        'e1_idx': [e1_token_id],
        'e2_idx': [e2_token_id]
    }
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask, edge_index_list, batch_data)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        
    return id2label[pred_idx.item()], confidence.item()

if __name__ == "__main__":
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    RELATION_LABELS = [
    'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
    'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
    'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
    'Content-Container(e1,e2)', 'Content-Container(e2,e1)',
    'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
    'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
    'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
    'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
    'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
    'Other'
    ]

    id2label = {idx: rel for idx, rel in enumerate(RELATION_LABELS)}

    device = device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")
    model = DistilBERTGATModel(num_classes=19).to(device)  
    gat_params = sum(p.numel() for p in model.parameters())
    print(f"DistillBertGAT Model:{gat_params/1e6:.1f}M parameters")
    model.load_state_dict(torch.load('gat_best.pth', map_location=device))
    print("Model loaded successfully!")

    test_samples = [
        "The <e1>child</e1> was inside the <e2>house</e2>.",
        "The <e1>author</e1> wrote a <e2>book</e2>.",
        "The <e1>storm</e1> caused a massive <e2>flood</e2>."
    ]

    print("\n--- Starting Inference ---\n")
    for sent in test_samples:
        pred_label, conf = predict_relation(sent)
        print(f"Input: {sent}")
        print(f"Predicted: {pred_label} ({conf:.1%})\n")