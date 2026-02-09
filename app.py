import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertTokenizerFast
from torch_geometric.nn import GATConv

st.set_page_config(page_title="GAT Relation Extractor", layout="wide")

@st.cache_resource
def load_nlp_resources():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    return nlp, tokenizer

nlp, tokenizer = load_nlp_resources()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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

class DistilBERTGATModel(nn.Module):
    def __init__(self, num_classes=19, gat_dim=256, gat_heads=8, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size
        
        self.gat1 = GATConv(self.hidden_size, gat_dim, heads=gat_heads, dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(gat_dim*gat_heads, gat_dim, heads=1, dropout=dropout, add_self_loops=True)
        
        combined_dim = self.hidden_size + gat_dim + self.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, edge_index, batch_data, return_attention=False):
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
                x_gat, (edge_index_attn, alpha) = self.gat1(x, edges, return_attention_weights=True)
                attention_weight_list.append((edge_index_attn, alpha))
            else:
                x_gat = self.gat1(x, edges)

            x_gat = F.relu(x_gat)
            x_gat = F.dropout(x_gat, p=0.3, training=self.training)
            x_gat = self.gat2(x_gat, edges)            
            
            gat_ent = torch.stack([x_gat[idx1], x_gat[idx2]]).mean(dim=0)
            gat_features.append(gat_ent)
            
        combined = torch.cat([cls_tokens, torch.stack(gat_features), torch.stack(raw_entity_features)], dim=1)
        logits = self.classifier(combined)
        
        if return_attention:
            return logits, attention_weight_list
        return logits

@st.cache_resource
def get_model():
    model = DistilBERTGATModel(num_classes=19).to(device)
    try:
        model.load_state_dict(torch.load('gat_best.pth', map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Weights 'gat_best.pth' not found. Please upload them.")
        return None

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
                if i != j: edges.append([i, j])
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    return edges, len(doc)

def visualize_attention(clean_sent, model, input_ids, mask, edge_index, batch_data):
    model.eval()
    with torch.no_grad():
        _, attn_data = model(input_ids, mask, [edge_index], batch_data, return_attention=True)
    
    edges, alpha = attn_data[0]
    weights = alpha.mean(dim=1).cpu().numpy()
    edges = edges.cpu().numpy()
    
    fig = plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    G = nx.DiGraph()
    doc = nlp(clean_sent)
    token_list = [t.text for t in doc]
    
    max_w = 0
    for i, (src, dst) in enumerate(zip(edges[0], edges[1])):
        if src < len(token_list) and dst < len(token_list):
            w = weights[i]
            if w > max_w: max_w = w
            if w > 0.05: 
                G.add_edge(token_list[src], token_list[dst], weight=w)

    pos = nx.spring_layout(G, k=0.8, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrows=True)
    
    if len(G.edges()) > 0:
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='r', alpha=0.7, arrows=True)
    
    plt.title(f"AI Model Focus (GAT Attention)", fontsize=14)

    plt.subplot(1, 2, 2)
    G_dep = nx.DiGraph()
    for token in doc:
        if token.head.i != token.i:
            G_dep.add_edge(token.text, token.head.text, label=token.dep_)
            
    pos_dep = nx.spring_layout(G_dep, k=0.8, seed=42)
    nx.draw(G_dep, pos_dep, with_labels=True, node_color='lightgreen', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G_dep, pos_dep, edge_labels=nx.get_edge_attributes(G_dep, 'label'), font_size=8)
    plt.title("Grammar Truth (Dependency Tree)", fontsize=14)
    
    plt.tight_layout()
    return fig

def run_inference(model, sentence, e1_word, e2_word):
    tagged_sentence = sentence.replace(e1_word, f"<e1>{e1_word}</e1>", 1)
    tagged_sentence = tagged_sentence.replace(e2_word, f"<e2>{e2_word}</e2>", 1)
    clean_sent = sentence 
    
    encoded = tokenizer(clean_sent, max_length=128, truncation=True, padding='max_length', return_tensors='pt', return_offsets_mapping=True)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    offset_mapping = encoded['offset_mapping'].squeeze(0)
    
    e1_start_char = clean_sent.find(e1_word)
    e2_start_char = clean_sent.find(e2_word)
    e1_token_id = 1
    e2_token_id = 2
    for i, (start, end) in enumerate(offset_mapping):
        if start <= e1_start_char < end:
            e1_token_id = i
        if start <= e2_start_char < end:
            e2_token_id = i

    edge_index, _ = build_dependency_graph(clean_sent)
    edge_index_list = [edge_index.to(device)]
    
    batch_data = {
        'e1_idx': [e1_token_id],
        'e2_idx': [e2_token_id]
    }
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask, edge_index_list, batch_data)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    return id2label[pred_idx.item()], confidence.item(), (clean_sent, input_ids, attention_mask, edge_index.to(device), batch_data)

st.title("Graph Attention Relation Extractor")
st.markdown("Combines **DistilBERT** (Semantic) + **GAT** (Structural) to find relationships.")

st.caption(f"Running on: {device}")
model = get_model()

st.subheader("Step 1: Input Text")
sentence = st.text_input("Type a sentence:", "The author wrote a book in the library.")

if sentence and model:
    tokens = sentence.split()
    st.subheader("Step 2: Select Entities")
    col1, col2 = st.columns(2)
    with col1:
        e1 = st.selectbox("Entity 1 (Source)", tokens, index=0)
    with col2:
        e2 = st.selectbox("Entity 2 (Target)", tokens, index=len(tokens)-1 if len(tokens)>0 else 0)

    if st.button("Analyze Relationship", type="primary"):
        with st.spinner("Analyzing semantics and dependency graph..."):
            try:
                prediction, conf, viz_data = run_inference(model, sentence, e1, e2)
                
                st.divider()
                r_col1, r_col2 = st.columns([1, 2])
                
                with r_col1:
                    st.metric("Prediction", prediction)
                    st.metric("Confidence", f"{conf:.2%}")
                    if prediction == "Other":
                        st.caption("'Other' means no specific relationship found.")
                
                with r_col2:
                    st.markdown("### Model Logic Visualization")
                    st.info("Left: Which words the AI focused on. Right: The actual grammar.")
                    
                    clean_sent, input_ids, mask, edge_index, batch_data = viz_data
                    fig = visualize_attention(clean_sent, model, input_ids, mask, edge_index, batch_data)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Inference Error: {e}")