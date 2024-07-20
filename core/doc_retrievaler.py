import torch
import faiss
from torch import nn
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import os

class SentenceEmbedding(nn.Module):
    def __init__(self, model_name, normalize=True, device="cuda:1"):
        super(SentenceEmbedding, self).__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.normalize = normalize
        print('self.device', self.device)
        self.model.to(self.device)

    def forward(self, text_list):
        device = self.model.device
        text_emb = self.tokenizer(text_list, return_tensors="pt", truncation=True, padding="max_length")
        model_output = self.model(**text_emb.to(device))
        embeddings = self.mean_pooling(model_output, text_emb['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def save_pretrained(self, output_path):
        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        


class DocRetrievaler:
    def __init__(self, index_path, index_dim=768):
        self.q_align = torch.load('core/alignment_model/{self.database}_question.pth')
        self.m_align = torch.load('core/alignment_model/{self.database}_manual.pth')
        self.sbert_model_path = rf"../sentence-transformers/all-mpnet-base-v2"
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.sbert_model = SentenceEmbedding(self.sbert_model_path, device=device)  # 创建SBERT模型对象


        self.index_path = index_path
        self.index_dim = index_dim

        self.index = None
        self.index_id = None

    def build_index(self):
        self.faiss_index = faiss.IndexFlatIP(self.index_dim)

        self.manuals_embeds = np.load('./core/sbert_embeds/{self.database}_manuals_data.npy')
        self.historical_questions_embeds = np.load('./core/sbert_embeds/{self.dataset}_retrieval_data.npy')
        
        with open('../data/dataset/manuals/{self.database}_manuals_data.json', 'r') as f:
            self.manuals_data = json.load(f)
        with open('../data/dataset/historical_questions/{self.dataset}_retrieval_data.json', 'r') as f:
            self.historical_questions_data = json.load(f)
        
        self.align_manuals_embeds = self.m_align(self.manuals_embeds)
        self.align_historical_questions_embeds = self.q_align(self.historical_questions_embeds)
        
        self.all_retrieval = {}
        self.all_retrieval_embds = np.concatenate([self.align_manuals_embeds, self.align_historical_questions_embeds], axis=0)


        self.faiss_index.add(self.all_retrieval_embds)
        faiss.write_index(self.faiss_index, self.index_path)


    def load_index(self):
        self.faiss_index = faiss.read_index(self.index_path)
        self.index_id = np.concatenate((list(self.manuals_data.keys()) , list(self.historical_questions_data.keys())), axis=0)

    def search(self, query, topk=5):
        if self.index is None and os.path.exists(self.index_path):
            self.load_index()
        else:
            self.build_index()

        query_embedding_list = self.sbert_model([query]).cpu().numpy()

        D, I = self.index.search(query_embedding_list, topk)
        
        return I

