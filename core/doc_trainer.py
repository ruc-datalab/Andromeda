import json
import numpy as np
import joblib
import os
import faiss
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from info_nce import InfoNCE, info_nce
import pickle
import copy

class AlignmentTrainDataset(Dataset):
    def __init__(self, train_list):
        self.data = train_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item[0][0], item[1][0], item[2][0], item[3][0], item[4][0]
    
class QuestionAlignment(nn.Module):
    def __init__(self, device, input_dim=768):
        super(QuestionAlignment, self).__init__()
        self.alignment = nn.Sequential([
            nn.Linear(input_dim, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 768),
            nn.LeakyReLU()
            ]
        )

    def forward(self, origin_embedding): ### [batchsize, 768], [batch_size, 1]
        question_embedding = self.alignment(origin_embedding)
        return question_embedding
    

class ManualAlignment(nn.Module):
    def __init__(self, device, input_dim=768):
        super(ManualAlignment, self).__init__()
        self.alignment = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 768),
            nn.LeakyReLU()
        )

    def forward(self, origin_embedding): ### [batchsize, 768], [batch_size, 1]
        manual_embedding = self.alignment(origin_embedding)
        return manual_embedding


class Trainer:
    def __init__(self, dataset, retrieve_topk = 5, lr = 0.001, epochs=10, nega_num=1):
        self.dataset = dataset
        self.retrieve_topk = retrieve_topk
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.epochs = epochs
        self.nega_num = nega_num
        return 
    
    def gen_positive(self):

        faiss_index1 = faiss.IndexFlatIP(self.historical_questions_embeds.shape[1])
        faiss_index1.add(self.historical_questions_embeds)
    
        D, I = faiss_index1.search(self.train_questions_embeds, self.retrieve_topk)
        question_positive_pairs = []

        for i, qid in enumerate(list(self.train_questions.keys())):
            question_positive_pairs.extend([(int(i), int(j)) for ind,j in enumerate(I[i]) if len(set(self.train_questions[qid]['parameter']) & set(list(self.historical_questions.keys())[j]['parameter'])) != 0])
        
        faiss_index2 = faiss.IndexFlatIP(self.manuals_embeds.shape[1])
        faiss_index2.add(self.manuals_embeds)
        D, I = faiss_index2.search(self.train_questions_embeds, self.retrieve_topk)
        manual_positive_pairs = []
        count = 0

    
        for i, qid in enumerate(list(self.train_questions.keys())):
            manual_positive_pairs.extend([(int(i), int(j)) for ind,j in enumerate(I[i]) if len(set(self.train_questions[qid]['parameter']) & set(self.manuals[list(self.manuals.keys())[j]]['parameters'])) != 0])

        question2positiveq = {}
        for i,j in question_positive_pairs:
            if i not in question2positiveq:
                question2positiveq[i] = []
            question2positiveq[i].append(rf"rq_{j}")
    

        question2positives = {}
        for i,j in manual_positive_pairs:
            if i not in question2positives:
                question2positives[i] = []
            question2positives[i].append(rf"rm_{j}")

        return question2positiveq, question2positives

    def gen_negative(self):
        faiss_index1 = faiss.IndexFlatIP(self.historical_questions_embeds.shape[1])
        faiss_index1.add(self.historical_questions_embeds)

        D, I = faiss_index1.search(self.train_questions_embeds, self.retrieve_topk)
        question_negative_pairs = []
        for qid in self.train_questions:
            question_negative_pairs.extend([(int(i), int(j)) for ind,j in enumerate(I[i]) if j != i and len(set(self.train_questions[qid]['parameter']) & set(self.historical_questions[list(self.historical_questions.keys())[j]]['parameter']))==0])

        faiss_index2 = faiss.IndexFlatIP(self.manuals_embeds.shape[1])
        faiss_index2.add(self.manuals_embeds)
        D, I = faiss_index2.search(self.train_questions_embeds, self.retrieve_topk)
        manual_negative_pairs = []
        
        for qid in self.train_questions:
            manual_negative_pairs.extend([(int(i), int(j)) for ind,j in enumerate(I[i]) if len(set(self.train_questions[qid]['parameter']) & set(self.manuals[list(self.manuals.keys())[j]]['parameters']))==0])

        question2negativeq = {}
        for i,j in question_negative_pairs:
            if i not in question2negativeq:
                question2negativeq[i] = []
            question2negativeq[i].append(rf"rq_{j}")

        question2negatives = {}
        for i,j in manual_negative_pairs:
            if i not in question2negatives:
                question2negatives[i] = []
            question2negatives[i].append(rf"rm_{j}")

        return question2negativeq, question2negatives
    
    def train_data(self):
        question2positiveq, question2positives  = self.gen_positive()
        question2negativeq, question2negatives = self.gen_negative()

        training_data = []

            #### qs
        for question_index in question2positives:
                
            for positive_sentence_index in question2positives[question_index]:
                if question_index not in question2negatives or question_index not in question2negativeq:
                        continue
                negative_list = question2negatives[question_index] + question2negativeq[question_index]
                if len(negative_list) == 0:
                    continue
                
                sampled_negative_index_list = random.sample(negative_list, self.nega_num)
                train_emb = self.train_questions_embeds[int(question_index)]

                positive_emb = self.manuals_embeds[int(positive_sentence_index.replace("rm_", ""))]
                nega_emb_list = []

                for neg_sentence_index in sampled_negative_index_list:
                    if 'rq_' in neg_sentence_index:
                        index = int(neg_sentence_index.replace("rq_", ''))
                        nega_emb_list.append(self.historical_questions_embeds[index])
                    else:
                        index = int(neg_sentence_index.replace("rm_", ''))
                        nega_emb_list.append(self.manuals_embeds[index])

                nega_emb_list = np.array(nega_emb_list)
                sampled_negative_index_list=  tuple(sampled_negative_index_list)
                item = [[query_emb], [positive_emb], [nega_emb_list], [positive_sentence_index], [sampled_negative_index_list]]
                training_data.append(item)


        #### qq
        for question_index in question2positiveq:
            for positive_question_index in question2positiveq[question_index]:
                if question_index not in question2negatives or question_index not in question2negativeq:
                    continue
                negative_list = question2negatives[question_index] + question2negativeq[question_index]
                if len(negative_list) < self.nega_num:
                    continue
            
                sampled_negative_index_list = random.sample(negative_list, self.nega_num)

                query_emb = self.train_questions_embeds[int(question_index)]
                positive_emb =self.historical_questions_embeds[int(positive_question_index.replace("rq_", ""))]

                nega_emb_list = []
                for neg_sentence_index in sampled_negative_index_list:
                    if 'rq_' in neg_sentence_index:
                        index = int(neg_sentence_index.replace("rq_", ''))
                        nega_emb_list.append(self.historical_questions_embeds[index])
                    else:
                        index = int(neg_sentence_index.replace("rm_", ''))
                        nega_emb_list.append(self.manuals_embeds[index])
                nega_emb_list = np.array(nega_emb_list)
                sampled_negative_index_list=  tuple(sampled_negative_index_list)
           
                item = [[query_emb], [positive_emb], [nega_emb_list], [positive_question_index], [sampled_negative_index_list]]
                training_data.append(item)
        
        training_data = AlignmentTrainDataset(training_data)
        self.train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

    def load_data(self):
        with open(rf'./data/dataset/augment_train/{self.dataset}_train_augment.json', "r") as f:
            self.train_questions = json.loads(f.read())
        with open(rf'./data/dataset/historical_questions/{self.dataset}_retrieval_data.json', "r") as f:
            self.historical_questions = json.loads(f.read())
        self.database = self.dataset.split("_")[0]
        with open(rf'./data/dataset/manuals/{self.database}_manuals_data.json', "r") as f:
            self.manuals = json.loads(f.read())

        self.train_questions_embeds = np.load(rf'./core/sbert_embeds/{self.dataset}_train_augment.npy', allow_pickle=True)
        self.historical_questions_embeds = np.load(rf'./core/sbert_embeds/{self.dataset}_retrieval_data.npy', allow_pickle=True)
        self.manuals_embeds = np.load(rf'./core/sbert_embeds/{self.database}_manuals_data.npy', allow_pickle=True)

    
    def train(self):
        self.load_data()
        self.train_data()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.question_alignment = QuestionAlignment()
        self.manual_alignment = ManualAlignment()

        self.opt_q = torch.optim.Adam(self.question_alignment.parameters(), lr=self.lr)
        self.opt_m = torch.optim.Adam(self.manual_alignment.parameters(), lr=self.lr)

        loss_func = InfoNCE(negative_mode='paired')

        for epoch in range(2*(self.epochs+1)):
            
            for index, batch in enumerate(self.train_loader):
                batch_query_emb, batch_positive_emb, batch_negatives_emb, batch_positive_id, batch_negative_ids = batch
                batch_positive_id = list(batch_positive_id)
                new_batch_negative_ids = []
                for i in range(len(batch_negative_ids[0])):
                    new_tmp = []
                    for j in range(self.nega_num):
                        new_tmp.append(batch_negative_ids[j][i])
                    new_batch_negative_ids.append(new_tmp)
                batch_negative_ids =new_batch_negative_ids

                batch_query_emb.to(device)
                batch_positive_emb.to(device)
                batch_negatives_emb.to(device)
                
                for bindex in range(len(batch_query_emb)):
                    query = torch.unsqueeze(batch_query_emb[bindex], 0)
                    positive = torch.unsqueeze(batch_positive_emb[bindex], 0)

                    negatives = torch.unsqueeze(batch_negatives_emb[bindex].reshape(self.nega_num, 768), 1)

                    positive_id = batch_positive_id[bindex]
                    negative_ids = batch_negative_ids[bindex]

                    n_rq = 0
                    n_rm = 0
                    if 'rq_' in positive_id:
                        
                        query_embedding = query
                        positive_embedding = self.question_alignment(query_embedding)
     
                        for nindex in range(len(negative_ids)):
                            negative_id = negative_ids[nindex]
                            if 'rq_' in negative_id:
                                negative_embedding = self.question_alignment(negatives[nindex])
                                n_rq += 1
                            else:
                                negative_embedding = self.manual_alignment(negatives[nindex])
                                n_rm += 1
                            if nindex == 0:
                                negative_embeddings = negative_embedding
                            else:
                                negative_embeddings = torch.cat((negative_embeddings, negative_embedding), 0)

                    else:
                        n_rq = 0
                        n_rm = 0
                        query_embedding = query
                    
                        positive_embedding = self.manual_alignment(positive) # [1,768]

                        negative_embeddings = []
                        for nindex in range(len(negative_ids)):
                            negative_id = negative_ids[nindex]
                            if 'rq_' in negative_id:
                                negative_embedding = self.question_alignment(negatives[nindex])
                                n_rq += 1
                            else:
                                negative_embedding = self.manual_alignment(negatives[nindex])
                                n_rm += 1
                            if nindex == 0:
                                negative_embeddings = negative_embedding
                            else:
                                negative_embeddings = torch.cat((negative_embeddings, negative_embedding), 0)
        
                    negative_embeddings = torch.unsqueeze(negative_embeddings, 0)
                    if bindex == 0:
                        batch_query_embedding = query_embedding
                        batch_positive_embedding = positive_embedding
                        batch_negative_embeddings = negative_embeddings
                    else:
                
                        batch_query_embedding = torch.cat((batch_query_embedding, query_embedding), 0)
                        batch_positive_embedding = torch.cat((batch_positive_embedding, positive_embedding), 0)
                        batch_negative_embeddings = torch.cat((batch_negative_embeddings, negative_embeddings), 0)


                try:     
                    loss = loss_func(batch_query_embedding, batch_positive_embedding, batch_negative_embeddings)
                    if epoch % 2 == 0:
                        self.opt_q.zero_grad()
                        loss.backward()
                        self.opt_q.step()
                    else:
                        self.opt_m.zero_grad()
                        loss.backward()
                        self.opt_m.step()

                except:
                    continue
            torch.save(self.opt_m, rf'./core/alignment_model/{self.database}_manual.pth')
            torch.save(self.opt_q, rf'./core/alignment_model/{self.database}_question.pth')
 


