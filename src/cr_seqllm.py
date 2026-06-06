import random
import pickle
import time
from re import L, S

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from collections import Counter
import numpy as np

from src.recsys_model import *
from src.cr_seqllm4rec import *
from utils import KMeans
from sentence_transformers import SentenceTransformer
from datetime import datetime

from tqdm import trange, tqdm

try:
    import habana_frameworks.torch.core as htcore
except:
    0
    

class llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        with open(f'./SeqRec/data_{args.inference_data}/text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)
        
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)

        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.all_embs = None
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.NDCG_20 = 0
        self.HIT_20 = 0
        self.NDCG_30 = 0
        self.HIT_30 = 0
        self.NDCG_40 = 0
        self.HIT_40 = 0
        self.NDCG_50 = 0
        self.HIT_50 = 0
        
        
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.num_user = 0
        self.yes = 0

        self.extract_embs_list = []
        self.group_outputs = []
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
            
        self.args.item_num = self.item_num
        self.llm = llm4rec(device=self.device, llm_model=args.llm, args = self.args)

        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
            nn.LayerNorm(self.llm.llm_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
        )
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)

        self.group_emb_proj = nn.Sequential(
                nn.Linear(self.recsys.hidden_units, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
        nn.init.xavier_normal_(self.group_emb_proj[0].weight)
        nn.init.xavier_normal_(self.group_emb_proj[3].weight)

        self.gate = nn.Embedding(
            num_embeddings=args.usernum+1,
            embedding_dim=1,
            padding_idx=args.usernum
        )
        nn.init.constant_(self.gate.weight, 0)

        self.cluster = KMeans(
            num_cluster=args.user_group_num,
            seed=42,
            hidden_size=self.recsys.hidden_units,
            gpu_id=0,
            device='cuda',
            niter=args.niter
        )
        
        self.users = 0.0
        self.NDCG = 0.0
        self.HT = 0.0

        self.group_centers = nn.Parameter(torch.Tensor(int(args.user_group_num), int(self.recsys.hidden_units)))
            
    def save_model(self, args, epoch2=None, best=False):
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'
        
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_'
        
        out_dir += f'{args.llm}_{epoch2}_'
        if args.train:
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            torch.save(self.group_emb_proj.state_dict(), out_dir + 'group_proj.pt')
            torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')
            torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
            torch.save(self.llm.res_adapter.state_dict(), out_dir+'res_adapter.pt')
            torch.save(self.group_centers.data, out_dir + 'group_centers.pt')

            if args.gated:
                torch.save(self.llm.gate_layer.state_dict(), out_dir + 'gate_layer.pt')

            if not args.token:
                if args.nn_parameter:
                    torch.save(self.llm.CLS.data, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.data, out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
            if args.token:
                torch.save(self.llm.llm_model.model.embed_tokens.state_dict(), out_dir + 'token.pt')
  
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None, best=False):
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'
        out_dir += f'{args.rec_pre_trained_data}_'

        out_dir += f'{args.llm}_{phase2_epoch}_'
        
        
        item_emb_proj = torch.load(out_dir + 'item_proj.pt', map_location = self.device)
        self.item_emb_proj.load_state_dict(item_emb_proj)
        del item_emb_proj

        group_emb_proj = torch.load(out_dir + 'group_proj.pt', map_location = self.device)
        self.group_emb_proj.load_state_dict(group_emb_proj)
        del group_emb_proj
        
        pred_user = torch.load(out_dir + 'pred_user.pt', map_location = self.device)
        self.llm.pred_user.load_state_dict(pred_user)
        del pred_user
        
        pred_item = torch.load(out_dir + 'pred_item.pt', map_location = self.device)
        self.llm.pred_item.load_state_dict(pred_item)
        del pred_item

        res_adpter = torch.load(out_dir + 'res_adapter.pt', map_location = self.device)
        self.llm.res_adapter.load_state_dict(res_adpter)
        del res_adpter

        group_centers = torch.load(out_dir + 'group_centers.pt', map_location = self.device)
        self.group_centers.data.copy_(group_centers)
        del group_centers

        if args.gated:
            gate_layer = torch.load(out_dir + 'gate_layer.pt', map_location = self.device)
            self.llm.gate_layer.load_state_dict(gate_layer)
            del gate_layer

        if not args.token:
            CLS = torch.load(out_dir + 'CLS.pt', map_location = self.device)
            self.llm.CLS.load_state_dict(CLS)
            del CLS
            
            CLS_item = torch.load(out_dir + 'CLS_item.pt', map_location = self.device)
            self.llm.CLS_item.load_state_dict(CLS_item)
            del CLS_item
        
        if args.token:
            token = torch.load(out_dir + 'token.pt', map_location = self.device)
            self.llm.llm_model.model.embed_tokens.load_state_dict(token)
            del token
            

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
        
    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        l = [datetime.utcfromtimestamp(int(self.text_name_dict['time'][i][user])/1000) for i in item]
        return [l_.strftime('%Y-%m-%d') for l_ in l]
    

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def get_item_emb(self, item_ids):
        with torch.no_grad():
            if self.args.nn_parameter:
                item_embs = self.recsys.model.item_emb[torch.LongTensor(item_ids).to(self.device)]
            else:
                item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
        
        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode=='generate_batch':
            self.generate_batch(data)
            print(self.args.save_dir, self.args.rec_pre_trained_data)
            print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.4f'
                    % (self.NDCG/self.users, self.HT/self.users, self.users))
            print('test (NDCG@20: %.4f, HR@20: %.4f), Num User: %.4f'
                    % (self.NDCG_20/self.users, self.HIT_20/self.users, self.users))
        if mode=='extract':
            self.extract_emb(data)

    def make_interact_text(self, interact_ids, interact_max_num, user):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        times = self.find_item_time(interact_ids, user)
        interact_text = []
        count = 1
        
            
        if interact_max_num =='all':
            times = self.find_item_time(interact_ids, user)
        else:
            times = self.find_item_time(interact_ids[-interact_max_num:], user)
        
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')

                count+=1
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                
                count+=1
            interact_ids = interact_ids[-interact_max_num:]
            
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids

    def make_group_interact_text(self, item_counts, interact_max_num='all', total_members=None):
        """
        item_counts: a Counter object containing {item_id: count}
        interact_max_num: the number of top popular items to extract
        total_members: the total number of members in the group, used to compute purchase ratios
        """
        if interact_max_num == 'all':
            top_items = item_counts.most_common()
        else:
            top_items = item_counts.most_common(interact_max_num)
        
        interact_ids = [item[0] for item in top_items]
        counts = [item[1] for item in top_items]

        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        
        interact_text = []
        count = 1

        for i, title in enumerate(interact_item_titles_):
            interact_text.append(
                f'Item No.{count}, Popularity: {counts[i]}, ' + title + '[HistoryEmb]'
            )
            count += 1
                
        interact_text = ', '.join(interact_text)

        return interact_text, interact_ids
    
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, task = 'ItemTask'):
        neg_item_id = []
        if candi_set == None:
            neg_item_id = []
            while len(neg_item_id)<99:
                t = np.random.randint(1, self.item_num+1)
                if not (t in interact_ids or t in neg_item_id):
                    neg_item_id.append(t)
        else:
            his = set(interact_ids)
            items = list(candi_set.difference(his))
            if len(items) >99:
                neg_item_id = random.sample(items, 99)
            else:
                while len(neg_item_id)<49:
                    t = np.random.randint(1, self.item_num+1)
                    if not (t in interact_ids or t in neg_item_id):
                        neg_item_id.append(t)
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        
        candidate_text = [f'The item title and item embedding are as follows: ' + target_item_title + "[HistoryEmb], then generate item representation token:[ItemOut]"]


        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(f'The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")

            candidate_ids.append(neg_candidate)
            
        return candidate_text, candidate_ids
    
    
    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, task = 'ItemTask'):
        neg_item_id = []
        neg_item_id = []
        while len(neg_item_id)<99:
            t = np.random.randint(1, self.item_num+1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        
        candidate_ids = candidate_ids + neg_item_id[:candidate_num - 1]
            
        return candidate_ids

    def pre_train_phase1(self, data, Group_ = True):
        '''
        Cluster all users into groups based on the full dataset.
        data: the global data for all users.
        '''
        print('cluster users into groups...')
        u, seq, pos, neg = data
        self.seq_all = seq

        log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
        log_emb = log_emb.detach().cpu().numpy()

        self.residuals = log_emb

        if Group_:
            self.cluster.train(log_emb)
            init_centers = self.cluster.centroids.to(self.device)
            self.group_centers.data.copy_(init_centers)

    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)
        optimizer.zero_grad()
        u, seq, pos, neg = data
        
        original_seq = seq.copy()
        
        mean_loss = 0
        
        text_input = [[], [], []]
        group_rep  = [[], [], []]
        interact_embs = [[], [], []]
        group_weight = []

        user_res = []
        candidates_pos = []
        candidate_embs_pos = []

        interact_max_num = self.args.interact_max_num

        # update the hard clustering labels
        self.update_hard_labels()
        
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            numpy_log = log_emb.cpu().numpy()
        for i in range(len(u)):
            user_idx = u[i] - 1
        
            temp = 1
            probs = self.compute_group_probabilities(log_emb[i].unsqueeze(0), temperature=temp).squeeze(0).cpu()
           
            sorted_indices = torch.argsort(probs, descending=True).numpy().tolist()
            
            top_groups = []
            cum_prob = 0.0
            
            for gid in sorted_indices:
                prob_val = probs[gid].item()
                top_groups.append(gid)
                cum_prob += prob_val
                if cum_prob >= self.args.group_weight_threshold or len(top_groups) >= 3:
                    break

            weights = probs[top_groups]
            weights = weights / weights.sum()

            padding_weights = np.zeros(3)
            padding_weights[:len(weights)] = weights.detach().numpy()

            group_weight.append(padding_weights)

            # ===== user-level stuff =====
            user_res.append(torch.tensor(self.residuals[user_idx]).unsqueeze(0))

            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(
                target_item_id, title_flag=True, description_flag=False)

            candidate_num = self.args.candidate_num
            candidate_text, candidate_ids = self.make_candidate_text(
                seq[i][seq[i]>0], candidate_num,
                target_item_id, target_item_title, task='RecTask')

            candidates_pos += candidate_text
            candidate_embs_pos.append(
                self.item_emb_proj(self.get_item_emb([candidate_ids])).squeeze(0)
            )

            # ===== build 3 levels =====
            for level in range(3):
                if level < len(top_groups):
                    gid = top_groups[level]
                    group_user_indices = np.where(self.labels == gid)[0]
                    group_seqs = self.seq_all[group_user_indices]

                    all_interacted_items = group_seqs.flatten()
                    all_interacted_items = all_interacted_items[all_interacted_items > 0]

                    item_counts = Counter(all_interacted_items)
                    num_members = len(group_user_indices)
                    interact_text, interact_ids = self.make_group_interact_text(
                        item_counts, interact_max_num, total_members=num_members)

                    input_text = ''
                
                    input_text += f'This is the group representation from recommendation models: [GroupRep]. To represent common interests of the group, the Top {interact_max_num} shared purchases made by most members are provided: '
                        
                    input_text += interact_text
                    
                    input_text +=". Based on this representative sequence and group representation, generate group representation token:[GroupOut]."
                    
                    text_input[level].append(input_text)

                    rep_group = self.group_emb_proj(
                        self.group_centers[gid]
                    )
                    group_rep[level].append(rep_group)
                    interact_embs[level].append(
                        self.item_emb_proj(self.get_item_emb(interact_ids))
                    )

                else:
                    # padding
                    text_input[level].append("")
                    group_rep[level].append(None)
                    interact_embs[level].append(None)
        
        candidate_embs = torch.cat(candidate_embs_pos)
        user_res = torch.cat(user_res)
        g = torch.sigmoid(self.gate(torch.from_numpy(u).to(self.args.device)))
        
        samples = {
            'text_input': text_input,        # [num_top][B]
            'group_rep': group_rep,          # [num_top][B][D]
            'interact': interact_embs,       # [num_top][B][T][D]
            'group_weight': torch.from_numpy(np.array(group_weight, dtype=np.float32)).to(self.device),
            'user_res': user_res,
            'log_emb': log_emb,
            'candidates_pos': candidates_pos,
            'candidate_embs': candidate_embs,
            'gate':g
        }
        
        # print('------state of llm------')
        loss, rec_loss, match_loss = self.llm(samples, mode=0)

        print("LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, rec_loss))
                            
        print("LLMRec model Matching loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, match_loss))
                    
    
        log_emb_F = log_emb
        group_centers = self.group_centers
        user_lambda = self.args.user_lambda
        center_lambda = self.args.center_lambda

        dist = torch.cdist(log_emb_F, group_centers)
        w = F.softmax(-dist,dim = -1)
        user_distance_loss = (w*dist).sum(dim = -1).mean()

        center_dist = torch.cdist(group_centers, group_centers)
        mask = ~torch.eye(group_centers.shape[0], dtype=torch.bool, device=self.device)
        center_dist = center_dist[mask]
        center_distance_loss = -center_dist.mean()

        print("LLMRec model user distance loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, user_distance_loss))
        print("LLMRec model center distance loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, center_distance_loss))

        
        loss += user_lambda * user_distance_loss + center_lambda * center_distance_loss

        loss.backward()
        if self.args.nn_parameter:
            htcore.mark_step()
        optimizer.step()
        if self.args.nn_parameter:
            htcore.mark_step()
           

    def split_into_batches(self,itemnum, m):
        numbers = list(range(1, itemnum+1))
        
        batches = [numbers[i:i + m] for i in range(0, itemnum, m)]
        
        return batches

    def extract_group_rep(self):
        '''
        inference phrase, extract group representation
        '''
        interact_max_num = self.args.interact_max_num

        with torch.no_grad():
            self.update_hard_labels()

            for i in tqdm(range(len(self.group_centers))):
                # i-th group
                text_input = []
                interact_embs = []
                group_rep = []

                group_user_indices = np.where(self.labels == i)[0]

                group_seqs = self.seq_all[group_user_indices]

                all_interacted_items = group_seqs.flatten()
                all_interacted_items = all_interacted_items[all_interacted_items > 0]
                item_counts = Counter(all_interacted_items)

                interact_text, interact_ids = self.make_group_interact_text(item_counts, interact_max_num)

                #no user
                input_text = ''
                
                input_text += f'This is the group representation from recommendation models: [GroupRep]. To represent common interests of the group, the Top {interact_max_num} shared purchases made by most members are provided: '
                    
                input_text += interact_text
                
                input_text +=". Based on this representative sequence and group representation, generate group representation token:[GroupOut]."

                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                group_rep.append(self.group_emb_proj(torch.tensor(self.group_centers[i], device=self.device)))

                max_input_length = 1024
                
                llm_tokens = self.llm.llm_tokenizer(
                    text_input,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=max_input_length,
                ).to(self.device)
                
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                
                inputs_embeds = self.llm.replace_out_token_all(llm_tokens, 
                                                    inputs_embeds, 
                                                    token = ['[HistoryEmb]',
                                                             '[GroupRep]',
                                                             '[GroupOut]'], 
                                                    embs= { '[HistoryEmb]':interact_embs, 
                                                            '[GroupRep]':group_rep})
                
                with torch.amp.autocast('cuda'):
                    outputs = self.llm.llm_model.forward(
                        inputs_embeds=inputs_embeds,
                        output_hidden_states=True
                    )
                    
                    indx = self.llm.get_embeddings(llm_tokens, '[GroupOut]')
                    self.group_outputs.append(torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))]).cpu())
            print("Group representation extraction done.")
    
    def generate_batch(self,data):
        if self.all_embs == None:
            batch_ = 128
            if self.args.llm =='llama':
                batch_ = 64
            if self.args.inference_data == 'Electronics' or self.args.inference_data == 'Books':
                batch_ = 64
                if self.args.llm =='llama':
                    batch_ = 32
            batches = self.split_into_batches(self.item_num, batch_)#128
            self.all_embs = []
            max_input_length = 1024
            for bat in tqdm(batches):
                candidate_text = []
                candidate_ids = []
                candidate_embs = []
                for neg_candidate in bat:
                    candidate_text.append('The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")
                    
                    candidate_ids.append(neg_candidate)
                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(self.device)
                    candidate_embs.append(self.item_emb_proj((self.get_item_emb(candidate_ids))))

                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':candidate_embs[0]})
                    
                    with torch.amp.autocast('cuda'):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds,
                            output_hidden_states=True
                        )
                        
                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                        item_outputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                        
                        item_outputs = self.llm.pred_item(item_outputs)

                    self.all_embs.append(item_outputs)

                    del candi_outputs
                    del item_outputs
                    torch.cuda.empty_cache()
                         
            self.all_embs = torch.cat(self.all_embs)

            # record the start time of user embedding
            self.user_start = time.time()

            if not self.args.train:
                self.extract_group_rep()

            
        u, seq, pos, neg, rank, candi_set, files = data
        original_seq = seq.copy()
        
        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
            numpy_log = log_emb.cpu().numpy()

            user_group_outputs = []

            for i in range(len(u)):

                temp = 1
                probs = self.compute_group_probabilities(log_emb[i].unsqueeze(0), temperature=temp).squeeze(0).cpu()
            
                sorted_indices = torch.argsort(probs, descending=True).numpy().tolist()
                
                top_groups = []
                cum_prob = 0.0
                
                for gid in sorted_indices:
                    prob_val = probs[gid].item()
                    top_groups.append(gid)
                    cum_prob += prob_val
                    if cum_prob >= self.args.group_weight_threshold or len(top_groups) >= 3:
                        break

                weights = probs[top_groups]
                weights = weights / weights.sum()

                weights = weights.view(-1, 1).to(self.device)  # [k,1]

                # ---- fetch cached group outputs ----
                group_vecs = torch.cat(
                    [self.group_outputs[gid].to(self.device) for gid in top_groups],
                    dim=0
                )                                       # [k, D]

                group_vecs = self.llm.pred_user(group_vecs)  # [k, D]

                outputs = (weights * group_vecs).sum(dim=0, keepdim=True)  # [1, D]

                user_group_outputs.append(outputs)

                candidate_embs = []
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                # interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i])
                
                candidate_num = 100
                candidate_ids = self.make_candidate(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, candi_set)
                
                candidate.append(candidate_ids)

            with torch.amp.autocast('cuda'):

                group_outputs = torch.cat(user_group_outputs, dim=0)

                user_idx = u - 1
                residuals_array = self.residuals[user_idx]
                residuals_tensor = torch.from_numpy(residuals_array).to(self.device)
                individual_outputs = self.llm.res_adapter(residuals_tensor)

                if self.args.gated:
                    g = torch.sigmoid(self.gate(torch.from_numpy(u).to(self.args.device)))
                    user_outputs = (1 - g) * group_outputs + g * individual_outputs
                else:
                    user_outputs = group_outputs + individual_outputs
                

                for i in range(len(candidate)):
                    
                    item_outputs = self.all_embs[np.array(candidate[i])-1]
                    
                    logits= torch.mm(item_outputs, user_outputs[i].unsqueeze(0).T).squeeze(-1)
                
                    logits = -1*logits
                    
                    rank = logits.argsort().argsort()[0].item()
                    
                    if rank < 10:
                        self.NDCG += 1 / np.log2(rank + 2)
                        self.HT += 1
                    if rank < 20:
                        self.NDCG_20 += 1 / np.log2(rank + 2)
                        self.HIT_20 += 1
                    if rank < 30:
                        self.NDCG_30 += 1 / np.log2(rank + 2)
                        self.HIT_30 += 1
                    if rank < 40:
                        self.NDCG_40 += 1 / np.log2(rank + 2)
                        self.HIT_40 += 1
                    if rank < 50:
                        self.NDCG_50 += 1 / np.log2(rank + 2)
                        self.HIT_50 += 1
                    self.users +=1
        return self.NDCG
                
    def extract_emb(self,data):    
        u, seq, pos, neg, original_seq, rank, files = data
            
        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):

                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i])

                input_text = ''
                    

                input_text += 'This user has made a series of purchases in the following order: '
                    
                input_text += interact_text
                

                input_text +=". Based on this sequence of purchases, generate user representation token:[UserOut]"
                
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                

            max_input_length = 1024
            
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)
            
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':interact_embs})

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,

                    output_hidden_states=True
                )
                
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)
                
                self.extract_embs_list.append(user_outputs.detach().cpu())
                
        return 0

    def compute_group_probabilities(self, log_emb, temperature=1.0):
        """
        """
        centers = self.group_centers  # [K, D]
        
        user_norm = (log_emb ** 2).sum(dim=1, keepdim=True)       # [B, 1]
        center_norm = (centers ** 2).sum(dim=1, keepdim=True).T   # [1, K]
        dot_product = torch.matmul(log_emb, centers.T)            # [B, K]
        
        distances = user_norm + center_norm - 2 * dot_product     # [B, K]
        distances = torch.clamp(distances, min=0.0)

        logits = -distances / temperature
        
        probs = torch.softmax(logits, dim=1)                      # [B, K]
        
        return probs

    def update_hard_labels(self):
        with torch.no_grad():
            all_log_emb = torch.from_numpy(self.residuals).to(self.device)
            dists = torch.cdist(all_log_emb, self.group_centers.data, p=2)
            self.labels = torch.argmin(dists, dim=1).cpu().numpy()