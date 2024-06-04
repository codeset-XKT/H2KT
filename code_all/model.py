import torch
import torch.nn as nn
from Embedding_module.Gen_graph import load_Graph
from Embedding_module.HeteroRGCN import HeteroRGCN
from Update_module.LSTM import Evolution
from Update_module.TransformerLayer import TransformerLayer
from Predict_module.Predict import Predict
import dgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class H2KT(nn.Module):
    def __init__(self, args):
        super(H2KT, self).__init__()
        self.args = args
        self.embed_dim = args.embed_dim
        if args.data_set == 'JunYi':
            self.G, self.ques2skill, self.ques2diff, self.ques2area, self.pre2post, self.post2pre = load_Graph(
                args.data_set, self.embed_dim)
        else:
            self.G, self.ques2skill, self.ques2diff, self.ques2area = load_Graph(args.data_set, self.embed_dim)
        self.HeteroRGCN = HeteroRGCN(self.G, self.embed_dim, self.embed_dim, self.embed_dim, self.args.num_layer)
        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(p=0.2)
        self.fusion = Fusion_Module(self.embed_dim, args.device)
        self.lstm_m = Evolution(2 * self.embed_dim, self.embed_dim)
        self.lstm_a = Evolution(self.embed_dim, self.embed_dim)
        self.predict = Predict(self.embed_dim, 1)
        self.embedding_q = nn.Embedding(args.ques_num, self.embed_dim)

        self.n_heads = 4
        self.search = TransformerLayer(self.embed_dim, self.embed_dim // self.n_heads, self.embed_dim, self.n_heads,
                                       0.2, kq_same=1)
        self.linear = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.concat = nn.Linear(2, self.embed_dim)
        self.mlp = nn.Linear(self.embed_dim, self.embed_dim)
        self.m = nn.Linear(1, self.embed_dim)

    def get_subgraph_embedding(self, q, next_q, embedding):
        bsz = q.size(0)
        seqlen = q.size(1)

        # Get skill embedding
        skill = nn.functional.embedding(q, self.ques2skill)
        skill = skill.view(bsz * seqlen, -1)
        skill = nn.functional.embedding_bag(skill, embedding['skill'], padding_idx=len(embedding['skill']) - 1)
        skill = skill.view(bsz, seqlen, -1)

        next_skill = nn.functional.embedding(next_q, self.ques2skill)
        next_skill = next_skill.view(bsz * seqlen, -1)
        next_skill = nn.functional.embedding_bag(next_skill, embedding['skill'], padding_idx=len(embedding['skill']) - 1)
        next_skill = next_skill.view(bsz, seqlen, -1)

        if self.args.data_set == 'JunYi':
            # Get the pre embedding
            pre = nn.functional.embedding(q, self.pre2post)
            pre = pre.view(bsz * seqlen, -1)
            pre = nn.functional.embedding_bag(pre, embedding['ques'], padding_idx=len(embedding['ques']) - 1)
            pre = pre.view(bsz, seqlen, -1)

            next_pre = nn.functional.embedding(next_q, self.pre2post)
            next_pre = next_pre.view(bsz * seqlen, -1)
            next_pre = nn.functional.embedding_bag(next_pre, embedding['ques'], padding_idx=len(embedding['ques']) - 1)
            next_pre = next_pre.view(bsz, seqlen, -1)

            # Get the post embedding
            post = nn.functional.embedding(q, self.post2pre)
            post = post.view(bsz * seqlen, -1)
            post = nn.functional.embedding_bag(post, embedding['ques'], padding_idx=len(embedding['ques']) - 1)
            post = post.view(bsz, seqlen, -1)

            next_post = nn.functional.embedding(next_q, self.post2pre)
            next_post = next_post.view(bsz * seqlen, -1)
            next_post = nn.functional.embedding_bag(next_post, embedding['ques'],
                                                    padding_idx=len(embedding['ques']) - 1)
            next_post = next_post.view(bsz, seqlen, -1)

        # Get the diff embedding
        diff = nn.functional.embedding(q, self.ques2diff).view(q.size(0), -1)
        diff = nn.functional.embedding(diff, embedding['diff'])
        next_diff = nn.functional.embedding(next_q, self.ques2diff).view(q.size(0), -1)
        next_diff = nn.functional.embedding(next_diff, embedding['diff'])

        # Get the area embedding
        area = nn.functional.embedding(q, self.ques2area).view(q.size(0), -1)
        area = nn.functional.embedding(area, embedding['area'])
        next_area = nn.functional.embedding(next_q, self.ques2area).view(q.size(0), -1)
        next_area = nn.functional.embedding(next_area, embedding['area'])

        # Get the question embedding
        q = nn.functional.embedding(q, embedding['ques'])
        next_q = nn.functional.embedding(next_q, embedding['ques'])

        if self.args.data_set == 'JunYi':
            q = (q + skill + diff + area + pre + post) / 6.0
            next_q = (next_q + next_skill + next_diff + next_area + next_pre + next_post) / 6.0
        else:
            q = (q + skill + diff + area) / 4.0
            next_q = (next_q + next_skill + next_diff + next_area) / 4.0
            next_q = (next_q + next_skill + next_diff + next_area) / 4.0

        q = self.q_linear(q)
        next_q = self.q_linear(next_q)

        return q, next_q

    def forward(self, q, a, next_q, timestap, attempt, answertime):
        bsz = q.size(0)
        seqlen = q.size(1)

        if self.args.subgraph_embed:
            q, next_q = self.get_subgraph_embedding(q, next_q, self.HeteroRGCN(self.G))
        else:
            embedding = self.HeteroRGCN(self.G)
            q = nn.functional.embedding(q, embedding['ques'])
            next_q = nn.functional.embedding(next_q, embedding['ques'])

        # Modeling students’ knowledge
        x = self.fusion(q, a)  # [bsz, sql_len, 2*embed_dim]
        hidden_master = self.lstm_m(x)  # [bsz, sql_len, embed_dim]

        # Modeling students’ question-solving ability
        attempt = attempt.view(bsz, seqlen, -1)
        answertime = answertime.view(bsz, seqlen, -1)

        ability_feature = torch.concat((attempt, answertime), dim=-1)  # [bsz, seq_len, 2]
        ability_feature = torch.relu(self.concat(ability_feature))
        hidden_ability = self.lstm_a(ability_feature)  # [bsz, seq_len, embed_dim]

        # Modeling students’ memory and forgetting factors
        hidden = torch.concat((hidden_master, hidden_ability), dim=-1)  # [bsz, seq_len, 2*embed_dim]
        hidden = self.linear(hidden)

        hidden = self.search(mask=1, query=hidden, key=hidden, values=hidden)  # [bsz,seq_len,embed_dim]
        timestap = nn.functional.normalize(timestap, p=2, dim=-1)
        timestap = timestap.view(bsz, seqlen, -1)
        hidden = hidden * torch.sigmoid(self.m(timestap))

        out = torch.concat((hidden, next_q), dim=-1)
        out = self.predict(out)
        return out


class Fusion_Module(nn.Module):
    def __init__(self, emb_dim, device):
        super(Fusion_Module, self).__init__()
        self.transform_matrix = torch.zeros(2, emb_dim * 2).to(device)
        self.transform_matrix[0][emb_dim:] = 1.0
        self.transform_matrix[1][:emb_dim] = 1.0

    def forward(self, ques_emb, pad_answer):
        ques_emb = torch.cat((ques_emb, ques_emb), -1)
        answer_emb = nn.functional.embedding(pad_answer, self.transform_matrix)
        input_emb = ques_emb * answer_emb
        return input_emb
