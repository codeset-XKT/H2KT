import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import os


# Get multi-hot encoding of questions to skills
def load_multihot_problem_to_skill(args):
    path = args.data_path + '\\' + args.data_set + '\graph\ques_skill.csv'
    problem_ids = []
    skill_ids = []
    with open(path) as r:
        data = r.read().split('\n')[1:]
        dict = {}
        for i in range(len(data)):
            edge = data[i].split(',')
            if len(edge) == 2:
                if edge[0] in dict:
                    dict[edge[0]] = dict[edge[0]] + ',' + edge[1]
                else:
                    dict[edge[0]] = edge[1]
                problem_ids.append(edge[0])
                skill_ids.append(edge[1])
    problem_ids = list(map(lambda x: int(x), problem_ids))
    skill_ids = list(map(lambda x: int(x), skill_ids))
    max_problem_id = max(list(set(problem_ids)))
    print("Number of questions", max_problem_id + 1)
    max_skill_id = max(list(set(skill_ids)))
    print("Number of skills", max_skill_id + 1)

    emb = [[] for i in range(max_problem_id + 2)]
    for key in dict:
        skills = dict[key].split(",")
        arr = [0 for i in range(max_skill_id + 1)]
        for item in skills:
            arr[int(item)] = 1
        emb[int(key)] = arr
    emb[-1] = [0 for i in range(max_skill_id + 1)]

    return dict, torch.tensor(emb).to(args.device)


def load_data(args):
    filePath_dict, dataList_dict, dataSet_dict, dataLoader_dict = dict(), dict(), dict(), dict()
    shuffle = {'train': True, 'test': False}
    global DEVICE
    DEVICE = torch.device(args.device)
    for train_or_test in ['train', 'test']:
        filePath_dict[train_or_test] = os.path.join(args.data_path, args.data_set, "train_test",
                                                    train_or_test + '_%s.txt' % args.input)
        dataList_dict[train_or_test] = file_to_list(filePath_dict[train_or_test], args.min_seq_len, args.max_seq_len,
                                                    args.data_set)
        dataSet_dict[train_or_test] = KTDataset(dataList_dict[train_or_test][0], dataList_dict[train_or_test][1],
                                                dataList_dict[train_or_test][2], dataList_dict[train_or_test][3],
                                                dataList_dict[train_or_test][4], dataList_dict[train_or_test][5])
        dataLoader_dict[train_or_test] = DataLoader(dataSet_dict[train_or_test], batch_size=args.batch_size,
                                                    collate_fn=collate_fn, shuffle=shuffle[train_or_test])

    print('load data done!')
    return dataLoader_dict


def file_to_list(filename, min_seq_len, max_seq_len, data_set, truncate=False):
    def split_func(_seq_len):
        _split_list = []
        while _seq_len > 0:
            if _seq_len >= max_seq_len:
                _split_list.append(max_seq_len)
                _seq_len -= max_seq_len
            elif _seq_len >= min_seq_len:
                _split_list.append(_seq_len)
                _seq_len -= _seq_len
            else:
                _seq_len -= min_seq_len
        return len(_split_list), _split_list

    seq_lens, ques_ids, timestaps, attempts, answertimes, answers = [], [], [], [], [], []
    k_split = -1
    with open(filename) as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if i % 6 == 0:
            seq_len = int(line)
            if seq_len < min_seq_len:
                i += 6
                continue
            else:
                k_split, split_list = split_func(seq_len)
                if truncate:
                    k_split = 1
                    seq_lens.append(split_list[0])
                else:
                    seq_lens += split_list
        else:
            line = line.split(',')
            if i % 6 == 1:
                array = [int(eval(e)) for e in line]
                for j in range(k_split):
                    ques_ids.append(array[max_seq_len * j: max_seq_len * (j + 1)])
            elif i % 6 == 2:
                array = [float(eval(e)) for e in line]
                for j in range(k_split):
                    timestaps.append(array[max_seq_len * j: max_seq_len * (j + 1)])
            elif i % 6 == 3:
                array = [float(eval(e)) for e in line]
                for j in range(k_split):
                    attempts.append(array[max_seq_len * j: max_seq_len * (j + 1)])
            elif i % 6 == 4:
                array = [float(eval(e)) for e in line]
                for j in range(k_split):
                    answertimes.append(array[max_seq_len * j: max_seq_len * (j + 1)])
            elif i % 6 == 5:
                array = [int(eval(e)) for e in line]
                for j in range(k_split):
                    answers.append(array[max_seq_len * j: max_seq_len * (j + 1)])

        i += 1
    # for integrity, check the lengths
    assert len(seq_lens) == len(ques_ids) == len(timestaps) == len(attempts) == len(answertimes) == len(answers)
    return seq_lens, ques_ids, timestaps, attempts, answertimes, answers
    # return seq_lens, ques_ids, answers


class KTDataset(Dataset):
    def __init__(self, seq_lens, ques_ids, timestaps, attempts, answertimes, answers):
        # def __init__(self, seq_lens, ques_ids, answers):
        self.seq_lens = seq_lens
        self.ques_ids = ques_ids
        self.timestaps = timestaps
        self.attempts = attempts
        self.answertimes = answertimes
        self.answers = answers

    def __len__(self):
        return len(self.seq_lens)

    def __getitem__(self, item):
        seq_len = self.seq_lens[item]
        ques_id = self.ques_ids[item]
        timestamp = self.timestaps[item]
        attempt = self.attempts[item]
        answertime = self.answertimes[item]
        answer = self.answers[item]

        sample_len = torch.tensor([seq_len - 1], dtype=torch.long)
        sample_exercise = torch.tensor(ques_id[:-1], dtype=torch.long)
        sample_answer = torch.tensor(answer[:-1], dtype=torch.long)
        sample_next_exercise = torch.tensor(ques_id[1:], dtype=torch.long)
        sample_next_answer = torch.tensor(answer[1:], dtype=torch.float)
        sample_timestap = torch.tensor(timestamp[:-1], dtype=torch.float)
        sample_attempt = torch.tensor(attempt[:-1], dtype=torch.float)
        sample_answertime = torch.tensor(answertime[:-1], dtype=torch.float)
        return (sample_len, sample_exercise, sample_answer, sample_next_exercise,
                sample_next_answer, sample_timestap, sample_attempt, sample_answertime)


def collate_fn(batch):
    # Sort the batch in the descending order
    batch = sorted(batch, key=lambda x: x[0], reverse=True)

    seq_lens = torch.cat([x[0] for x in batch])
    questions = pad_sequence([x[1] for x in batch], batch_first=True)
    answers = pad_sequence([x[2] for x in batch], batch_first=True)
    next_questions = pad_sequence([x[3] for x in batch], batch_first=True)
    next_answers = pad_sequence([x[4] for x in batch], batch_first=True)
    timestaps = pad_sequence([x[5] for x in batch], batch_first=True)
    attempts = pad_sequence([x[6] for x in batch], batch_first=True)
    answertimes = pad_sequence([x[7] for x in batch], batch_first=True)

    return [i.to(DEVICE) for i in [seq_lens, questions, answers, next_questions, next_answers, timestaps, attempts, answertimes]]

