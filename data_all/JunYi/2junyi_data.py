from junyi_utils import *
import os
import pandas as pd
import numpy as np
import time


class DataProcess:
    def __init__(self):
        self.data_path = dataPath
        self.save_folder = saveFolder
        self.question_id_dict = None
        self.skill_id_dict = None
        self.area_id_dict = None
        self.train_user_id_dict = None
        self.quesID2skillIDs_dict = {}

        self.logger = Logger()

    def process_csv(self, isRemoveEmptySkill=True, min_inter_num=10):
        print("### processing data ###")
        # read original csv file
        df = pd.read_csv(self.data_path, encoding="ISO-8859-1")
        self.logger.count("originally", df)

        # 1. remove duplicated rows by 'time_done'
        df.drop_duplicates(subset=['user_id', 'exercise', 'time_done'], keep='first', inplace=True)
        self.logger.count('After removing duplicated rows', df)

        # 2. sort records by 'time_done' in ascending order
        df.sort_values(by=['time_done'], ascending=True, inplace=True)

        # 3. remove records without skill or fill empty skill
        if isRemoveEmptySkill:
            df.drop(df[df['topic'].isna()].index, inplace=True)
            self.logger.count('After removing empty skill', df)
        else:
            df['topic'].fillna(value='UNK', inplace=True)

        # 6. delete the users whose interaction number is less than min_inter_num
        users = df.groupby(['user_id'], as_index=True)
        delete_users = []
        for u in users:
            if len(u[1]) < min_inter_num:
                delete_users.append(u[0])
        print('deleted user number based min-inters %d' % len(delete_users))
        df = df[~df['user_id'].isin(delete_users)]
        self.logger.count('After deleting some users', df)

        # 7. select 5000 users
        select_users = np.random.choice(np.array(df['user_id'].unique()), size=5000, replace=False)
        df = df[df['user_id'].isin(select_users)]
        self.logger.count('After selecting users', df)

        return df

    def get_train_test_df(self, df, train_user_ratio=0.8):
        print("\n### splitting train and test df ###")
        # get train df & test df
        all_users = list(df['user_id'].unique())
        num_train_user = int(len(all_users) * train_user_ratio)

        train_users = list(np.random.choice(all_users, size=num_train_user, replace=False))
        test_users = list(set(all_users) - set(train_users))
        open(os.path.join(self.save_folder, 'train_test', 'train_users.txt'), 'w').write(str(train_users))
        open(os.path.join(self.save_folder, 'train_test', 'test_users.txt'), 'w').write(str(test_users))
        train_df = df[df['user_id'].isin(train_users)]
        test_df = df[df['user_id'].isin(test_users)]

        # 3. remove the questions from test df, which do not exist in train df
        train_questions = list(train_df['exercise'].unique())
        test_df = test_df[test_df['exercise'].isin(train_questions)]

        # save train df & test df
        train_df.to_csv(os.path.join(self.save_folder, 'train_test', 'train_df.csv'))
        test_df.to_csv(os.path.join(self.save_folder, 'train_test', 'test_df.csv'))

        return train_df, test_df

    def encode_entity(self, train_df, test_df):
        print("\n### encoding entities ###")
        df = pd.concat([train_df, test_df], ignore_index=True)

        # encode questions
        problems = df['exercise'].unique()
        self.question_id_dict = dict(zip(problems, range(len(problems))))
        save_dict(self.question_id_dict, os.path.join(self.save_folder, "encode", "question_id_dict.txt"))
        print('question number: %d' % len(problems))

        # encode skills
        skills = df['topic'].unique()
        skill_set = set(skills)

        index, self.skill_id_dict = 0, dict()
        for skill in skill_set:
            self.skill_id_dict[skill] = index
            index += 1

        save_dict(self.skill_id_dict, os.path.join(self.save_folder, "encode", "skill_id_dict.txt"))
        print('skill number %d' % len(skills))

        # encode areas
        areas = df['area'].unique()
        self.area_id_dict = dict(zip(areas, range(len(areas))))
        save_dict(self.area_id_dict, os.path.join(self.save_folder, "encode", "area_id_dict.txt"))
        print('area number %d' % len(areas))

        # encode train_users
        train_users = train_df['user_id'].unique()
        self.train_user_id_dict = dict(zip(train_users, range(len(train_users))))
        save_dict(self.train_user_id_dict, os.path.join(self.save_folder, "encode", "train_user_id_dict.txt"))
        print('train_user number: %d' % len(train_users))

    def generate_user_sequence(self, df, seq_file):
        # generate user interaction sequence
        ui_df = df.groupby(['user_id'], as_index=True)

        user_inters = []
        for ui in ui_df:
            tmp_user, tmp_inter = ui[0], ui[1]
            tmp_inter.sort_values(by=['time_done'], ascending=True, inplace=True)
            tmp_questions = list(tmp_inter['exercise'])
            tmp_skills = list(tmp_inter['topic'])
            tmp_timestap = list(tmp_inter['time_done'])
            time_attempt = list(tmp_inter['count_attempts'])
            time_answertime = list(tmp_inter['time_taken'])
            tmp_ans = list(tmp_inter['correct'])
            user_inters.append(
                [[len(tmp_inter)], tmp_skills, tmp_questions, tmp_timestap, time_attempt, time_answertime, tmp_ans])
        write_list(os.path.join(self.save_folder, "train_test", seq_file), user_inters)

    def encode_user_sequence(self, train_or_test):
        with open(os.path.join(self.save_folder, "train_test", '%s_data.txt' % train_or_test), 'r') as f:
            lines = f.readlines()

        index = 0
        seqLen_list, skills_list, questions_list, timestap_list, attempt_list, answertime_list, answers_list = [], [], [], [], [], [], []
        while index < len(lines):
            tmp_skills = eval(lines[index + 1])
            tmp_skills = [self.skill_id_dict[str(ele)] for ele in tmp_skills]
            tmp_pro = eval(lines[index + 2])
            tmp_pro = [self.question_id_dict[ele] for ele in tmp_pro]
            tem_timestap = eval(lines[index + 3])
            time_attempt = eval(lines[index + 4])
            time_answertime = eval(lines[index + 5])
            tmp_ans = eval(lines[index + 6])
            real_len = len(tmp_pro)

            seqLen_list.append(real_len)
            skills_list.append(tmp_skills)
            questions_list.append(tmp_pro)
            timestap_list.append(tem_timestap)
            attempt_list.append(time_attempt)
            answertime_list.append(time_answertime)
            answers_list.append(tmp_ans)

            index += 7

        with open(os.path.join(self.save_folder, "train_test", "%s_all_feature.txt" % train_or_test), 'w') as w:
            for user in range(len(seqLen_list)):
                w.write('%d\n' % seqLen_list[user])
                w.write('%s\n' % ','.join([str(i) for i in questions_list[user]]))
                w.write('%s\n' % ','.join([str(i) for i in timestap_list[user]]))
                w.write('%s\n' % ','.join([str(i) for i in attempt_list[user]]))
                w.write('%s\n' % ','.join([str(i) for i in answertime_list[user]]))
                w.write('%s\n' % ','.join([str(i) for i in answers_list[user]]))

        # generate input data using template_id
        # get_train_test_template(train_or_test)

    def get_ques_skill_mat(self):
        df = pd.read_csv(os.path.join(self.save_folder, "graph", "ques_skill.csv"))
        num_ques, num_skill = df['ques'].max() + 1, df['skill'].max() + 1
        ques_skill_mat = np.zeros(shape=(num_ques, num_skill), dtype=np.int64)
        for index, row in df.iterrows():
            quesID, skillID = int(row['ques']), int(row['skill'])
            ques_skill_mat[quesID][skillID] = 1
        np.save(os.path.join(self.save_folder, "graph", "ques_skill_mat.npy"), ques_skill_mat)

    def build_ques_interaction_graph(self, train_df, test_df):
        """
        build ques_skill interaction graph
        build ques_area interaction graph
        """
        print("building question interaction graph")
        df = pd.concat([train_df, test_df], ignore_index=True)

        ques_skill_set, ques_area_set, ques_type_set = set(), set(), set()
        for ques in self.question_id_dict.keys():
            quesID = self.question_id_dict[ques]
            tmp_df = df[df['exercise'] == ques]
            tmp_df_0 = tmp_df.iloc[0]

            # build ques-skill graph
            if quesID not in self.quesID2skillIDs_dict.keys():
                self.quesID2skillIDs_dict[quesID] = set()
            skillID = self.skill_id_dict[tmp_df_0['topic']]
            ques_skill_set.add((quesID, skillID))
            self.quesID2skillIDs_dict[quesID].add(skillID)

            # build ques-area graph
            tmp_area = tmp_df_0['area']
            ques_area_set.add((quesID, self.area_id_dict[tmp_area]))

        save_graph(ques_skill_set, os.path.join(self.save_folder, 'graph', 'ques_skill.csv'), ['ques', 'skill'])
        save_graph(ques_area_set, os.path.join(self.save_folder, 'graph', 'ques_area.csv'), ['ques', 'area'])

        # save ques_skill matrix
        self.get_ques_skill_mat()

    def get_ques_attribute(self, train_df):
        print("\n### getting question attributes ###")
        # calculate question difficulty using train records
        quesID2diffValue_dict = get_quesDiff(train_df, self.question_id_dict)
        save_dict(quesID2diffValue_dict, os.path.join(self.save_folder, "attribute", "quesID2diffValue_dict.txt"))

        # calculate average elapsed_time
        quesID2AvgMsRsp_dict = get_quesAvgMsRsp(train_df, self.question_id_dict)
        save_dict(quesID2AvgMsRsp_dict, os.path.join(self.save_folder, "attribute", "quesID2AvgMsRsp_dict.txt"))

        # categorize difficulty into 10 discrete levels
        quesID2diffLevel_df = pd.DataFrame.from_dict(quesID2diffValue_dict, orient='index', columns=['diff'])
        quesID2diffLevel_df.index.names = ['ques']
        quesID2diffLevel_df['diff'] = quesID2diffLevel_df['diff'].apply(lambda x: int(x * 10))
        quesID2diffLevel_df.to_csv(os.path.join(self.save_folder, "graph", "ques_diff.csv"))

        # calculate question discrimination (refer to AKTHE)
        user_score_list = []
        for user in self.train_user_id_dict.keys():
            tmp_df = train_df[train_df['user_id'] == user]
            score = tmp_df['correct'].sum()
            user_score_list.append((user, score))
        user_score_list.sort(key=lambda x: x[1], reverse=True)

        ratio, num_user = 0.5, len(user_score_list)
        top_users, _ = zip(*user_score_list[:int(ratio * num_user)])
        btm_users, _ = zip(*user_score_list[-int(ratio * num_user):])
        top_df = train_df[train_df['user_id'].isin(top_users)]
        btm_df = train_df[train_df['user_id'].isin(btm_users)]
        quesID2topDiff = get_quesDiff(top_df, self.question_id_dict)
        quesID2btmDiff = get_quesDiff(btm_df, self.question_id_dict)

        quesID2discValue_dict = dict()
        for quesID in quesID2diffValue_dict.keys():
            disc = quesID2topDiff[quesID] - quesID2btmDiff[quesID]
            quesID2discValue_dict[quesID] = disc
        # quesID_disc_list = sorted(quesID2discValue_dict.items(), key=lambda x: x[1], reverse=True)
        # print(quesID_disc_list)
        save_dict(quesID2discValue_dict, os.path.join(self.save_folder, "attribute", "quesID2discValue_dict.txt"))

        # categorize discrimination into 4 discrete levels
        with open(os.path.join(self.save_folder, "graph", "ques_disc.csv"), 'w', encoding='utf-8') as writer:
            writer.write("ques,disc\n")
            for quesID, discValue in quesID2discValue_dict.items():
                if discValue < 0.2:
                    discLevel = 0
                elif 0.2 <= discValue < 0.3:
                    discLevel = 1
                elif 0.3 <= discValue < 0.4:
                    discLevel = 2
                else:
                    discLevel = 3
                writer.write("%d,%d\n" % (quesID, discLevel))

    def build_stu_interaction_graph(self, train_df):
        """
        build stu_skill interaction graph
        build stu_question interaction graph
        """
        print("\n### building student interaction graph ###")
        df = train_df.copy()
        df['time_taken'] = feature_normalize(df['time_taken'])

        stu_skill_set, stu_ques_set = set(), set()
        num_train_stu, num_skill = len(self.train_user_id_dict), len(self.skill_id_dict.keys())
        stu_skill_mat = np.zeros(shape=(num_train_stu, num_skill), dtype=np.float32)
        for stu in self.train_user_id_dict.keys():  # traverse all students in train dataset
            stuID = self.train_user_id_dict[stu]
            tmp_df = df[df['user_id'] == stu].copy()
            tmp_df.sort_values(by=['time_done'], ascending=True, inplace=True)

            # # build stu-skill graph, using combined skills
            # for skill in tmp_df['topic'].unique():
            #     skillID = self.skill_id_dict[skill]
            #     skill_df = tmp_df[tmp_df['topic'] == skill]
            #     crtRatio = skill_df['correct'].mean()
            #     wrgRatio = 1 - crtRatio
            #     stu_skill_mat[stuID][skillID] = crtRatio - wrgRatio
            #     stu_skill_set.add((stuID, skillID, crtRatio))

            # build stu_skill graph, using atom skills
            skill2crts_dict = dict()
            for index, row in tmp_df.iterrows():
                quesID = self.question_id_dict[row['exercise']]
                for skillID in self.quesID2skillIDs_dict[quesID]:
                    if skillID not in skill2crts_dict.keys():
                        skill2crts_dict[skillID] = []
                    skill2crts_dict[skillID].append(int(row['correct']))
            for skillID, correct_list in skill2crts_dict.items():
                crtRatio = np.mean(correct_list)
                wrgRatio = 1 - crtRatio
                stu_skill_mat[stuID][skillID] = crtRatio - wrgRatio
                stu_skill_set.add((stuID, skillID, crtRatio))

            # build stu_ques graph
            timeStep = 1
            for index, row in tmp_df.iterrows():
                quesID = self.question_id_dict[row['exercise']]
                correct = row['correct']
                timePoint = timeStep / len(tmp_df)
                elapsed_time = row['time_taken']
                stu_ques_set.add((stuID, quesID, correct, timePoint, elapsed_time))
                timeStep += 1

        names = ['stu', 'skill', 'crtRatio']
        save_graph(stu_skill_set, os.path.join(self.save_folder, "graph", "stu_skill.csv"), names)
        names = ['stu', 'ques', 'correct', 'timePoint', 'time_taken']
        save_graph(stu_ques_set, os.path.join(self.save_folder, "graph", "stu_ques.csv"), names)
        np.save(os.path.join(self.save_folder, "graph", "stu_skill_mat.npy"), stu_skill_mat)

        # cluster students and skills according to stu_skill_mat
        for num_cluster in [60, 80, 100, 120]:
            get_cluster(num_cluster, stu_skill_mat,
                        os.path.join(self.save_folder, "graph", "stu_cluster_%d.csv" % num_cluster),
                        ['stu', 'cluster'])

        skill_stu_mat = np.transpose(stu_skill_mat)
        for num_cluster in [4, 8, 16]:
            get_cluster(num_cluster, skill_stu_mat,
                        os.path.join(self.save_folder, "graph", "skill_cluster_%d.csv" % num_cluster),
                        ['skill', 'cluster'])


if __name__ == '__main__':
    t = time.time()
    dataPath = "../junyi.csv"
    saveFolder = './'

    DP = DataProcess()
    DF = DP.process_csv()
    trainDF, testDF = DP.get_train_test_df(DF)
    DP.encode_entity(trainDF, testDF)

    DP.build_ques_interaction_graph(trainDF, testDF)
    DP.get_ques_attribute(trainDF)
    # DP.build_stu_interaction_graph(trainDF)

    DP.generate_user_sequence(trainDF, 'train_data.txt')
    DP.generate_user_sequence(testDF, 'test_data.txt')
    DP.encode_user_sequence(train_or_test='train')
    DP.encode_user_sequence(train_or_test='test')

    print("consume %d seconds" % (time.time() - t))
