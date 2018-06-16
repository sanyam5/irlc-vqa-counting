from __future__ import print_function
import json
import cPickle as pkl
import os
import re


def find_image_ids():

    # read locations
    train_targets_loc = "./data/cache/train_target.pkl"
    val_targets_loc = "./data/cache/val_target.pkl"
    hmq_ids_loc = "./data/how_many_qa/HowMany-QA/question_ids.json"

    # write locations
    hmq_image_ids_loc = "./data/how_many_qa/image_ids.json"
    if os.path.isfile(hmq_image_ids_loc):
        print("The file {} already exists. Skipping finding image ids.".format(hmq_image_ids_loc))
        return

    train_targets = pkl.load(open(train_targets_loc, "rb"))
    val_targets = pkl.load(open(val_targets_loc, "rb"))

    hmq_ids = json.load(open(hmq_ids_loc, "rb"))
    qids = {
        "train": set(hmq_ids["train"]["vqa"]),
        "test": set(hmq_ids["test"]),
        "dev": set(hmq_ids["dev"]),
    }

    image_ids = {
        "test": [],
        "train": [],
        "dev": [],
    }

    # train
    for i, ans in enumerate(train_targets):
        if i % 10000 == 0:
            print(i)
        if ans["question_id"] in qids["train"]:
            image_ids["train"].append(ans["image_id"])
        if ans["question_id"] in qids["test"] or ans["question_id"] in qids["dev"]:
            raise Exception("found train question id {} in qids marked for test and dev")

    # dev and test
    for i, ans in enumerate(val_targets):
        if i % 10000 == 0:
            print(i)
        if ans["question_id"] in qids["train"]:
            raise Exception("found validation question id {} in qids marked for training")
        if ans["question_id"] in qids["test"]:
            image_ids["test"].append(ans["image_id"])
        if ans["question_id"] in qids["dev"]:
            image_ids["dev"].append(ans["image_id"])

    unique_image_ids = {
        "test": list(set(image_ids["test"])),
        "train": list(set(image_ids["train"])),
        "dev": list(set(image_ids["dev"])),
    }

    assert len(unique_image_ids["train"]) == 31932
    assert len(unique_image_ids["dev"]) == 13119
    assert len(unique_image_ids["test"]) == 2483

    print("writing image ids for how many QA to disk..")
    json.dump(unique_image_ids, open(hmq_image_ids_loc, "wb"))
    print("Done.")
    return


def prepare_visual_genome():

    # read locations
    hmqa_qids_loc = "./data/how_many_qa/HowMany-QA/question_ids.json"
    all_vg_loc = "./data/how_many_qa/HowMany-QA/visual_genome_question_answers.json"
    vg_image_data_loc = "./data/how_many_qa/HowMany-QA/visual_genome_image_data.json"

    # write locations
    vgn_ques_loc = "./data/how_many_qa/vgn_ques.json"
    if os.path.isfile(vgn_ques_loc):
        print("The file {} already exists. Skipping preparing visual genome.".format(vgn_ques_loc))
        return

    hmqa_qids = json.load(open(hmqa_qids_loc, "rb"))
    hmqa_vg_qids = set(hmqa_qids["train"]["visual_genome"])
    all_vg = json.load(open(all_vg_loc))
    vg_image_data = json.load(open(vg_image_data_loc))

    vg_entries = []

    for qaset in all_vg:
        # setid = qaset['id']
        for entry in qaset['qas']:
            if entry["qa_id"] in hmqa_vg_qids:
                vg_entries.append(entry)

    vg_image_id2coco_id = {x["image_id"]: x["coco_id"] for x in vg_image_data}

    vgn_ques = [
        {'image_id': vg_image_id2coco_id[x['image_id']], 'question': x['question'], 'question_id': x['qa_id']
         } for x in vg_entries
    ]

    json.dump(vgn_ques, open(vgn_ques_loc, "wb"))


def vg_ans2count(s):
    # replace all non-alphanums with space
    r = re.sub('[^0-9a-zA-Z]+', ' ', s)

    r = r.lower()
    r = r.split(' ')

    word2num = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20
    }

    cands = []
    for word in r:
        try:
            cands.append(int(word))
        except:
            pass
        try:
            cands.append(word2num[word])
        except:
            pass

    cands = list(set(cands))  # merging duplicates

    if len(cands) != 1:
        print(s, cands)
        if (s == "3 or 4." and cands == [3, 4]):
            print("manually correcting '{}'".format(s))
            cands = cands[:1]

    assert len(cands) == 1
    count = cands[0]

    assert 0 <= count <= 20

    return count


def find_counts():

    # read locations
    _hmq_ids_loc = "./data/how_many_qa/HowMany-QA/question_ids.json"
    vqa_train_entries_loc = "./data/cache/train_target.pkl"
    test_dev_entries_loc = "./data/cache/val_target.pkl"
    label2ans_loc = "./data/cache/trainval_label2ans.pkl"
    all_vg_loc = "./data/how_many_qa/HowMany-QA/visual_genome_question_answers.json"

    # write locations
    qid2count_loc = "./data/how_many_qa/qid2count.json"
    qid2count2score_loc = "./data/how_many_qa/qid2count2score.json"

    if os.path.isfile(qid2count_loc) and os.path.isfile(qid2count2score_loc):
        print("The file {} and {} already exists. Skipping finding counts.".format(qid2count_loc, qid2count2score_loc))
        return

    _hmq_ids = json.load(open(_hmq_ids_loc, "rb"))
    hmq_ids = {
        "train": {
            "vqa": set(_hmq_ids["train"]["vqa"]),
            "visual_genome": set(_hmq_ids["train"]["visual_genome"]),
        },
        "test": set(_hmq_ids["test"]),
        "dev": set(_hmq_ids["dev"]),
    }

    vqa_train_entries = pkl.load(open(vqa_train_entries_loc, "rb"))
    test_dev_entries = pkl.load(open(test_dev_entries_loc, "rb"))
    label2ans = pkl.load(open(label2ans_loc, "rb"))

    qid2count = {
        "train": {
            "vqa": {},
            "visual_genome": {}
        },
        "test": {},
        "dev": {},
    }

    qid2count2score = {
        "train": {
            "vqa": {},
            "visual_genome": {}
        },
        "test": {},
        "dev": {},
    }

    # vqa train
    for entry in vqa_train_entries:
        qid = entry['question_id']

        if qid not in hmq_ids["train"]["vqa"]:
            continue

        gt_cands = []
        max_occurence_count = 0

        for occurence_count, score, label in zip(entry["counts"], entry["scores"], entry["labels"]):
            try:
                count = int(label2ans[label])
                assert count <= 20, "No {} is more (score: {})".format(count, score)

                if occurence_count > max_occurence_count:
                    max_occurence_count = occurence_count
                    gt_cands = [count]
                elif occurence_count == max_occurence_count:
                    gt_cands.append(count)

                if qid2count2score["train"]["vqa"].get(qid) is None:
                    qid2count2score["train"]["vqa"][qid] = [0] * 21  # count2score list mapping
                qid2count2score["train"]["vqa"][qid][count] = score

            except Exception as e:
                print(e)
                pass

        # select the answer with highest occurence count, in case of a tie select the minimum
        qid2count["train"]["vqa"][qid] = min(gt_cands)

    ##### VISUAL GENOME  #######

    hmqa_qids = json.load(open(_hmq_ids_loc, "rb"))
    hmqa_vg_qids = set(hmqa_qids["train"]["visual_genome"])
    all_vg = json.load(open(all_vg_loc))

    vg_entries = []

    for qaset in all_vg:
        # setid = qaset['id']
        for entry in qaset['qas']:
            if entry["qa_id"] in hmqa_vg_qids:
                vg_entries.append(entry)

    for entry in vg_entries:
        qid = entry["qa_id"]
        count = vg_ans2count(entry["answer"])
        assert qid2count["train"]["visual_genome"].get(qid) is None
        assert qid2count2score["train"]["visual_genome"].get(qid) is None

        qid2count["train"]["visual_genome"][qid] = count
        qid2count2score["train"]["visual_genome"][qid] = [0] * 21
        qid2count2score["train"]["visual_genome"][qid][count] = 1

    ##################################

    # test and dev
    for entry in test_dev_entries:
        qid = entry['question_id']

        test_entry = qid in hmq_ids["test"]
        dev_entry = qid in hmq_ids["dev"]

        if not (test_entry or dev_entry):
            continue

        if test_entry and dev_entry:
            raise Exception("Found qid {} that is marked for both test set and train set!!".format(qid))

        gt_cands = []
        max_occurence_count = 0

        for occurence_count, score, label in zip(entry["counts"], entry["scores"], entry["labels"]):
            try:
                count = int(label2ans[label])
                assert count <= 20, "No {} is more (score: {})".format(count, score)

                if occurence_count > max_occurence_count:
                    max_occurence_count = occurence_count
                    gt_cands = [count]
                elif occurence_count == max_occurence_count:
                    gt_cands.append(count)

                if test_entry:

                    if qid2count2score["test"].get(qid) is None:
                        qid2count2score["test"][qid] = [0] * 21  # count2score list mapping
                    qid2count2score["test"][qid][count] = score

                if dev_entry:

                    if qid2count2score["dev"].get(qid) is None:
                        qid2count2score["dev"][qid] = [0] * 21  # count2score list mapping
                    qid2count2score["dev"][qid][count] = score

            except Exception as e:
                print(e)
                pass

        # select the answer with highest occurence count, in case of a tie select the minimum
        if test_entry:
            qid2count["test"][qid] = min(gt_cands)
        if dev_entry:
            assert not test_entry
            qid2count["dev"][qid] = min(gt_cands)

    assert len(qid2count["train"]["vqa"]) == 47542
    assert len(qid2count["test"]) == 5000
    assert len(qid2count["dev"]) == 17714

    json.dump(qid2count, open(qid2count_loc, "w"))
    json.dump(qid2count2score, open(qid2count2score_loc, "w"))
    return


def main():
    find_image_ids()
    prepare_visual_genome()
    find_counts()


if __name__ == '__main__':
    main()

