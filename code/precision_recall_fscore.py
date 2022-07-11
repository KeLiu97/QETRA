
import argparse
import nltk.translate.gleu_score as gleu
from nlgeval import getListRouge, compute_metrics
import pandas as pd



def mrr_N_sentence(ground, pred, n):
    score = 0.0
    for rank, item in enumerate(pred[:n]):
        if str(item) in ground:
            score = 1.0 / (rank + 1.0)
            break

    return score


def mrr_N_List(preds, gloden,n):
    if "txt" in preds:
        preds = txt2DataFrame(preds,n)
    else:
        preds = pd.read_csv(preds,header=None)
    if "txt" in gloden:
        gloden = txt2DataFrame(gloden,1)
    else:
        gloden = pd.read_csv(gloden,header=None)
    score = 0
    total = 0
    for i in range(gloden.shape[0]):
        score += mrr_N_sentence(gloden.iloc[i,0], preds.iloc[i], n)
        total=total+1

    return score/total


def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)



def txt2DataFrame(file,n):
    file = open(file, 'r', encoding='utf8')
    txt = file.readlines()
    data = []
    one = []
    for index in range(len(txt)):
        one.append(txt[index])
        if (index + 1) % n == 0:
            data.append(one)
            one = []
    data = pd.DataFrame(data)
    return data

# exact accuracy goes through each integer in each array in the actual and prediction arrays.
# good for multilabel classification problems (this is the same as the Exact Match metric)
def EM(preds, gloden,n):
    if "txt" in preds:
        preds = txt2DataFrame(preds,n)
    else:
        preds = pd.read_csv(preds,header=None)
    if "txt" in gloden:
        gloden = txt2DataFrame(gloden,1)
    else:
        gloden = pd.read_csv(gloden,header=None)
    correct = 0
    total = 0
    for i in range(gloden.shape[0]):
        for j in range(n):
            if gloden.iloc[i,0] == preds.iloc[i,j]:
                correct=correct+1
                break
        total=total+1

    return correct/total

def compute(preds, gloden):
    t = open(gloden, 'r', encoding='utf8')
    p = open(preds, 'r', encoding='utf8')
    tline = t.readlines()
    pline = p.readlines()
    gleu_result = score_gleu(tline, pline)
    metrics_dict = compute_metrics(hypothesis=preds,
                                   references=[gloden], no_skipthoughts=True, no_glove=True)
    metrics_dict['gleu'] = gleu_result
    return metrics_dict

if __name__ == "__main__":

    preds1 = "target_preds.csv"
    gloden = "target_gloden.csv"
    preds5 = "target_preds_5.csv"
    preds10 = "target_preds_10.csv"

    metrics_dict = compute(preds1, gloden)
    EM1 = EM(preds1, gloden, 1)
    EM5 = EM(preds5, gloden, 5)
    EM10 = EM(preds10, gloden, 10)
    MRR = mrr_N_List(preds5, gloden, 5)
    metrics_dict['EM1'] = EM1
    metrics_dict['EM5'] = EM5
    metrics_dict['EM10'] = EM10
    metrics_dict['MRR'] = MRR
    print(metrics_dict)
