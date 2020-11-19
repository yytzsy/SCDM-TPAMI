import json
import numpy as np
import copy
import argparse


def get_iou(pred, gt):
    start_pred, end_pred = pred
    start, end = gt
    intersection = max(0, min(end, end_pred) - max(start, start_pred))
    union = min(max(end, end_pred) - min(start, start_pred), end - start + end_pred - start_pred)
    iou = float(intersection) / (union + 1e-8)
    return iou


def nms_detections(proposals, overlap=0.8):
    if len(proposals) == 0:
        return proposals
    props = np.array([item['segment'] for item in proposals])
    scores = np.array([item['score'] for item in proposals])
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]
    out_proposals = []
    for idx in range(nms_props.shape[0]):
        prop = nms_props[idx].tolist()
        score = float(nms_scores[idx])
        out_proposals.append({'segment': prop, 'score': score})
    return out_proposals


def final_proposal(prob_list, high_threshold, low_threshold):
    prob_list = np.array(prob_list)
    prob_list_high_low_origin = copy.deepcopy(prob_list)
    prob_list[prob_list < low_threshold] = 10
    prob_list[prob_list < high_threshold] = 0.1
    prob_list[prob_list == 10] = 0
    prob_list[prob_list >= high_threshold] = 1

    prob_list_high_low = copy.deepcopy(prob_list)
    prob_list_high_low_origin = prob_list_high_low

    prob_list[prob_list > 0] = 1

    start_list = []
    end_list = []
    if prob_list[0] == 1:
        start_list.append(0)
    for i in range(1, len(prob_list)):
        if prob_list[i] - prob_list[i - 1] == 1:
            start_list.append(i)
        if prob_list[i] - prob_list[i - 1] == -1:
            end_list.append(i - 1)

    if len(start_list) - len(end_list) == 1:
        end_list.append(len(prob_list) - 1)
    if len(start_list) != len(end_list):
        print('the function high_proposal is wrong!')

    high_proposal_result = []
    for i in range(len(start_list)):
        score = prob_list_high_low[start_list[i]:end_list[i] + 1].mean()
        score_origin = prob_list_high_low_origin[start_list[i]:end_list[i] + 1].mean()
        if score == 0.1:
            continue
        high_proposal_result.append(
            {'score': score_origin, 'segment': [start_list[i], end_list[i]]})
    return high_proposal_result




def tag_group(prob):
    high_range1 = np.arange(0.43, 1, 0.28)
    low_range1 = np.arange(0.0, 0.6, 0.015)
    high_range2 = np.arange(0.24, 0.49, 0.28)
    low_range2 = np.arange(0.0, 0.22, 0.015)
    high_range3 = np.arange(0.19, 0.24, 0.28)
    low_range3 = np.arange(0.05, 0.18, 0.015)

    # high_range1 = np.arange(0.8, 1, 0.01)
    # low_range1 = np.arange(0.0, 0.6, 0.01)
    # high_range2 = np.arange(0.6, 0.9, 0.01)
    # low_range2 = np.arange(0.0, 0.4, 0.01)
    # high_range3 = np.arange(0.4, 0.7, 0.01)
    # low_range3 = np.arange(0.0, 0.3, 0.01)


    temp_result = []
    for high_threshold in high_range1:
        for change in low_range1:
            low_threshold = high_threshold - change
            one_result = final_proposal(prob, high_threshold, low_threshold)
            temp_result = temp_result + one_result
    for high_threshold in high_range2:
        for change in low_range2:
            low_threshold = high_threshold - change
            one_result = final_proposal(prob, high_threshold, low_threshold)
            temp_result = temp_result + one_result
    for high_threshold in high_range3:
        for change in low_range3:
            low_threshold = high_threshold - change
            one_result = final_proposal(prob, high_threshold, low_threshold)
            temp_result = temp_result + one_result

    temp_result9 = nms_detections(temp_result, overlap=0.9)
    return temp_result9


