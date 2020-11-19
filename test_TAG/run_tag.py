import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rc, font_manager
import math
import random
import numpy
import cPickle as pkl
import numpy as np
import logging
from TAG import *
from opt import *
import operator



eps_list = [5.0,6.0,7.0]
sigma_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
lamda_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
bp_overlap_list = [1.0,0.95,0.9,0.85,0.8,0.75,0.7]

# eps = 5.0
# sigma = 0.9
# lamda = 0.6
options = default_options()


def generate_anchor(feat_len,feat_ratio,max_len):
    anchor_list = []
    element_span = max_len / feat_len 
    span_list = []
    for kk in feat_ratio:
        span_list.append(kk * element_span)
    for i in range(feat_len):
        inner_list = []
        for span in span_list:
            left =   i*element_span + (element_span * 1 / 2 - span / 2)
            right =  i*element_span + (element_span * 1 / 2 + span / 2) 
            inner_list.append([left,right])
        anchor_list.append(inner_list)
    return anchor_list


def generate_all_anchor():
    all_anchor_list = []
    for i in range(len(options['feature_map_len'])):
        anchor_list = generate_anchor(options['feature_map_len'][i],options['scale_ratios_anchor'+str(i+1)],options['sample_len'])
        all_anchor_list.append(anchor_list)
    return all_anchor_list


def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick
    union = map(operator.sub, x2, x1) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)
        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick


def calculate_IOU(groundtruth, predict):

    groundtruth_init = max(0,groundtruth[0])
    groundtruth_end = groundtruth[1]

    predict_init = max(0,predict[0])
    predict_end = predict[1]

    init_min = min(groundtruth_init,predict_init)
    end_max = max(groundtruth_end,predict_end)

    init_max = max(groundtruth_init,predict_init)
    end_min = min(groundtruth_end,predict_end)

    if end_min < init_max:
        return 0

    IOU = ( end_min - init_max ) * 1.0 / ( end_max - init_min)
    return IOU



def analysis_iou(result, epoch, logging):

    threshold_list = [0.1,0.3,0.5,0.7]
    rank_list = [1,5,10]
    result_dict = {}
    top1_iou = []

    for i in range(len(result)):
        video_name = result[i][0]
        ground_truth_interval = result[i][1]
        predict_list = result[i][3]
        video_duration = result[i][4]

        iou_list = []
        for predict_interval in predict_list:
            iou_list.append(calculate_IOU(ground_truth_interval,predict_interval))
        top1_iou.append(iou_list[0])

        for rank in rank_list:
            for threshold in threshold_list:
                key_str = 'Recall@'+str(rank)+'_iou@'+str(threshold)
                if key_str not in result_dict:
                    result_dict[key_str] = 0

                for jj in range(rank):
                    if iou_list[jj] >= threshold:
                        result_dict[key_str] += 1
                        break

    logging.info('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    logging.info('epoch '+str(epoch)+': ')
    for key_str in result_dict:
        logging.info(key_str+': '+str(result_dict[key_str]*1.0/len(result)))
    logging.info('mean iou: '+str(np.mean(top1_iou)))
    logging.info('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')


def find_boundary_position(prob,origin_point,duration,eps,sigma,lamda):
    delta = int(duration / eps)
    left = int(origin_point - delta)
    right = int(origin_point + delta)
    left = max(0,min(left,options['sample_len']-1))
    right = max(0,min(right,options['sample_len']-1))
    max_score = -1
    max_position = -1
    i = left
    while i <= right:
        if prob[i] > max_score:
            max_score = prob[i]
            max_position = i
        i+=1
    if max_score >= sigma:
        final_position = lamda * origin_point + (1-lamda) * max_position
    else:
        final_position = origin_point
    return final_position


def get_nms_metric_value(epoch,result,bns_result,overlap,logging,eps,sigma,lamda,bp_overlap):

    all_anchor_list = generate_all_anchor()
    expand_anchor_list = []
    for anchor_group_id in range(len(options['feature_map_len'])):
        for anchor_id in range(options['feature_map_len'][anchor_group_id]):
            for kk in range(4):
                expand_anchor_list.append(all_anchor_list[anchor_group_id][anchor_id][kk])

    new_result = []
    for i in range(len(result)):
        print i
        start_prob = bns_result[i][-1][0,:,0]
        end_prob = bns_result[i][-1][0,:,1]
        middle_prob = bns_result[i][-1][0,:,2]
        bsn_proposal = tag_group(middle_prob)

        video_name = result[i][0]
        ground_truth_interval = result[i][1]
        video_duration = result[i][4]
        predict_overlap_list = result[i][6]
        predict_center_list = result[i][7]
        predict_width_list = result[i][8]
        a_left = []
        a_right = []
        a_score = []
        for index in range(len(predict_overlap_list)):
            anchor = expand_anchor_list[index]
            anchor_center = (anchor[1] - anchor[0]) * 0.5 + anchor[0]
            anchor_width = anchor[1] - anchor[0]
            center_offset = predict_center_list[index]
            width_offset = predict_width_list[index]
            p_center = anchor_center+0.1*anchor_width*center_offset
            p_width =anchor_width*np.exp(0.1*width_offset)
            p_left = max(0, p_center-p_width*0.5)
            p_right = min(options['sample_len'], p_center+p_width*0.5)
            if p_right - p_left < 1.0:
                continue
            if p_right - p_left > video_duration:
                continue
            ssad_duration = p_right - p_left + 1
            refined_left = find_boundary_position(start_prob,p_left,ssad_duration,eps,sigma,lamda)
            refined_right = find_boundary_position(end_prob,p_right,ssad_duration,eps,sigma,lamda)
            if refined_right != p_right or refined_left != p_left:   
                a_left.append(refined_left)
                a_right.append(refined_right)
                a_score.append(predict_overlap_list[index]*1.15)
            else:
                a_left.append(refined_left)
                a_right.append(refined_right)
                a_score.append(predict_overlap_list[index])      

        picks = nms_temporal(a_left,a_right,a_score,overlap)
        process_segment = []
        process_score = []
        for pick in picks:
            ssad_left = a_left[pick]
            ssad_right = a_right[pick]
            ssad_seg = [ssad_left,ssad_right]
            bp_iou_list = []
            for kkk in range(len(bsn_proposal)):
                bp = bsn_proposal[kkk]
                bp_seg = bp['segment']
                bp_iou_list.append(calculate_IOU(bp_seg,ssad_seg))
            if max(bp_iou_list) >= bp_overlap:
                final_ssad_seg = bsn_proposal[np.argsort(bp_iou_list)[-1]]['segment']
            else:
                final_ssad_seg = ssad_seg
            process_segment.append(final_ssad_seg)
            process_score.append(a_score[pick])

        new_result.append([video_name,ground_truth_interval,[],process_segment,video_duration])
    analysis_iou(new_result, epoch, logging)


test_num = 0 #set your test model 
path = '../../result/scdm_newshuffle_BSN_lr0.0001_dimhidden256_Center50.0_Width20.0_pos100.0_hardneg50.0_easyneg50.0_regular0.001_posTh0.5_negTh0.3_predictHidden64_hardneg2.0/'
pkl_path = path + str(test_num)+'.pkl'
content = pkl.load(open(pkl_path))
ssad_content = content

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
log_file_name = 'tag_new_'+str(test_num)+'.log'
fh = logging.FileHandler(filename=log_file_name, mode='w', encoding='utf-8')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logging.root.addHandler(fh)

overlap = 0.7
for bp_overlap in bp_overlap_list:
    for eps in eps_list:
        for sigma in sigma_list:
            for lamda in lamda_list:
                logging.info('*************************************************************')
                logging.info('bp_overlap'+str(bp_overlap)+'_eps'+str(eps)+'_lamda'+str(lamda)+'_sigma'+str(sigma))
                get_nms_metric_value(test_num,ssad_content,content,overlap,logging,eps,sigma,lamda,bp_overlap)
                logging.info('*************************************************************')





