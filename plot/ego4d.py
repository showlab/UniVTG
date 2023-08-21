import os
import cv2
import pdb
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

import sys
sys.path.append('/Users/kevin/univtg')
from utils.basic_utils import load_jsonl
from utils.temporal_nms import compute_temporal_iou

def norm(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

def apply_template(frame, template_path):
    # add template to frame image
    frame = Image.fromarray(frame)
    template = Image.open(template_path).convert("RGBA")
    width, height = frame.size
    new_size = (width, int(height * 1.4))
    white_background = Image.new('RGBA', new_size, (255, 255, 255, 255))
    white_background.paste(frame, (0, int(height * 0.19))) # 调整位置
    template = template.resize(new_size, Image.ANTIALIAS)
    result = Image.alpha_composite(white_background, template)
    result = result.convert('RGB')
    result = np.array(result)
    return result

def plot_video(pred_json, gt_json, save_dir_i, fig_num=None, template=True):
    duration = gt_json['duration']-clip_len
    t_min, t_max = 0, duration
    x = np.arange(t_min, t_max, clip_len)
    
    if fig_num is None:
        fig_num = round(duration / gap)
    
    fig, axs = plt.subplots(nrows=1, ncols=fig_num, figsize=(40, 20), gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

    # pred_json["vid"] = "5533ab65-7463-47a6-b040-c0c6d65b8cf5"
    # video_path = "/Users/kevin/dataset/Ego4D/data_chunked/0a02a1ed-a327-4753-b270-e95298984b96"
    # pred_json['vid'] = '0'
    vid_exists = os.path.exists(os.path.join(video_path, pred_json['vid'] + '.mp4'))
    assert vid_exists, f'Video {pred_json["vid"]} does not exist!'
    if vid_exists:
        cap = cv2.VideoCapture(os.path.join(video_path, pred_json['vid'] + '.mp4'))
        for i, t in enumerate(np.linspace( t_min, t_max, fig_num)):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1e3)
            rval, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_with_template = apply_template(frame, template_path)
            # axs[i].imshow(frame) 
            axs[i].imshow(frame_with_template) 
            axs[i].axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.savefig(os.path.join(save_dir_i, '0_vid.jpg'), bbox_inches='tight', pad_inches=0, dpi=100)

    image = Image.open(os.path.join(save_dir_i, '0_vid.jpg'))
    image = np.array(image)
    query = pred_json['query']
    text1 = 'QUERY: '
    text2 = '{}'.format(query)
    image_pil = Image.fromarray(image)

    draw = ImageDraw.Draw(image_pil)
    font1 = ImageFont.truetype(font_path2, fontsize2)  # Font for 'QUERY: '
    font2 = ImageFont.truetype(font_path3, fontsize2)  # Font for the query
    text_width1, text_height1 = draw.textsize(text1, font=font1)
    text_width2, text_height2 = draw.textsize(text2, font=font2)
    total_text_width = text_width1 + text_width2  # Total width of the text
    total_text_height = max(text_height1, text_height2)  # Total height of the text
    text_image = Image.new("RGB", (image_pil.width, total_text_height), "white")
    text_draw = ImageDraw.Draw(text_image)

    x1 = (text_image.width - total_text_width) / 2  # Start 'QUERY:' from here
    y1 = (text_image.height - text_height1) / 2
    text_draw.text((x1, y1), text1, fill="black", font=font1)

    x2 = x1 + text_width1  # Start query right after 'QUERY:'
    y2 = (text_image.height - text_height2) / 2
    text_draw.text((x2, y2), text2, fill="black", font=font2)

    # Concatenate the text image and the original image
    final_image = Image.new("RGB", (image_pil.width, image_pil.height + text_image.height))
    final_image.paste(text_image, (0, 0))
    final_image.paste(image_pil, (0, text_image.height))
    final_image.save(os.path.join(save_dir_i, f'0_vid_query.jpg'))
    return final_image

def plot_mr(pred_json, gt_json, save_dir_i, only_one_gt=False, pred_num=1, base_json=None):
    if only_one_gt:
        assert len(gt_json['relevant_windows']) == 1

    plt.figure(figsize=(1, 1))
    offset = gt_json['duration'] * 0.01

    if not base_json:
        fig, ax = plt.subplots(1,1, figsize=(50, 2))
        ax.barh(["UniVTG's Prediction", "GT Interval"], [gt_json['duration'], gt_json['duration']], color="white", edgecolor="black", height=0.6, left=0, linewidth=2)
    
        ax.set_xlim(left=0, right=gt_json['duration'])
        if pred_num is None:
            pred_num = len(gt_json['relevant_windows'])
        for i in range(len(gt_json['relevant_windows'][:pred_num])):
            gt_start, gt_end = gt_json['relevant_windows'][i][0], gt_json['relevant_windows'][i][1]
            ax.barh("GT Interval", gt_end - gt_start, color=color1, edgecolor=color1_dark, height=0.6, left=gt_start, linewidth=2)
            if gt_start > 2 * offset:
                ax.text(gt_start, 1, f'{gt_start:.1f}', va='center', ha='right', color=color1_dark,  fontproperties=font_prop1)
            if gt_end < gt_json['duration'] - offset:
                ax.text(gt_end, 1, f'{gt_end:.1f}', va='center', ha='left',  color=color1_dark, fontproperties=font_prop1)
        
        if pred_num is None:
            pred_num = len(gt_json['relevant_windows'])
        for i in range(len(pred_json['pred_relevant_windows'][:pred_num])):
            interval = pred_json['pred_relevant_windows'][i]
            ax.barh("UniVTG's Prediction", interval[1] - interval[0], color=color2, edgecolor=color2_dark, height=0.6, left=interval[0], linewidth=2)
            if interval[0] > 2 * offset:
                ax.text(interval[0], 0, f'{interval[0]:.1f}', va='center', ha='right',  color=color2_dark,  fontproperties=font_prop1)
            if interval[1] < gt_json['duration'] - offset:
                ax.text(interval[1], 0, f'{interval[1]:.1f}', va='center', ha='left',color=color2_dark,  fontproperties=font_prop1)    

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["", ""])
        
        offset = gt_json['duration'] * 0.01
        ax.text(offset, -0.75, '0.0', va='center', ha='center', color="black", fontproperties=font_prop1)
        ax.text(gt_json['duration'] -offset, -0.75, f'{gt_json["duration"]:.1f}', va='center', ha='center',  color="black", fontproperties=font_prop1)

        ax.set_xticklabels([])

        ax.text(0, 1, "    GT Interval", va='center', ha='left', color=color1_dark, fontproperties=font_prop3)
        ax.text(0, 0, "    UniVTG's Prediction", va='center', ha='left', color=color2_dark, fontproperties=font_prop3)
        # pred_interval = pred_json['pred_relevant_windows'][0]
        # gt_interval = gt_json['relevant_windows'][0]
        # pred_mid = (pred_interval[0] + pred_interval[1]) / 2
        # gt_mid = (gt_interval[0] + gt_interval[1]) / 2
        # ax.text(pred_interval[0], 1, "GT Interval", va='center', ha='center', color=color1_dark, fontproperties=font_prop3)
        # ax.text(gt_interval[0], 0, "UniVTG's Prediction", va='center', ha='center', color=color2_dark, fontproperties=font_prop3)
    else:
        fig, ax = plt.subplots(1,1, figsize=(50, 3))
        ax.barh(["MomentDETR's Prediction", "UniVTG's Prediction", "GT Interval"], [gt_json['duration'], gt_json['duration'], gt_json['duration']], color="white", edgecolor="black", height=0.6, left=0, linewidth=2)
    
        ax.set_xlim(left=0, right=gt_json['duration'])
        for i in range(len(gt_json['relevant_windows'])):
            gt_start, gt_end = gt_json['relevant_windows'][i][0], gt_json['relevant_windows'][i][1]
            ax.barh("GT Interval", gt_end - gt_start, color=color1, edgecolor=color1_dark, height=0.6, left=gt_start, linewidth=2)
            if gt_start > 2 * offset:
                ax.text(gt_start, 2, f'{gt_start:.1f}', va='center', ha='right', color=color1_dark,  fontproperties=font_prop1)
            if gt_end < gt_json['duration'] - offset:
                ax.text(gt_end, 2, f'{gt_end:.1f}', va='center', ha='left',  color=color1_dark, fontproperties=font_prop1)
        
        if pred_num is None:
            pred_num = len(gt_json['relevant_windows'])

        for i in range(len(pred_json['pred_relevant_windows'][:pred_num])):
            interval = pred_json['pred_relevant_windows'][i]
            ax.barh("UniVTG's Prediction", interval[1] - interval[0], color=color2, edgecolor=color2_dark, height=0.6, left=interval[0], linewidth=2)
            if interval[0] > 2 * offset:
                ax.text(interval[0], 1, f'{interval[0]:.1f}', va='center', ha='right',  color=color2_dark,  fontproperties=font_prop1)
            if interval[1] < gt_json['duration'] - offset:
                ax.text(interval[1], 1, f'{interval[1]:.1f}', va='center', ha='left', color=color2_dark,  fontproperties=font_prop1)    

        for i in range(len(base_json['pred_relevant_windows'][:pred_num])):
            interval = base_json['pred_relevant_windows'][i]
            ax.barh("MomentDETR's Prediction", interval[1] - interval[0], color=color3, edgecolor=color3_dark, height=0.6, left=interval[0], linewidth=2)
            if interval[0] > 2 * offset:
                ax.text(interval[0], 0, f'{interval[0]:.1f}', va='center', ha='right',  color=color3_dark,  fontproperties=font_prop1)
            if interval[1] < gt_json['duration'] - offset:
                ax.text(interval[1], 0, f'{interval[1]:.1f}', va='center', ha='left', color=color3_dark, fontproperties=font_prop1)    

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["", "", ""])
        
        ax.text(offset, -0.75, '0.0', va='center', ha='center', color="black", fontproperties=font_prop1)
        ax.text(gt_json['duration'] -offset, -0.75, f'{gt_json["duration"]:.1f}', va='center', ha='center',  color="black", fontproperties=font_prop1)

        ax.set_xticklabels([])
        gt_interval = gt_json['relevant_windows'][0]
        pred_interval = pred_json['pred_relevant_windows'][0]
        pred_base = base_json['pred_relevant_windows'][0]
        # ax.text(gt_interval[0], 2, "    GT Interval", va='center', ha='left', color=color1_dark, fontproperties=font_prop3)
        # ax.text(pred_interval[0], 1, "    UniVTG's Prediction", va='center', ha='left', color=color2_dark, fontproperties=font_prop3)
        # ax.text(pred_base[0], 0, "    MomentDETR's Prediction", va='center', ha='left', color=color3_dark, fontproperties=font_prop3)
        ax.text(0, 2, "    GT Interval", va='center', ha='left', color=color1_dark, fontproperties=font_prop3)
        ax.text(0, 1, "    UniVTG's Prediction", va='center', ha='left', color=color2_dark, fontproperties=font_prop3)
        ax.text(0, 0, "    MomentDETR's Prediction", va='center', ha='left', color=color3_dark, fontproperties=font_prop3)

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.savefig(os.path.join(save_dir_i, '1_mr.jpg'), bbox_inches='tight', pad_inches=0.2, dpi=100)
    return
    
def plot_hl(pred_json, gt_json, save_dir_i,  base_json=None):
    pred_saliency = np.array(pred_json['pred_saliency_scores'])
    pred_saliency = norm(pred_saliency)
    
    gt_saliency = np.zeros_like(pred_saliency)
    gt_saliency_valid = np.array(gt_json['saliency_scores']).mean(-1)
    relevant_idx = gt_json["relevant_clip_ids"]
    gt_saliency[relevant_idx] = norm(gt_saliency_valid)
    
    duration = gt_json['duration']
    t_min, t_max = 0, duration
    x = np.arange(t_min, t_max, clip_len)

    plt.figure(figsize=(1, 1))
    if not base_json:
        fig, ax = plt.subplots(1,1, figsize=(50, 2))
        
        plt.plot(x, gt_saliency, label='GT Saliency', color=color1_dark, linewidth=6, linestyle='solid')
        plt.plot(x, pred_saliency, label='UniVTG\'s Prediction', color=color2_dark, linewidth=6, linestyle='solid')
    
    else:
        fig, ax = plt.subplots(1,1, figsize=(50, 3))

        # ax.set_yticks([0, 1, 2])
        # ax.set_yticklabels(["", "", ""])
        # ax.set_xticklabels([])

        base_saliency = np.array(base_json['pred_saliency_scores'])
        base_saliency = norm(base_saliency)
        
        plt.plot(x, gt_saliency, label='GT Saliency', color=color1_dark, linewidth=6, linestyle='solid')
        plt.plot(x, pred_saliency, label='UniVTG\'s Prediction', color=color2_dark, linewidth=6, linestyle='solid')
        plt.plot(x, base_saliency, label='MomentDETR\'s Prediction', color=color3_dark, linewidth=6, linestyle='solid')
    
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop1)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop1)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(left=0, right=duration - clip_len)

    offset = gt_json['duration'] * 0.01
    # ax.set_xticks(np.arange(gap/2, gt_json['duration'] - gap/2, gap))
    ax.text(offset, -0.2, '0.0', va='center', ha='center', color="black", fontproperties=font_prop1)
    ax.text(gt_json['duration']-clip_len-offset, -0.2, f'{gt_json["duration"]:.1f}', va='center', ha='center',  color="black", fontproperties=font_prop1)
    for i in np.arange(0, gt_json['duration'] + gap/2, gap)[1:-1]:
        ax.text(i, -0.2, '{:.1f}'.format(i), va='center', ha='center', color="black", fontproperties=font_prop1)

    # # ax.xaxis.set_tick_params(which='both', direction='in', length=15)  # Set the direction and length of the x ticks
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) 
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(axis='both', labelsize=fontsize1)
    
    legend = ax.legend(prop=font_prop3, loc='upper left', bbox_to_anchor=(0, 1.1))
    lines, labels = legend.get_lines(), legend.get_texts()
    for line, label in zip(lines, labels):
        label.set_color(line.get_color())

    for position in ['top', 'right']:
        ax.spines[position].set_visible(False)
    for position in ['bottom', 'left']:
        ax.spines[position].set_visible(True)
        ax.spines[position].set_linewidth(2)  # Change the line width here
    plt.savefig(os.path.join(save_dir_i, '2_hl.jpg'), bbox_inches='tight', pad_inches=0.2, dpi=100)
    return

def plot_sample(sample_id):
    pred_json = pred_json_val[sample_id]
    gt_json = gt_json_val[sample_id]
    base_json = base_json_val[sample_id] if base_json_val is not None else None
        
    global gap
    gap = round(gt_json['duration'] / seg_num)
    iou = compute_temporal_iou(pred_json['pred_relevant_windows'][0], gt_json['relevant_windows'][0])
    iou = round(iou, 2)
    save_dir_i = os.path.join(save_dir, '_'.join([str(iou), gt_json['vid'],str(sample_id)]))
    
    if not os.path.exists(save_dir_i):
        os.mkdir(save_dir_i)
     
    plot_mr(pred_json, gt_json, save_dir_i, only_one_gt=only_one_gt, pred_num=pred_num, base_json=base_json)
    plot_video(pred_json, gt_json, save_dir_i, fig_num=fig_num, template=True)
    # plot_hl(pred_json, gt_json, save_dir_i, base_json=base_json)
    
    image1 = Image.open(os.path.join(save_dir_i, '0_vid_query.jpg'))
    image2 = Image.open(os.path.join(save_dir_i, '1_mr.jpg'))
    image2 = image2.resize(image1.size)
    # image3 = Image.open(os.path.join(save_dir_i, '2_hl.jpg'))
    # image3 = image3.resize(image1.size)

    # new_image = Image.new('RGB', (image1.width, image1.height + image2.height + image3.height))
    new_image = Image.new('RGB', (image1.width, image1.height + image2.height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, image1.height))
    # new_image.paste(image3, (0, 2 * image1.height))

    # Save the new image
    new_image.save(os.path.join(save_dir_i, 'combined.jpg')) 
    return

if __name__ == "__main__":
    # settings
    fig_num = None  # if None, will be set automatically
    clip_len = 2
    seg_num = 15
    
    only_one_gt = False
    pred_num = 1
    # only_one_gt = False
    # pred_num = None
    
    # load prediction, ground truth, and video
    video_path="/Users/kevin/dataset/ego4d_nlq/val_segment"
    gt_path="/Users/kevin/univtg/data/ego4d/metadata/nlq_val.jsonl"

    dset_path="/Users/kevin/univtg/results/mr-ego4d"
    checkpoint_dir="aio_unified_epo6__f50_b10g1_s0.1_1-slowfast_clip-clip-2023_05_29_06"
    json_file="best_ego4d_val_preds_nms_thd_0.7.jsonl"
    
    # baseline
    base_checkpoint_dir = "/Users/kevin/univtg_baseline/moment_detr/results/ego4d-video_tef-exp-2023_03_01_08_42_54"
    base_json_file = "best_ego4d_val_preds.jsonl"
    
    pred_json_val = load_jsonl(os.path.join(dset_path, checkpoint_dir, json_file))
    gt_json_val = load_jsonl(gt_path)
    
    base_json_val = load_jsonl(os.path.join(base_checkpoint_dir, base_json_file))
    # base_json_val = None
    
    # other settings
    template_path = "/Users/kevin/univtg/plot/settings/template.png"
    color1 = '#90ee90'  # green
    color2 = '#add8e6'  # blue
    color3 = '#D8BFD8'  # purple
    color1_dark = '#008000'  # Dark Green
    color2_dark = '#00008B'  # Dark Blue
    color3_dark = '#800080'  # Dark Purple
    fontsize1 = 30
    fontsize2 = 75
    
    font_path1 = "/Users/kevin/univtg/plot/settings/calibri.ttf"
    font_path2 = "/Users/kevin/univtg/plot/settings/calibri-bold.ttf"
    font_path3 = "/Users/kevin/univtg/plot/settings/calibri-italy.ttf"
    font_prop1 = FontProperties(fname=font_path1, size=fontsize1)
    font_prop2 = FontProperties(fname=font_path2, size=fontsize2)
    font_prop3 = FontProperties(fname=font_path2, size=fontsize1)

    save_dir = os.path.join(os.path.join('/Users/kevin/univtg/plot', 'ego4d', checkpoint_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # SINGLE VIDEO
    # sample_id = 555
    # plot_sample(sample_id)

    for sample_id in range(len(gt_json_val)):
        if float(gt_json_val[sample_id]['duration']) <= 500:
            continue
        try:
            plot_sample(sample_id)
        except Exception as e:
            print(f'Error in sample {sample_id} with error {e}')
            continue
