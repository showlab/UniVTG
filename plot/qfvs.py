import os
import cv2
import pdb
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches

import sys
import datetime
sys.path.append('/Users/kevin/univtg')
from utils.basic_utils import load_jsonl
from utils.temporal_nms import compute_temporal_iou

def norm(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

def convert_seconds_to_hms(seconds):
    return str(datetime.timedelta(seconds=seconds))

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

def plot_video(pred_json, save_dir_i, fig_num=None, template=True, gt_only=False):
    duration = pred_json['shots']-clip_len
    t_min, t_max = 1, 1 + duration * 5
    x = np.arange(t_min, t_max, clip_len)
    
    if fig_num is None:
        fig_num = round(duration / gap)

    if not gt_only:
        top_pred = sorted(pred_json['top_pred'])
    else:
        top_pred = sorted(pred_json['gt'])
    top_pred = [i * clip_len for i in top_pred]
    top_pred = random.sample(top_pred, fig_num)
    top_pred = sorted(top_pred)
    
    # select top clips w/ top scores, but there are highly similar
    # top_pred = pred_json['top_pred'][:fig_num]
    # top_pred = sorted(top_pred)

    fig, axs = plt.subplots(nrows=1, ncols=fig_num, figsize=(40, 20), gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

    # for i, t in enumerate(np.linspace( t_min, t_max, fig_num)):
    for i, t in enumerate(top_pred):
        t = int(t)
        image_path = os.path.join(video_path, f'{t:03d}.jpg')
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_with_template = apply_template(frame, template_path)
        axs[i].imshow(frame_with_template) 
        axs[i].axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if not gt_only:
        plt.savefig(os.path.join(save_dir_i, '0_vid.jpg'), bbox_inches='tight', pad_inches=0, dpi=100)
        image = Image.open(os.path.join(save_dir_i, '0_vid.jpg'))
    else:
        plt.savefig(os.path.join(save_dir_i, '0_vid_gt.jpg'), bbox_inches='tight', pad_inches=0, dpi=100)
        image = Image.open(os.path.join(save_dir_i, '0_vid_gt.jpg'))
    
    image = np.array(image)
    query = pred_json['concept1'] + ' and ' + pred_json['concept2']
    text1 = 'KEYWORDS: '
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
    if not gt_only:
        final_image.save(os.path.join(save_dir_i, f'0_vid_query.jpg'))
    else:
        final_image.save(os.path.join(save_dir_i, f'0_vid_query_gt.jpg'))
    return final_image
 
def plot_vs(pred_json, save_dir_i,  base_json=None):
    pred_saliency = np.array(pred_json['top_pred'])
    gt_saliency = np.array(pred_json['gt'])
    total_cells = pred_json['shots']

    t_min, t_max = 0, total_cells
    x = np.arange(t_min, t_max, clip_len)
    x = x[:len(pred_saliency)]
    
    colors1 = ['white'] * total_cells
    colors2 = ['white'] * total_cells

    # for idx in gt_saliency:
    #     colors[idx] = 'green'
    for idx in gt_saliency:
        colors1[idx] = color1_dark
    for idx in pred_saliency:
        colors2[idx] = color2_dark

    # plt.figure(figsize=(50, 2))
    # fig, ax = plt.subplots(2,1, figsize=(50, 2))
    fig, ax = plt.subplots(2,1, figsize=(50, 2), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})

    # plt.bar(range(total_cells), np.ones(total_cells), color=colors, width=2,)
    # plt.axis('off')
    # rect = patches.Rectangle((0, 0), total_cells, 1, linewidth=1, edgecolor='black', facecolor='none')
    # ax.add_patch(rect)
    
    bars1 = ax[0].bar(range(total_cells), np.ones(total_cells), color=colors1, width=2, label='GT Summary ↑')
    ax[0].axis('off')
    rect1 = patches.Rectangle((0, 0), total_cells, 1, linewidth=1, edgecolor='black', facecolor='none')
    ax[0].add_patch(rect1)
    legend = ax[0].legend(loc='upper right', prop=font_prop3, handlelength=0)
    # legend.get_texts().set_color(color1_dark)
    for text in legend.get_texts(): text.set_color(color1_dark)

    bars2 = ax[1].bar(range(total_cells), np.ones(total_cells), color=colors2, width=2, label='UniVTG\'s Summary ↓')
    ax[1].axis('off')
    rect2 = patches.Rectangle((0, 0), total_cells, 1, linewidth=1, edgecolor='black', facecolor='none')
    ax[1].add_patch(rect2)
    legend = ax[1].legend(loc='upper right', prop=font_prop3, handlelength=0)
    # legend.get_texts().set_color(color2_dark)
    for text in legend.get_texts(): text.set_color(color2_dark)

    # if not base_json:
    #     fig, ax = plt.subplots(1,1, figsize=(50, 2))
    #     plt.plot(x, gt_saliency, label='GT Saliency', color=color1_dark, linewidth=6, linestyle='solid')
    #     plt.plot(x, pred_saliency, label='UniVTG\'s Prediction', color=color2_dark, linewidth=6, linestyle='solid')
    
    # else:
    #     fig, ax = plt.subplots(1,1, figsize=(50, 3))

    #     # ax.set_yticks([0, 1, 2])
    #     # ax.set_yticklabels(["", "", ""])
    #     # ax.set_xticklabels([])

    #     base_saliency = np.array(base_json['pred_saliency_scores'])
    #     base_saliency = norm(base_saliency)
        
    #     plt.plot(x, gt_saliency, label='GT Saliency', color=color1_dark, linewidth=6, linestyle='solid')
    #     plt.plot(x, pred_saliency, label='UniVTG\'s Prediction', color=color2_dark, linewidth=6, linestyle='solid')
    #     plt.plot(x, base_saliency, label='MomentDETR\'s Prediction', color=color3_dark, linewidth=6, linestyle='solid')
    
    # for label in ax.get_xticklabels():
    #     label.set_fontproperties(font_prop1)
    # for label in ax.get_yticklabels():
    #     label.set_fontproperties(font_prop1)

    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    ax[0].set_xlim(left=0, right=total_cells)
    ax[1].set_xlim(left=0, right=total_cells)

    offset = pred_json['shots'] * 0.01
    # # ax.set_xticks(np.arange(gap/2, gt_json['duration'] - gap/2, gap))
    start_time = convert_seconds_to_hms(0)
    ax[1].text(offset, -0.3, start_time, va='center', ha='center', color="black", fontproperties=font_prop1)
    end_time = convert_seconds_to_hms(pred_json['shots'] * 5)
    ax[1].text(pred_json['shots']-offset, -0.3, end_time, va='center', ha='center',  color="black", fontproperties=font_prop1)
    # for i in np.arange(0, pred_json['shots'] + gap/2, gap)[1:-1]:
    #     ax.text(i, -0.2, '{:.1f}'.format(i), va='center', ha='center', color="black", fontproperties=font_prop1)

    # # # ax.xaxis.set_tick_params(which='both', direction='in', length=15)  # Set the direction and length of the x ticks
    # # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) 
    
    # ax.set_yticks([])
    # ax.set_xticks([])
    # ax.tick_params(axis='both', labelsize=fontsize1)
    
    # legend = ax.legend(prop=font_prop3, loc='upper left', bbox_to_anchor=(0, 1.1))
    # lines, labels = legend.get_lines(), legend.get_texts()
    # for line, label in zip(lines, labels):
    #     label.set_color(line.get_color())

    # for position in ['top', 'right']:
    #     ax.spines[position].set_visible(False)
    # for position in ['bottom', 'left']:
    #     ax.spines[position].set_visible(True)
    #     ax.spines[position].set_linewidth(2)  # Change the line width here
    plt.savefig(os.path.join(save_dir_i, '2_vs.jpg'), bbox_inches='tight', pad_inches=0.2, dpi=100)
    return

def plot_sample(sample_id):
    pred_json = pred_json_val[sample_id]
    # vid = pred_json['vid']
    vid = domain
    f1 = str(round(pred_json['f1'], 2))
    concept = pred_json['concept1'] + '_' + pred_json['concept2']
        
    global gap
    gap = round(pred_json['shots']) / seg_num
    save_dir_i = os.path.join(save_dir, '_'.join([f1, concept]))
    
    if not os.path.exists(save_dir_i):
        os.mkdir(save_dir_i)
     
    # plot_mr(pred_json, gt_json, save_dir_i, only_one_gt=only_one_gt, pred_num=pred_num, base_json=base_json)
    plot_video(pred_json, save_dir_i, fig_num=fig_num, template=True, gt_only=True)
    plot_video(pred_json, save_dir_i, fig_num=fig_num, template=True)
    plot_vs(pred_json, save_dir_i)
    
    image1 = Image.open(os.path.join(save_dir_i, '0_vid_query_gt.jpg'))
    image3 = Image.open(os.path.join(save_dir_i, '2_vs.jpg'))
    image3 = image3.resize((image1.width, image3.height))
    image4 = Image.open(os.path.join(save_dir_i, '0_vid.jpg'))

    # new_image = Image.new('RGB', (image1.width, image1.height + image2.height + image3.height))
    new_image = Image.new('RGB', (image1.width, image1.height + image3.height + image4.height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image3, (0, image1.height))
    new_image.paste(image4, (0, image1.height + image3.height))
    # new_image.paste(image3, (0, 2 * image1.height))

    # Save the new image
    new_image.save(os.path.join(save_dir_i, 'combined.jpg')) 
    return

if __name__ == "__main__":
    # settings
    fig_num = None  # if None, will be set automatically
    clip_len = 5
    seg_num = 15
    
    only_one_gt = False
    pred_num = 1
    # only_one_gt = False
    # pred_num = None
    
    # load prediction, ground truth, and video
    # video_path="/Users/kevin/dataset/UTE_frames"
    
    for domain in ["P03"]:
    # for domain in ["P01"]:
        pred_json_val = load_jsonl(f"/Users/kevin/univtg/plot/qfvs/{domain}.jsonl")
        video_path=f"/Users/kevin/dataset/UTE_frames/{domain}"
        
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
        fontsize3 = 25
        
        font_path1 = "/Users/kevin/univtg/plot/settings/calibri.ttf"
        font_path2 = "/Users/kevin/univtg/plot/settings/calibri-bold.ttf"
        font_path3 = "/Users/kevin/univtg/plot/settings/calibri-italy.ttf"
        font_prop1 = FontProperties(fname=font_path1, size=fontsize1)
        font_prop2 = FontProperties(fname=font_path2, size=fontsize2)
        font_prop3 = FontProperties(fname=font_path2, size=fontsize3)

        save_dir = os.path.join(os.path.join('/Users/kevin/univtg/plot', 'qfvs', domain))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # SINGLE VIDEO
        # sample_id = 0
        # plot_sample(sample_id)

        for sample_id in range(len(pred_json_val)):
            try:
                plot_sample(sample_id)
            except Exception as e:
                print(f'Error in sample {sample_id} with error {e}')
                continue
