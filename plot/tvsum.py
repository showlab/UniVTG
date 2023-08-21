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

def plot_video(pred_json, save_dir_i, fig_num=None, template=True):
    duration = pred_json['duration']-clip_len
    t_min, t_max = 0, duration
    x = np.arange(t_min, t_max, clip_len)
    
    if fig_num is None:
        fig_num = round(duration / gap)
    
    fig, axs = plt.subplots(nrows=1, ncols=fig_num, figsize=(40, 20), gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

    vid_exists = os.path.exists(os.path.join(video_path, pred_json['vid'] + '.mp4'))
    assert vid_exists, f"Video {pred_json['vid']} does not exist!"
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
    query = pred_json['title']
    text1 = 'VIDEO TITLE: '
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
 
def plot_hl(pred_json, save_dir_i,  base_json=None):
    pred_saliency = np.array(pred_json['pred'])
    pred_saliency = norm(pred_saliency)
    
    gt_saliency = np.array(pred_json['gt'])
    gt_saliency = norm(gt_saliency)
    
    duration = pred_json['duration']
    t_min, t_max = 0, duration
    x = np.arange(t_min, t_max, clip_len)
    x = x[:len(pred_saliency)]
    
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

    offset = pred_json['duration'] * 0.01
    # ax.set_xticks(np.arange(gap/2, gt_json['duration'] - gap/2, gap))
    ax.text(offset, -0.2, '0.0', va='center', ha='center', color="black", fontproperties=font_prop1)
    ax.text(pred_json['duration']-clip_len-offset, -0.2, f'{pred_json["duration"]:.1f}', va='center', ha='center',  color="black", fontproperties=font_prop1)
    for i in np.arange(0, pred_json['duration'] + gap/2, gap)[1:-1]:
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
    vid = pred_json['vid']
        
    global gap
    gap = round(pred_json['duration']) / seg_num
    save_dir_i = os.path.join(save_dir, vid)
    
    if not os.path.exists(save_dir_i):
        os.mkdir(save_dir_i)
     
    # plot_mr(pred_json, gt_json, save_dir_i, only_one_gt=only_one_gt, pred_num=pred_num, base_json=base_json)
    plot_video(pred_json, save_dir_i, fig_num=fig_num, template=True)
    plot_hl(pred_json, save_dir_i)
    
    image1 = Image.open(os.path.join(save_dir_i, '0_vid_query.jpg'))
    # image2 = Image.open(os.path.join(save_dir_i, '1_mr.jpg'))
    # image2 = image2.resize(image1.size)
    image3 = Image.open(os.path.join(save_dir_i, '2_hl.jpg'))
    image3 = image3.resize(image1.size)

    # new_image = Image.new('RGB', (image1.width, image1.height + image2.height + image3.height))
    new_image = Image.new('RGB', (image1.width, image1.height + image3.height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image3, (0, image1.height))
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
    video_path="/Users/kevin/dataset/tvsum/ydata-tvsum50-v1_1/video"
    
    for domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]:
    # domain = 'BK'
        pred_json_val = load_jsonl(f"/Users/kevin/univtg/plot/tvsum/{domain}.jsonl")
        
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

        save_dir = os.path.join(os.path.join('/Users/kevin/univtg/plot', 'tvsum', domain))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # SINGLE VIDEO
        sample_id = 0
        try:
            plot_sample(sample_id)
        except:
            continue
