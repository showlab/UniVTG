import os
import pdb
import time
import torch
import gradio as gr
import numpy as np
import argparse
import subprocess
from run_on_video import clip, vid2clip, txt2clip

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='./tmp')
parser.add_argument('--resume', type=str, default='./results/omni/model_best.ckpt')
parser.add_argument("--gpu_id", type=int, default=2)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

#################################
model_version = "ViT-B/32"
output_feat_size = 512
clip_len = 2
overwrite = True
num_decoding_thread = 4
half_precision = False

clip_model, _ = clip.load(model_version, device=args.gpu_id, jit=False)

import logging
import torch.backends.cudnn as cudnn
from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def load_model():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse(args)
    # pdb.set_trace()
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    return model

vtg_model = load_model()

def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def load_data(save_dir):
    vid = np.load(os.path.join(save_dir, 'vid.npz'))['features'].astype(np.float32)
    txt = np.load(os.path.join(save_dir, 'txt.npz'))['features'].astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    clip_len = 2
    ctx_l = vid.shape[0]

    timestamp =  ( (torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    if True:
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

    src_vid = vid.unsqueeze(0).cuda()
    src_txt = txt.unsqueeze(0).cuda()
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

def forward(model, save_dir, query):
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(save_dir)
    src_vid = src_vid.cuda(args.gpu_id)
    src_txt = src_txt.cuda(args.gpu_id)
    src_vid_mask = src_vid_mask.cuda(args.gpu_id)
    src_txt_mask = src_txt_mask.cuda(args.gpu_id)

    model.eval()
    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)
    
    # prepare the model prediction
    pred_logits = output['pred_logits'][0].cpu()
    pred_spans = output['pred_spans'][0].cpu()
    pred_saliency = output['saliency_scores'].cpu()

    # prepare the model prediction
    pred_windows = (pred_spans + timestamp) * ctx_l * clip_len
    pred_confidence = pred_logits
    
    # grounding
    top1_window = pred_windows[torch.argmax(pred_confidence)].tolist()
    top5_values, top5_indices = torch.topk(pred_confidence.flatten(), k=5)
    top5_windows = pred_windows[top5_indices].tolist()
    
    # print(f"The video duration is {convert_to_hms(src_vid.shape[1]*clip_len)}.")
    q_response = f"For query: {query}"

    mr_res =  " - ".join([convert_to_hms(int(i)) for i in top1_window])
    mr_response = f"The Top-1 interval is: {mr_res}"
    
    hl_res = convert_to_hms(torch.argmax(pred_saliency) * clip_len)
    hl_response = f"The Top-1 highlight is: {hl_res}"
    return '\n'.join([q_response, mr_response, hl_response])
    
def extract_vid(vid_path, state):
    history = state['messages']
    vid_features = vid2clip(clip_model, vid_path, args.save_dir)
    history.append({"role": "user", "content": "Finish extracting video features."}) 
    history.append({"role": "system", "content": "Please Enter the text query."}) 
    chat_messages = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history),2)]
    return '', chat_messages, state

def extract_txt(txt):
    txt_features = txt2clip(clip_model, txt, args.save_dir)
    return

def download_video(url, save_dir='./examples', size=768):
    save_path = f'{save_dir}/{url}.mp4'
    cmd = f'yt-dlp -S ext:mp4:m4a --throttled-rate 5M -f "best[width<={size}][height<={size}]" --output {save_path} --merge-output-format mp4 https://www.youtube.com/embed/{url}'
    if not os.path.exists(save_path):
        try:
            subprocess.call(cmd, shell=True)
        except:
            return None
    return save_path

def get_empty_state():
    return {"total_tokens": 0, "messages": []}

def submit_message(prompt, state):
    history = state['messages']

    if not prompt:
        return gr.update(value=''), [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history)-1, 2)], state

    prompt_msg = { "role": "user", "content": prompt }
    
    try:
        history.append(prompt_msg)
        # answer = vlogger.chat2video(prompt)
        # answer = prompt
        extract_txt(prompt)
        answer = forward(vtg_model, args.save_dir, prompt)
        history.append({"role": "system", "content": answer}) 

    except Exception as e:
        history.append(prompt_msg)
        history.append({
            "role": "system",
            "content": f"Error: {e}"
        })

    chat_messages = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history)-1, 2)]
    return '', chat_messages, state


def clear_conversation():
    return gr.update(value=None, visible=True), gr.update(value=None, interactive=True), None, gr.update(value=None, visible=True), get_empty_state()


def subvid_fn(vid):
    save_path = download_video(vid)
    return gr.update(value=save_path)


css = """
      #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
      #video_inp {min-height: 100px}
      #chatbox {min-height: 100px;}
      #header {text-align: center;}
      #hint {font-size: 1.0em; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2em; }
      """

with gr.Blocks(css=css) as demo:
    
    state = gr.State(get_empty_state())


    with gr.Column(elem_id="col-container"):
        gr.Markdown("""## 🤖️ UniVTG: Towards Unified Video-Language Temporal Grounding
                    Given a video and text query, return relevant window and highlight.
                    https://github.com/showlab/UniVTG/""",
                    elem_id="header")

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="video_input")
                gr.Markdown("👋 **Step1**: Select a video in Examples (bottom) or input youtube video_id in this textbox, *e.g.* *G7zJK6lcbyU* for https://www.youtube.com/watch?v=G7zJK6lcbyU", elem_id="hint")
                with gr.Row():
                    video_id = gr.Textbox(value="", placeholder="Youtube video url", show_label=False)
                    vidsub_btn = gr.Button("(Optional) Submit Youtube id")

            with gr.Column():
                vid_ext = gr.Button("Step2: Extract video feature, may takes a while")
                # vlog_outp = gr.Textbox(label="Document output", lines=40)
                total_tokens_str = gr.Markdown(elem_id="total_tokens_str")
                
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_message = gr.Textbox(show_label=False, placeholder="Enter text query and press enter", visible=True).style(container=False)
                btn_submit = gr.Button("Step3: Enter your text query")
                btn_clear_conversation = gr.Button("🔃 Clear")

        examples = gr.Examples(
            examples=[
                # ["./examples/youtube.mp4"], 
                ["./examples/charades.mp4"], 
                # ["./examples/ego4d.mp4"],
            ],
            inputs=[video_inp],
        )

    gr.HTML('''<br><br><br><center>You can duplicate this Space to skip the queue:<a href="https://huggingface.co/spaces/anzorq/chatgpt-demo?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a><br></center>''')

    btn_submit.click(submit_message, [input_message, state], [input_message, chatbot])
    input_message.submit(submit_message, [input_message, state], [input_message, chatbot])
    # btn_clear_conversation.click(clear_conversation, [], [input_message, video_inp, chatbot, vlog_outp, state])
    btn_clear_conversation.click(clear_conversation, [], [input_message, video_inp, chatbot, state])
    vid_ext.click(extract_vid, [video_inp, state], [input_message, chatbot])
    vidsub_btn.click(subvid_fn, [video_id], [video_inp])

    demo.load(queur=False)


demo.queue(concurrency_count=10)
demo.launch(height='800px', server_port=2253, debug=True, share=True)
