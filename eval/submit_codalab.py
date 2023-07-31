import os
import pdb
import shutil
import argparse
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str)
parser.add_argument("--nms_thd", type=float)

opt = parser.parse_args()
# pdb.set_trace()

# val = os.path.join(opt.resume.split('/')[:-1], 'inference')
# test = os.path.join(opt.resume.split('/')[:-1],  )

val_jsonl = os.path.join('/'.join(opt.resume.split('/')[:-1]), f'best_qvhighlights_val_preds_nms_thd_{opt.nms_thd}.jsonl')
test_jsonl = os.path.join('/'.join(opt.resume.split('/')[:-1]), f'inference_qvhighlights_test_None_preds_nms_thd_{opt.nms_thd}.jsonl')

name = opt.resume.split('/')[-2]
save_zip_jsonl = os.path.join('/'.join(opt.resume.split('/')[:-1]), f'codalab_{name}_nms_thd_{opt.nms_thd}.zip')
save_val_jsonl = os.path.join('/'.join(opt.resume.split('/')[:-1]), f'hl_val_submission.jsonl')
save_test_jsonl = os.path.join('/'.join(opt.resume.split('/')[:-1]), f'hl_test_submission.jsonl')


shutil.copy(val_jsonl, save_val_jsonl)
shutil.copy(test_jsonl, save_test_jsonl)

newZip = zipfile.ZipFile(save_zip_jsonl, 'w')
newZip.write(save_val_jsonl, 'hl_val_submission.jsonl', compress_type=zipfile.ZIP_DEFLATED)
newZip.write(save_test_jsonl, 'hl_test_submission.jsonl', compress_type=zipfile.ZIP_DEFLATED)
newZip.close()