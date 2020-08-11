import os
import glob
import subprocess
from subprocess import PIPE
from pathlib import Path
import datetime


img = 640
batch = 32
epoch_num = 10
model = "yolov5m"
resume = True
weights = "./weights/yolov5m.pt"


def get_train_cmd(weights_path, from_resume=False):
    cmdline =  f"python3 train.py"\
               f" --img {img}"\
               f" --batch {batch}"\
               f" --epochs {epoch_num}"\
               f" --data ./config/fdlpd_colab.yaml"\
               f" --cfg ./models/{model}.yaml"\
               f" --cache-images"
    if from_resume:
        cmdline += f" --resume {weights_path}"
    else:
        cmdline += f" --weights {weights_path}"
    return cmdline


def get_latest_exp_dir(search_dir='./runs'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    last_pt = max(last_list, key=os.path.getctime)
    last_dir = os.path.dirname(last_pt)
    return Path(last_dir).parent


train_cmd = get_train_cmd(weights, resume)
print(train_cmd)
train_proc = subprocess.run(train_cmd, shell=True, stdout=PIPE, stderr=PIPE)
train_result = train_proc.stdout
print(train_result)

last_exp_dir = get_latest_exp_dir()
today = datetime.date.today().isoformat()
rsync_cmd = f"rsync -ahv --progress"\
            f" {last_exp_dir}"\
            f" /s3/runs-${today}-${model}-epoch${epoch_num}"
rsync_proc = subprocess.run(rsync_cmd, shell=True, stdout=PIPE, stderr=PIPE)
rsync_result = rsync_proc.stdout
print(rsync_result)

