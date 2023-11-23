import os

import argparse
from train import CRFA_train
from ensemble import CRFA_ensemble


parser = argparse.ArgumentParser(description="CRFAEmotionNet")

parser.add_argument("--data", "-d", default="data/", help="data path")
parser.add_argument("--emotion", "-e", default="both", help="Which emotion needs to be trained. By default, both Valence and Arousal are trained")
parser.add_argument("--stream", "-s", default="both", help="Which stream needs to be trained. By default, both static and dynamic streams are trained" )
parser.add_argument("--mode", "-m", default="train", help="train or test")

args = parser.parse_args()


def mkdir_dir(tmp_path):
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
        
        

def check_tmp_path(emotion, stream):
    if emotion == "both" and stream == "both":
        mkdir_dir("tmp/valence/static/")
        mkdir_dir("tmp/valence/dynamic/")
        mkdir_dir("tmp/arousal/static/")
        mkdir_dir("tmp/arousal/dynamic/")
    
    elif emotion == "both":
        mkdir_dir("tmp/valence/{}/".format(stream))
        mkdir_dir("tmp/arousal/{}/".format(stream))
    
    elif stream == "both":
        mkdir_dir("tmp/{}/static/".format(emotion))
        mkdir_dir("tmp/{}/dynamic/".format(emotion))
    else:
        mkdir_dir("tmp/{}/{}/".format(emotion, stream))

if __name__ == "__main__":
    check_tmp_path(args.emotion, args.stream)
    
    
    if args.mode == "train":
        if args.emotion == "valence" or args.emotion == "both":
            if args.stream == "static" or args.stream == "both":
                CRFA_train(args.data, "tmp/valence/", 0, "static")
            if args.stream == "dynamic" or args.stream == "both":
                CRFA_train(args.data, "tmp/valence/", 0, "dynamic")
                
        if args.emotion == "arousal" or args.emotion == "both":
            if args.stream == "static" or args.stream == "both":
                CRFA_train(args.data, "tmp/arousal/", 1, "static")
            if args.stream == "dynamic" or args.stream == "both":
                CRFA_train(args.data, "tmp/arousal/", 1, "dynamic")
                
                
    if args.mode == "test":
        if args.emotion == "valence" or args.emotion == "both":
            CRFA_ensemble("tmp/valence/", 0)
                
        if args.emotion == "arousal" or args.emotion == "both":
            CRFA_ensemble("tmp/arousal/", 1)
    