import os
import glob

data_root = "/home/ab3379/work/data/MMEW_Final/Macro_Expression"
emotion_map = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "sadness": 4, "surprise": 5}

subjects = sorted(os.listdir(data_root))
# Subject-independent split: e.g., first 22 subjects for training, remaining for testing
train_subjects = subjects[:22]
test_subjects = subjects[22:]

os.makedirs("./annotation", exist_ok=True)

def create_list(subj_list, out_file):
    records = []
    for subj in subj_list:
        subj_path = os.path.join(data_root, subj)
        if not os.path.isdir(subj_path):
            continue
        for emotion, label in emotion_map.items():
            emo_path = os.path.join(subj_path, emotion)
            if os.path.isdir(emo_path):
                # Check if it contains images directly or subfolders
                frames = glob.glob(os.path.join(emo_path, "*.jpg"))
                if len(frames) > 0:
                    records.append(f"{emo_path} {len(frames)} {label}")
                else:
                    # If nested clips exist
                    for clip in os.listdir(emo_path):
                        clip_path = os.path.join(emo_path, clip)
                        clip_frames = glob.glob(os.path.join(clip_path, "*.jpg"))
                        if len(clip_frames) > 0:
                            records.append(f"{clip_path} {len(clip_frames)} {label}")
    
    with open(out_file, "w") as f:
        f.write("\n".join(records) + "\n")
    print(f"Generated {out_file} with {len(records)} samples.")

create_list(train_subjects, "./annotation/MMEW_macro_train.txt")
create_list(test_subjects, "./annotation/MMEW_macro_test.txt")