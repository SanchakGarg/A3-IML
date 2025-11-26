import pandas as pd
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

def subsample_artemis(input_csv='artemis-dataset.csv', output_dir='wikiart_subsample', target_count=10000, size=(224, 224)):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Original dataset size: {len(df)}")
    
    # 1. Filter nulls
    df = df.dropna(subset=['utterance'])
    
    # 2. Determine dominant emotion for each painting
    print("Determining dominant emotion for each painting...")
    painting_emotions = df.groupby(['art_style', 'painting'])['emotion'].agg(lambda x: x.mode().iloc[0]).reset_index()
    
    print(f"Total unique paintings: {len(painting_emotions)}")
    
    # 3. Balance unique paintings by emotion
    emotions = painting_emotions['emotion'].unique()
    print(f"Emotions found: {emotions}")
    
    n_emotions = len(emotions)
    samples_per_emotion = target_count // n_emotions
    print(f"Targeting approximately {samples_per_emotion} unique paintings per emotion.")
    
    sampled_paintings_list = []
    for emotion in emotions:
        emotion_group = painting_emotions[painting_emotions['emotion'] == emotion]
        if len(emotion_group) >= samples_per_emotion:
            sampled_paintings_list.append(emotion_group.sample(n=samples_per_emotion, random_state=42))
        else:
            print(f"Warning: Not enough paintings for {emotion} (found {len(emotion_group)}, wanted {samples_per_emotion}). Taking all.")
            sampled_paintings_list.append(emotion_group)
            
    selected_paintings_df = pd.concat(sampled_paintings_list)
    print(f"Selected unique paintings: {len(selected_paintings_df)}")
    print("Painting distribution by dominant emotion:")
    print(selected_paintings_df['emotion'].value_counts())
    
    # 4. Retrieve ALL captions for the selected paintings
    final_df = df.merge(selected_paintings_df[['art_style', 'painting']], on=['art_style', 'painting'], how='inner')
    
    print(f"Final dataset size (captions): {len(final_df)}")
    
    # 5. Copy images
    print(f"Processing images to {output_dir} (Resizing to {size})...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    copied_count = 0
    missing_count = 0
    
    for idx, row in selected_paintings_df.iterrows():
        art_style = row['art_style']
        painting = row['painting']
        
        src_path = os.path.join('wikiart', art_style, f"{painting}.jpg")
        
        dest_dir = os.path.join(output_dir, art_style)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            
        dest_path = os.path.join(dest_dir, f"{painting}.jpg")
        
        if os.path.exists(src_path):
            if not os.path.exists(dest_path):
                try:
                    with Image.open(src_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_resized = img.resize(size, Image.Resampling.LANCZOS)
                        img_resized.save(dest_path)
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")
                    continue
            print(f"Image copied: {copied_count}")
            copied_count += 1

        else:
            missing_count += 1
            
    print(f"Finished copying. Copied: {copied_count}, Missing: {missing_count}")
    
    final_df.to_csv('artemis_subsample.csv', index=False)
    print("Saved artemis_subsample.csv")

def get_processed_image(file_path, size=(224, 224)):
    try:
        with Image.open(file_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            return img_array
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    subsample_artemis()
