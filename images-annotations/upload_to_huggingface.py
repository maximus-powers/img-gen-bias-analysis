import os
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Image

def create_huggingface_dataset():
    base_dir = ""
    
    model_configs = {
        "dall_e_3": {
            "csv": "gpt.csv",
            "model_name": "dall-e-3"
        },
        "flux_dev": {
            "csv": "flux.csv", 
            "model_name": "fal-ai/flux/dev"
        },
        "gemini_imagen_4_0": {
            "csv": "gemini.csv",
            "model_name": "imagen-4.0-generate-001"
        },
        "grok_2_image": {
            "csv": "grok.csv",
            "model_name": "grok-2-image"
        },
        "sdxl_turbo": {
            "csv": "sdxl.csv",
            "model_name": "stabilityai/sdxl-turbo"
        }
    }
    
    dataset_dict = {}
    
    for model_name, config in model_configs.items():
        model_dir = os.path.join(base_dir, model_name)
        csv_path = os.path.join(model_dir, config["csv"])
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file {csv_path} not found, skipping {model_name}")
            continue
            
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        processed_data = []
        for _, row in df.iterrows():
            relative_path = row['image_file']
            # skip missing img or emptry rows
            if pd.isna(relative_path) or relative_path == '' or str(relative_path).strip() == '':
                continue
            image_path_parts = str(relative_path)
            if image_path_parts.startswith("/"):
                image_path_parts = image_path_parts[1:]
                
            full_image_path = os.path.join(model_dir, image_path_parts)
            if os.path.exists(full_image_path):
                processed_data.append({
                    'image': full_image_path,
                    'gender': row.get('gender', ''),
                    'race': row.get('race', ''),
                    'category': row['category'],
                    'setting': row['setting'],
                    'filepath': image_path_parts 
                })
            else:
                print(f"Warning: Image not found: {full_image_path}")
        
        print(f"Found {len(processed_data)} valid images for {model_name}")
        
        # create dataset split for model
        if processed_data:
            features = Features({
                'image': Image(),
                'gender': Value('string'),
                'race': Value('string'),
                'category': Value('string'),
                'setting': Value('string'),
                'filepath': Value('string'),
            })
            dataset = Dataset.from_list(processed_data, features=features)
            dataset_dict[model_name] = dataset
    
    if dataset_dict:
        full_dataset = DatasetDict(dataset_dict)
        return full_dataset
    else:
        print("No valid datasets avail.")
        return None



dataset = create_huggingface_dataset()

if dataset:
    print(f"Dataset created with splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} samples")
    try:
        dataset.push_to_hub(
            repo_id="",
            token="", 
            private=False, 
        )
    except Exception as e:
        print(f"Upload failed: {e}")
else:
    print("Failed to create dataset!")