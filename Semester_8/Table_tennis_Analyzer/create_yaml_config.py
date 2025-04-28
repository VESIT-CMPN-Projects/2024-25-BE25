# create_yaml_config.py

import os

def create_yaml_config(dataset_path, class_names):
    """Create YAML configuration file for YOLOv8 training"""
    
    config_content = f"""
# Table Tennis Detection Dataset
path: {dataset_path}  # dataset root directory
train: images/train  # train images (relative to 'path')
val: images/val  # validation images (relative to 'path')

# Classes
names:
"""
    for idx, name in enumerate(class_names):
        config_content += f"  {idx}: {name}\n"

    config_path = os.path.join(dataset_path, "table_tennis.yaml")
    with open(config_path, "w") as f:
        f.write(config_content.strip())
    
    print(f"✅ Created YAML config at: {config_path}")
    return config_path

if __name__ == "__main__":
    dataset_path = "dataset"
    class_names = ["table", "net", "ball"]
    yaml_path = create_yaml_config(dataset_path, class_names)
