import yaml

data = {
  'train': 'dataset/train',
  'val': 'dataset/valid',
  'nc': 1,
  'names': ['document'] # Thêm dòng này để chỉ định sử dụng GPU 0 và 1
}

with open('config.yaml', 'w') as yaml_file:
  yaml.dump(data, yaml_file, default_flow_style=False)