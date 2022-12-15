"""
Code by Sho
github: RC-Sho0
:)))
"""

import os
os.system("pip install --upgrade segments-ai")
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset

def clone_data(api_key = "5d62708313ee28458ba0b93c8028aa98819cad77", data_source="hoangsonvothanh/4VNFood",output_type='semantic-color', version="v1"):
  # Initialize a SegmentsDataset from the release file
  client = SegmentsClient(api_key)
  release = client.get_release(data_source, version) # Alternatively: release = 'flowers-v1.0.json'
  dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])
  export_dataset(dataset, export_format=output_type)

def create_fol(path):
  os.system(f"mkdir {path}/ {path}/pho {path}/comtam {path}/banhmi {path}/banhtrang")
  os.system(f"mkdir {path}/pho/images/ {path}/comtam/images/ {path}/banhmi/images/ {path}/banhtrang/images/")
  os.system(f"mkdir {path}/pho/labels/ {path}/comtam/labels/ {path}/banhmi/labels/ {path}/banhtrang/labels/")


def extract2fol(path, new_path, output_type="semantic_colored"):
  create_fol(new_path)
  if output_type=="semantic_colored":
    string="_label_ground-truth_semantic_colored.png"
  elif output_type=="semantic":
    string="_label_ground-truth.png"
  
  for item in os.listdir(path):
    if string in item:
      name = item.split(string)[0]
      # print(name)
      if "bm_" in name:
        os.system(f"mv {path}/{name}.jpg {new_path}/banhmi/images/{name}.jpg")
        os.system(f"mv {path}/{name}{string}  {new_path}/banhmi/labels/{name}.png")
      elif "ct_" in name:
        os.system(f"mv {path}/{name}.jpg {new_path}/comtam/images/{name}.jpg")
        os.system(f"mv {path}/{name}{string}  {new_path}/comtam/labels/{name}.png")
      elif "btn_" in name:
        os.system(f"mv {path}/{name}.jpg {new_path}/banhtrang/images/{name}.jpg")
        os.system(f"mv {path}/{name}{string}  {new_path}/banhtrang/labels/{name}.png")
      else:
        os.system(f"mv {path}/{name}.jpg {new_path}/pho/images/pho_{name}.jpg")
        os.system(f"mv {path}/{name}{string}  {new_path}/pho/labels/pho_{name}.png")
  print("============> All Done")