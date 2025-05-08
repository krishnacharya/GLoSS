from pathlib import Path

def project_root():
    current_dir = Path(__file__).absolute().parent.parent.parent
    return current_dir

def data_root():
    res = project_root()/"data"
    res.mkdir(parents=True, exist_ok=True)
    return res

def raw_data_root():
    res = data_root() / "raw"
    res.mkdir(parents=True, exist_ok=True)
    return res

def processed_data_root() -> Path:
    res = data_root() / "processed"
    res.mkdir(parents=True, exist_ok=True)
    return res

def processed_data_dir(dataset:str):
    res = processed_data_root()
    res = res / dataset
    res.mkdir(parents=True, exist_ok = True)
    return res

def raw_data_ds(dataset):
    res = raw_data_root()
    res = res / dataset
    res.mkdir(parents=True, exist_ok = True)
    return res

def uid_to_group_dir(dataset:str):
    res = processed_data_root() / dataset / "uid_to_group"
    res.mkdir(parents=True, exist_ok=True)
    return res

def output_root():
    res =  project_root() / "output" 
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_src_dir():
    return project_root() / "src" 

def get_hfdata_dir():
    res = project_root() / "hfdataset"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_reviews_raw2018_dir():
    res = project_root() / "reviews" / "raw-2018"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_reviews_raw2014_dir():
    res = project_root() / "reviews" / "raw-2014"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_analysis_dir():
    return get_src_dir() / "analysis" 

def get_training_scripts_dir():
    return get_src_dir() / "training"

def get_eval_scripts_dir():
    return get_src_dir() / "eval"

def get_bm25_indexes_dir():
    res = project_root() / "bm25_indexes"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_llama_modelsave_dir():
    res = project_root() / "llama_modelsave"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_gen_dir():
    res = project_root() / "genqueries"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_gen_dir_model(dataset:str, model_name:str):
    res = get_gen_dir_dataset(dataset) / model_name
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_gen_dir_dataset(dataset:str):
    res = get_gen_dir() / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res

def ft_configs_dir():
    res = project_root() / "ft_configs"
    res.mkdir(parents=True, exist_ok=True)
    return res