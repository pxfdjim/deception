import os, sys, ast
from pathlib import Path

pwd = Path(__file__).parent

# DECEPTION 目录本身就是项目根目录
# utils/, dataloaders/ 等都在这一层
project_dir = pwd

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(project_dir))

# 导入工具类
from utils.tools import DotDict

arg_blacklist=["num_labels", "text", "audio", "visual"]

def __arg_val_type(val_str):
    """
    完善的类型转换函数，能够正确解析log_str中的所有参数值
    """
    val_str = val_str.strip()
    
    # 处理布尔值
    if val_str.lower() in ('true', 'false'):
        return val_str.lower() == 'true'
    
    # 处理None值
    if val_str.lower() == 'none':
        return None
    
    # 处理列表和字典等复杂结构
    if val_str.startswith('[') and val_str.endswith(']') or \
       val_str.startswith('{') and val_str.endswith('}'):
        try:
            return ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            pass
    
    # 处理数字
    # 整数
    try:
        if '.' not in val_str and 'e' not in val_str.lower():
            return int(val_str)
    except ValueError:
        pass
    
    # 浮点数
    try:
        return float(val_str)
    except ValueError:
        pass
    
    # 默认返回字符串
    return val_str

def parse_args_from_log_str(log_str, sep="-"):

    # 尝试修复"k:v1-v2-v3"->[k,v1][v2][v3]...->{k1:"v1-v2-v3"}
    arg_kv_list=[arg_kv.split(':') for arg_kv in log_str.split(sep)]
    args_dict={}
    fix_list=[]
    for i in list(range(len(arg_kv_list)))[::-1]:
        arg_kv=arg_kv_list[i]
        # print(len(arg_kv),arg_kv)
        if any([blacklist_arg in arg_kv_list[i][0] for blacklist_arg in arg_blacklist]):continue
        if len(arg_kv) == 1:
            fix_list.append(arg_kv[0])
        elif len(arg_kv) == 2 and fix_list:
            fix_list.append(arg_kv[1])#[k, v][1]
            args_dict[arg_kv[0]]=__arg_val_type(sep.join(reversed(fix_list)))
            fix_list.clear()
        else:
            args_dict[arg_kv[0]]=__arg_val_type(arg_kv[1])


    return DotDict(args_dict)

if __name__ == '__main__':
    print(parse_args_from_log_str("gpus:[0]__run_type:0__model_id:10__pretrained:False__load_epoch:None__save_interval:5__dataset_root:/Data/WSD_Data/__dataset_name:MDPE__video_num:200__feature_type:I3D__inp_feat_num:6656__out_feat_num:1024__class_num:2__scale_factor:20.0__T:0.5__w:0.5__mod_dims:5120,768,768__lambda_align:0.1__mmd_sigma:1.0__batch_size:8__lr:5e-05__weight_decay:0.0005__dropout:0.6__seed:42__max_epoch:100__mu_num:8__mu_queue_len:5__em_iter:2__lambda_a:0.1__lambda_b:0.1__lambda_s:0.5__warmup_epoch:1000__class_threshold:0.1__start_threshold:0.001__end_threshold:0.04__threshold_interval:0.002__decay_type:0__changeLR_list:[80, 1000]__mode:grid__grid_out_feat_num:1024,2048__grid_lr:0.00005,0.0001,0.0005__grid_batch_size:8,16,12__grid_max_epoch:100,200__grid_lambda_a:0.1,0.2,0.5__grid_lambda_b:0.1,0.2,0.5__grid_lambda_s:0.5,1.0,1.5__grid_w:0.3,0.5,0.7__grid_T:0.1,0.2,0.5__gpu_ids:2", sep="__"))