
source /volume/pt-train/users/wzhang/ghchen/zh/miniconda3/bin/activate router

python test_dynamic.py --datasets big_math --probe_types dirichlet --max_samples 4000 --save_dir /volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/base/probe

python test_dynamic.py --datasets mmlu_train --probe_types dirichlet --max_samples 4000 --save_dir /volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/base/probe

python test_dynamic.py --datasets alpaca_5k --probe_types dirichlet --max_samples 4000 --save_dir /volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/base/probe

python test_dynamic.py  --datasets alpaca_5k mmlu_train big_math --probe_types dirichlet --max_samples 4000

python test_dynamic.py  --datasets  alpaca_5k mmlu_train big_math\
 --probe_types dirichlet --max_samples 12000 \
 --use_input_dependent \
 --dropout 0.5\
 --save_dir /volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/dynamic