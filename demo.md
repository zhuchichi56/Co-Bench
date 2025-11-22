llf) root@cgh-gpu-0:/volume/pt-train/users/wzhang/ghchen/zh/Cobench/CoBench/src# python run.py 
17:20:50 | INFO     | 🚀 Starting Complete Pipeline Evaluation
17:20:50 | INFO     | 📊 Datasets: ['mmlu']
17:20:50 | INFO     | 🤖 Router Type: self_questioning
17:20:50 | INFO     | 📝 Step 1: Evaluating models on datasets
Evaluating mmlu...
DEBUG: Model name/path: 
DEBUG: Has chat_template: False
17:20:50 | INFO     | Prompt list's first 1 element: <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the multiple choice question by selecting the correct option. Put your final answer in \boxed{}.

Question: The following is a multiple choice question about Abstract Algebra. Please select the correct answer.

Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

Options:
A. 0
B. 4
C. 2
D. 6

Please choose the correct answer from A, B, C, or D.
Answer:<|im_end|>
<|im_start|>assistant

17:20:50 | INFO     | Chunk 0 size: 20
17:20:50 | INFO     | Chunk 1 size: 20
17:21:01 | INFO     | Prompt list's first 1 element: <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the multiple choice question by selecting the correct option. Put your final answer in \boxed{}.

Question: The following is a multiple choice question about Abstract Algebra. Please select the correct answer.

Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

Options:
A. 0
B. 4
C. 2
D. 6

Please choose the correct answer from A, B, C, or D.
Answer:<|im_end|>
<|im_start|>assistant

17:21:01 | INFO     | Chunk 0 size: 20
17:21:01 | INFO     | Chunk 1 size: 20
mmlu - Small model: 0.150, Large model: 0.825
17:21:27 | INFO     | 🎯 Step 2: Generating router scores using self_questioning router
17:21:27 | INFO     | 🤔 Setting up Self-Questioning Router
17:21:27 | INFO     | Prompt list's first 1 element: <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: The following is a multiple choice question about Abstract Algebra. Please select the correct answer.

Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

Options:
A. 0
B. 4
C. 2
D. 6

Please choose the correct answer from A, B, C, or D.

On a scale of 0-100, how easy is this question for you to answer correctly? Rate 0 for very difficult, 100 for very easy. Please write your answer as \boxed{number}.<|im_end|>
<|im_start|>assistant

17:21:27 | INFO     | Chunk 0 size: 20
17:21:27 | INFO     | Chunk 1 size: 20
17:21:27 | SUCCESS  | ✅ Generated router scores for mmlu: 40 samples
17:21:27 | INFO     | 📊 Step 3: Calculating evaluation metrics
Evaluating metrics for mmlu...
mmlu: AUROC=0.4387, LPM=0.1718, HPM=0.0000

Results saved to: metric_results
17:21:27 | SUCCESS  | 🎉 Pipeline Evaluation Complete!
17:21:27 | INFO     | 📈 mmlu: Average router score = 0.466
17:21:27 | INFO     | 🏆 mmlu: AUROC = N/A
(llf) root@cgh-gpu-0:/volume/pt-train/users/wzhang/ghchen/zh/Cobench/CoBench/src# python run.py 
17:24:28 | INFO     | 🚀 Starting Complete Pipeline Evaluation
17:24:28 | INFO     | 📊 Datasets: ['mmlu']
17:24:28 | INFO     | 🤖 Router Type: self_questioning
17:24:28 | INFO     | 📝 Step 1: Evaluating models on datasets
Evaluating mmlu...
DEBUG: Model name/path: 
DEBUG: Has chat_template: False
17:24:28 | INFO     | Prompt list's first 1 element: <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the multiple choice question by selecting the correct option. Put your final answer in \boxed{}.

Question: The following is a multiple choice question about Abstract Algebra. Please select the correct answer.

Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

Options:
A. 0
B. 4
C. 2
D. 6

Please choose the correct answer from A, B, C, or D.
Answer:<|im_end|>
<|im_start|>assistant

17:24:28 | INFO     | Chunk 0 size: 20
17:24:28 | INFO     | Chunk 1 size: 20
17:24:38 | INFO     | Prompt list's first 1 element: <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the multiple choice question by selecting the correct option. Put your final answer in \boxed{}.

Question: The following is a multiple choice question about Abstract Algebra. Please select the correct answer.

Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

Options:
A. 0
B. 4
C. 2
D. 6

Please choose the correct answer from A, B, C, or D.
Answer:<|im_end|>
<|im_start|>assistant

17:24:38 | INFO     | Chunk 0 size: 20
17:24:38 | INFO     | Chunk 1 size: 20
mmlu - Small model: 0.775, Large model: 0.825
17:25:04 | INFO     | 🎯 Step 2: Generating router scores using self_questioning router
17:25:04 | INFO     | 🤔 Setting up Self-Questioning Router
17:25:05 | INFO     | Prompt list's first 1 element: <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: The following is a multiple choice question about Abstract Algebra. Please select the correct answer.

Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

Options:
A. 0
B. 4
C. 2
D. 6

Please choose the correct answer from A, B, C, or D.

On a scale of 0-100, how easy is this question for you to answer correctly? Rate 0 for very difficult, 100 for very easy. Please write your answer as \boxed{number}.<|im_end|>
<|im_start|>assistant

17:25:05 | INFO     | Chunk 0 size: 20
17:25:05 | INFO     | Chunk 1 size: 20
17:25:05 | SUCCESS  | ✅ Generated router scores for mmlu: 40 samples
17:25:05 | INFO     | 📊 Step 3: Calculating evaluation metrics
Evaluating metrics for mmlu...
mmlu: AUROC=0.7292, LPM=0.7963, HPM=0.9287

Results saved to: metric_results
17:25:05 | SUCCESS  | 🎉 Pipeline Evaluation Complete!
17:25:05 | INFO     | 📈 mmlu: Average router score = 0.466
17:25:05 | INFO     | 🏆 mmlu: AUROC = N/A