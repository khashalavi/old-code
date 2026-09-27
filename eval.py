from dataclasses import dataclass, field          # for defining typed argument containers (ModelArguments/DataArguments)
from tqdm import tqdm                              # progress bar for the evaluation loop
from typing import Optional                        # type hints for optional dataclass fields
import transformers                                 # HuggingFace library: tokenizers, models, argument parsing
import torch                                        # PyTorch: tensors, model inference, device handling
import json                                         # reading/writing JSON (prompt text, output results)
from torch.utils.data import DataLoader             # batches the dataset for evaluation
import os, json, random, pickle,re                  # os: paths/dirs; random: sampling test subset; re: regex for step-type parsing
import numpy as np                                  # imported but not directly used below
from huggingface_hub import login                   # authenticates with the HF Hub (needed for gated models like Llama)
from load_data.preprocess import GSMData, AquaData, StrategyQAData, StrategyQAData_Ours, CommonsenseQAData_Ours, TruthfulQAData_Ours  # dataset-specific loaders/answer-checkers
from model.generation_utils import make_sparse_mask  # builds a custom sparse attention mask when enabled
from model.load_model import MyAutoModelForCausalLM   # custom model loader wrapping HF's AutoModelForCausalLM (adds soft-prompt/sparse support)
from model.peft_model import MyPeftModelForCausalLM   # custom PEFT (LoRA) wrapper for loading adapter weights

INVALID_ANS = "[invalid]"                            # sentinel string used elsewhere to mark unparsable answers (not used directly in this file)


@dataclass
class ModelArguments:
    # CLI/config arguments describing the model to evaluate and how to load/run it.
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "pre-trained language model name on Huggingface, or path to a checkpoint."},)   # checkpoint to evaluate (base model or fine-tuned dir)
    base_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "pre-trained language model name on Huggingface, or path to a checkpoint."},)   # underlying base model when using PEFT (LoRA/prompt-tuning) adapters
    cache_dir: Optional[str] = field(default=None)                      # HF cache directory for downloaded weights/tokenizers
    output_dir: Optional[str] = field(default='./save_data', metadata={"help": "Path to the output dir."})  # where evaluation results get written
    save_result: Optional[bool] = field(default=True)                   # whether to dump generated outputs to a JSON file
    max_length: Optional[int] = field(default=512)                      # max total sequence length for generation
    decoding_scheme: Optional[str] = field(default="greedy")            # decoding strategy label (not actually branched on below; generate() uses HF defaults)
    load_in_8bit: Optional[bool] = field(default=False)                 # load model weights in 8-bit precision to save memory
    use_calculator: Optional[bool] = field(default=False)               # flag for enabling a calculator tool (not used in this file)
    add_soft_prompts: Optional[bool] = field(default=False)             # whether the model was trained with learned "soft prompt" embeddings/tokens
    add_hard_prompts: Optional[bool] = field(default=False)             # whether to inject fixed textual ("hard") prompt snippets instead
    only_at_front: Optional[bool] = field(default=False)                # restrict prompt insertion to only the front of the input (duplicated field, see line below)
    use_sparse_attention: Optional[bool] = field(default=False)         # enable the custom sparse attention mask during generation
    parameter_efficient_mode: Optional['str'] = field(default='none',
        metadata={"choices": ["none", "prompt-tuning", "lora", "lora+prompt-tuning"]})  # which PEFT method (if any) was used to fine-tune the model
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Require for llama family."})  # HF auth token needed to download gated Llama weights
    enable_cpu_offload: Optional[bool] = field(default=False)           # flag for offloading model layers to CPU (declared but not used below)
    only_at_front: Optional[bool] = field(default=False)                # NOTE: duplicate of the field above; this second definition silently overrides the first
    plan_first: Optional[bool] = field(default=False)                   # whether the model generates a "plan" before the answer
    plan_only: Optional[bool] = field(default=False)                    # whether the model outputs only the plan (no final answer)
    extract_step_type_tokens: Optional[str] = field(default="none",
        metadata={"choices": ["none", "+-*/", "vae", "tf-idf", "k-means","memory","other"]})  # strategy for classifying/tagging reasoning "step types" in output
    num_plan_types: Optional[int] = field(default=5)                    # number of distinct plan/step categories expected

@dataclass
class DataArguments:
    # CLI/config arguments describing the evaluation dataset and batching/sampling behavior.
    dataset: str = field(default=None, metadata={"help": "dataset name."})            # which dataset to evaluate on (selects a data_class below)
    batch_size: Optional[int] = field(default=16)                                     # evaluation batch size
    use_demonstrations: Optional[bool] = field(default=False)                         # whether to include few-shot demonstrations in the prompt
    demo_selection: Optional[str] = field(default="uniform")                          # strategy for picking few-shot demonstrations
    candidate_size: Optional[int] = field(default=100)                                # pool size to draw demonstrations from
    k_shot: Optional[int] = field(default=4)                                          # number of few-shot examples per prompt
    seed: Optional[int] = field(default=42)                                           # random seed for reproducibility (declared; the code below hardcodes 42 separately)
    num_test: Optional[int] = field(default=1000)                                     # max number of test examples to evaluate on
    prompt_template: Optional[str] = field(default=None)                              # optional template string used to format each example's prompt
    embedding_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")   # model used to compute embeddings (e.g. for demonstration selection)


def main():

    # Parse the two dataclasses above from CLI arguments (transformers' HfArgumentParser turns argparse-style flags into these objects).
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    login(token=model_args.hf_hub_token)              # authenticate with HuggingFace Hub so gated models (e.g. Llama-2) can be downloaded
    print(model_args.extract_step_type_tokens)         # debug print of the chosen step-type extraction strategy


    if model_args.output_dir is None:
        model_args.output_dir = model_args.model_name_or_path   # fall back to the model path as the output location
    else:
        os.makedirs(model_args.output_dir, exist_ok = True)     # ensure the output directory exists

    # Pick the right tokenizer class: LlamaTokenizer for llama2/alpaca-family models, otherwise the generic AutoTokenizer.
    if 'llama2' in model_args.model_name_or_path or 'alpaca' in model_args.model_name_or_path:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    print("loaded tokenizer")

    tokenizer.pad_token_id = 0        # define a pad token id (many Llama-family tokenizers lack one by default)
    tokenizer.padding_side = "left"   # pad on the left so generation continues naturally from the end of the sequence

    prompt_text = {}            # maps prompt-segment names -> literal/special-token text to splice into inputs
    prompt_tokens = []           # token ids of any special "soft prompt" tokens
    num_new_tokens = 0           # count of special tokens added to the vocabulary
    step_type_predictor = None   # optional helper object that classifies each output step as e.g. "reason" vs "memory"
    step_type_ids = None         # placeholder passed to the dataset class (unused/unset here)

    # now lets look what new tokens the train.py have generated --> therefor look into prompt_text.json
    if model_args.add_soft_prompts:
        # Soft prompts: load the special-token vocabulary that train.py produced and saved alongside the checkpoint.
        prompt_text_file =  f'{os.path.dirname(model_args.model_name_or_path)}/prompt_text.json'
        special_tokens_list = []

        if os.path.exists(prompt_text_file):

            prompt_text = json.load(open(prompt_text_file))   # dict of segment-name -> string containing the special tokens, e.g. "<tok1><tok2>"
            for k in prompt_text:
                tokens = prompt_text[k].split('>')             # split on '>' to isolate each "<...>"-style special token
                special_tokens_list += [tok+'>' for tok in tokens[:-1]]   # re-append the '>' delimiter, drop the trailing empty fragment
                # special_tokens_list looks like ['<prefix_0>', '<prefix_1>', ...] ,so to keep the IDs for prompt-tokens


            if "memory" in model_args.extract_step_type_tokens:
                # Only when step types include a "memory" category: define an inline classifier that
                # scans generated text for "[reason]:" / "[rag]:" markers and labels each occurrence.
                class StepType:
                    def __init__(self):
                        self.vocab = ['reason','memory']   # the two step categories this predictor recognizes

                    def predict(self, text: str, start=0):

                        pattern = re.compile(r"^\[(.*?)\]:", re.MULTILINE)   # matches lines starting with "[label]:"
                        matches = pattern.findall(text)

                        result = []
                        for match in matches:
                            if match == 'reason':
                                result.append('reason')
                            elif match == 'rag':
                                result.append('memory')   # "rag" marker in the text is mapped to the "memory" category
                        return result


                step_type_predictor = StepType()   # instantiate the classifier defined above

                for s in step_type_predictor.vocab:
                    prompt_text[s] = ''   # register empty placeholder entries for 'reason'/'memory' in prompt_text


        prompt_tokens = tokenizer.convert_tokens_to_ids(special_tokens_list)   # map special token strings to their vocabulary ids
        num_new_tokens = len(special_tokens_list)                             # how many special tokens were added

    elif model_args.add_hard_prompts:
        # Hard prompts: instead of learned tokens, splice in fixed literal text (e.g. "Plan: ", " addition ") into the input.
        if model_args.only_at_front and not model_args.plan_first:
            prompt_text = {'prefix': 'Plan: '}   # only prepend a generic "Plan:" marker
        else:
            prompt_text = {'prefix': 'Plan: ', 'answer': ' answer', 'assignment': ' assignment',
                                '+': ' addition ', '-': ' deduction', '*': ' multiplication', '/': ' division'}
            # richer set of literal markers, including verbalized arithmetic operators, used to annotate reasoning steps

    # Decide which checkpoint path to actually load weights from: for PEFT modes, load the base model
    # first and apply the adapter afterward; otherwise load model_name_or_path directly.
    if model_args.parameter_efficient_mode != 'none':
        model_name = model_args.base_model_name_or_path
    else:
        model_name = model_args.model_name_or_path

    # For prompt-tuning, locate the saved embedding tensors (input/output) produced during training.
      # will be executed with "Prompt-tuning" and "prompt-tuning+lora"
      # load new_embedding and the new_linear weights of soft-prompt.
    if 'prompt-tuning' in model_args.parameter_efficient_mode:        

        input_embedding_file = model_args.model_name_or_path + '/embeddings.pt'
        output_embedding_file = None
        if not os.path.exists(input_embedding_file):
            # fall back to separate input/output embedding files if the combined one isn't present
            input_embedding_file = model_args.model_name_or_path + '/input_embeddings.pt'
            output_embedding_file = model_args.model_name_or_path + '/output_embeddings.pt'
    else:
        input_embedding_file = None
        output_embedding_file = None

    if model_args.load_in_8bit:
        # Load the (custom) causal LM in 8-bit quantized mode, spreading layers across available devices
        # and offloading to disk ("offload" folder) if GPU/CPU memory runs out.
        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            input_embedding_file=input_embedding_file,
            output_embedding_file=output_embedding_file,
            sparse=model_args.use_sparse_attention,
            prompt_tokens=prompt_tokens,
            pretrained_model_name_or_path=model_name,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            cache_dir=model_args.cache_dir, torch_dtype=torch.float32,
            device_map="auto", load_in_8bit=True,
            offload_folder="offload", offload_state_dict = True,
        )

    else:
        # Same as above but in full precision (float32), no 8-bit quantization.
        model = MyAutoModelForCausalLM.from_pretrained(n_tokens=num_new_tokens,
            input_embedding_file=input_embedding_file,
            output_embedding_file=output_embedding_file,
            sparse=model_args.use_sparse_attention,
            prompt_tokens=prompt_tokens,
            pretrained_model_name_or_path=model_name,
            parameter_efficient_mode=model_args.parameter_efficient_mode,
            cache_dir=model_args.cache_dir,
            device_map="auto", torch_dtype=torch.float32,
            offload_folder="offload", offload_state_dict = True
        )


    if 'lora' in model_args.parameter_efficient_mode:
        # If LoRA was used, wrap the base model with the LoRA adapter weights from model_name_or_path.
        model = MyPeftModelForCausalLM.from_pretrained(model,
            model_args.model_name_or_path,
            load_embeddings=model_args.add_soft_prompts,
            n_tokens=num_new_tokens)

    print("loaded model.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'   # pick GPU if available, else CPU
    model.eval()   # switch model to evaluation mode (disables dropout etc.)

    # Select the dataset-specific loader/answer-checker class based on the --dataset flag.
    if data_args.dataset == "gsm8k":
        data_class = GSMData
    elif data_args.dataset == "aqua":
        data_class = AquaData
    elif data_args.dataset == "qa":
        data_class = StrategyQAData
    elif data_args.dataset == "stratgeqa_agent":
        data_class = StrategyQAData_Ours
    elif data_args.dataset == "commonsenseqa_agent":
        data_class = CommonsenseQAData_Ours
    elif data_args.dataset == "truthfulqa_agent" or data_args.dataset == "truthfulqa_agent_crossdomain":
        data_class = TruthfulQAData_Ours

    # Instantiate the test split, passing through all the prompt-construction options gathered above.
    dataset = data_class("test", prompt_text,
                        add_soft_prompts=model_args.add_soft_prompts or model_args.add_hard_prompts,
                        only_at_front=model_args.only_at_front,
                        plan_first=model_args.plan_first,
                        plan_only=model_args.plan_only,
                        prompt_template=data_args.prompt_template,
                        step_type_ids=step_type_ids, tokenizer=tokenizer,
                        step_type_predictor=step_type_predictor,)
    random.seed(42)   # fix the RNG so the subsampling below is reproducible
    if len(dataset) > data_args.num_test:
        # If the dataset is larger than the requested test size, randomly subsample (with replacement) num_test examples.
        idx = random.choices(list(range(len(dataset))), k=data_args.num_test)
        new_x = []
        new_y = []
        for i in idx:
            new_x.append(dataset[i]['x'])   # input text for example i
            new_y.append(dataset[i]['y'])   # target/label text for example i
        dataset.x = new_x   # replace the full dataset's inputs with the sampled subset
        dataset.y = new_y   # replace the full dataset's targets with the sampled subset
    assert len(dataset) <= data_args.num_test   # sanity check that subsampling worked
    print(dataset[0], len(dataset))   # debug print: first example and dataset size



    print("loaded dataset")

    dataloader = DataLoader(dataset, batch_size=data_args.batch_size, shuffle=False)   # batches examples in original order (no shuffling for eval)


    prompt_ts = {}   # will hold, per step-type, the literal marker string to search/count in generated text

    if step_type_predictor is not None:
        generated_planning_token_dist = {}   # running counts of how often each step-type marker appears in model outputs
        gt_planning_token_dist ={}           # running counts of how often each step-type marker appears in ground-truth targets
        for k in step_type_predictor.vocab:
            prompt_ts[k] = prompt_text[k].strip().split('>')[0] + '>'   # normalize each step-type's marker to "<label>" form


    num_correct = 0    # running count of correctly answered examples
    num_all = 0         # running count of total examples processed
    output_data = []    # collected per-example results to optionally save to disk

    for i, batch in tqdm(enumerate(dataloader)):   # iterate over batches, showing progress
        x_text, y_text = batch['x'], batch['y']    # x_text: list of input prompts; y_text: list of ground-truth answers


        encoding = tokenizer(x_text, padding=True, return_tensors='pt').to(device)   # tokenize + pad the batch, move tensors to GPU/CPU
        max_length = min(model_args.max_length, encoding['input_ids'].size(1) + 512)   # computed but unused below (generate() uses model_args.max_length directly)
        if model_args.use_sparse_attention:
            print("use sparse attention")
            sparese_attention_mask = make_sparse_mask(encoding['input_ids'], prompt_tokens).to(device)   # build the custom sparse attention mask
            encoding["attention_mask"] = (encoding["attention_mask"], sparese_attention_mask)             # pass both the normal and sparse masks together
        with torch.no_grad():   # disable gradient tracking during generation (inference only)
            generated_ids = model.generate(**encoding,
                 max_length=model_args.max_length)   # autoregressively generate output token ids for the batch


        try:
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   # convert generated token ids back to strings
            print(generated_texts)
        except:
            print("cannot decode: ")
            print(generated_ids)   # if decoding fails, just dump the raw ids (generated_texts stays undefined -> would error below)

        for text, x, y in zip(generated_texts, x_text, y_text):   # iterate per-example over the decoded batch
            text, x, y = str(text), str(x), str(y)   # ensure plain strings

            if step_type_predictor is not None:
                # Tally how many times each step-type marker occurs in the generated text vs. the ground truth.
                for k in prompt_ts:
                    n_generated_k = text.count(prompt_ts[k])
                    n_gt_k = y.count(prompt_ts[k])
                    if k in generated_planning_token_dist:
                        generated_planning_token_dist[k] += n_generated_k
                    else:
                        generated_planning_token_dist[k] = n_generated_k
                    if k in gt_planning_token_dist:
                        gt_planning_token_dist[k] += n_gt_k
                    else:
                        gt_planning_token_dist[k] = n_gt_k
            print(text)   # debug print of the generated text
            result = ''
            if dataset.is_correct(text, y):   # delegate correctness checking to the dataset class (extracts/compares final answers)
                num_correct += 1
                print('correct')
                result = 'correct'

            else:
                print('wrong')
                result = 'wrong'

            output_data.append({
            'generated_text': text,
            'result': result
            })   # record this example's generation and correctness for later saving

            num_all += 1


        print("Accuracy: ", num_correct/num_all)   # running accuracy printed after each batch
        if step_type_predictor is not None:
            print("groundtruth planning token dist: ", gt_planning_token_dist)
            print("generated planning token dist: ", generated_planning_token_dist)



    if model_args.save_result:
        # Write all collected generations/results to a JSON file under output_dir/<dataset>/<base_model>_output.json
        output_file = os.path.join(model_args.output_dir, f"{data_args.dataset}/{model_args.base_model_name_or_path}_output.json")
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("Accuracy: ", num_correct/num_all)   # final overall accuracy
    print("num test: ", num_all)               # total number of examples evaluated




if __name__ == "__main__":
    main()   # entry point: parse args, load model/tokenizer/dataset, run generation loop, save results
