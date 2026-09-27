import re, os, json
import sys
# Make the parent directory importable (so `load_data.xxx` imports work
# regardless of the current working directory the script is run from).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_dataset          # HuggingFace `datasets` library, used to pull public datasets (gsm8k, aqua_rat, ...)
from torch.utils.data import Dataset       # Base class so these can be used with PyTorch DataLoaders
from load_data.utils import _strip_string, delete_extra_zero, compare_both_string_and_number_format  # helper string/number normalizers (imported but not used directly in this file)
from load_data.data_agent import get_label  # imported but not used directly in this file

class BaseAgentData(Dataset):
    """
    Base dataset class for the 'agent' variant of the pipeline (datasets whose
    chain-of-thought steps were produced/labeled by an external 'agent', e.g.
    the *_Ours classes further down that load pre-generated JSON files).
    Subclasses must implement load_data() and parse_q_a().
    """

    def __init__(self, split: str, soft_prompt_text: list,
                 invalid_ans="[invalid]", add_soft_prompts=False,
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(BaseAgentData, self).__init__()
        # Load raw examples for the requested split ('train'/'test'/...); subclass-defined.
        self.data = self.load_data(split)
        self.split = split

        # soft_prompt_text: dict mapping a "step type" name (e.g. 'prefix', 'answer',
        # 'assignment', or custom types) -> the literal soft-prompt token/string to insert.
        self.soft_prompt_text = soft_prompt_text
        self.INVALID_ANS = invalid_ans
        # Regex used later to pull the final answer out of generated text, e.g.
        # "... The answer is: 42" or "... The answer is: true"
        self.ANS_RE = re.compile(r"The answer is: (\-?[0-9\.\,]+|true|false)", re.IGNORECASE)
        self.add_soft_prompts = add_soft_prompts   # whether to inject soft-prompt tokens into the target text
        self.only_at_front = only_at_front         # if True, put a single soft prompt before the whole CoT instead of per-step
        self.plan_first = plan_first               # if True, prepend a "plan" (concatenated step-type tokens) before the solution
        self.plan_only = plan_only                 # if True, target is just the plan + final answer (no actual CoT steps)
        self.step_type_ids = step_type_ids         # optional token ids used to detect step types via tokenizer instead of predictor/regex
        self.tokenizer = tokenizer
        self.step_type_predictor = step_type_predictor  # optional model/heuristic that predicts a step type per CoT step
        self.step_type_re = None                   # will hold a regex alternation of soft-prompt "keyword" step types, if applicable

        if self.add_soft_prompts:
            if step_type_ids is None:
                # Build a regex that matches any soft-prompt key that ISN'T one of the
                # generic/structural ones ('prefix', 'answer', 'assignment') or one that
                # the step_type_predictor already knows how to produce.
                skip_list = ['prefix', 'answer', 'assignment']
                if self.step_type_predictor is not None:
                    skip_list += list(self.step_type_predictor.vocab)
                # Only bother building the regex if there are soft-prompt keys beyond
                # the ones already covered by skip_list (i.e. custom keyword-based types).
                if len(soft_prompt_text) > len(skip_list):
                    self.step_type_re = r"{}".format('|'.join([re.escape(x)
                        for x in soft_prompt_text if x not in skip_list]))
            elif self.step_type_predictor is None:
                # If step types are detected via tokenizer/token-ids, a tokenizer is required.
                assert tokenizer is not None, "Tokenizer must be provided if step_type_ids is not None."

        if prompt_template is not None:
            # Optional instruction-tuning style template (e.g. Alpaca format) loaded from JSON.
            file_name = f"./load_data/prompt_templates/{prompt_template}.json"
            assert os.path.exists(file_name), f"Prompt template {prompt_template} not found."
            self.prompt_template = json.load(open(file_name, "r"))
        else:
            self.prompt_template = None

        self.x = []   # will hold the model inputs (questions/prompts)
        self.y = []   # will hold the model targets (solutions/answers)
        self.prepare_data()   # populate self.x / self.y from self.data
        print(len(self.x))
        print(len(self.y))
        assert len(self.x) == len(self.y)   # sanity check: every input has a matching target
        print("Data prepared")

    def load_data(self, split: str):
        # Abstract: subclasses must return the raw examples for this split.
        raise NotImplementedError

    def parse_q_a(self, example):
        # Abstract: subclasses must turn one raw example into
        # (instruction, question, list_of_cot_steps, answer_string).
        raise NotImplementedError

    def extract_step_type(self, q, cot_steps):
        """
        Determine, for each CoT step, which soft-prompt "type" tags apply to it.
        Returns a list (one entry per step) of lists of type-name strings.
        """
        if len(cot_steps) == 0:
            return None

        # Every step starts tagged as 'prefix'; more tags get appended below.
        step_types = [['prefix'] for _ in cot_steps]
        if self.step_type_predictor is not None:
            # Build one big text blob (question + all steps) and ask the predictor
            # to tag every step at once (more context than tagging steps individually).
            text = "Question: " + q + ' \n' + ' \n'.join(cot_steps) + ' \n'
            start = len(q.split('\n'))   # index offset where the CoT steps begin within `text`

            new_step_types = self.step_type_predictor.predict(text, start)

            # Sanity-check: predictor should return one tag per step. If it returns
            # MORE tags than steps, something is badly wrong -> abort the whole run.
            # (If it returns fewer, this just logs and proceeds, appending what it has.)
            if len(new_step_types) != len(cot_steps):
                print(new_step_types)
                print(cot_steps)
                if len(new_step_types) >  len(cot_steps):
                    exit(1)
            for i, t in enumerate(new_step_types):
                    step_types[i].append(t)

        if self.step_type_ids is None:
            # Fallback: use the keyword regex (self.step_type_re) built in __init__
            # to find literal soft-prompt keywords occurring inside each step's text.
            if self.step_type_re is not None:
                for i, step in enumerate(cot_steps):
                    step_types[i] += re.findall(self.step_type_re, step)

        else:
            # Alternative detection path: tokenize each step and check whether any of
            # the configured step_type_ids appear among its token ids.
            for i, step in enumerate(cot_steps):
                text_ids = self.tokenizer.encode(step)
                for j in self.step_type_ids:
                    if j in text_ids:
                        # NOTE: appends the token for index `i` (the step index), not `j`
                        # (the matched id) — looks like a pre-existing bug, but left as-is.
                        step_types[i].append(self.tokenizer.convert_ids_to_tokens(i))

        # Any step that only ever got the default 'prefix' tag (no predictor/regex
        # match) is treated as a plain 'assignment' step.
        for i, step in enumerate(cot_steps):
            if len(step_types[i]) == 1:
                step_types[i].append('assignment')

        return step_types

    def prepare_data(self):
        """
        Turns each raw example into a (x, y) pair: x = the prompt/question given to
        the model, y = the target text (CoT steps + final answer, optionally
        decorated with soft-prompt tokens) it should learn to produce.
        """
        i = 0   # running counter of "steps processed" (not used for anything beyond incrementing)
        for ex_idx, ex in enumerate(self.data):
            plans = ''   # accumulates just the soft-prompt tokens, used when plan_first/plan_only
            sol = ''     # accumulates the actual solution text
            inst, q, cot_steps, ans = self.parse_q_a(ex)
            if cot_steps is None:
                # parse_q_a signals "skip this example" by returning None steps.
                continue

            if self.add_soft_prompts:

                if self.only_at_front:
                    # Single soft prompt before the whole CoT block instead of per-step tagging.
                    sol += self.soft_prompt_text['prefix'] + ' ' + ' \n'.join(cot_steps) + '\n'
                    i += len(cot_steps)
                else:
                    # Per-step soft-prompt tagging path.
                    prompt_types = self.extract_step_type(q, cot_steps)

                    if prompt_types is None:
                        continue
                    for step, prompt_type in zip(cot_steps, prompt_types):
                        # Strip any bracketed prefix annotation already in the raw step text,
                        # e.g. "[fact]: some sentence" -> "some sentence".
                        step = re.sub(r'\[.*?\]: ', '', step)

                        if len(self.soft_prompt_text) > 1:
                            # Multiple distinct soft-prompt tokens configured: emit one
                            # token per detected type, both into the solution and the plan.
                            for p_type in prompt_type:
                                sol += self.soft_prompt_text[p_type]
                                plans += self.soft_prompt_text[p_type]
                            plans += '\n'
                        else:
                            # Only one soft-prompt token configured: always use 'prefix'.
                            sol += self.soft_prompt_text['prefix']
                        sol += ' ' + step + '\n'
                        i += 1

            else:
                # No soft prompts: just join the CoT steps as plain text.
                sol += ' \n'.join(cot_steps) + '\n'
                i += len(cot_steps)

            if self.add_soft_prompts:
                if not self.only_at_front:
                    # Append a soft-prompt marker before the final answer too.
                    sol += self.soft_prompt_text['prefix']
                    if len(self.soft_prompt_text) > 1:
                        sol += self.soft_prompt_text['answer']
                        plans += self.soft_prompt_text['answer'] + '\n'
                sol += ' The answer is: ' + ans + '\n'
                if self.plan_first:
                    # Prepend the accumulated plan tokens before the full solution text.
                    sol = plans + sol
                if self.plan_only:
                    # Overwrite: target becomes just the plan + the final answer line
                    # (no actual reasoning steps) — used to train a "planner" head/mode.
                    sol = plans + ' The answer is: ' + ans + '\n '
            else:
                sol += ' The answer is: ' + ans + '\n '
            i += 1

            self.y.append(sol)

            if self.prompt_template is not None:
                # Use the instruction-tuning template to format the input.
                x = self.prompt_template["prompt_input"].format(
                    instruction=inst, input=q)
            else:
                x = "Question: " + q + '\n '

            # At eval/inference time (not 'train'), and unless we're doing plan_first/
            # plan_only, prime the input with the soft-prompt 'prefix' token so the
            # model continues naturally in the same format it was trained on.
            if self.add_soft_prompts and self.split != 'train' and not self.plan_first and not self.plan_only:
                x += self.soft_prompt_text['prefix']

            self.x.append(x)

    def extract_answer(self, completion):
        # Pull the final numeric/boolean answer out of a completion string using ANS_RE.
        match = self.ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")   # strip thousands separators, e.g. "1,000" -> "1000"
            return match_str
        else:
            return self.INVALID_ANS

    def is_correct(self, model_completion, gt_example):
        # Compare a model's generated completion against the ground-truth text.
        gt_answer = self.extract_answer(gt_example)
        assert gt_answer != self.INVALID_ANS   # ground truth must always be parseable
        try:
            pred_answer = self.extract_answer(model_completion)
        except:
            pred_answer = self.INVALID_ANS
        return pred_answer == gt_answer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Standard PyTorch Dataset interface: return one (input, target) pair.
        return dict(x=self.x[idx], y=self.y[idx])






class BaseData(Dataset):
    """
    Near-identical twin of BaseAgentData, used as the base class for the
    "vanilla" (non-agent) datasets below (GSM8K, AQuA, StrategyQA, ...).
    The logic is duplicated here rather than shared via inheritance/mixins —
    likely an artifact of iterative development rather than a design choice.
    """

    def __init__(self, split: str, soft_prompt_text: list,
                 invalid_ans="[invalid]", add_soft_prompts=False,
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(BaseData, self).__init__()
        self.data = self.load_data(split)
        self.split = split

        self.soft_prompt_text = soft_prompt_text
        self.INVALID_ANS = invalid_ans
        # self.ANS_RE = re.compile(r"The answer is: (\-?[0-9\.\,]+)")   # older, numeric-only version (left commented for reference)
        self.ANS_RE = re.compile(r"The answer is: (\-?[0-9\.\,]+|true|false)", re.IGNORECASE)
        self.add_soft_prompts = add_soft_prompts
        self.only_at_front = only_at_front
        self.plan_first = plan_first
        self.plan_only = plan_only
        self.step_type_ids = step_type_ids
        self.tokenizer = tokenizer
        self.step_type_predictor = step_type_predictor
        self.step_type_re = None


        if self.add_soft_prompts:
            if step_type_ids is None:
                skip_list = ['prefix', 'answer', 'assignment']
                if self.step_type_predictor is not None:
                    skip_list += list(self.step_type_predictor.vocab)
                if len(soft_prompt_text) > len(skip_list):
                    self.step_type_re = r"{}".format('|'.join([re.escape(x)
                        for x in soft_prompt_text if x not in skip_list]))
            elif self.step_type_predictor is None:
                assert tokenizer is not None, "Tokenizer must be provided if step_type_ids is not None."


        if prompt_template is not None:
            file_name = f"./load_data/prompt_templates/{prompt_template}.json"
            assert os.path.exists(file_name), f"Prompt template {prompt_template} not found."
            self.prompt_template = json.load(open(file_name, "r"))
        else:
            self.prompt_template = None

        self.x = []
        self.y = []
        self.prepare_data()
        assert len(self.x) == len(self.y)
        # NOTE: unlike BaseAgentData, no print(len(...)) / "Data prepared" here.

    def load_data(self, split: str):
        raise NotImplementedError

    def parse_q_a(self, example):
        raise NotImplementedError

    def extract_step_type(self, q, cot_steps):
        # Same logic as BaseAgentData.extract_step_type, with one difference noted below.
        if len(cot_steps) == 0:
            return None

        step_types = [['prefix'] for _ in cot_steps]
        if self.step_type_predictor is not None:
            text = "Question: " + q + ' \n' + ' \n'.join(cot_steps) + ' \n'
            start = len(q.split('\n'))

            new_step_types = self.step_type_predictor.predict(text, start)

            # NOTE: mismatch case is silent here (no print) compared to BaseAgentData,
            # but still hard-exits if the predictor over-produces tags.
            if len(new_step_types) != len(cot_steps):
                if len(new_step_types) >  len(cot_steps):
                    exit(1)
            for i, t in enumerate(new_step_types):
                    step_types[i].append(t)


        if self.step_type_ids is None:
            if self.step_type_re is not None:
                for i, step in enumerate(cot_steps):
                    step_types[i] += re.findall(self.step_type_re, step)
        else:
            for i, step in enumerate(cot_steps):
                text_ids = self.tokenizer.encode(step)
                for j in self.step_type_ids:
                    if j in text_ids:
                        step_types[i].append(self.tokenizer.convert_ids_to_tokens(i))

        for i, step in enumerate(cot_steps):
            if len(step_types[i]) == 1:
                step_types[i].append('assignment')

        return step_types

    def prepare_data(self):
        # Identical structure to BaseAgentData.prepare_data, except:
        #  - no `[\[.*?\]: ]` stripping of bracketed prefixes from steps
        #  - a debug print of every solution for the first 1000 examples (see below)
        i = 0
        for ex_idx, ex in enumerate(self.data):
            plans = ''
            sol = ''
            inst, q, cot_steps, ans = self.parse_q_a(ex)
            if cot_steps is None:
                continue

            if self.add_soft_prompts:
                if self.only_at_front:
                    sol += self.soft_prompt_text['prefix'] + ' ' + ' \n'.join(cot_steps) + '\n'
                    i += len(cot_steps)
                else:
                    prompt_types = self.extract_step_type(q, cot_steps)
                    if prompt_types is None:
                        continue
                    for step, prompt_type in zip(cot_steps, prompt_types):
                        if len(self.soft_prompt_text) > 1:
                            for p_type in prompt_type:
                                sol += self.soft_prompt_text[p_type]
                                plans += self.soft_prompt_text[p_type]
                            plans += '\n'
                        else:
                            sol += self.soft_prompt_text['prefix']
                        sol += ' ' + step + '\n'
                        i += 1

            else:
                sol += ' \n'.join(cot_steps) + '\n'
                i += len(cot_steps)

            if self.add_soft_prompts:
                if not self.only_at_front:
                    sol += self.soft_prompt_text['prefix']
                    if len(self.soft_prompt_text) > 1:
                        sol += self.soft_prompt_text['answer']
                        plans += self.soft_prompt_text['answer'] + '\n'
                sol += ' The answer is: ' + ans + '\n'
                if self.plan_first:
                    sol = plans + sol
                if self.plan_only:
                    sol = plans + ' The answer is: ' + ans + '\n '
            else:
                sol += ' The answer is: ' + ans + '\n '
            i += 1

            # Debug logging: print the first 1000 constructed targets to stdout.
            if i < 1000:
                print(sol)
            self.y.append(sol)

            if self.prompt_template is not None:
                x = self.prompt_template["prompt_input"].format(
                    instruction=inst, input=q)
            else:
                x = "Question: " + q + '\n '

            if self.add_soft_prompts and self.split != 'train' and not self.plan_first and not self.plan_only:
                x += self.soft_prompt_text['prefix']

            self.x.append(x)

    def extract_answer(self, completion):

        match = self.ANS_RE.search(completion)

        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.INVALID_ANS

    def is_correct(self, model_completion, gt_example):
        gt_answer = self.extract_answer(gt_example)
        print(gt_answer)   # debug print of the ground-truth answer being checked against

        assert gt_answer != self.INVALID_ANS
        try:
            pred_answer = self.extract_answer(model_completion)
            print(pred_answer)   # debug print of the predicted answer
        except:
            pred_answer = self.INVALID_ANS
        return pred_answer == gt_answer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return dict(x=self.x[idx], y=self.y[idx])



class GSMData(BaseData):
    """Dataset wrapper for GSM8K (grade-school math word problems), loaded via HuggingFace."""

    def __init__(self, split: str, soft_prompt_text: list,
                 invalid_ans="[invalid]", add_soft_prompts=False,
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        # Just forwards everything to BaseData.__init__ — no GSM-specific state here.
        super(GSMData, self).__init__(split, soft_prompt_text,
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)

    def load_data(self, split: str):
        # Pull the 'main' config of gsm8k from the HuggingFace Hub for the given split.
        return load_dataset('gsm8k', 'main')[split]

    def parse_q_a(self, example):
        # GSM8K answers are formatted as "<reasoning> #### <final answer>".
        cot, ans = example["answer"].split("####")
        cot = cot.strip()
        cot = cot.split('. ')   # naive sentence split on ". "
        cot_steps = []
        for step in cot:
            # Further split on newlines within a "sentence" (GSM8K sometimes has
            # multiple calculation lines per sentence), producing individual steps.
            for s in step.strip().split('\n'):
                s = s.strip()
                if len(s) == 0:
                    continue
                if s[-1] != '.':
                    s += '.'   # normalize: every step should end with a period
                cot_steps.append(s)

        ans = ans.strip()
        q = example["question"].strip()
        inst = 'Solve the following problem step by step, and give a numerical answer.'
        return inst, q, cot_steps, ans

    def extract_step_type(self, q, cot_steps):
        # GSM-specific override: GSM8K steps often contain inline calculator
        # annotations like "<<3*4=12>>", so step-type keywords are searched for
        # INSIDE those annotations first, falling back to the whole step text.
        step_types = [['prefix'] for _ in cot_steps]
        if self.step_type_predictor is not None:
            text = "Question: " + q + '\n ' + ' \n'.join(cot_steps) + '\n'
            start = len(q.split('\n'))
            new_step_types = self.step_type_predictor.predict(text, start)
            if len(new_step_types) != len(cot_steps):
                print(new_step_types)
                print(cot_steps)
                # NOTE: unlike the base class, this override does NOT exit(1) on mismatch.
            for i, t in enumerate(new_step_types):
                step_types[i].append(t)

        if self.step_type_ids is None:
            if self.step_type_re is not None:
                rgx = r"<<.*>>"   # matches GSM8K's inline "<<expression=result>>" annotations
                for i, step in enumerate(cot_steps):
                    matches = re.findall(rgx, step)
                    if len(matches) > 0:
                        # Prefer scanning inside the calculator annotation(s) for step-type keywords.
                        for match in matches:
                            step_types[i] += re.findall(self.step_type_re, match)
                            # print(match)
                            # print(step_types)
                    else:
                        # No calculator annotation present: scan the whole step text instead.
                        step_types[i] += re.findall(self.step_type_re, step)
        else:
            for i, step in enumerate(cot_steps):
                text_ids = self.tokenizer.encode(step)
                for j in self.step_type_ids:
                    if j in text_ids:
                        step_types[i].append(self.tokenizer.convert_ids_to_tokens(i))

        for i, step in enumerate(cot_steps):
            if len(step_types[i]) == 1:
                step_types[i].append('assignment')

        return step_types


    def clean_text(self, rgx_list, text):
        # Utility: strip every regex pattern in rgx_list out of `text`. Not called
        # anywhere else in this file — presumably used elsewhere in the pipeline.
        new_text = text
        for rgx_match in rgx_list:
            new_text = re.sub(rgx_match, '', new_text)
        return new_text



class AquaData(BaseData):
    """Dataset wrapper for AQuA-RAT (algebraic word problems with multiple-choice answers)."""

    def __init__(self, split: str, soft_prompt_text: list,
                 invalid_ans="[invalid]", add_soft_prompts=False,
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(AquaData, self).__init__(split, soft_prompt_text,
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)

    def load_data(self, split: str):
        return load_dataset('aqua_rat', 'raw')[split]

    def parse_q_a(self, example):
        # Build a "(A ... (B ... (C ..." style answer-choices string from the options list.
        choice = "(" + "(".join(e.strip() for e in example["options"])
        choice = choice.replace("(", " (").replace(")", ") ")
        choice = "Answer Choices:" + choice

        q = example["question"].strip() + '\n' + choice

        ans = example["correct"].strip()   # e.g. "A", "B", ...
        cot = example["rationale"].strip()
        cot = cot.split('. ')
        cot_steps = []
        for step in cot:
            for s in step.strip().split('\n'):
                s = s.strip()
                if len(s) == 0:
                    continue
                cot_steps.append(s)

        # Some rationales start with a literal "Explanation" header — strip it.
        if 'Explanation' in cot_steps[0]:
            print(cot_steps[0])
            cot_steps[0] = cot_steps[0][12:].strip()   # 12 == len("Explanation") + 1, drops the header

        # Drop a near-empty leading step (e.g. leftover punctuation) after the header removal.
        if len(cot_steps[0]) < 2:
            cot_steps = cot_steps[1:]

        if len(cot_steps) == 0:
            # Nothing usable left: signal parse_q_a's caller to skip this example.
            return None, None, None, None

        # Drop a trailing step if it's just restating the answer choice (short "Ans: B"-style
        # lines) rather than being part of the reasoning.
        if 'Ans' in cot_steps[-1] or 'ANS' in cot_steps[-1] or 'ans' in cot_steps[-1] or 'Option' in cot_steps[-1]:
            print(cot_steps[-1])
            if len(cot_steps[-1]) < 30:
                cot_steps = cot_steps[:-1]
        elif len(cot_steps[-1]) < 5:
            print(cot_steps[-1])
            cot_steps = cot_steps[:-1]

        inst = 'Solve the following problem step by step, and choose the best answer from the given choices.'

        return inst, q, cot_steps, ans

    def extract_answer(self, completion):
        # AQuA-specific override: answers are letters (A-E), not numbers/booleans,
        # so this bypasses self.ANS_RE entirely and does its own letter extraction.
        preds = completion.split("The answer is:")
        pred = preds[-1].strip()
        pred = pred.upper()
        pred = re.findall(r'A|B|C|D|E', pred)
        if len(preds) > 1:
            # "The answer is:" marker was found — trust the first letter found after it.
            if len(pred) > 0:
                return pred[0]
            else:
                return self.INVALID_ANS
        else:
            # No marker found at all — fall back to the last letter anywhere in the text.
            try:
                return pred[-1]
            except:
                print(preds)
                return self.INVALID_ANS




class StrategyQAData(BaseData):
    """Dataset wrapper for StrategyQA (yes/no questions requiring implicit multi-hop reasoning)."""

    def __init__(self, split: str, soft_prompt_text: dict,
                 invalid_ans="[invalid]", add_soft_prompts=False,
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):
        super(StrategyQAData, self).__init__(split, soft_prompt_text,
            invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
            prompt_template, step_type_ids, tokenizer, step_type_predictor)

    def load_data(self, split: str):
        dataset = load_dataset("ChilleD/StrategyQA")[split]
        # 返回前 n 条数据 dataset.select(range(3))
        # (Chinese comment, left as in the original: "return the first n rows:
        #  dataset.select(range(3))" — a leftover debugging note, not active code.)
        return dataset

    def parse_q_a(self, example):
        inst = 'Answer the following question Ture or False step by step using the supporting facts in your knowledge.'
        question = example["question"]

        answer = example["answer"]
        answer = 'True' if answer else 'False'   # dataset stores answer as a bool; convert to the string format ANS_RE expects

        # Use the dataset's supporting "facts" field as the CoT steps, split on '.'.
        cot_steps = [sentence.strip() for sentence in example['facts'].split('.') if sentence.strip()]

        return inst, question, cot_steps, answer





class StrategyQAData_Ours(BaseAgentData):
    """
    'Ours' variant of StrategyQA: loads pre-built CoT data (with steps already
    segmented/labeled, presumably by the internal 'agent' pipeline) from a local
    JSON file instead of re-deriving CoT from the raw HF dataset.
    """

    def __init__(self, split: str, soft_prompt_text: dict,
                 invalid_ans="[invalid]", add_soft_prompts=False,
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):


        super(StrategyQAData_Ours, self).__init__(split, soft_prompt_text,
                                                  invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
                                                  prompt_template, step_type_ids, tokenizer, step_type_predictor)


    def load_data(self, split: str):
        # Fixed local path convention: one JSON file per split, e.g.
        # "StrategyQA_train_clean.json" / "StrategyQA_test_clean.json".
        json_file = "./load_data/dataset_folder/StrategyQA_{}_clean.json".format(split)
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                print("Loading data from json file")
                data = json.load(f)
            # Extra filter: keep only entries whose own "split" field matches
            # (in case one file contains a mix of splits).
            data = [entry for entry in data if entry.get("split") == split]
        # NOTE: if the file doesn't exist, `data` is never defined -> this will
        # raise an UnboundLocalError on return rather than a clear "file missing" error.
        return data


    def parse_q_a(self, example):
        # if the json file exists, load the data from the json file
        inst = 'Answer the following question True or False step by step using the supporting facts in your knowledge.'
        question = example["question"]
        cot_steps = example["cot_steps"]   # pre-segmented steps, already stored in the JSON
        answer = example["answer"]
        return inst, question, cot_steps, answer





class CommonsenseQAData_Ours(BaseAgentData):
    """'Ours' variant for CommonsenseQA, same local-JSON loading pattern as StrategyQAData_Ours."""

    def __init__(self, split: str, soft_prompt_text: dict,
                 invalid_ans="[invalid]", add_soft_prompts=False,
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):

        super(CommonsenseQAData_Ours, self).__init__(split, soft_prompt_text,
                                                  invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
                                                  prompt_template, step_type_ids, tokenizer, step_type_predictor)


    def load_data(self, split: str):
        json_file = './load_data/dataset_folder/commonsense_qa_{}_clean_CC.json'.format(split)
        print(json_file)
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                print("Loading data from json file")
                data = json.load(f)
            data = [entry for entry in data if entry.get("split") == split]

        else:
            # Unlike StrategyQAData_Ours, this one fails loudly (and immediately)
            # if the expected file is missing, instead of silently erroring later.
            print("No matching data found.")
            exit(111)

        return data


    def parse_q_a(self, example):

        inst = 'Answer the following question True or False step by step using the supporting facts in your knowledge.'
        qu = example["question"]
        question = qu.replace('Question: ', '', 1)   # strip a redundant "Question: " prefix if the stored text already has one

        cot_steps = example["cot_steps"]

        if cot_steps == []:
            cot_steps = ['']   # ensure at least one (empty) step so downstream code doesn't treat this as "no steps -> skip"
        answer = example["answer"]

        return inst, question, cot_steps, answer

    def extract_answer(self, completion):
        # Same letter-based (A-E) extraction as AquaData.extract_answer.
        preds = completion.split("The answer is:")
        pred = preds[-1].strip()
        pred = pred.upper()
        pred = re.findall(r'A|B|C|D|E', pred)
        if len(preds) > 1:
            if len(pred) > 0:
                return pred[0]
            else:
                return self.INVALID_ANS
        else:
            try:
                return pred[-1]
            except:
                print(preds)
                return self.INVALID_ANS


class TruthfulQAData_Ours(BaseAgentData):
    """'Ours' variant for TruthfulQA, same local-JSON loading pattern, with a wider answer-letter range (A-N)."""

    def __init__(self, split: str, soft_prompt_text: dict,
                 invalid_ans="[invalid]", add_soft_prompts=False,
                 only_at_front=False, plan_first=False, plan_only=False,
                 prompt_template=None, step_type_ids=None, tokenizer=None,
                 step_type_predictor=None):

        super(TruthfulQAData_Ours, self).__init__(split, soft_prompt_text,
                                                  invalid_ans, add_soft_prompts, only_at_front, plan_first, plan_only,
                                                  prompt_template, step_type_ids, tokenizer, step_type_predictor)


    def load_data(self, split: str):
        json_file = './load_data/dataset_folder/truthful_qa_{}_clean_CC.json'.format(split)
        print(json_file)
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                print("Loading data from json file")
                data = json.load(f)
            data = [entry for entry in data if entry.get("split") == split]
        else:
            print("No matching data found.")
            exit(111)

        return data


    def parse_q_a(self, example):
        print(example)   # debug: dump every raw example to stdout while parsing

        inst = 'Answer the following question True or False step by step using the supporting facts in your knowledge.'
        question  = example["question"]
        cot_steps = example["cot_steps"]

        if cot_steps == []:
            cot_steps = ['']
        answer = example["answer"]

        return inst, question, cot_steps, answer

    def extract_answer(self, completion):
        # TruthfulQA has more answer options than the other multiple-choice sets,
        # so the letter range extends to N instead of stopping at E.
        preds = completion.split("The answer is:")
        pred = preds[-1].strip()
        pred = pred.upper()
        pred = re.findall(r'A|B|C|D|E|F|G|H|I|J|K|L|M|N', pred)
        if len(preds) > 1:
            if len(pred) > 0:
                return pred[0]
            else:
                return self.INVALID_ANS
        else:
            try:
                return pred[-1]
            except:
                print(preds)
                return self.INVALID_ANS
