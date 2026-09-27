# Import required libraries
import torch
import os
from typing import Dict, Optional, Sequence, Union, Callable

from transformers import AutoModelForCausalLM, PreTrainedModel
import torch.nn.functional as F
from model.sparse_models import SparseGPT2LMHeadModel, SparseLlamaForCausalLM

# Custom module for handling input embeddings with support for new tokens
# This allows extending the vocabulary of a pre-trained model with new prompt tokens
class InputEmbedding(torch.nn.Module):
    """
    Wrapper around the original input embedding layer that allows adding new tokens
    while keeping the original vocabulary intact.
    """
    def __init__(self, original_embedding, n_new_tokens, initialize_tokens=None):
        """
        Initialize the custom input embedding layer.

        Args:
            original_embedding: The original embedding layer from the model
            n_new_tokens: Number of new tokens to add to the vocabulary
            initialize_tokens: Optional indices to initialize new token embeddings
        """
        super(InputEmbedding, self).__init__()
        self.original_embedding = original_embedding
        self.num_original_tokens = original_embedding.weight.size(0)
        print("original vocab size: ", self.num_original_tokens)
        self.n_new_tokens = n_new_tokens

        # Create embeddings for new tokens if any
        if n_new_tokens > 0:
            # Initialize new embedding layer with same dimensions as original
            self.new_embedding = torch.nn.Embedding(n_new_tokens,
                original_embedding.weight.size(1)).to(original_embedding.weight.device)

            # Initialize new token weights based on provided tokens or average of original
            if initialize_tokens is not None:
                new_embeddings = self.original_embedding(initialize_tokens)
                self.new_embedding.weight.data = new_embeddings
            else:
                # Use mean of original embeddings as initialization
                self.new_embedding.weight.data = original_embedding.weight.mean(
                    dim=0, keepdim=True).repeat(n_new_tokens, 1)
        else:
            self.new_embedding = None

    def forward(self, input_ids):
        """
        Forward pass that handles both original and new token embeddings.

        Args:
            input_ids: Token indices (may contain IDs >= num_original_tokens for new tokens)

        Returns:
            Embedding vectors for all input tokens
        """
        # Check if any token ID is from the new tokens
        if input_ids.max() >= self.num_original_tokens:
            # All tokens are new tokens
            if input_ids.min() >= self.num_original_tokens:
                return self.new_embedding(input_ids - self.num_original_tokens)
            # Mix of original and new tokens - need to handle both
            else:
                # Create masks to separate new and original tokens
                prompt_mask = input_ids >= self.num_original_tokens
                text_mask = input_ids < self.num_original_tokens

                # Get embeddings for new tokens
                prompt_embd = self.new_embedding(input_ids[prompt_mask] - self.num_original_tokens)
                # Get embeddings for original tokens
                original_embd = self.original_embedding(input_ids[text_mask])

                # Combine embeddings for both token types
                all_embd = torch.zeros((input_ids.size(0), input_ids.size(1),
                        self.original_embedding.weight.size(1)),
                        dtype=original_embd.dtype,
                        device=input_ids.device)
                all_embd[prompt_mask.unsqueeze(-1).repeat(1, 1,
                        self.original_embedding.weight.size(1))] = prompt_embd.flatten()
                all_embd[text_mask.unsqueeze(-1).repeat(1, 1,
                        self.original_embedding.weight.size(1))] = original_embd.flatten()
                return all_embd
        # All tokens are original tokens
        else:
            return self.original_embedding(input_ids)
        


# Custom module for handling output embeddings (language model head) with new tokens
class OutputEmbedding(torch.nn.Module):
    """
    Wrapper around the original output linear layer (language model head) that allows
    producing logits for new tokens while keeping the original vocabulary intact.
    """
    def __init__(self, original_linear, n_new_tokens, initialize_tokens=None):
        """
        Initialize the custom output embedding layer.

        Args:
            original_linear: The original output linear layer from the model
            n_new_tokens: Number of new tokens to add
            initialize_tokens: Optional indices to initialize new token weights
        """
        super(OutputEmbedding, self).__init__()
        self.original_linear = original_linear
        self.n_new_tokens = n_new_tokens

        # Create a separate linear layer for new tokens if needed
        if n_new_tokens > 0:
            # New linear layer outputs logits for n_new_tokens
            self.new_linear = torch.nn.Linear(original_linear.weight.size(1),
                n_new_tokens).to(original_linear.weight.device)

            # Initialize weights based on provided tokens or average of original
            if initialize_tokens is not None:
                new_embeddings = F.embedding(initialize_tokens,
                                             self.original_linear.weight.data)
                self.new_linear.weight.data = new_embeddings
            else:
                # Use mean of original weights as initialization
                self.new_linear.weight.data = original_linear.weight.mean(dim=0,
                    keepdim=True).repeat(n_new_tokens, 1)
        else:
            self.new_linear = None

    def forward(self, inputs):
        """
        Forward pass that produces logits for both original and new tokens.

        Args:
            inputs: Hidden states from the model

        Returns:
            Concatenated logits for all tokens (original + new)
        """
        # Get logits for original vocabulary
        original_token_logits = self.original_linear(inputs)

        # Concatenate with new token logits if they exist
        if self.n_new_tokens > 0:
            new_token_logits = self.new_linear(inputs)
            return torch.cat((original_token_logits, new_token_logits), dim=-1)
        else:
            return original_token_logits
        


# Function to load pre-trained embedding weights from checkpoint files
def load_embeddings(model, input_embedding_file, output_embedding_file,
                    n_tokens, orig_vocab_size):
    """
    Load saved input and output embeddings from checkpoint files into the model.

    Args:
        model: The model to update with loaded embeddings
        input_embedding_file: Path to saved input embeddings
        output_embedding_file: Path to saved output embeddings
        n_tokens: Number of new tokens
        orig_vocab_size: Size of original vocabulary
    """
    # Load and set input embeddings
    assert os.path.isfile(input_embedding_file)
    new_token_embeddings = torch.load(input_embedding_file)
    print(new_token_embeddings)

    try:
        # Handle different embedding sizes
        if new_token_embeddings.weight.size(0) == n_tokens + orig_vocab_size:
            # Full embedding layer (original + new)
            model.set_input_embeddings(new_token_embeddings)
        elif new_token_embeddings.weight.size(0) == n_tokens:
            # Only new token embeddings
            model.set_input_embeddings(InputEmbedding(
                model.get_input_embeddings(), n_tokens))
            model.get_input_embeddings().new_embedding = \
                new_token_embeddings
        else:
            print("new token embeddings size does not match: ",
                    new_token_embeddings.weight.size(0))
            exit(1)
    except:
        # Fallback for tensor instead of embedding module
        assert new_token_embeddings.size(0) == n_tokens + orig_vocab_size
        model.get_input_embeddings().weight.data = new_token_embeddings
    print("input embeddings loaded from file")

    # Load and set output embeddings
    if output_embedding_file is not None:
        assert os.path.isfile(output_embedding_file)
        new_token_embeddings = torch.load(output_embedding_file)

        if new_token_embeddings.weight.size(0) == n_tokens + orig_vocab_size:
            # Full output layer (original + new)
            model.set_output_embeddings(new_token_embeddings)
        elif new_token_embeddings.weight.size(0) == n_tokens:
            # Only new token weights
            model.set_output_embeddings(OutputEmbedding(
                model.get_output_embeddings(), n_tokens))
            model.get_output_embeddings().new_linear = \
                new_token_embeddings
        else:
            print("new token embeddings size does not match: ",
                    new_token_embeddings.weight.size(0))
            exit(1)
        print("output embeddings loaded from file")
    else:
        # Tie input and output embeddings if no separate output file
        model.tie_weights()
    
            

# Function to save new embeddings to checkpoint (used for prompt-tuning mode)
def save_pretrained(
    self,
    save_directory: Union[str, os.PathLike],
    **kwargs,
):
    """
    Save the new token embeddings to disk. This replaces the default save_pretrained
    method when using prompt-tuning mode.

    Args:
        self: Model instance
        save_directory: Directory to save embeddings to
        **kwargs: Additional arguments (unused)
    """
    # Validate and create save directory
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
    os.makedirs(save_directory, exist_ok=True)

    # Save new token embeddings from input embedding layer
    torch.save(self.get_input_embeddings().new_embedding,
                os.path.join(save_directory, "input_embeddings.pt"))
    # Save new token weights from output embedding layer
    torch.save(self.get_output_embeddings().new_linear,
                os.path.join(save_directory, "output_embeddings.pt"))


# Custom wrapper around AutoModelForCausalLM to support new tokens and parameter-efficient fine-tuning
class MyAutoModelForCausalLM(AutoModelForCausalLM):
    """
    Extended version of AutoModelForCausalLM that supports:
    - Adding new prompt tokens to the vocabulary
    - Sparse model variants (SparseLLaMA, SparseGPT2)
    - Parameter-efficient fine-tuning modes (prompt-tuning)
    """

    def __init__(self, n_tokens=0, sparse=False,
                 parameter_efficient_mode=False, **kwargs):
        """
        Initialize the custom causal language model.

        Args:
            n_tokens: Number of new prompt tokens to add
            sparse: Whether to use sparse model variants
            parameter_efficient_mode: Mode for parameter-efficient fine-tuning
            **kwargs: Additional arguments passed to parent class
        """
        self = super().__init__(**kwargs)
        self.n_tokens = n_tokens
        self.sparse = sparse
        self.parameter_efficient_mode = parameter_efficient_mode
        
    @classmethod
    def from_pretrained(cls, n_tokens=0, input_embedding_file=None, output_embedding_file=None,
                        sparse=False, parameter_efficient_mode='none',
                        prompt_tokens=None, initialize_tokens=None, **kwargs):
        """
        Load a pre-trained model with support for new tokens and parameter-efficient fine-tuning.

        Args:
            n_tokens: Number of new prompt tokens to add to vocabulary
            input_embedding_file: Path to saved input embeddings checkpoint
            output_embedding_file: Path to saved output embeddings checkpoint
            sparse: Whether to load sparse model variant
            parameter_efficient_mode: 'none', 'prompt-tuning', or other mode
            prompt_tokens: Prompt token IDs
            initialize_tokens: Token IDs to use for initializing new embeddings
            **kwargs: Arguments passed to model's from_pretrained (including pretrained_model_name_or_path)

        Returns:
            Loaded model with new tokens configured
        """
        # Replace save_pretrained for prompt-tuning mode to only save new embeddings
        if parameter_efficient_mode == 'prompt-tuning':
            PreTrainedModel.save_pretrained = save_pretrained

        # Load base model - either sparse variant or standard AutoModelForCausalLM
        if sparse:
            if 'llama' in kwargs['pretrained_model_name_or_path'] or 'alpaca' in kwargs['pretrained_model_name_or_path']:
                model = SparseLlamaForCausalLM.from_pretrained(**kwargs)
            elif 'gpt2' in kwargs['pretrained_model_name_or_path']:
                model = SparseGPT2LMHeadModel.from_pretrained(**kwargs)
            else:
                raise NotImplementedError
        else:
            model = AutoModelForCausalLM.from_pretrained(**kwargs, trust_remote_code=True)

        # Store model configuration
        model.n_tokens = n_tokens
        model.sparse = sparse
        model.parameter_efficient_mode = parameter_efficient_mode
        model.prompt_tokens = prompt_tokens

        # Handle adding new tokens if requested
        if n_tokens > 0:
            orig_vocab_size = model.get_input_embeddings().weight.size(0)
            print("original vocab size: ", orig_vocab_size)

            # Convert initialize_tokens to tensor if provided
            if initialize_tokens is not None:
                initialize_tokens = torch.tensor(initialize_tokens,
                                    dtype=torch.long, device=model.device)

            # Parameter-efficient mode: wrap embeddings with custom modules
            if parameter_efficient_mode != 'none':
                # Update model config with new vocabulary size
                model.config.vocab_size = orig_vocab_size + n_tokens

                # Load pre-saved embeddings or create new ones
                if input_embedding_file is not None:
                    load_embeddings(model, input_embedding_file, output_embedding_file,
                                    n_tokens, orig_vocab_size)
                else:
                    # Wrap embedding layers with custom classes that support new tokens
                    model.set_input_embeddings(InputEmbedding(
                        model.get_input_embeddings(), n_tokens, initialize_tokens))
                    model.set_output_embeddings(OutputEmbedding(
                        model.get_output_embeddings(), n_tokens, initialize_tokens))

            # Standard mode: resize embeddings and directly set new token values
            elif initialize_tokens is not None:
                # Expand embedding matrices to accommodate new tokens
                model.resize_token_embeddings(orig_vocab_size + n_tokens)
                new_vocab_size = model.get_input_embeddings().weight.size(0)
                assert new_vocab_size == n_tokens + orig_vocab_size

                # Initialize new input embeddings based on provided tokens
                new_embeddings = model.get_input_embeddings()(initialize_tokens)
                model.get_input_embeddings().weight.data[-n_tokens:] = new_embeddings

                # Initialize new output embeddings based on provided tokens
                new_embeddings = F.embedding(initialize_tokens,
                                             model.get_output_embeddings().weight.data)
                model.get_output_embeddings().weight.data[-n_tokens:] = new_embeddings

        return model
    


# Example usage - load a pre-trained LLaMA model with 10 new prompt tokens
if __name__ == "__main__":
    # Load LLaMA-7B model with 10 new prompt tokens, using 8-bit quantization
    model = MyAutoModelForCausalLM.from_pretrained(n_tokens=10,
        pretrained_model_name_or_path="../pretrained_models/llama-7b-hf",
        device_map="auto", load_in_8bit=True,
        offload_folder="offload", offload_state_dict=True)

    print(model)