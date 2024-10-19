import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from clica import add_and_init_special_tokens, InteractiveCLI, TransformerAgent, InteractivePythonEnv
from clica.code_env import ENV_SPECIAL_TOKENS
from clica.interface.cli import suppress_cli_warnings


@hydra.main(version_base=None, config_path='../conf', config_name='default')
def run_cli(config: DictConfig):
    suppress_cli_warnings()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    config = OmegaConf.create(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.get('tokenizer_name', config.model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    # Add special tokens
    add_and_init_special_tokens(ENV_SPECIAL_TOKENS, tokenizer, model)
        
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
        
    vocab = tokenizer.get_vocab()
    
    agent = TransformerAgent(model, tokenizer, max_gen_length=128)
    cli = InteractiveCLI(
        agent = agent,
        make_env = InteractivePythonEnv,
        model_save_dir = config.model_save_dir,
        env_kwargs = dict(tokenizer=agent.tokenizer),
    )

    cli.run()


if __name__ == '__main__':
    run_cli()
