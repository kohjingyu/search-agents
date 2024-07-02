# Tree Search for Language Model Agents

[<a href="https://jykoh.com/search-agents">Website</a>] 
[<a href="http://arxiv.org/abs/2407.01476">Paper</a>] 

![Overview](media/search_overview.gif)

We propose an inference-time tree search algorithm to enable language model agents to perform exploration and multi-step planning in interactive web environments. This repository demonstrates how to run our method on the [VisualWebArena](https://jykoh.com/vwa) and [WebArena](https://webarena.dev/) benchmarks.

## TODOs
- [ ] Add other options besides gpt-4o for the value function

## News
- [06/19/2024]: GitHub repo released.

## Install
```bash
# Python 3.10 or 3.11 recommended
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e .
```

## End-to-end Evaluation on (V)WA
1. Setup the standalone environments.
Please check out [this page](environment_docker/README.md) for details.

2. Configurate the urls for each website.
First, export the `DATASET` to be `visualwebarena`:
```bash
export DATASET=visualwebarena
```
Then, set the URL for the websites

```bash
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export HOMEPAGE="<your_homepage_domain>:4399"
```

If you want to run on the WebArena tasks instead, make sure to also set up the [CMS](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#e-commerce-content-management-system-cms), [GitLab](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#gitlab-website), and [map](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#map) environments, and then set their respective environment variables:
```bash
export DATASET=webarena
export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"
```

3. Generate config files for each test example:
```bash
python scripts/generate_test_data.py
```
You will see `*.json` files generated in the [config_files](./config_files) folder. Each file contains the configuration for one test example.

4. Obtain and save the auto-login cookies for all websites:
```
bash prepare.sh
```

5. Set up API keys.

If using OpenAI models, set a valid OpenAI API key (starting with `sk-`) as the environment variable:
```
export OPENAI_API_KEY=your_key
```

6. Launch the evaluation. For example, to reproduce our GPT-4o + Search agent, you can run the script provided:

```bash
bash scripts/run_vwa_shopping_search.sh
```

This script will run the search agent with the default hyperparams from our paper on the full set of VWA shopping tasks. Note that the baselines that include a captioning model run on GPU by default (e.g., BLIP-2-T5XL as the captioning model will take up approximately 12GB of GPU VRAM). Similarly, the other bash scripts in `scripts/` reproduce the results on the other VWA sites and the text-only WA environment.

By default, the scripts run experiments with the agents with search. If you wish to reproduce the baseline results (without search), set  `--agent_type  prompt` when executing `run.py`.

### Running Llama-3 models

If you wish to run the Llama-3 models we have in our paper, first set up a [vLLM OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). Then, update the `OPENAI_BASE_URL` environment variable in `scripts/run_llama_vwa_shopping_search.sh` to reflect the URL that the vLLM server is running on. This particular script shows how to run the Llama-3 agent on the VWA shopping environment; it is otherwise very similar to the OpenAI scripts for running on the other environments.


## Citation
If you methods or code useful, please consider citing our paper:
```
@article{koh2024tree,
  title={Tree Search for Language Model Agents},
  author={Koh, Jing Yu and McAleer, Stephen and Fried, Daniel and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2407.01476},
  year={2024}
}
```

## Acknowledgements

Our code is heavily based off the <a href="https://github.com/web-arena-x/visualwebarena" target="_blank">VisualWebArena codebase</a> and the <a href="https://github.com/web-arena-x/webarena" target="_blank">WebArena codebase</a>.
