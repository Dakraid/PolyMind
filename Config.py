from pathlib import Path
import sys
import os
import json
import importlib.util

mem = {}
vismem = {}
blipcache = {}

script_dir = Path(os.path.abspath(__file__)).parent


class LLMParameters:
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    max_temp: int
    min_temp: int
    max_new_tokens: int
    max_new_tokens_gatekeeper: int
    few_shot: str
    eos: str
    begin_sep: str
    end_sep: str
    sys_begin_sep: str
    sys_end_sep: str

    def __init__(self, temperature: float, top_k: int, top_p: float, min_p: float, repetition_penalty: float,
                 max_temp: int, min_temp: int, max_new_tokens: int, max_new_tokens_gatekeeper: int, few_shot: str,
                 eos: str, begin_sep: str, end_sep: str, sys_begin_sep: str, sys_end_sep: str) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.max_new_tokens = max_new_tokens
        self.max_new_tokens_gatekeeper = max_new_tokens_gatekeeper
        self.few_shot = few_shot
        self.eos = eos
        self.begin_sep = begin_sep
        self.end_sep = end_sep
        self.sys_begin_sep = sys_begin_sep
        self.sys_end_sep = sys_end_sep


class TabbyApi:
    api_key: str
    host: str
    port: int
    api_endpoint_uri: str

    def __init__(self, api_key: str, host: str, port: int) -> None:
        self.api_key = api_key
        if api_key == "your-tabby-api-key" or api_key == "":
            print(
                "\033[93m WARN: You have not set an API key, You probably want to set this if using TabbyAPI. \033[0m")
        self.host = (
            host.rstrip("/")
            if host.endswith("/")
            else host)
        self.port = port
        self.api_endpoint_uri = (
            f"{self.host}/" if self.host.lower().startswith("https") else
            (f"{self.host}:{self.port}/" if self.host.lower().startswith("http") else f"{self.host}:{self.port}/")
        )


class TogetherAi:
    api_key: str
    model: str
    tokenizer_model: str

    def __init__(self, api_key: str, model: str, tokenizer_model: str) -> None:
        self.api_key = api_key
        if api_key == "":
            print("\033[93m ERR: You have not set a TogetherAI API key. Exiting... \033[0m")
            sys.exit()
        self.model = model
        self.tokenizer_model = tokenizer_model


class MistralAi:
    api_key: str
    model: str
    tokenizer_model: str
    use_embed: bool

    def __init__(self, api_key: str, model: str, tokenizer_model: str, use_embed: bool) -> None:
        self.api_key = api_key
        if api_key == "":
            print("\033[93m ERR: You have not set a MistralAI API key. Exiting... \033[0m")
            sys.exit()
        self.model = model
        self.tokenizer_model = tokenizer_model
        self.use_embed = use_embed


class Config:
    def __init__(self):
        with open(os.path.join(script_dir, "config.json")) as config_file:
            loaded_config = json.load(config_file)

        self.backend = loaded_config["backend"]

        if self.backend == "tabbyapi":
            self.backend_config = TabbyApi(loaded_config["backend_config"]["tabbyapi"]["api_key"],
                                           loaded_config["backend_config"]["tabbyapi"]["host"],
                                           loaded_config["backend_config"]["tabbyapi"]["port"])
        elif self.backend == "togetherai":
            self.backend_config = TogetherAi(loaded_config["backend_config"]["togetherai"]["api_key"],
                                             loaded_config["backend_config"]["togetherai"]["model"],
                                             loaded_config["backend_config"]["togetherai"]["tokenizer_model"])
        elif self.backend == "mistralai":
            self.backend_config = MistralAi(loaded_config["backend_config"]["togetherai"]["api_key"],
                                            loaded_config["backend_config"]["togetherai"]["model"],
                                            loaded_config["backend_config"]["togetherai"]["tokenizer_model"],
                                            loaded_config["backend_config"]["togetherai"]["use_embed"])

        self.listen = loaded_config["listen"]
        self.llm_parameters = LLMParameters(loaded_config["LLM_parameters"]["temperature"],
                                            loaded_config["LLM_parameters"]["top_k"],
                                            loaded_config["LLM_parameters"]["top_p"],
                                            loaded_config["LLM_parameters"]["min_p"],
                                            loaded_config["LLM_parameters"]["repetition_penalty"],
                                            loaded_config["LLM_parameters"]["max_temp"],
                                            loaded_config["LLM_parameters"]["min_temp"],
                                            loaded_config["LLM_parameters"]["max_new_tokens"],
                                            loaded_config["LLM_parameters"]["max_new_tokens_gatekeeper"],
                                            loaded_config["LLM_parameters"]["few_shot"],
                                            loaded_config["LLM_parameters"]["eos"],
                                            loaded_config["LLM_parameters"]["begin_sep"],
                                            loaded_config["LLM_parameters"]["end_sep"],
                                            loaded_config["LLM_parameters"]["sys_begin_sep"],
                                            loaded_config["LLM_parameters"]["sys_end_sep"])

        try:
            self.plugins = loaded_config["plugins"]
        except KeyError:
            print("Plugins disabled.")
            self.plugins = []

        self.system = loaded_config['system_prompt']
        self.enabled_features = loaded_config["enabled_features"]
        self.admin_ip = loaded_config["admin_ip"]
        self.ctx_length = loaded_config["max_seq_len"]
        self.reserve_space = loaded_config["reserve_space"]

        try:
            self.compat = loaded_config["compatibility_mode"]
            self.token_model = loaded_config["compat_tokenizer_model"]
        except KeyError:
            self.compat = False
            self.token_model = ""

        self.address = "0.0.0.0" if self.listen else "127.0.0.1"
        print("Loaded config")


values = Config()

loaded_file = {}

if values.enabled_features["file_input"]["enabled"] and "retrieval_count" not in values.enabled_features["file_input"]:
    print("\033[91mERROR: retrieval_count missing from file_input config. Update your config. Exiting... \033[0m")
    sys.exit()

if values.compat:
    if values.token_model == "":
        print("\033[91mERROR: Compatibility_mode is set to true but no tokenizer model is set. Exiting... \033[0m")
        sys.exit()

if values.enabled_features["wolframalpha"]["enabled"]:
    if (values.enabled_features["wolframalpha"]["app_id"] == ""
            or values.enabled_features["wolframalpha"]["app_id"] == "your-wolframalpha-app-id"):
        values.enabled_features["wolframalpha"]["enabled"] = False
        print("\033[93m WARN: Wolfram Alpha has been disabled because no app_id was provided. \033[0m")


def import_plugin(plugin_directory, plugin_name):
    main_path = os.path.join(plugin_directory, plugin_name, 'main.py')
    spec = importlib.util.spec_from_file_location(f"{plugin_name}.main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plugins():
    config_plugins = values.plugins
    plugin_dict = {}
    if len(config_plugins) < 1:
        return [], {}
    manifests = []

    for folder_name in os.listdir(os.path.join(script_dir, 'plugins')):
        if folder_name in config_plugins:
            print(f"loading plugin: {folder_name}")
            manifest_path = os.path.join(script_dir, 'plugins', folder_name, 'manifest.json')
            try:
                with open(manifest_path, 'r') as file:
                    loaded_json = json.load(file)
                    manifests.append(loaded_json)
                    plugin_dict[loaded_json['module_name']] = import_plugin(os.path.join(script_dir, "plugins"),
                                                                            loaded_json['module_name'])
            except FileNotFoundError:
                print(f"Manifest file not found for plugin: {folder_name}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from manifest of plugin: {folder_name}")

    return manifests, plugin_dict


plugin_manifests, loaded_plugins = load_plugins()
