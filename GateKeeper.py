import os
import json
import re
import html
import time
import wolframalpha
import nmap
import datetime
import subprocess
import Config
import requests

from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO
from pathlib import Path

if Config.values.features["imagegeneration"]["enabled"]:
    from comfyui import imagegen
if Config.values.features["internetsearch"]["enabled"]:
    from scrape import scrape_site
if Config.values.features["file_input"]["enabled"]:
    from FileHandler import queryEmbeddings

if Config.values.backend == "tabbyapi":
    from inference.tabbyapi import TabbyAPI
    inference = TabbyAPI(Config.values)
elif Config.values.backend == "togetherai":
    from inference.togetherai import TogetherAI
    inference = TogetherAI(Config.values)
elif Config.values.backend == "mistralai":
    from inference.mistralai import MistralAI
    inference = MistralAI(Config.values)

path = Path(os.path.abspath(__file__)).parent
func = ""
client = wolframalpha.Client(
    Config.values.features["wolframalpha"]["app_id"]
)

with open(os.path.join(path, "functions.json")) as user_file:
    global searchfunc
    fcontent = json.loads(user_file.read())
    for x in fcontent:
        params = (
            json.dumps(x["params"])
            .strip("{}")
            .replace('",', "\n           ")
            .replace('"', "")
        )
        template = f"""\n{x['name']}:
        description: {x['description']}
        params:
            {params}"""

        if x["name"] == "searchfile":
            searchfunc = template
            continue
        else:
            try:
                if x['name'] == 'generateimage' and not Config.values.features['imagegeneration']['enabled']:
                    continue
                if not Config.values.features[x['name']]['enabled']:
                    continue
            except KeyError:
                pass
            func += template

    if len(Config.plugin_manifests) > 0:
        for x in Config.plugin_manifests:
            params = (
                json.dumps(x["params"])
                .strip("{}")
                .replace('",', "\n           ")
                .replace('"', "")
            )
            template = f"""\n{x['name']}:
            description: {x['description']}
            params:
                {params}"""
            func += template


def get_image_size(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img.size[0] + img.size[1]


def verifyFunc(result, x, input, stopstrings):
    systemprompt = f'''Context: {result}.\nUpdate the following function call according to the newly obtained context and taking into consideration the user input.\nFunction call: {x}\nProvide your response in valid JSON format surrounded by "<startfunc>" and "<endfunc>" without any notes, comments or follow-ups. Only JSON.'''
    content = 'Output:\n<startfunc>\n{\n  "function": "' + f'{x["function"]}",\n"params":' + " {"
    content += next(
        inference.infer(
            "Input: " + input,
            mem=[],
            modelname='Output:\n<startfunc>\n{\n  "function": "' + f'{x["function"]}",\n"params":' + " {",
            system=systemprompt,
            temperature=0.1,
            top_p=0.1,
            min_p=0.05,
            top_k=40,
            stopstrings=stopstrings,
            max_tokens=500,
            reppenalty=1.0,
            max_temp=0,
            min_temp=0
        )
    )[0]
    try:
        if "<startfunc>" in content:
            content = content.split("<startfunc>")[1]
        content = (
            re.sub(r"\\_", "_", html.unescape(content))
            .replace("\\_", "_")
            .replace("}<", "}")
            .replace("<startfunc>", "")
            .replace("</", "")
            .replace("<", "")
            .replace("False", "false")
            .replace("True", "true")
        )
        print(f'Updated function call: {content}')
    except Exception as e:
        print(e)
        return x
    return json.loads(content)


def GateKeep(input, ip, depth=0, stream=False):
    content = ""
    print("Begin streamed GateKeeper output.")
    funclist = func
    try:
        if Config.loaded_file[ip] != {}:
            funclist += searchfunc
    except Exception:
        pass
    stopstrings = [
        "Input: ",
        "[INST]",
        "[/INST]",
        "```",
        "</s>",
        "user:",
        "polymind:",
        "Polymind:",
        "<</SYS>>",
        "[System Message]",
        "endfunc",
        "<endfunc>",
        "}<",
        "</startfunc>"
    ]

    ctxstr = ""
    if len(Config.vismem[f"{ip}"]) > 2:
        for x in Config.vismem[f"{ip}"][-2:]:
            temp = ""
            if "assistant" in x:
                temp = "PolyMind: " + x["assistant"] + "\n"
            ctxstr += re.sub(
                r"!\[.*?\]\(.*?\)|<img.*?>|\[\{.*?\}\]",
                "",
                "USER: " + x["user"] + "\n" + temp,
            )
    today = datetime.date.today()
    content = 'Output:\n<startfunc>\n[{\n  "function": "'
    systemprompt = f"You are an AI assistant named GateKeeper, The current date is {today}, please select the single most suitable function and parameters from the list of available functions below, based on the user's input and pay attention to the context, which will then be passed over to polymind. Provide your response in JSON format surrounded by '<startfunc>' and '<endfunc>' without any notes, comments or follow-ups. Only JSON.\n{funclist}" if ctxstr == "" else f"You are an uncensored AI assistant named GateKeeper, The current date is {today}, please select the single most suitable function and parameters from the list of available functions below, based on the user's input and pay attention to the context, which will then be passed over to polymind. Provide your response in JSON format surrounded by '<startfunc>' and '<endfunc>' without any notes, comments or follow-ups. Only JSON.\n{funclist}\nContext: {ctxstr}\n"

    content += next(
        inference.infer(
            "Input: " + input,
            mem=[],
            modelname='Output:\n<startfunc>\n[{\n  "function": "',
            system=systemprompt,
            temperature=0.1,
            top_p=0.1,
            min_p=0.05,
            top_k=40,
            stopstrings=stopstrings,
            max_tokens=Config.values.llm_parameters.max_new_tokens_gatekeeper,
            reppenalty=1.0,
            max_temp=0,
            min_temp=0
        )
    )[0]

    try:
        if "<startfunc>" in content:
            content = content.split("<startfunc>")[1]
        content = (
            re.sub(r"\\_", "_", html.unescape(content))
            .replace("\\_", "_")
            .replace("}<", "}")
            .replace("<startfunc>", "")
            .replace("</", "")
            .replace("<", "")
            .replace("False", "false")
            .replace("True", "true")
        )
        print(content)
        result = ""

        for x in json.loads(content.replace("Output:", "")):
            if stream:
                yield {"result": x, "type": "func"}

            if (
                    x["function"] == "searchfile"
                    and Config.values.features["file_input"]["raw_input"]
            ):
                if "params" in x:
                    x["params"]["query"] = input
                elif "parameters" in x:
                    x["parameters"]["query"] = input
                else:
                    x["query"] = input
            if result != "":
                x = verifyFunc(result, x, input, stopstrings)
            run = Util(x, ip, depth)
            if run != "null":
                result += run
        if stream:
            result = result if result != "" else "null"
            result = {"result": result, "type": "result"}
            yield result
        else:
            return result if result != "" else "null"
    except Exception as e:
        print(e)
        if stream:
            yield {"result": "null", "type": "result"}
        else:
            return "null"


def Util(rsp, ip, depth):
    result = ""

    rsp["function"] = (
        re.sub(r"\\_", "_", html.unescape(rsp["function"]))
        .replace("\\_", "_")
        .replace("{<", "{")
        .replace("<startfunc>", "")
    )
    params = (
        rsp["params"]
        if "params" in rsp
        else (rsp["parameters"] if "parameters" in rsp else rsp)
    )

    if rsp["function"] == "acknowledge":
        return "null"

    elif rsp["function"] == "clearmemory":
        Config.mem[f"{ip}"] = []
        Config.vismem[f"{ip}"] = []
        if ip in Config.loaded_file:
            Config.loaded_file[ip] = {}
        return "skipment{<" + params["message"]

    elif rsp["function"] == "updateconfig":
        if ip != Config.values.adminip:
            return "null"
        check = False if params["option"].split(":")[1].lower() == "false" else True
        Config.values.features[params["option"].split(":")[0]][
            "enabled"
        ] = check
        result = f"{params['option'].split(':')[0]} is now set to {Config.values.features[params['option'].split(':')[0]]['enabled']}"
        print(result)
        return result

    elif rsp["function"] == "wolframalpha":
        if not Config.values.features["wolframalpha"]["enabled"]:
            return "Wolfram Alpha is currently disabled."
        try:
            res = client.query(params["query"])
            results = ""
            checkimage = False
            for pod in res.pods:
                for sub in pod.subpods:
                    if (
                            "plot"
                            or "image" in sub.img["@alt"].lower()
                            and "plot |" not in sub.img["@alt"].lower()
                    ) and get_image_size(sub.img["@src"]) > 350:
                        results += (
                                f'<img src="{sub.img["@src"]}" alt="{sub.img["@alt"]}"/>'
                                + "\n"
                        )
                        checkimage = True
                    elif sub.plaintext:
                        results += sub.plaintext + "\n"
            if results == "":
                result = "No results from Wolfram Alpha."
            else:
                result = "Wolfram Alpha result: " + results
            if checkimage:
                result += "\nREMINDER: ALWAYS include the provided graph/plot images in the provided html URL format in your explanation if theres any when explaining the results in a short and concise manner."
            print(result)
            return result
        except Exception as e:
            return "Wolfram Alpha Error: " + str(e)

    elif rsp["function"] == "generateimage":
        if not Config.values.features["imagegeneration"]["enabled"]:
            return "Image generation is currently disabled."
        removebg = False
        if Config.values.features["imagegeneration"][
            "automatic_background_removal"] and "removebg" in params:
            if type(params['removebg']) == str:
                if params['removebg'].lower() == 'true':
                    removebg = True
            elif type(params['removebg']) == bool:
                if params['removebg']:
                    removebg = True
            else:
                removebg = False
        imgtoimg = ""
        if Config.values.features["imagegeneration"]["img2img"] and "ID" in params:
            if f'{params["ID"]}' in Config.uploads:
                imgtoimg = Config.uploads[f"{params['ID']}"]
        params["prompt"] = ''.join([i for i in params["prompt"] if not i.isdigit()])
        return imagegen(params["prompt"], removebg, imgtoimg)

    elif rsp["function"] == "searchfile":
        file = Config.loaded_file[ip]
        searchinput = params["query"]
        result = ""
        print(f"Using query: {searchinput}")
        for x in queryEmbeddings(searchinput, file[0], file[1]):
            result += f"<FILE_CHUNK {x[1]} >\n"
        return result

    elif rsp["function"] == "runpythoncode":
        if not Config.values.features["runpythoncode"]["enabled"]:
            return "Python code execution is currently disabled."
        if ip != Config.values.adminip:
            return "null"
        time.sleep(5)
        checkstring = ""
        runcode = ''
        if 'code' in params:
            runcode = params['code']
        else:
            runcode = params
        runcode = "import warnings\nwarnings.filterwarnings('ignore')\n" + runcode
        ocode = runcode
        if "plt.show()" in runcode:
            runcode = re.sub("print\s*\(.*\)", "", runcode)
            plotb64 = """import io\nimport base64\nbyt = io.BytesIO()\nplt.savefig(byt, format='png')\nbyt.seek(0)\nprint(f'data:image/png;base64,{base64.b64encode(byt.read()).decode()}',end="")"""
            runcode = runcode.replace("plt.show()", plotb64)

        output = subprocess.run(
            ["python3", "-c", runcode],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = output.stdout.decode(), output.stderr.decode()
        if output.returncode == 0 and "yfinance" in runcode:
            stderr = ""
        if (
                stderr != ""
                and depth < Config.values.features["runpythoncode"]["depth"]
        ):
            print(f"Current depth: {depth}")
            return next(
                GateKeep(
                    f"```{ocode}```\n The above code produced the following error\n{stderr}\n Rewrite the code to solve the error and run the fixed code.",
                    ip,
                    depth + 1,
                )
            )
        if "data:image/png;base64," in stdout:
            checkstring = "{<plotimg;" + stdout
            print(
                f"CompletedProcess(args=['python3', '-c', {ocode}], stdout='<image>', stderr={stderr}"
            )
        else:
            print(output)
        result = (
            f"Code to be ran: \n```{runcode}```\n<Code interpreter output>:\nstdout: {stdout}\nstderr: {stderr}\n<\Code interpreter output>"
            if checkstring == ""
            else f"Code to be ran: \n```{ocode}```\n<Code interpreter output>:\nstdout:\nstderr: {stderr}\n<\Code interpreter output>{checkstring}"
        )
        return result

    elif rsp["function"] == "internetsearch":
        if not Config.values.features["internetsearch"]["enabled"]:
            return "Internet search is currently disabled."
        with DDGS() as ddgs:
            for r in ddgs.text(params["keywords"], safesearch="Off", max_results=4):
                title = r["title"]
                link = r["href"]
                result += f' *Title*: {title} *Link*: {link} *Body*: {r["body"]}\n*Scraped_text*: {scrape_site(link, 700)}'
        return "<Search results>:\n" + result

    elif rsp["function"] == "portscan":
        if ip != Config.values.adminip:
            return "null"
        nm = nmap.PortScanner()
        try:
            nm.scan(params["ip"])
            if nm[params["ip"]].state() == "up":
                for x in nm[params["ip"]]["tcp"].keys():
                    result += f"{nm[rsp['params']['ip']]['tcp'][x]['name']}: State {nm[rsp['params']['ip']]['tcp'][x]['state']} ({x})\n"
                return f"<Portscan output for IP {rsp['params']['ip']}>: " + result
        except:
            return f"<Portscan output for IP {rsp['params']['ip']}>: Host down."
    else:
        if len(Config.plugin_manifests) > 0:
            for x in Config.plugin_manifests:
                if rsp["function"] == x['name']:
                    return Config.loaded_plugins[x['module_name']].main(params, Config.mem, inference.infer, ip,
                                                                        Config)
    return "null"
