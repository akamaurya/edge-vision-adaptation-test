"""Can a VLM run on my laptop, and how fast? Times a local VLM through Ollama on your own frames.

"decide" = answer Yes/No, 1 output token (pick an answer, don't write text)
"talk"   = write a 6-field JSON answer, 60 output tokens (the naive way)

Setup (once):  install Ollama from ollama.com, open it, then in Terminal:
    ollama pull qwen3.5:0.8b
    ollama pull qwen3.5:2b
Run (frames from the Colab zip, e.g. ~/Downloads/edge_vision_results_real/frames):
    python3 mac_vlm_speed.py ~/Downloads/edge_vision_results_real/frames/*.jpg
Standard library only. Close Chrome first, 8 GB of RAM is tight.
"""
import base64, json, platform, statistics, subprocess, sys, urllib.request

MODELS = ["qwen3.5:0.8b", "qwen3.5:2b"]
DECIDE = "This photo shows a metal tray on a table, seen from a fixed camera. Is the tray empty? Answer Yes or No."   # same question as the notebook
TALK = "Describe the tray as JSON with keys part_on_tray, hand_visible, tray_visible, lighting, object_type, confidence."


def ask(model, prompt, img, tokens):
    body = {"model": model, "prompt": prompt, "images": [img], "stream": False, "think": False,
            "options": {"num_predict": tokens, "temperature": 0}}
    req = urllib.request.Request("http://localhost:11434/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["total_duration"] / 1e9, r.get("eval_count", 0), r["response"].strip().replace("\n", " ")


paths = sys.argv[1:]
if not paths:
    sys.exit(__doc__)
imgs = [base64.b64encode(open(p, "rb").read()).decode() for p in paths]
try:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
except OSError:
    chip = platform.machine()
print(f"Machine: {chip} | {len(paths)} frames\n")

for m in MODELS:
    try:
        ask(m, DECIDE, imgs[0], 1)                                   # warm-up, loads the model into memory
    except Exception as ex:
        print(f"{m}: skipped ({ex}). Is Ollama open, and did you 'ollama pull {m}'?\n")
        continue
    d = [ask(m, DECIDE, i, 1) for i in imgs]
    t = [ask(m, TALK, i, 60) for i in imgs[:3]]
    print(f"{m}: decide {statistics.median(x[0] for x in d):.2f} s/frame | "
          f"talk {statistics.median(x[0] for x in t):.2f} s/frame ({t[0][1]} tokens)")
    for p, x in zip(paths, d):
        print(f"   {p.split('/')[-1]:>18}  ->  {x[2]}")
    print(f"   talk example: {t[0][2][:160]}\n")
print("Need on the Orin: 10 cameras x 0.5 FPS = 5 frames/s, i.e. 0.2 s per frame for the whole box.")
