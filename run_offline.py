import threading
import subprocess

# define your running script
def run_script(script_path):
    try:
        result = subprocess.run(['bash', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"[{script_path}] Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[{script_path}] Error:\n{e.stderr}")

# scripts to be exectuted: add the scripts here
scripts = [
    './scripts/online/inference_gpt4o.sh',
    './scripts/online/inference_gpt4o_cot.sh',
    './scripts/online/inference_o1.sh',
    './scripts/online/inference_claude.sh',
    './scripts/online/inference_claude_thinking.sh',
    './scripts/online/inference_gemini_flash.sh',
    './scripts/online/inference_gemini_pro.sh'
]

# create and launch threads
threads = []
for script in scripts:
    t = threading.Thread(target=run_script, args=(script,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
