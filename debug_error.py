import subprocess
import sys

print("Running scripts.01_demographics...")
try:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.01_demographics"],
        capture_output=True,
        text=True,
        check=True
    )
    print("SUCCESS!")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("FAILED!")
    print("STDOUT:")
    print(e.stdout)
    print("STDERR:")
    print(e.stderr)
    
    with open("error_log.txt", "w") as f:
        f.write(e.stderr)
