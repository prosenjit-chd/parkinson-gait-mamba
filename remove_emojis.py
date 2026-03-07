import os
import re

# Emojis and miscellaneous symbols block
emoji_pattern = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U00002702-\U000027b0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

def remove_emojis_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if there's any emoji to strip
    new_content = emoji_pattern.sub(r'', content)
    
    # Also strip some specific ones we might see like 🧠, 🚀, ⚡, ✅, ⏱️, ❌, 🎨, 📈
    extra_emojis = ['🧠', '🚀', '⚡', '✅', '⏱️', '❌', '🎨', '📈', '✨', '🎉']
    for e in extra_emojis:
        new_content = new_content.replace(e, '')
        
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Removed emojis from: {filepath}")

def main():
    base_dir = r"c:\project_time_series\parkinson_mamba_project"
    
    # Walk through the directory finding .py files
    for root, dirs, files in os.walk(base_dir):
        # Skip virtual environment or hidden directories
        if '.venv' in root or '.git' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                remove_emojis_from_file(filepath)

if __name__ == "__main__":
    main()
