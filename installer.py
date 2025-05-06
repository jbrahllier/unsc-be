import subprocess
import os

def run_install_script():
    # define the path to install.sh
    install_script_path = os.path.join(os.path.dirname(__file__), 'install.sh')
    
    try:
        # make sure install.sh is executable
        subprocess.run(['chmod', '+x', install_script_path], check=True)
        
        # run install.sh
        subprocess.run([install_script_path], check=True)
        print("install.sh executed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute install.sh: {e}")
        exit(1)
